#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAFL CIFAR-10 experiment suite.

Adds:
- Attacks: original static attacks plus adaptive norm-evasion, staleness-exploit, collusion, ALIE, and joint adaptive attacks.
- Aggregators/baselines: FedAsync, FedBuff-style buffered mean, mean, median,
  trimmed mean, Krum, ASB/RAFL, AFLGuard-style norm screening, Zeno++-style
  validation-loss screening.
- Sensitivity sweeps: staleness coefficient, ET threshold, Byzantine ratio,
  max delay, clients, communication savings.
- Metrics: test accuracy/loss, suppression rate, transmitted updates,
  communication reduction, staleness, rho proxy, nu2 proxy, wall-clock.

Examples:
    python rafl_cifar10_experiment_suite.py --suite smoke
    python rafl_cifar10_experiment_suite.py --suite attacks
    python rafl_cifar10_experiment_suite.py --suite baselines
    python rafl_cifar10_experiment_suite.py --suite sensitivity
    python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine
    
    # Communication-efficiency sensitivity experiments
    python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 0.5

    python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 1.0

    python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 2.0

    python rafl_cifar10_experiment_suite.py --suite single --aggregator asb --attack signflip --enable_byzantine --trigger_eps 5.0
"""

import os
import time
import json
import copy
import random
import argparse
import multiprocessing as mp
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Callable, Any

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
import asyncio
import nest_asyncio
nest_asyncio.apply()


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + self.short(x)
        return F.relu(y)


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_ch = 32
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_ch, n, stride):
        layers = [block(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, n):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return F.log_softmax(y, dim=1)


def make_model():
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=10)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def create_directory(config: Dict[str, Any], base_dir="results_rafl_cifar10_suite"):
    byz_count_tag = config.get("byz_count_override", None)
    if byz_count_tag is None:
        byz_count_tag = "auto"
    name = (
        f"{config['suite']}_agg_{config['aggregator']}_attack_{config['attack']}_"
        f"C_{config['num_clients']}_J_{config['num_clients_per_round']}_"
        f"R_{config['num_rounds']}_alpha_{config['alpha_stale']}_"
        f"eps_{config['trigger_eps']}_byz_{config['byz_frac']}_B_{byz_count_tag}_"
        f"tau_{config['tau_max_rounds']}_seed_{config['seed']}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    dn = os.path.join(base_dir, name)
    os.makedirs(dn, exist_ok=True)
    return dn


def plot_series(y, title, save_path, xlabel="Rounds", ylabel="Value"):
    if y is None or len(y) == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_xy(x, y, title, save_path, xlabel="x", ylabel="y"):
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def npar(x):
    return np.array(x, dtype=np.float32)


def clone_state_dict_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def dict_subtract(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: (a[k] - b[k])
        for k in a.keys()
        if k in b and a[k].dtype.is_floating_point and b[k].dtype.is_floating_point
    }


def dict_to_vec(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [p.detach().flatten().float().cpu() for p in state.values() if p.dtype.is_floating_point]
    if len(parts) == 0:
        return torch.zeros(1)
    return torch.cat(parts)


def vec_to_update(vec: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    idx = 0
    for k, v in template.items():
        if not v.dtype.is_floating_point:
            continue
        n = v.numel()
        out[k] = vec[idx:idx + n].view_as(v).cpu().float()
        idx += n
    return out


def l2_norm_of_update(update: Dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        return float(torch.norm(dict_to_vec(update), p=2).item())


def add_update_to_model(server_model: nn.Module, update: Dict[str, torch.Tensor], gamma_t: float):
    with torch.no_grad():
        cur_sd = server_model.state_dict()
        for k, upd in update.items():
            if k not in cur_sd or not cur_sd[k].dtype.is_floating_point:
                continue
            cur_sd[k].add_(gamma_t * upd.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype))
        server_model.load_state_dict(cur_sd, strict=False)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def partition_non_iid(dataset, num_clients, alpha=0.5, num_classes=10):
    data_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        data_by_class[int(label)].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        cls_idxs = data_by_class[c]
        np.random.shuffle(cls_idxs)
        props = np.random.dirichlet([alpha] * num_clients)
        props = (props * len(cls_idxs)).astype(int)

        start = 0
        for i, p in enumerate(props):
            end = start + p
            client_indices[i].extend(cls_idxs[start:end])
            start = end

        for i, idx in enumerate(cls_idxs[start:]):
            client_indices[i % num_clients].append(idx)

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        if len(client_indices[i]) == 0:
            client_indices[i].append(random.randrange(len(dataset)))
    return client_indices


def build_client_loaders(
    dataset,
    client_indices: List[List[int]],
    batch_size: int,
    val_size_per_client: int = 128,
    pin_memory: bool = False,
):
    train_loaders, val_loaders = [], []
    for idxs in client_indices:
        if len(idxs) <= val_size_per_client:
            vsize = max(1, len(idxs) // 2)
        else:
            vsize = val_size_per_client

        val_ids = idxs[-vsize:]
        train_ids = idxs[:-vsize] if len(idxs) > vsize else idxs

        train_loaders.append(DataLoader(
            Subset(dataset, train_ids),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=False,
        ))
        val_loaders.append(DataLoader(
            Subset(dataset, val_ids),
            batch_size=min(128, max(1, vsize)),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        ))
    return train_loaders, val_loaders


@torch.no_grad()
def get_one_batch(loader: DataLoader, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    for x, y in loader:
        return x.to(device), y.to(device)
    return None


def flip_labels(y, num_classes=10):
    return (num_classes - 1) - y


# ---------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------
def byzantine_corrupt_update(
    update: Dict[str, torch.Tensor],
    mode="signflip",
    scale=10.0,
    benign_mean: Optional[Dict[str, torch.Tensor]] = None,
):
    bad = {}
    for k, v in update.items():
        if not v.dtype.is_floating_point:
            continue
        t = v.detach().cpu().float()

        if mode == "signflip":
            bad[k] = -float(scale) * t
        elif mode == "gaussian":
            bad[k] = t + float(scale) * torch.randn_like(t)
        elif mode == "model_replacement":
            # Common model-replacement style: amplify the malicious local update.
            bad[k] = float(scale) * t
        elif mode == "zero":
            bad[k] = torch.zeros_like(t)
        elif mode == "random":
            bad[k] = float(scale) * torch.randn_like(t)
        elif mode == "mean_shift" and benign_mean is not None and k in benign_mean:
            bad[k] = benign_mean[k].float() + float(scale) * torch.sign(benign_mean[k].float())
        else:
            bad[k] = t
    return bad


def zero_like_update(template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: torch.zeros_like(v.detach().cpu().float())
        for k, v in template.items()
        if v.dtype.is_floating_point
    }


def scale_update_to_norm(update: Dict[str, torch.Tensor], target_norm: float) -> Dict[str, torch.Tensor]:
    """Rescale an update to an exact global l2 norm, preserving direction."""
    target_norm = max(0.0, float(target_norm))
    cur = l2_norm_of_update(update)
    if cur <= 1e-12:
        return zero_like_update(update)
    fac = target_norm / cur
    return {k: fac * v.detach().cpu().float() for k, v in update.items() if v.dtype.is_floating_point}


def negate_update(update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: -v.detach().cpu().float() for k, v in update.items() if v.dtype.is_floating_point}


def cosine_between_updates(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    va, vb = dict_to_vec(a), dict_to_vec(b)
    na = float(torch.norm(va, p=2).item())
    nb = float(torch.norm(vb, p=2).item())
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(torch.dot(va, vb).item() / (na * nb))


def coordinate_alie_update(
    honest_updates: List[Dict[str, torch.Tensor]],
    z: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """
    ALIE-style omniscient attack: for each coordinate use mu - z * sigma.
    Computed tensor-by-tensor to avoid flattening all client models at once.
    """
    if len(honest_updates) == 0:
        return {}
    keys = [k for k, v in honest_updates[0].items() if v.dtype.is_floating_point]
    out = {}
    for k in keys:
        X = torch.stack([u[k].detach().cpu().float() for u in honest_updates], dim=0)
        mu = X.mean(dim=0)
        sigma = X.std(dim=0, unbiased=False)
        out[k] = mu - float(z) * sigma
    return out


def craft_round_adaptive_attacks(
    updates_J: List[Dict[str, torch.Tensor]],
    selected_clients: List[int],
    byz_clients: set,
    attack: str,
    byz_scale: float,
    trigger_eps: float,
    adaptive_norm_quantile: float,
    adaptive_norm_ratio: float,
    adaptive_z: float,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, float]]:
    """
    Apply update-space Byzantine attacks *before* event-triggering.

    Adaptive attacks use the honest updates in the same round as an omniscient
    reference. This is deliberately stronger than the paper's original static
    attacks and directly stress-tests the ASB largest-norm filter.

    - norm_evasion: opposite-to-honest-mean direction, projected below a chosen
      honest norm quantile so malicious updates can evade largest-norm trimming.
    - collusion: all Byzantine clients send the same coordinated opposite update.
    - staleness_exploit: update-space part is norm-evasion; the federated loop also
      forces Byzantine arrivals to tau=0, maximizing their staleness-aware weight.
    - alie: coordinate-wise mu - z sigma attack from honest round updates.
    - adaptive_joint: combines norm evasion, coordination, fresh-arrival weight
      exploitation, and trigger awareness.
    """
    attacked = [copy.deepcopy(u) for u in updates_J]
    if len(byz_clients) == 0 or attack in ["none", "label_flip"]:
        return attacked, {
            "honest_norm_reference": 0.0,
            "crafted_target_norm": 0.0,
            "attack_cosine_with_honest_mean": 0.0,
        }

    honest_idx = [j for j, cid in enumerate(selected_clients) if cid not in byz_clients]
    byz_idx = [j for j, cid in enumerate(selected_clients) if cid in byz_clients]
    if len(honest_idx) == 0:
        # Failure-boundary sweeps normally retain >=1 honest client. This fallback
        # keeps the simulator defined even in the all-Byzantine corner case.
        honest_idx = list(range(len(updates_J)))

    honest_updates = [updates_J[j] for j in honest_idx]
    honest_mean = weighted_mean_updates(
        honest_updates, np.ones(len(honest_updates), dtype=np.float32) / max(1, len(honest_updates))
    )
    honest_norms = np.array([l2_norm_of_update(u) for u in honest_updates], dtype=np.float64)
    q = float(np.clip(adaptive_norm_quantile, 0.0, 1.0))
    norm_ref = float(np.quantile(honest_norms, q)) if honest_norms.size else 0.0
    target_norm = max(0.0, float(adaptive_norm_ratio) * norm_ref)

    # If communication is event-triggered, an adaptive attacker can choose a norm
    # just above epsilon. When epsilon exceeds the norm-evasion target this is no
    # longer a stealthy norm attack, and the diagnostics reveal the larger norm.
    trigger_target = float(trigger_eps) * 1.001 if trigger_eps > 0 else 0.0

    if attack in ["norm_evasion", "staleness_exploit", "adaptive_joint"]:
        poison = negate_update(honest_mean)
        if not poison:
            poison = copy.deepcopy(updates_J[byz_idx[0]])
        final_target = max(target_norm, trigger_target)
        poison = scale_update_to_norm(poison, final_target)
        for j in byz_idx:
            attacked[j] = copy.deepcopy(poison)

    elif attack == "collusion":
        poison = negate_update(honest_mean)
        if not poison:
            poison = copy.deepcopy(updates_J[byz_idx[0]])
        # Coordinated attack is intentionally not norm-capped; it tests whether
        # robust filters reject a coherent high-amplitude malicious direction.
        base_norm = max(norm_ref, 1e-12)
        final_target = max(float(byz_scale) * base_norm, trigger_target)
        poison = scale_update_to_norm(poison, final_target)
        for j in byz_idx:
            attacked[j] = copy.deepcopy(poison)

    elif attack == "alie":
        poison = coordinate_alie_update(honest_updates, z=adaptive_z)
        # Keep ALIE adaptive to the event trigger but otherwise preserve its
        # coordinate-wise construction.
        if trigger_target > 0 and l2_norm_of_update(poison) < trigger_target:
            poison = scale_update_to_norm(poison, trigger_target)
        for j in byz_idx:
            attacked[j] = copy.deepcopy(poison)

    else:
        # Existing static update-space attacks are preserved.
        benign_mean = honest_mean
        for j in byz_idx:
            attacked[j] = byzantine_corrupt_update(
                updates_J[j], mode=attack, scale=byz_scale, benign_mean=benign_mean
            )

    byz_updates_after = [attacked[j] for j in byz_idx]
    byz_mean = weighted_mean_updates(
        byz_updates_after,
        np.ones(len(byz_updates_after), dtype=np.float32) / max(1, len(byz_updates_after)),
    ) if byz_updates_after else {}

    return attacked, {
        "honest_norm_reference": norm_ref,
        "crafted_target_norm": target_norm,
        "attack_cosine_with_honest_mean": cosine_between_updates(byz_mean, honest_mean) if byz_mean else 0.0,
    }


# ---------------------------------------------------------------------
# Client training
# ---------------------------------------------------------------------
def _train_client_sync(
    init_state_cpu: Dict[str, torch.Tensor],
    model_ctor: Callable[[], nn.Module],
    train_loader: DataLoader,
    device_str: str,
    local_epochs: int,
    loss_fn,
    eta_client: float,
    accumulation_steps: int,
    early_stopping_patience: int,
    attack_mode: str = "none",
    num_classes: int = 10,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[float], int, float]:
    device = torch.device(device_str)
    model = model_ctor().to(device)
    model.load_state_dict(init_state_cpu, strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=eta_client)
    client_losses, best, patience = [], float("inf"), 0

    start = time.time()
    for epoch in range(local_epochs):
        model.train()
        epoch_loss = 0.0
        pending_step = False

        for b, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if attack_mode == "label_flip":
                y = flip_labels(y, num_classes=num_classes)

            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            pending_step = True

            if (b + 1) % max(1, accumulation_steps) == 0:
                opt.step()
                pending_step = False

            epoch_loss += float(loss.item())

        if pending_step:
            opt.step()

        avg = epoch_loss / max(1, len(train_loader))
        client_losses.append(avg)

        if avg < best - 1e-9:
            best, patience = avg, 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    wall = time.time() - start
    local_state_cpu = clone_state_dict_cpu(model.state_dict())
    delta_cpu = dict_subtract(local_state_cpu, init_state_cpu)
    return local_state_cpu, delta_cpu, client_losses, len(train_loader.dataset), wall


# ---------------------------------------------------------------------
# Aggregators / baselines
# ---------------------------------------------------------------------
def normalized_stale_weights(tau_list: List[int], alpha_stale: float, w_min: float, w_max: float) -> np.ndarray:
    J = len(tau_list)
    raw = np.array([1.0 / (1.0 + alpha_stale * max(0.0, float(tau))) for tau in tau_list], dtype=np.float32)
    raw_sum = raw.sum()
    if raw_sum <= 0:
        raw = np.ones_like(raw)
        raw_sum = raw.sum()

    wtilde = raw / raw_sum
    wtilde = np.clip(wtilde, w_min / max(1, J), w_max / max(1, J))
    wtilde = wtilde / max(1e-12, wtilde.sum())
    return wtilde


def weighted_mean_updates(updates: List[Dict[str, torch.Tensor]], weights: np.ndarray) -> Dict[str, torch.Tensor]:
    if len(updates) == 0:
        return {}
    all_keys = [k for k, v in updates[0].items() if v.dtype.is_floating_point]
    agg = {k: torch.zeros_like(updates[0][k].float().cpu()) for k in all_keys}
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / max(1e-12, weights.sum())
    for u, w in zip(updates, weights):
        for k in all_keys:
            agg[k] += float(w) * u[k].float().cpu()
    return agg


def median_aggregate(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(updates) == 0:
        return {}
    keys = [k for k, v in updates[0].items() if v.dtype.is_floating_point]
    return {
        k: torch.median(torch.stack([u[k].float().cpu() for u in updates], dim=0), dim=0).values
        for k in keys
    }


def trimmed_mean_aggregate(updates: List[Dict[str, torch.Tensor]], trim_ratio=0.2) -> Dict[str, torch.Tensor]:
    if len(updates) == 0:
        return {}
    keys = [k for k, v in updates[0].items() if v.dtype.is_floating_point]
    out = {}
    for k in keys:
        X = torch.stack([u[k].float().cpu() for u in updates], dim=0)
        n = X.shape[0]
        b = int(trim_ratio * n)
        if b <= 0 or n <= 2 * b:
            out[k] = X.mean(dim=0)
        else:
            Xs, _ = torch.sort(X, dim=0)
            out[k] = Xs[b:n - b].mean(dim=0)
    return out


def krum_select_index(updates: List[Dict[str, torch.Tensor]], f=1) -> int:
    if len(updates) == 0:
        return -1
    if len(updates) <= 2:
        return 0

    vecs = [dict_to_vec(u) for u in updates]
    n = len(vecs)
    f = min(int(f), max(0, n - 3))
    m = max(1, n - f - 2)

    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dists.append(float(torch.norm(vecs[i] - vecs[j], p=2).item() ** 2))
        dists.sort()
        scores.append(sum(dists[:m]))
    return int(np.argmin(scores))


def krum_aggregate(updates: List[Dict[str, torch.Tensor]], f=1) -> Dict[str, torch.Tensor]:
    idx = krum_select_index(updates, f=f)
    if idx < 0:
        return {}
    return copy.deepcopy(updates[idx])


def asb_aggregate(
    updates_S: List[Dict[str, torch.Tensor]],
    taus_J: List[int],
    indices_S_in_J: List[int],
    alpha_stale: float,
    w_min: float,
    w_max: float,
    trim_B: int = 0,
) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, Any]]:
    if len(updates_S) == 0:
        return {}, 0.0, {"kept_indices": [], "dropped_indices": [], "trim_B": 0}

    wtilde_J = normalized_stale_weights(taus_J, alpha_stale, w_min, w_max)
    w_S = np.array([wtilde_J[j_idx] for j_idx in indices_S_in_J], dtype=np.float32)
    w_S = w_S / max(1e-12, w_S.sum())

    mean_before = weighted_mean_updates(updates_S, w_S)

    trim_B = int(max(0, min(trim_B, max(0, len(updates_S) - 1))))
    if trim_B > 0:
        norms = [l2_norm_of_update(u) for u in updates_S]
        order = np.argsort(norms)[::-1]
        drop = set(order[:trim_B].tolist())
        kept_indices = [i for i in range(len(updates_S)) if i not in drop]
        dropped_indices = sorted(drop)
        kept_updates = [updates_S[i] for i in kept_indices]
        kept_weights = np.array([w_S[i] for i in kept_indices], dtype=np.float32)
        kept_weights = kept_weights / max(1e-12, kept_weights.sum())
    else:
        kept_indices = list(range(len(updates_S)))
        dropped_indices = []
        kept_updates = updates_S
        kept_weights = w_S

    mean_after = weighted_mean_updates(kept_updates, kept_weights)

    keys = list(mean_after.keys())
    if not keys:
        return {}, 0.0, {
            "kept_indices": kept_indices,
            "dropped_indices": dropped_indices,
            "trim_B": trim_B,
        }

    vec_diff = dict_to_vec({k: mean_before[k] - mean_after[k] for k in keys})
    rho_hat_sq = float(torch.dot(vec_diff, vec_diff).item())
    return mean_after, rho_hat_sq, {
        "kept_indices": kept_indices,
        "dropped_indices": dropped_indices,
        "trim_B": trim_B,
    }


def aflguard_screen(updates: List[Dict[str, torch.Tensor]], tau_list: List[int], tau_cutoff: int, norm_factor=2.5):
    """
    Lightweight AFLGuard-style screen:
    keep fresh-enough updates and reject extreme update norms.
    This is not the official AFLGuard implementation, but is useful as a transparent
    screening baseline for reviewer experiments.
    """
    if len(updates) == 0:
        return [], []

    norms = np.array([l2_norm_of_update(u) for u in updates], dtype=np.float32)
    med = float(np.median(norms))
    mad = float(np.median(np.abs(norms - med))) + 1e-12
    cutoff = med + norm_factor * 1.4826 * mad

    kept_updates, kept_indices = [], []
    for i, (u, tau) in enumerate(zip(updates, tau_list)):
        if tau <= tau_cutoff and norms[i] <= cutoff:
            kept_updates.append(u)
            kept_indices.append(i)

    if len(kept_updates) == 0:
        # Avoid empty aggregation by keeping the smallest norm update.
        j = int(np.argmin(norms))
        kept_updates = [updates[j]]
        kept_indices = [j]

    return kept_updates, kept_indices


def evaluate_loss_from_state(
    model_ctor: Callable[[], nn.Module],
    state_cpu: Dict[str, torch.Tensor],
    val_loader: DataLoader,
    device: torch.device,
    loss_fn,
    max_batches: int = 2,
) -> float:
    model = model_ctor().to(device)
    model.load_state_dict(state_cpu, strict=False)
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(val_loader):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
    return total_loss / max(1, total)


def zeno_screen(
    updates: List[Dict[str, torch.Tensor]],
    server_state_cpu: Dict[str, torch.Tensor],
    model_ctor: Callable[[], nn.Module],
    val_loader: DataLoader,
    device: torch.device,
    loss_fn,
    gamma_t: float,
    keep_ratio: float = 0.7,
):
    """
    Zeno++-style validation-loss screening:
    score each update by validation loss after a tentative server step.
    Keep the best keep_ratio updates, then aggregate them.

    This is a practical proxy/surrogate, not a verbatim Zeno++ implementation.
    """
    if len(updates) == 0:
        return [], []

    scored = []
    for i, u in enumerate(updates):
        candidate = clone_state_dict_cpu(server_state_cpu)
        for k, upd in u.items():
            if k in candidate and candidate[k].dtype.is_floating_point:
                candidate[k] = candidate[k].float() + gamma_t * upd.float()
        loss_val = evaluate_loss_from_state(model_ctor, candidate, val_loader, device, loss_fn, max_batches=2)
        scored.append((loss_val, i))

    scored.sort(key=lambda x: x[0])
    keep_n = max(1, int(np.ceil(keep_ratio * len(updates))))
    kept_idx = [i for _, i in scored[:keep_n]]
    return [updates[i] for i in kept_idx], kept_idx


def aggregate_dispatch(
    aggregator: str,
    updates_S: List[Dict[str, torch.Tensor]],
    taus_J: List[int],
    indices_S_in_J: List[int],
    alpha_stale: float,
    w_min: float,
    w_max: float,
    byz_frac: float,
    byz_budget_count: int,
    asb_trim: bool,
    server_state_cpu: Dict[str, torch.Tensor],
    model_ctor: Callable[[], nn.Module],
    zeno_val_loader: Optional[DataLoader],
    device: torch.device,
    loss_fn,
    gamma_t: float,
    tau_max_rounds: int,
    fedbuff_buffer: Optional[List[Dict[str, torch.Tensor]]] = None,
    fedbuff_size: int = 10,
):
    rho_hat_sq = 0.0
    if len(updates_S) == 0:
        return {}, rho_hat_sq, 0, {"kept_indices": [], "dropped_indices": []}

    agg = aggregator.lower()
    n = len(updates_S)
    B = int(max(0, min(byz_budget_count, max(0, n - 1))))

    if agg == "fedasync":
        wtilde_J = normalized_stale_weights(taus_J, alpha_stale, w_min, w_max)
        w_S = np.array([wtilde_J[j_idx] for j_idx in indices_S_in_J], dtype=np.float32)
        return weighted_mean_updates(updates_S, w_S), 0.0, n, {
            "kept_indices": list(range(n)), "dropped_indices": []
        }

    if agg == "fedbuff":
        if fedbuff_buffer is None:
            fedbuff_buffer = []
        fedbuff_buffer.extend(copy.deepcopy(updates_S))
        if len(fedbuff_buffer) < fedbuff_size:
            return {}, 0.0, 0, {"kept_indices": [], "dropped_indices": []}
        batch = fedbuff_buffer[:fedbuff_size]
        del fedbuff_buffer[:fedbuff_size]
        return weighted_mean_updates(batch, np.ones(len(batch)) / len(batch)), 0.0, len(batch), {
            "kept_indices": None, "dropped_indices": None
        }

    if agg == "mean":
        return weighted_mean_updates(updates_S, np.ones(n) / n), 0.0, n, {
            "kept_indices": list(range(n)), "dropped_indices": []
        }

    if agg == "median":
        return median_aggregate(updates_S), 0.0, n, {
            "kept_indices": None, "dropped_indices": None
        }

    if agg == "trimmed_mean":
        trim_ratio = min(0.499999, float(B) / max(1, n))
        return trimmed_mean_aggregate(updates_S, trim_ratio=trim_ratio), 0.0, max(1, n - 2 * B), {
            "kept_indices": None, "dropped_indices": None, "trim_B": B
        }

    if agg == "krum":
        winner = krum_select_index(updates_S, f=B)
        out = copy.deepcopy(updates_S[winner]) if winner >= 0 else {}
        return out, 0.0, 1 if winner >= 0 else 0, {
            "kept_indices": [winner] if winner >= 0 else [],
            "dropped_indices": [i for i in range(n) if i != winner] if winner >= 0 else [],
            "krum_f": B,
        }

    if agg == "aflguard":
        tau_S = [taus_J[j] for j in indices_S_in_J]
        kept, kept_local_idx = aflguard_screen(updates_S, tau_S, tau_cutoff=tau_max_rounds)
        return weighted_mean_updates(kept, np.ones(len(kept)) / len(kept)), 0.0, len(kept), {
            "kept_indices": kept_local_idx,
            "dropped_indices": [i for i in range(n) if i not in set(kept_local_idx)],
        }

    if agg == "zeno":
        if zeno_val_loader is None:
            return weighted_mean_updates(updates_S, np.ones(n) / n), 0.0, n, {
                "kept_indices": list(range(n)), "dropped_indices": []
            }
        kept, kept_idx = zeno_screen(
            updates_S, server_state_cpu, model_ctor, zeno_val_loader, device, loss_fn, gamma_t, keep_ratio=0.7
        )
        return weighted_mean_updates(kept, np.ones(len(kept)) / len(kept)), 0.0, len(kept), {
            "kept_indices": kept_idx,
            "dropped_indices": [i for i in range(n) if i not in set(kept_idx)],
        }

    if agg == "asb":
        trim_B = B if asb_trim else 0
        u_hat, rho_hat_sq, diag = asb_aggregate(
            updates_S=updates_S,
            taus_J=taus_J,
            indices_S_in_J=indices_S_in_J,
            alpha_stale=alpha_stale,
            w_min=w_min,
            w_max=w_max,
            trim_B=trim_B,
        )
        return u_hat, rho_hat_sq, len(diag.get("kept_indices", [])), diag

    raise ValueError(f"Unknown aggregator: {aggregator}")


# ---------------------------------------------------------------------
# Evaluation and heterogeneity
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, loss_fn) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def grad_vector_at_batch(
    model_ctor: Callable[[], nn.Module],
    state_cpu: Dict[str, torch.Tensor],
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    loss_fn,
) -> torch.Tensor:
    model = model_ctor().to(device)
    model.load_state_dict(state_cpu, strict=False)
    model.train()
    for p in model.parameters():
        p.grad = None
    x, y = batch
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.requires_grad and p.grad is not None]
    return torch.cat(grads) if grads else torch.zeros(1, device=device)


def heterogeneity_proxy_nu2(
    model_ctor: Callable[[], nn.Module],
    server_state_cpu: Dict[str, torch.Tensor],
    selected_clients: List[int],
    client_val_loaders: List[DataLoader],
    device: torch.device,
    loss_fn,
) -> float:
    grad_list = []
    for cid in selected_clients:
        batch = get_one_batch(client_val_loaders[cid], device)
        if batch is None:
            continue
        g = grad_vector_at_batch(model_ctor, server_state_cpu, batch, device, loss_fn)
        if g.numel() > 0:
            grad_list.append(g)
    if len(grad_list) <= 1:
        return 0.0
    G = torch.stack(grad_list, dim=0)
    g_bar = torch.mean(G, dim=0)
    vals = torch.sum((G - g_bar) ** 2, dim=1)
    return float(torch.mean(vals).item())


# ---------------------------------------------------------------------
# Federated loop
# ---------------------------------------------------------------------
async def federated_learning_rafl(
    clients_model_ctors: List[Callable[[], nn.Module]],
    server_model: nn.Module,
    clients_train_loaders: List[DataLoader],
    clients_val_loaders: List[DataLoader],
    test_loader: DataLoader,
    config: Dict[str, Any],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = F.nll_loss

    num_rounds = config["num_rounds"]
    local_epochs = config["local_epochs"]
    max_clients_per_round = config["num_clients_per_round"]
    max_parallel_clients = config["max_parallel_clients"]
    eta_client = config["eta_client"]
    zeta0 = config["zeta0"]
    alpha_stale = config["alpha_stale"]
    beta_et = config["beta_et"]
    L_smooth = config["L_smooth"]
    tau_max_rounds = config["tau_max_rounds"]
    trigger_eps = config["trigger_eps"]
    w_min = config["w_min"]
    w_max = config["w_max"]
    enable_byzantine = config["enable_byzantine"]
    byz_frac = config["byz_frac"]
    byz_count_override = config.get("byz_count_override", None)
    attack = config["attack"]
    byz_scale = config["byz_scale"]
    adaptive_norm_quantile = config.get("adaptive_norm_quantile", 0.50)
    adaptive_norm_ratio = config.get("adaptive_norm_ratio", 0.95)
    adaptive_z = config.get("adaptive_z", 1.5)
    asb_trim = config["asb_trim"]
    aggregator = config["aggregator"]
    accumulation_steps = config["accumulation_steps"]
    early_stopping_patience = config["early_stopping_patience"]
    fedbuff_size = config["fedbuff_size"]
    failure_tail_rounds = int(config.get("failure_tail_rounds", 20))

    I = local_epochs
    C_total = len(clients_model_ctors)

    sem = asyncio.Semaphore(max_parallel_clients)
    server_model.to(device)
    server_state_hist = deque(maxlen=tau_max_rounds + 1)
    server_state_hist.append(clone_state_dict_cpu(server_model.state_dict()))

    per_client_losses = [[] for _ in range(C_total)]
    selected_clients_log, exec_times_by_round = [], []
    fedbuff_buffer = []

    metrics = {
        "server_loss_proxy": [],
        "tau_bar": [],
        "p_t": [],
        "zeta_t": [],
        "gamma_t": [],
        "rho_sb_hat": [],
        "s_min": [],
        "nu2": [],
        "test_loss": [],
        "test_acc": [],
        "num_selected": [],
        "num_transmitted": [],
        "num_suppressed": [],
        "num_used_by_aggregator": [],
        "comm_reduction": [],
        "round_wall_time": [],
        "byzantine_count": [],
        "byzantine_count_transmitted": [],
        "actual_byz_frac_selected": [],
        "actual_byz_frac_transmitted": [],
        "byz_weight_mass_transmitted": [],
        "byz_survival_rate": [],
        "honest_update_norm_median": [],
        "byz_update_norm_mean": [],
        "byz_to_honest_norm_ratio": [],
        "adaptive_honest_norm_reference": [],
        "adaptive_crafted_target_norm": [],
        "attack_cosine_with_honest_mean": [],
        "asb_trimmed_count": [],
    }

    kappa_sq = max((w_max - 1.0) ** 2, (1.0 - w_min) ** 2)
    D_T_accum = 0.0
    alpha_t = (eta_client ** 2) * (L_smooth ** 2) * I * (I - 1)

    if alpha_t >= 0.5:
        print(f"[WARN] eta^2 L^2 I(I-1)={alpha_t:.3f} >= 1/2. Consider smaller eta/local_epochs.")

    async def launch_client(i_client: int, init_state_cpu: Dict[str, torch.Tensor], attack_mode: str):
        async with sem:
            return await asyncio.to_thread(
                _train_client_sync,
                init_state_cpu,
                clients_model_ctors[i_client],
                clients_train_loaders[i_client],
                "cuda" if torch.cuda.is_available() else "cpu",
                I,
                loss_fn,
                eta_client,
                accumulation_steps,
                early_stopping_patience,
                attack_mode,
                10,
            )

    for t in tqdm(range(num_rounds), desc=f"{aggregator}/{attack}"):
        round_start = time.time()

        selected = random.sample(range(C_total), min(max_clients_per_round, C_total))
        selected_clients_log.append(selected)
        num_selected = len(selected)

        # Exact B-count override is essential for failure-boundary studies because
        # small per-round participation makes nominal fractions highly discrete.
        if enable_byzantine and attack != "none":
            if byz_count_override is not None:
                B_count = int(max(0, min(int(byz_count_override), num_selected)))
            else:
                B_count = int(np.floor(float(byz_frac) * num_selected))
        else:
            B_count = 0
        byz_clients = set(random.sample(selected, B_count)) if B_count > 0 else set()

        tau_cap = min(t, tau_max_rounds)
        tau_c_map = {i: random.randint(0, tau_cap) for i in selected}

        # Adaptive staleness-weight exploit: malicious clients strategically arrive
        # fresh (tau=0), maximizing w_c = 1/(1+alpha*tau_c). No false metadata is
        # required; this is an arrival-timing attack against the weighting rule.
        if attack in ["staleness_exploit", "adaptive_joint"]:
            for i in byz_clients:
                tau_c_map[i] = 0

        tasks = []
        for i in selected:
            tau_i = tau_c_map[i]
            hist_idx = -1 - tau_i
            if -len(server_state_hist) <= hist_idx < 0:
                init_state_cpu = server_state_hist[hist_idx]
            else:
                init_state_cpu = server_state_hist[0]

            client_attack_mode = "label_flip" if (i in byz_clients and attack == "label_flip") else "none"
            tasks.append(launch_client(i, init_state_cpu, client_attack_mode))

        results = await asyncio.gather(*tasks)

        updates_J, taus_J, exec_times, client_losses_round = [], [], [], []
        for i_client, (local_state_cpu, delta_cpu, client_losses, num_samples, wall) in zip(selected, results):
            exec_times.append(wall)
            client_losses_round.append(sum(client_losses) / max(1, len(client_losses)))
            updates_J.append(delta_cpu)
            taus_J.append(tau_c_map[i_client])
            per_client_losses[i_client].extend(client_losses)

        # IMPORTANT for Comment 6: adaptive update attacks are crafted BEFORE the
        # event trigger, so the adversary can deliberately manipulate update norms
        # and transmission eligibility.
        updates_J, attack_diag = craft_round_adaptive_attacks(
            updates_J=updates_J,
            selected_clients=selected,
            byz_clients=byz_clients,
            attack=attack,
            byz_scale=byz_scale,
            trigger_eps=trigger_eps,
            adaptive_norm_quantile=adaptive_norm_quantile,
            adaptive_norm_ratio=adaptive_norm_ratio,
            adaptive_z=adaptive_z,
        )

        norms_J = [l2_norm_of_update(u) for u in updates_J]
        honest_norms = [norms_J[j] for j, cid in enumerate(selected) if cid not in byz_clients]
        byz_norms = [norms_J[j] for j, cid in enumerate(selected) if cid in byz_clients]
        honest_norm_median = float(np.median(honest_norms)) if honest_norms else 0.0
        byz_norm_mean = float(np.mean(byz_norms)) if byz_norms else 0.0
        norm_ratio = byz_norm_mean / max(1e-12, honest_norm_median) if byz_norms else 0.0

        # Event-triggered communication after malicious norm manipulation.
        S_indices_in_J = [j for j, nrm in enumerate(norms_J) if nrm >= trigger_eps]
        updates_S = [updates_J[j] for j in S_indices_in_J]
        selected_S_clients = [selected[j] for j in S_indices_in_J]
        byz_S_local_indices = [k for k, cid in enumerate(selected_S_clients) if cid in byz_clients]
        B_transmitted = len(byz_S_local_indices)

        num_transmitted = len(S_indices_in_J)
        num_suppressed = num_selected - num_transmitted
        p_t = 1.0 - (num_transmitted / max(1, num_selected))
        comm_reduction = num_suppressed / max(1, num_selected)

        wtilde_J = normalized_stale_weights(taus_J, alpha_stale, w_min, w_max)
        tau_bar_t = float(np.sum(wtilde_J * np.array(taus_J, dtype=np.float32)))

        # Weight mass of Byzantine transmitted updates after re-normalization on S.
        if num_transmitted > 0:
            w_S = np.array([wtilde_J[j] for j in S_indices_in_J], dtype=np.float64)
            w_S = w_S / max(1e-12, float(w_S.sum()))
            byz_weight_mass = float(sum(w_S[k] for k in byz_S_local_indices))
        else:
            byz_weight_mass = 0.0

        A_t = (1.0 + alpha_stale * tau_bar_t) * (1.0 + beta_et * p_t)
        zeta_t = zeta0 / (np.sqrt(t + 1.0) * A_t)
        zeta_cap = 1.0 / (4.0 * L_smooth)
        zeta_t = min(float(zeta_t), float(zeta_cap))
        gamma_t = float(zeta_t / max(1e-12, eta_client * I))
        D_T_accum += A_t

        server_state_cpu_now = clone_state_dict_cpu(server_model.state_dict())
        zeno_val_loader = clients_val_loaders[selected[0]] if len(selected) > 0 else None

        # Robust methods are given the same configured per-round Byzantine budget B,
        # but never the Byzantine identities. For event-triggered rounds, the budget
        # is conservatively capped by the number of transmitted updates.
        B_budget_for_aggregation = min(B_count, max(0, num_transmitted - 1)) if num_transmitted > 0 else 0

        u_hat, rho_hat_sq, used_count, agg_diag = aggregate_dispatch(
            aggregator=aggregator,
            updates_S=updates_S,
            taus_J=taus_J,
            indices_S_in_J=S_indices_in_J,
            alpha_stale=alpha_stale,
            w_min=w_min,
            w_max=w_max,
            byz_frac=byz_frac,
            byz_budget_count=B_budget_for_aggregation,
            asb_trim=asb_trim,
            server_state_cpu=server_state_cpu_now,
            model_ctor=clients_model_ctors[0],
            zeno_val_loader=zeno_val_loader,
            device=device,
            loss_fn=loss_fn,
            gamma_t=gamma_t,
            tau_max_rounds=tau_max_rounds,
            fedbuff_buffer=fedbuff_buffer,
            fedbuff_size=fedbuff_size,
        )

        kept_idx = agg_diag.get("kept_indices", None)
        if kept_idx is not None and B_transmitted > 0:
            kept_set = set(kept_idx)
            byz_survivors = sum(1 for k in byz_S_local_indices if k in kept_set)
            byz_survival_rate = byz_survivors / max(1, B_transmitted)
        else:
            byz_survival_rate = float("nan")

        if u_hat:
            add_update_to_model(server_model, u_hat, gamma_t)
            server_state_hist.append(clone_state_dict_cpu(server_model.state_dict()))
        else:
            server_state_hist.append(server_state_hist[-1])

        nu2_t = heterogeneity_proxy_nu2(
            model_ctor=clients_model_ctors[0],
            server_state_cpu=server_state_cpu_now,
            selected_clients=selected,
            client_val_loaders=clients_val_loaders,
            device=device,
            loss_fn=loss_fn,
        )

        test_loss, test_acc = evaluate_model(server_model, test_loader, device, loss_fn)

        metrics["server_loss_proxy"].append(sum(client_losses_round) / max(1, len(client_losses_round)))
        metrics["tau_bar"].append(tau_bar_t)
        metrics["p_t"].append(p_t)
        metrics["zeta_t"].append(zeta_t)
        metrics["gamma_t"].append(gamma_t)
        metrics["rho_sb_hat"].append(rho_hat_sq)
        metrics["s_min"].append(w_min * (1.0 - float(B_count) / max(1, num_selected)))
        metrics["nu2"].append(nu2_t)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)
        metrics["num_selected"].append(num_selected)
        metrics["num_transmitted"].append(num_transmitted)
        metrics["num_suppressed"].append(num_suppressed)
        metrics["num_used_by_aggregator"].append(used_count)
        metrics["comm_reduction"].append(comm_reduction)
        metrics["round_wall_time"].append(time.time() - round_start)
        metrics["byzantine_count"].append(B_count)
        metrics["byzantine_count_transmitted"].append(B_transmitted)
        metrics["actual_byz_frac_selected"].append(B_count / max(1, num_selected))
        metrics["actual_byz_frac_transmitted"].append(B_transmitted / max(1, num_transmitted))
        metrics["byz_weight_mass_transmitted"].append(byz_weight_mass)
        metrics["byz_survival_rate"].append(byz_survival_rate)
        metrics["honest_update_norm_median"].append(honest_norm_median)
        metrics["byz_update_norm_mean"].append(byz_norm_mean)
        metrics["byz_to_honest_norm_ratio"].append(norm_ratio)
        metrics["adaptive_honest_norm_reference"].append(attack_diag.get("honest_norm_reference", 0.0))
        metrics["adaptive_crafted_target_norm"].append(attack_diag.get("crafted_target_norm", 0.0))
        metrics["attack_cosine_with_honest_mean"].append(attack_diag.get("attack_cosine_with_honest_mean", 0.0))
        metrics["asb_trimmed_count"].append(int(agg_diag.get("trim_B", 0)) if aggregator.lower() == "asb" else 0)
        exec_times_by_round.append(exec_times)

    metrics["kappa_sq"] = kappa_sq
    metrics["D_T"] = D_T_accum / max(1, num_rounds)
    metrics["alpha_t"] = alpha_t
    metrics["final_test_acc"] = metrics["test_acc"][-1] if metrics["test_acc"] else None
    metrics["final_test_loss"] = metrics["test_loss"][-1] if metrics["test_loss"] else None
    tail = metrics["test_acc"][-max(1, failure_tail_rounds):]
    metrics["tail_mean_test_acc"] = float(np.mean(tail)) if tail else None
    metrics["mean_comm_reduction"] = float(np.mean(metrics["comm_reduction"])) if metrics["comm_reduction"] else 0.0
    metrics["mean_round_wall_time"] = float(np.mean(metrics["round_wall_time"])) if metrics["round_wall_time"] else 0.0
    metrics["mean_actual_byz_frac_selected"] = float(np.mean(metrics["actual_byz_frac_selected"])) if metrics["actual_byz_frac_selected"] else 0.0
    metrics["mean_actual_byz_frac_transmitted"] = float(np.mean(metrics["actual_byz_frac_transmitted"])) if metrics["actual_byz_frac_transmitted"] else 0.0
    metrics["mean_byz_weight_mass_transmitted"] = float(np.mean(metrics["byz_weight_mass_transmitted"])) if metrics["byz_weight_mass_transmitted"] else 0.0
    finite_survival = [x for x in metrics["byz_survival_rate"] if np.isfinite(x)]
    metrics["mean_byz_survival_rate"] = float(np.mean(finite_survival)) if finite_survival else None

    return server_model, per_client_losses, selected_clients_log, exec_times_by_round, metrics


# ---------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------
def default_config():
    return {
        "suite": "single",
        "seed": 42,
        "num_clients": 10,
        "alpha_dirichlet": 0.5,
        "num_rounds": 100,
        "local_epochs": 10,
        "num_clients_per_round": 5,
        "max_parallel_clients": 5,
        "batch_size": 64,
        "eta_client": 5e-3,
        "zeta0": 1e-2,
        "L_smooth": 1.0,
        "alpha_stale": 0.2,
        "beta_et": 0.5,
        "tau_max_rounds": 3,
        "trigger_eps": 0.0,
        "w_min": 0.5,
        "w_max": 1.5,
        "enable_byzantine": False,
        "byz_frac": 0.3,
        "byz_count_override": None,
        "attack": "none",
        "byz_scale": 10.0,
        "adaptive_norm_quantile": 0.50,
        "adaptive_norm_ratio": 0.95,
        "adaptive_z": 1.5,
        "asb_trim": True,
        "aggregator": "asb",
        "fedbuff_size": 10,
        "accumulation_steps": 1,
        "early_stopping_patience": 10,
        "failure_tail_rounds": 20,
        "failure_drop_threshold": 0.10,
        "failure_acc_floor": 0.20,
        "base_dir": "results_rafl_cifar10_suite",
    }


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def build_experiment_grid(args):
    base = default_config()
    base.update(vars(args))
    suite = args.suite
    grid = []

    if suite == "single":
        grid = [base]

    elif suite == "smoke":
        base["num_rounds"] = min(base["num_rounds"], 5)
        base["num_clients"] = min(base["num_clients"], 10)
        base["num_clients_per_round"] = min(base["num_clients_per_round"], 5)
        for agg in ["fedasync", "asb"]:
            c = copy.deepcopy(base)
            c["aggregator"] = agg
            c["attack"] = "signflip"
            c["enable_byzantine"] = True
            c["byz_frac"] = 0.2
            grid.append(c)

    elif suite == "attacks":
        for attack in ["none", "signflip", "gaussian", "model_replacement", "label_flip"]:
            c = copy.deepcopy(base)
            c["suite"] = "attacks"
            c["aggregator"] = args.aggregator
            c["attack"] = attack
            c["enable_byzantine"] = attack != "none"
            grid.append(c)

    elif suite == "adaptive_attacks":
        # Direct response to Reviewer #2, Comment 6.
        adaptive_attacks = parse_csv_list(args.adaptive_attack_list)
        for attack in adaptive_attacks:
            c = copy.deepcopy(base)
            c["suite"] = "adaptive_attacks"
            c["aggregator"] = args.aggregator
            c["attack"] = attack
            c["enable_byzantine"] = True
            grid.append(c)

    elif suite == "baselines":
        for agg in ["fedasync", "fedbuff", "mean", "median", "trimmed_mean", "krum", "aflguard", "zeno", "asb"]:
            c = copy.deepcopy(base)
            c["suite"] = "baselines"
            c["aggregator"] = agg
            c["attack"] = args.attack if args.attack != "none" else "signflip"
            c["enable_byzantine"] = True
            grid.append(c)

    elif suite == "sensitivity":
        sweep_specs = [
            ("alpha_stale", [0.0, 0.1, 0.2, 0.5]),
            ("trigger_eps", [0.0, 0.005, 0.01, 0.05, 0.1]),
            ("byz_frac", [0.0, 0.1, 0.2, 0.3, 0.4]),
            ("tau_max_rounds", [0, 3, 5, 10, 20]),
            ("num_clients", [10, 20, 50]),
        ]
        for param, values in sweep_specs:
            for val in values:
                c = copy.deepcopy(base)
                c["suite"] = f"sensitivity_{param}"
                c["aggregator"] = args.aggregator
                c["attack"] = args.attack if args.attack != "none" else "signflip"
                c["enable_byzantine"] = c["attack"] != "none" and float(val if param == "byz_frac" else c["byz_frac"]) > 0
                c[param] = val
                if param == "num_clients":
                    c["num_clients_per_round"] = min(c["num_clients_per_round"], max(5, int(val // 5)))
                    c["max_parallel_clients"] = min(c["max_parallel_clients"], c["num_clients_per_round"])
                grid.append(c)

    elif suite == "failure_boundary":
        # Exact B/J sweep avoids misleading nominal fractions for small J.
        aggregators = parse_csv_list(args.failure_aggregators)
        if args.failure_byz_counts.strip():
            byz_counts = sorted(set(parse_int_csv(args.failure_byz_counts)))
        else:
            # Include both sides of the ASB q<1/2 theoretical condition.
            byz_counts = list(range(0, max(1, base["num_clients_per_round"])))

        J = int(base["num_clients_per_round"])
        for agg in aggregators:
            for B in byz_counts:
                if B < 0 or B >= J:
                    continue
                c = copy.deepcopy(base)
                c["suite"] = "failure_boundary"
                c["aggregator"] = agg
                c["byz_count_override"] = int(B)
                c["byz_frac"] = float(B) / max(1, J)
                c["attack"] = "none" if B == 0 else args.failure_attack
                c["enable_byzantine"] = B > 0
                grid.append(c)

    else:
        raise ValueError(f"Unknown suite: {suite}")

    return grid


async def run_one_experiment(config: Dict[str, Any]):
    set_seed(config["seed"])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    client_indices = partition_non_iid(
        train_dataset,
        config["num_clients"],
        alpha=config["alpha_dirichlet"],
        num_classes=10,
    )

    pin = torch.cuda.is_available()
    train_loaders, val_loaders = build_client_loaders(
        train_dataset,
        client_indices,
        batch_size=config["batch_size"],
        val_size_per_client=256,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    clients_model_ctors = [make_model for _ in range(config["num_clients"])]
    server_model = make_model()

    results_dir = create_directory(config, base_dir=config["base_dir"])
    save_json(config, os.path.join(results_dir, "config.json"))

    server_model, per_client_losses, selected_clients, exec_times, metrics = await federated_learning_rafl(
        clients_model_ctors=clients_model_ctors,
        server_model=server_model,
        clients_train_loaders=train_loaders,
        clients_val_loaders=val_loaders,
        test_loader=test_loader,
        config=config,
    )

    save_json(metrics, os.path.join(results_dir, "metrics.json"))

    with open(os.path.join(results_dir, "selected_clients.csv"), "w") as f:
        f.write("Round,Selected Clients\n")
        for r, clist in enumerate(selected_clients, 1):
            f.write(f"{r},{','.join(map(str, clist))}\n")

    with open(os.path.join(results_dir, "execution_times.csv"), "w") as f:
        f.write("Round,ClientIndexInRound,ExecutionTimeSeconds\n")
        for r, times in enumerate(exec_times, 1):
            for idx, tsec in enumerate(times):
                f.write(f"{r},{idx},{tsec:.6f}\n")

    for key, title, ylabel in [
        ("server_loss_proxy", "Server Loss Proxy", "Loss"),
        ("test_loss", "Test Loss per Round", "Loss"),
        ("test_acc", "Test Accuracy per Round", "Accuracy"),
        ("tau_bar", "Weighted Average Staleness", "tau_bar"),
        ("p_t", "Suppression Rate", "p_t"),
        ("comm_reduction", "Communication Reduction", "Reduction"),
        ("num_transmitted", "Transmitted Updates per Round", "Count"),
        ("num_used_by_aggregator", "Updates Used by Aggregator", "Count"),
        ("zeta_t", "Composite Step Size", "zeta_t"),
        ("gamma_t", "Server Step Size", "gamma_t"),
        ("rho_sb_hat", "Robustness Residual Proxy", "rho_hat_sq"),
        ("nu2", "Heterogeneity Proxy", "nu2"),
        ("round_wall_time", "Wall-clock Time per Round", "Seconds"),
        ("actual_byz_frac_transmitted", "Actual Byzantine Fraction Among Transmissions", "B/|S|"),
        ("byz_weight_mass_transmitted", "Byzantine Staleness-Weight Mass", "Weight mass"),
        ("byz_to_honest_norm_ratio", "Byzantine/Honest Update Norm Ratio", "Norm ratio"),
        ("byz_survival_rate", "Byzantine Survival Rate After Filtering", "Survival rate"),
        ("attack_cosine_with_honest_mean", "Attack Cosine vs Honest Mean", "Cosine"),
    ]:
        plot_series(metrics.get(key, []), title, os.path.join(results_dir, f"{key}.png"), ylabel=ylabel)

    plot_xy(
        metrics.get("comm_reduction", []),
        metrics.get("test_acc", []),
        "Accuracy vs Communication Reduction",
        os.path.join(results_dir, "accuracy_vs_comm_reduction.png"),
        xlabel="Communication Reduction",
        ylabel="Test Accuracy",
    )

    torch.save(server_model.state_dict(), os.path.join(results_dir, "server_model_final.pt"))

    requested_B = config.get("byz_count_override", None)
    actual_B = int(round(float(np.mean(metrics["byzantine_count"])))) if metrics["byzantine_count"] else 0
    summary = {
        "results_dir": results_dir,
        "suite": config["suite"],
        "seed": config["seed"],
        "aggregator": config["aggregator"],
        "attack": config["attack"],
        "num_clients": config["num_clients"],
        "num_clients_per_round": config["num_clients_per_round"],
        "alpha_stale": config["alpha_stale"],
        "trigger_eps": config["trigger_eps"],
        "requested_byz_frac": config["byz_frac"],
        "requested_byz_count": requested_B,
        "actual_byz_count_per_round": actual_B,
        "actual_byz_frac_selected": metrics["mean_actual_byz_frac_selected"],
        "actual_byz_frac_transmitted": metrics["mean_actual_byz_frac_transmitted"],
        "tau_max_rounds": config["tau_max_rounds"],
        "final_test_acc": metrics["final_test_acc"],
        "tail_mean_test_acc": metrics["tail_mean_test_acc"],
        "final_test_loss": metrics["final_test_loss"],
        "mean_comm_reduction": metrics["mean_comm_reduction"],
        "mean_round_wall_time": metrics["mean_round_wall_time"],
        "mean_byz_weight_mass_transmitted": metrics["mean_byz_weight_mass_transmitted"],
        "mean_byz_survival_rate": metrics["mean_byz_survival_rate"],
        "asb_theory_q_lt_half": bool(metrics["mean_actual_byz_frac_transmitted"] < 0.5),
    }

    save_json(summary, os.path.join(results_dir, "summary.json"))
    print(json.dumps(summary, indent=2))
    return summary


def write_failure_boundary_report(all_summaries: List[Dict[str, Any]], args):
    """
    Quantify two boundaries for each aggregator:
      1) degradation boundary: first q where tail accuracy drops by at least
         --failure_drop_threshold from that method's own q=0 reference;
      2) collapse boundary: first q where tail accuracy falls below
         --failure_acc_floor or becomes non-finite.
    """
    if not all_summaries:
        return

    groups = defaultdict(list)
    for row in all_summaries:
        groups[row["aggregator"]].append(row)

    detailed_rows = []
    boundary_rows = []
    for agg, rows in groups.items():
        rows = sorted(rows, key=lambda r: (r.get("actual_byz_frac_selected", 0.0), r.get("actual_byz_count_per_round", 0)))
        clean_candidates = [r for r in rows if int(r.get("actual_byz_count_per_round", 0)) == 0]
        if not clean_candidates:
            continue
        clean_acc = float(clean_candidates[0]["tail_mean_test_acc"])

        drop_boundary = None
        collapse_boundary = None
        for r in rows:
            q = float(r.get("actual_byz_frac_selected", 0.0))
            acc = r.get("tail_mean_test_acc", None)
            acc_f = float(acc) if acc is not None else float("nan")
            abs_drop = clean_acc - acc_f if np.isfinite(acc_f) else float("inf")
            retained = acc_f / max(1e-12, clean_acc) if np.isfinite(acc_f) else 0.0
            failed_drop = (not np.isfinite(acc_f)) or (abs_drop >= float(args.failure_drop_threshold))
            failed_floor = (not np.isfinite(acc_f)) or (acc_f <= float(args.failure_acc_floor))

            if failed_drop and drop_boundary is None and q > 0:
                drop_boundary = q
            if failed_floor and collapse_boundary is None and q > 0:
                collapse_boundary = q

            detailed_rows.append({
                "aggregator": agg,
                "attack": r["attack"],
                "B": r.get("actual_byz_count_per_round", 0),
                "q_selected": q,
                "q_transmitted": r.get("actual_byz_frac_transmitted", 0.0),
                "tail_mean_acc": acc_f,
                "clean_tail_mean_acc": clean_acc,
                "absolute_accuracy_drop": abs_drop,
                "retained_accuracy_ratio": retained,
                "failed_drop_criterion": failed_drop,
                "failed_floor_criterion": failed_floor,
                "asb_theory_q_lt_half": r.get("asb_theory_q_lt_half", None),
                "results_dir": r["results_dir"],
            })

        boundary_rows.append({
            "aggregator": agg,
            "clean_tail_mean_acc": clean_acc,
            "drop_threshold_definition": float(args.failure_drop_threshold),
            "first_q_accuracy_drop_ge_threshold": drop_boundary,
            "collapse_floor_definition": float(args.failure_acc_floor),
            "first_q_accuracy_le_floor": collapse_boundary,
        })

    os.makedirs(args.base_dir, exist_ok=True)
    save_json(detailed_rows, os.path.join(args.base_dir, "failure_boundary_detailed.json"))
    save_json(boundary_rows, os.path.join(args.base_dir, "failure_boundary_thresholds.json"))

    if detailed_rows:
        keys = list(detailed_rows[0].keys())
        with open(os.path.join(args.base_dir, "failure_boundary_detailed.csv"), "w") as f:
            f.write(",".join(keys) + "\n")
            for row in detailed_rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

        plt.figure(figsize=(10, 6))
        for agg in sorted(set(r["aggregator"] for r in detailed_rows)):
            rr = sorted([r for r in detailed_rows if r["aggregator"] == agg], key=lambda x: x["q_selected"])
            plt.plot([r["q_selected"] for r in rr], [r["tail_mean_acc"] for r in rr], marker="o", label=agg)
        plt.axvline(0.5, linestyle="--", linewidth=1.0, label="q=0.5 theory boundary")
        plt.axhline(float(args.failure_acc_floor), linestyle=":", linewidth=1.0, label="collapse floor")
        plt.xlabel("Actual Byzantine fraction per selected round, q=B/J")
        plt.ylabel(f"Mean test accuracy over final {args.failure_tail_rounds} rounds")
        plt.title(f"Failure-boundary comparison under {args.failure_attack}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.base_dir, "failure_boundary_accuracy.png"))
        plt.close()

    if boundary_rows:
        keys = list(boundary_rows[0].keys())
        with open(os.path.join(args.base_dir, "failure_boundary_thresholds.csv"), "w") as f:
            f.write(",".join(keys) + "\n")
            for row in boundary_rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


async def main_async(args):
    grid = build_experiment_grid(args)
    os.makedirs(args.base_dir, exist_ok=True)

    all_summaries = []
    for idx, config in enumerate(grid, 1):
        print(f"\n===== Experiment {idx}/{len(grid)} =====")
        print(json.dumps({
            "suite": config["suite"],
            "aggregator": config["aggregator"],
            "attack": config["attack"],
            "C": config["num_clients"],
            "J": config["num_clients_per_round"],
            "R": config["num_rounds"],
            "alpha_stale": config["alpha_stale"],
            "trigger_eps": config["trigger_eps"],
            "byz_frac": config["byz_frac"],
            "byz_count_override": config.get("byz_count_override", None),
            "tau_max_rounds": config["tau_max_rounds"],
        }, indent=2))

        summary = await run_one_experiment(config)
        all_summaries.append(summary)

        with open(os.path.join(args.base_dir, "all_summaries.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)

    csv_path = os.path.join(args.base_dir, "all_summaries.csv")
    if all_summaries:
        keys = list(all_summaries[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in all_summaries:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    if args.suite == "failure_boundary":
        write_failure_boundary_report(all_summaries, args)

    print(f"\n[DONE] All summaries saved to {args.base_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite", type=str, default="single",
        choices=["single", "smoke", "attacks", "adaptive_attacks", "baselines", "sensitivity", "failure_boundary"]
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--alpha_dirichlet", type=float, default=0.5)
    p.add_argument("--num_rounds", type=int, default=400)
    p.add_argument("--local_epochs", type=int, default=10)
    p.add_argument("--num_clients_per_round", type=int, default=5)
    p.add_argument("--max_parallel_clients", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--eta_client", type=float, default=1e-3)
    p.add_argument("--zeta0", type=float, default=1e-2)
    p.add_argument("--L_smooth", type=float, default=1.0)
    p.add_argument("--alpha_stale", type=float, default=0.2)
    p.add_argument("--beta_et", type=float, default=0.5)
    p.add_argument("--tau_max_rounds", type=int, default=3)
    p.add_argument("--trigger_eps", type=float, default=0.0)
    p.add_argument("--w_min", type=float, default=0.5)
    p.add_argument("--w_max", type=float, default=1.5)

    p.add_argument("--enable_byzantine", action="store_true")
    p.add_argument("--byz_frac", type=float, default=0.3)
    p.add_argument("--byz_count_override", type=int, default=None,
                   help="Exact Byzantine count per selected round; overrides floor(byz_frac*J).")
    p.add_argument(
        "--attack", type=str, default="none",
        choices=[
            "none", "signflip", "gaussian", "model_replacement", "label_flip", "zero", "random", "mean_shift",
            "norm_evasion", "staleness_exploit", "collusion", "alie", "adaptive_joint",
        ],
    )
    p.add_argument("--byz_scale", type=float, default=10.0)
    p.add_argument("--adaptive_norm_quantile", type=float, default=0.50,
                   help="Honest norm quantile used as the norm-evasion reference.")
    p.add_argument("--adaptive_norm_ratio", type=float, default=0.95,
                   help="Malicious norm = ratio * chosen honest norm quantile.")
    p.add_argument("--adaptive_z", type=float, default=1.5,
                   help="ALIE coordinate shift multiplier.")
    p.add_argument("--adaptive_attack_list", type=str,
                   default="norm_evasion,staleness_exploit,collusion,alie,adaptive_joint")
    p.add_argument("--asb_trim", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument(
        "--aggregator", type=str, default="asb",
        choices=["fedasync", "fedbuff", "mean", "median", "trimmed_mean", "krum", "aflguard", "zeno", "asb"],
    )
    p.add_argument("--fedbuff_size", type=int, default=10)

    # Failure-boundary experiment controls (Reviewer #2, Comment 6).
    p.add_argument("--failure_attack", type=str, default="adaptive_joint",
                   choices=["signflip", "gaussian", "model_replacement", "label_flip", "norm_evasion", "staleness_exploit", "collusion", "alie", "adaptive_joint"])
    p.add_argument("--failure_aggregators", type=str,
                   default="median,trimmed_mean,krum,aflguard,zeno,asb")
    p.add_argument("--failure_byz_counts", type=str, default="",
                   help="Comma-separated exact B values. Empty means B=0,...,J-1.")
    p.add_argument("--failure_tail_rounds", type=int, default=20)
    p.add_argument("--failure_drop_threshold", type=float, default=0.10,
                   help="Failure if tail accuracy drops this many absolute points from q=0.")
    p.add_argument("--failure_acc_floor", type=float, default=0.20,
                   help="Catastrophic collapse floor for CIFAR-10 accuracy.")

    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--base_dir", type=str, default="results_rafl_cifar10_suite")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
