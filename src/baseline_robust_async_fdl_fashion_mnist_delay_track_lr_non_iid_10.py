#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 23:23:52 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Robust AFL (baseline) on Fashion-MNIST
- True asynchrony: clients train concurrently (asyncio.to_thread)
- Non-IID Dirichlet split
- Robust aggregators: 'mean', 'stale_mean', 'median', 'trimmed_mean', 'krum'
- Optional Byzantine simulation (signflip/gaussian)
- Saves: per-client losses, server loss proxy, selected clients, exec times,
         test loss/acc, and staleness stats (tau_bar) for comparison to RAFL.

Design differences vs your RAFL:
- Fixed server step-size gamma (no ζ_t / γ_t schedule)
- No event-trigger suppression p_t, no ASB trimming
- Simple staleness weighting (only for 'stale_mean')
"""

import os
import time
import random
import json
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Callable

# ---- HPC/Headless safety
import multiprocessing as mp
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

# ----------------- Reproducibility -----------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ----------------- Model (Fashion-MNIST, 1x28x28) -----------------
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
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + self.short(x)
        return F.relu(y)

class ResNetFashion(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_ch = 16
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_ch, n, stride):
        layers = [block(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, n):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y); y = self.layer2(y); y = self.layer3(y)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return F.log_softmax(y, dim=1)

# ----------------- Utils -----------------
def create_directory(tag: str, base_dir="results_afl_baseline_fmnist"):
    dn = os.path.join(base_dir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dn, exist_ok=True)
    return dn

def plot_series(y, title, save_path, xlabel="Rounds", ylabel="Value"):
    plt.figure(figsize=(10, 6))
    plt.plot(y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True); plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def npar(x): return np.array(x, dtype=np.float32)

def clone_state_dict_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}

def dict_subtract(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (a[k] - b[k]) for k in a.keys()
            if a[k].dtype.is_floating_point and b[k].dtype.is_floating_point}

def dict_to_vec(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [p.detach().flatten().float() for p in state.values() if p.dtype.is_floating_point]
    if len(parts) == 0:
        return torch.zeros(1)
    return torch.cat(parts)

def l2_norm_of_update(update: Dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        v = dict_to_vec(update)
        return float(torch.norm(v, p=2).item())

# ----------------- Dataset split -----------------
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
    return client_indices

def build_client_loaders(
    dataset, client_indices: List[List[int]], batch_size: int,
    val_size_per_client: int = 128, num_workers: int = 0, pin_memory: bool = False
):
    train_loaders, val_loaders = [], []
    for idxs in client_indices:
        if len(idxs) <= val_size_per_client:
            vsize = max(1, len(idxs)//2)
        else:
            vsize = val_size_per_client
        val_ids = idxs[-vsize:]
        train_ids = idxs[:-vsize] if len(idxs) > vsize else idxs

        train_loaders.append(DataLoader(
            Subset(dataset, train_ids),
            batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin_memory, persistent_workers=False
        ))
        val_loaders.append(DataLoader(
            Subset(dataset, val_ids),
            batch_size=min(128, vsize), shuffle=False,
            num_workers=0, pin_memory=pin_memory
        ))
    return train_loaders, val_loaders

def get_one_batch(loader: DataLoader, device: torch.device):
    for x, y in loader:
        return x.to(device), y.to(device)
    return None

# ----------------- Byzantine simulation -----------------
def byzantine_corrupt_update(update: Dict[str, torch.Tensor], mode="signflip", scale=10.0):
    bad = {}
    for k, v in update.items():
        if not v.dtype.is_floating_point:  # skip buffers like num_batches_tracked
            continue
        t = v.detach().cpu().float()
        if mode == "signflip":
            bad[k] = (-t) * float(scale)
        elif mode == "gaussian":
            bad[k] = t + torch.randn_like(t) * float(scale)
        else:
            bad[k] = t
    return bad

# ----------------- Client train (sync in thread) -----------------
def _train_client_sync(
    init_state_cpu: Dict[str, torch.Tensor],
    model_ctor: Callable[[], nn.Module],
    train_loader: DataLoader,
    device_str: str,
    local_epochs: int,
    loss_fn,
    eta_client: float,
    accumulation_steps: int,
    early_stopping_patience: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[float], int, float]:
    device = torch.device(device_str)
    model = model_ctor().to(device)
    model.load_state_dict(init_state_cpu, strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=eta_client)
    client_losses, best, patience = [], float("inf"), 0

    start = time.time()
    for epoch in range(local_epochs):
        model.train(); epoch_loss = 0.0
        for b, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            if (b + 1) % accumulation_steps == 0:
                opt.step()
            epoch_loss += float(loss.item())
        if (len(train_loader) % max(1, accumulation_steps)) != 0:
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

# ----------------- Robust aggregators -----------------
def _keys_float(updates: List[Dict[str, torch.Tensor]]):
    return [k for k, v in updates[0].items() if v.dtype.is_floating_point]

def agg_mean(updates: List[Dict[str, torch.Tensor]], weights: Optional[np.ndarray] = None):
    keys = _keys_float(updates)
    if weights is None:
        weights = np.ones(len(updates), dtype=np.float32) / max(1, len(updates))
    agg = {k: torch.zeros_like(updates[0][k].float().cpu()) for k in keys}
    for w, u in zip(weights, updates):
        for k in keys:
            agg[k] += float(w) * u[k].float().cpu()
    return agg

def agg_median(updates: List[Dict[str, torch.Tensor]]):
    keys = _keys_float(updates)
    agg = {}
    for k in keys:
        stacked = torch.stack([u[k].float().cpu() for u in updates], dim=0)
        agg[k] = torch.median(stacked, dim=0).values
    return agg

def agg_trimmed_mean(updates: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.2):
    keys = _keys_float(updates)
    J = len(updates)
    t = int(np.floor(trim_ratio * J))
    kept = max(1, J - 2 * t)
    agg = {}
    for k in keys:
        stacked = torch.stack([u[k].float().cpu() for u in updates], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[t: t + kept]
        agg[k] = trimmed.mean(dim=0)
    return agg

def agg_krum(updates: List[Dict[str, torch.Tensor]], byz_frac: float = 0.3, m_multi: int = 1):
    vecs = [dict_to_vec(u).cpu() for u in updates]
    J = len(vecs)
    if J == 1:
        return updates[0]
    dmat = torch.zeros((J, J))
    for i in range(J):
        for j in range(i + 1, J):
            d = torch.sum((vecs[i] - vecs[j]) ** 2)
            dmat[i, j] = dmat[j, i] = d
    f = int(np.floor(byz_frac * J))
    keep_k = max(1, J - f - 2)
    scores = []
    for i in range(J):
        row = dmat[i].clone()
        row_sorted, _ = torch.sort(row)
        scores.append(torch.sum(row_sorted[1:1 + keep_k]).item())
    order = np.argsort(np.array(scores))
    chosen_idx = order[:max(1, m_multi)]
    if len(chosen_idx) == 1:
        return updates[int(chosen_idx[0])]
    chosen_updates = [updates[int(i)] for i in chosen_idx]
    return agg_mean(chosen_updates)

def normalized_stale_weights(taus: List[int], alpha_stale: float, w_min: float, w_max: float):
    J = len(taus)
    raw = np.array([1.0 / (1.0 + alpha_stale * max(0.0, float(t))) for t in taus], dtype=np.float32)
    raw_sum = raw.sum()
    if raw_sum <= 0:
        raw = np.ones_like(raw); raw_sum = raw.sum()
    w = raw / raw_sum
    w = np.clip(w, w_min / J, w_max / J)
    w = w / w.sum()
    return w

def aggregate_updates(
    updates: List[Dict[str, torch.Tensor]],
    aggregator: str,
    taus: Optional[List[int]] = None,
    alpha_stale: float = 0.2,
    w_min: float = 0.5, w_max: float = 1.5,
    trim_ratio: float = 0.2,
    byz_frac: float = 0.3,
    krum_m: int = 1
):
    if len(updates) == 0:
        return {}
    aggregator = aggregator.lower()
    if aggregator == "mean":
        return agg_mean(updates)
    elif aggregator == "stale_mean":
        if taus is None:
            return agg_mean(updates)
        w = normalized_stale_weights(taus, alpha_stale, w_min, w_max)
        return agg_mean(updates, w)
    elif aggregator == "median":
        return agg_median(updates)
    elif aggregator == "trimmed_mean":
        return agg_trimmed_mean(updates, trim_ratio=trim_ratio)
    elif aggregator == "krum":
        return agg_krum(updates, byz_frac=byz_frac, m_multi=krum_m)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")

# ----------------- Eval -----------------
@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, loss_fn):
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

# ----------------- Federated baseline (async) -----------------
async def federated_learning_afl_baseline(
    clients_model_ctors: List[Callable[[], nn.Module]],
    server_model: nn.Module,
    clients_train_loaders: List[DataLoader],
    test_loader: DataLoader,
    num_rounds=500, local_epochs=3, max_clients_per_round=5,
    max_parallel_clients=None,
    loss_fn=None,
    eta_client=1e-2,
    gamma_server=1e-2,           # fixed server step-size
    aggregator="stale_mean",     # 'mean'|'stale_mean'|'median'|'trimmed_mean'|'krum'
    alpha_stale=0.2,
    w_min=0.5, w_max=1.5,
    trim_ratio=0.2,
    krum_m=1,
    enable_byzantine=False, byz_frac=0.0, byz_mode="signflip", byz_scale=10.0,
    accumulation_steps=1, early_stopping_patience=5,
    tau_max_rounds=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    I = local_epochs
    J_per_round = max_clients_per_round
    C_total = len(clients_model_ctors)

    if max_parallel_clients is None:
        max_parallel_clients = max_clients_per_round
    sem = asyncio.Semaphore(max_parallel_clients)

    server_model.to(device)
    server_state_hist = deque(maxlen=tau_max_rounds + 1)
    server_state_hist.append(clone_state_dict_cpu(server_model.state_dict()))

    per_client_losses = [[] for _ in range(C_total)]
    server_losses = []
    selected_clients_log = []
    exec_times_by_round = []
    tau_bar = []
    test_loss_series, test_acc_series = [], []

    async def launch_client(i_client: int, init_state_cpu: Dict[str, torch.Tensor]):
        async with sem:
            local_state_cpu, delta_cpu, client_losses, num_samples, wall = await asyncio.to_thread(
                _train_client_sync,
                init_state_cpu,
                clients_model_ctors[i_client],
                clients_train_loaders[i_client],
                "cuda" if torch.cuda.is_available() else "cpu",
                I,
                loss_fn,
                eta_client,
                accumulation_steps,
                early_stopping_patience
            )
            return i_client, local_state_cpu, delta_cpu, client_losses, num_samples, wall

    for t in tqdm(range(num_rounds), desc="AFL Baseline Rounds (Fashion-MNIST)"):
        selected = random.sample(range(C_total), J_per_round)
        selected_clients_log.append(selected)

        # assign staleness τ_i ∈ {0..tau_max_rounds}
        tau_cap = min(t, tau_max_rounds)
        tau_c_map = {i: random.randint(0, tau_cap) for i in selected}

        # launch clients with (possibly stale) initial states
        tasks = []
        for i in selected:
            tau_i = tau_c_map[i]
            hist_idx = -1 - tau_i
            init_state_cpu = server_state_hist[hist_idx] if -len(server_state_hist) <= hist_idx < 0 else server_state_hist[0]
            tasks.append(launch_client(i, init_state_cpu))
        results = await asyncio.gather(*tasks)

        updates_J, taus_J, exec_times, client_losses_round = [], [], [], []
        for i_client, local_state_cpu, delta_cpu, client_losses, num_samples, wall in results:
            exec_times.append(wall)
            client_losses_round.append(sum(client_losses) / max(1, len(client_losses)))
            updates_J.append(delta_cpu)
            taus_J.append(tau_c_map[i_client])
            per_client_losses[i_client].extend(client_losses)

        exec_times_by_round.append(exec_times)
        server_losses.append(sum(client_losses_round) / max(1, len(client_losses_round)))

        # Byzantine simulation (before aggregation)
        if enable_byzantine and len(updates_J) > 0:
            B_count = int(np.floor(byz_frac * len(updates_J)))
            byz_idx = set(random.sample(range(len(updates_J)), B_count)) if B_count > 0 else set()
            for idx in byz_idx:
                updates_J[idx] = byzantine_corrupt_update(updates_J[idx], mode=byz_mode, scale=byz_scale)

        # Aggregate
        u_hat = aggregate_updates(
            updates_J,
            aggregator=aggregator,
            taus=taus_J,
            alpha_stale=alpha_stale,
            w_min=w_min, w_max=w_max,
            trim_ratio=trim_ratio,
            byz_frac=byz_frac,
            krum_m=krum_m
        )

        # Apply server update with fixed gamma
        if u_hat:
            with torch.no_grad():
                cur_sd = server_model.state_dict()
                for k, upd in u_hat.items():
                    if k not in cur_sd:
                        continue
                    if not cur_sd[k].dtype.is_floating_point:
                        continue
                    cur_sd[k].add_(float(gamma_server) * upd.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype))
                server_model.load_state_dict(cur_sd, strict=False)
            server_state_hist.append(clone_state_dict_cpu(server_model.state_dict()))
        else:
            server_state_hist.append(server_state_hist[-1])

        # Monitor staleness
        if len(taus_J) > 0:
            if aggregator == "stale_mean":
                w = normalized_stale_weights(taus_J, alpha_stale, w_min, w_max)
                tau_bar.append(float(np.sum(w * np.array(taus_J, dtype=np.float32))))
            else:
                tau_bar.append(float(np.mean(taus_J)))
        else:
            tau_bar.append(0.0)

        # Test each round
        test_loss, test_acc = evaluate_model(server_model, test_loader, device, loss_fn)
        test_loss_series.append(test_loss)
        test_acc_series.append(test_acc)

    metrics = {
        "tau_bar": tau_bar,
        "test_loss": test_loss_series,
        "test_acc": test_acc_series,
        "aggregator": aggregator,
        "gamma_server": float(gamma_server),
        "alpha_stale": float(alpha_stale),
        "w_min": float(w_min), "w_max": float(w_max),
        "trim_ratio": float(trim_ratio),
        "krum_m": int(krum_m),
        "byzantine": bool(enable_byzantine),
        "byz_frac": float(byz_frac),
    }
    return server_model, per_client_losses, server_losses, selected_clients_log, exec_times_by_round, metrics

# ----------------- Main -----------------
async def main():
    # ---- Hyperparams (aligned with your FMNIST RAFL defaults) ----
    num_clients = 10
    alpha_dirichlet = 0.5
    num_rounds = 500
    local_epochs = 3
    num_clients_per_round = 5
    max_parallel_clients = 5
    batch_size = 128

    eta_client = 1e-2
    gamma_server = 1e-2       # fixed server LR (contrast to γ_t in RAFL)
    aggregator = "stale_mean" # 'mean'|'stale_mean'|'median'|'trimmed_mean'|'krum'

    # staleness/robust params
    alpha_stale = 0.2
    w_min, w_max = 0.5, 1.5
    trim_ratio = 0.2
    krum_m = 1

    # Byzantine toggles
    enable_byzantine = False
    byz_frac = 0.3
    byz_mode = "signflip"
    byz_scale = 10.0

    accumulation_steps = 1
    early_stopping_patience = 5
    tau_max_rounds = 3

    # ---- Data ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST stats
    ])
    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    client_indices = partition_non_iid(train_dataset, num_clients, alpha=alpha_dirichlet, num_classes=10)
    pin = torch.cuda.is_available()

    NUM_WORKERS_TRAIN = 0  # critical for thread safety
    NUM_WORKERS_TEST = 0

    train_loaders, _ = build_client_loaders(
        train_dataset, client_indices, batch_size=batch_size,
        val_size_per_client=128, num_workers=NUM_WORKERS_TRAIN, pin_memory=pin
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=NUM_WORKERS_TEST, pin_memory=pin)

    def make_model():
        return ResNetFashion(BasicBlock, [2, 2, 2], num_classes=10)

    clients_model_ctors = [make_model for _ in range(num_clients)]
    server_model = make_model()
    loss_fn = F.nll_loss

    tag = (f"fmnist_clients_{num_clients}_rounds_{num_rounds}_epochs_{local_epochs}_"
           f"cpr_{num_clients_per_round}_agg_{aggregator}")
    results_dir = create_directory(tag, base_dir="results_afl_baseline_fmnist")

    # ---- Train ----
    server_model, per_client_losses, server_losses, selected_clients, exec_times, metrics = await federated_learning_afl_baseline(
        clients_model_ctors=clients_model_ctors,
        server_model=server_model,
        clients_train_loaders=train_loaders,
        test_loader=test_loader,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        max_clients_per_round=num_clients_per_round,
        max_parallel_clients=max_parallel_clients,
        loss_fn=loss_fn,
        eta_client=eta_client,
        gamma_server=gamma_server,
        aggregator=aggregator,
        alpha_stale=alpha_stale,
        w_min=w_min, w_max=w_max,
        trim_ratio=trim_ratio,
        krum_m=krum_m,
        enable_byzantine=enable_byzantine, byz_frac=byz_frac, byz_mode=byz_mode, byz_scale=byz_scale,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        tau_max_rounds=tau_max_rounds
    )

    # ---- Save outputs (mirrors your RAFL layout for easy comparison) ----
    for i, cl in enumerate(per_client_losses):
        np.save(os.path.join(results_dir, f"client_{i}_losses.npy"), npar(cl))
        plot_series(cl, f"Client {i} Training Losses",
                    os.path.join(results_dir, f"client_{i}_training_loss.png"),
                    xlabel="Local epochs (accumulated)", ylabel="Loss")

    np.save(os.path.join(results_dir, "server_losses.npy"), npar(server_losses))
    plot_series(server_losses, "Server Loss Proxy (avg client loss per round)",
                os.path.join(results_dir, "server_training_loss.png"))

    with open(os.path.join(results_dir, "selected_clients.csv"), "w") as f:
        f.write("Round,Selected Clients\n")
        for r, clist in enumerate(selected_clients, 1):
            f.write(f"{r},{','.join(map(str, clist))}\n")

    with open(os.path.join(results_dir, "execution_times.csv"), "w") as f:
        f.write("Round,ClientIndex,ExecutionTime(s)\n")
        for r, times in enumerate(exec_times, 1):
            for idx, tsec in enumerate(times):
                f.write(f"{r},{idx},{tsec:.6f}\n")

    save_json(metrics, os.path.join(results_dir, "afl_baseline_metrics.json"))
    plot_series(metrics["tau_bar"], "Weighted Avg Staleness $\\bar{\\tau}_t$",
                os.path.join(results_dir, "tau_bar.png"))
    plot_series(metrics["test_loss"], "Test Loss per Round",
                os.path.join(results_dir, "test_loss.png"))
    plot_series(metrics["test_acc"], "Test Accuracy per Round",
                os.path.join(results_dir, "test_acc.png"), ylabel="Accuracy")

    print(f"[DONE] Baseline AFL (Fashion-MNIST) results saved in: {results_dir}")

# ----------------- Entry -----------------
if __name__ == "__main__":
    asyncio.run(main())
