#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Asynchronous Federated Learning (RAFL) on CIFAR-10
with round-wise test evaluation and heterogeneity proxy ν̂_t^2.

THIS VERSION: identical algorithm & hyperparameters.
Stability fixes for HPC deadlock when iterating DataLoader from threads:
- Train DataLoader uses num_workers=0 (critical).
- Multiprocessing start method 'spawn' (safe).
- Matplotlib backend 'Agg' (headless-safe).
"""

import os
import time
import random
import json
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Callable

# ---- HPC stability: safe start method before importing torch DataLoader workers
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

# Headless plotting safety
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
import asyncio
import nest_asyncio
nest_asyncio.apply()

# ------------------------- Reproducibility -------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------- Model: ResNet (small) -------------------------
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_ch = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
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

# ------------------------- Utilities -------------------------
def create_directory(num_clients, num_rounds, local_epochs, max_clients_per_round, base_dir="results_rafl"):
    dn = os.path.join(
        base_dir,
        f"cifar_clients_{num_clients}_rounds_{num_rounds}_epochs_{local_epochs}_"
        f"clients_per_round_{max_clients_per_round}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(dn, exist_ok=True)
    return dn

def plot_series(y, title, save_path, xlabel="Rounds", ylabel="Value"):
    plt.figure(figsize=(10, 6))
    plt.plot(y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True); plt.tight_layout()
    plt.savefig(save_path); plt.close()

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def npar(x): return np.array(x, dtype=np.float32)

def dict_add_inplace(dst: Dict[str, torch.Tensor], src: Dict[str, torch.Tensor], alpha: float = 1.0):
    for k in dst.keys():
        if not (dst[k].dtype.is_floating_point and src[k].dtype.is_floating_point):
            continue
        dst[k].add_(alpha * src[k])

def dict_scaled_copy(src: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    return {k: (alpha * v.clone()) for k, v in src.items() if v.dtype.is_floating_point}

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

def clone_state_dict_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}

# ------------------------- Data partition (Dirichlet) -------------------------
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
    dataset,
    client_indices: List[List[int]],
    batch_size: int,
    val_size_per_client: int = 256,
    num_workers: int = 0,           # CRITICAL: 0 to avoid deadlocks in threads
    pin_memory: bool = False
):
    """
    Split each client's indices into train and a small validation slice (for ν̂_t^2).
    num_workers MUST be 0 when iterating loaders from background threads.
    """
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
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,              # <--- keep 0 (important)
            pin_memory=pin_memory,
            persistent_workers=False
        ))
        val_loaders.append(DataLoader(
            Subset(dataset, val_ids),
            batch_size=min(128, vsize),
            shuffle=False,
            num_workers=0,              # safe
            pin_memory=pin_memory
        ))
    return train_loaders, val_loaders

def get_one_batch(loader: DataLoader, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    for x, y in loader:
        return x.to(device), y.to(device)
    return None

# ------------------------- Byzantine simulation -------------------------
def byzantine_corrupt_update(update: Dict[str, torch.Tensor], mode="signflip", scale=10.0):
    bad = {}
    for k, v in update.items():
        if not v.dtype.is_floating_point:
            continue
        t = v.detach().cpu().float()
        if mode == "signflip":
            bad[k] = (-t) * float(scale)
        elif mode == "gaussian":
            bad[k] = t + torch.randn_like(t) * float(scale)
        else:
            bad[k] = t
    return bad

# ------------------------- Client training (sync in a thread) -------------------------
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
        # handle leftover grads if accumulation_steps > 1 (no hyperparam change; accumulation_steps=1 by default)
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

# ------------------------- Staleness weights & ASB aggregation -------------------------
def normalized_stale_weights(tau_list: List[int], alpha_stale: float, w_min: float, w_max: float) -> np.ndarray:
    J = len(tau_list)
    raw = np.array([1.0 / (1.0 + alpha_stale * max(0.0, float(tau))) for tau in tau_list], dtype=np.float32)
    raw_sum = raw.sum()
    if raw_sum <= 0:
        raw = np.ones_like(raw); raw_sum = raw.sum()
    wtilde = raw / raw_sum
    wtilde = np.clip(wtilde, w_min / J, w_max / J)
    wtilde = wtilde / wtilde.sum()
    return wtilde

def weighted_mean_updates(updates: List[Dict[str, torch.Tensor]], weights: np.ndarray) -> Dict[str, torch.Tensor]:
    all_keys = [k for k, v in updates[0].items() if v.dtype.is_floating_point]
    agg = {k: torch.zeros_like(updates[0][k].float().cpu()) for k in all_keys}
    for u, w in zip(updates, weights):
        for k in all_keys:
            agg[k] += float(w) * u[k].float().cpu()
    return agg

def asb_aggregate(
    updates_S: List[Dict[str, torch.Tensor]],
    taus_J: List[int],
    indices_S_in_J: List[int],
    alpha_stale: float,
    w_min: float, w_max: float,
    trim_B: int = 0
) -> Tuple[Dict[str, torch.Tensor], float]:
    if len(updates_S) == 0:
        return {}, 0.0

    wtilde_J = normalized_stale_weights(taus_J, alpha_stale, w_min, w_max)
    w_S = np.array([wtilde_J[j_idx] for j_idx in indices_S_in_J], dtype=np.float32)
    w_S = w_S / w_S.sum()

    mean_before = weighted_mean_updates(updates_S, w_S)

    if trim_B > 0 and trim_B < len(updates_S):
        norms = [l2_norm_of_update(u) for u in updates_S]
        order = np.argsort(norms)[::-1]
        drop = set(order[:trim_B].tolist())
        kept_updates = [u for i, u in enumerate(updates_S) if i not in drop]
        kept_weights = np.array([w for i, w in enumerate(w_S) if i not in drop], dtype=np.float32)
        kept_weights = kept_weights / kept_weights.sum()
    else:
        kept_updates = updates_S
        kept_weights = w_S

    mean_after = weighted_mean_updates(kept_updates, kept_weights)

    keys = list(mean_after.keys())
    if not keys:
        return {}, 0.0
    vec_diff = dict_to_vec({k: mean_before[k] - mean_after[k] for k in keys})
    rho_hat_sq = float(torch.dot(vec_diff, vec_diff).item())
    return mean_after, rho_hat_sq

# ------------------------- Eval & Heterogeneity -------------------------
@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, loss_fn) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

def grad_vector_at_batch(model_ctor: Callable[[], nn.Module],
                         state_cpu: Dict[str, torch.Tensor],
                         batch: Tuple[torch.Tensor, torch.Tensor],
                         device: torch.device, loss_fn) -> torch.Tensor:
    model = model_ctor().to(device)
    model.load_state_dict(state_cpu, strict=False)
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    x, y = batch
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if len(grads) == 0:
        return torch.zeros(1, device=device)
    return torch.cat(grads)

def heterogeneity_proxy_nu2(
    model_ctor: Callable[[], nn.Module],
    server_state_cpu: Dict[str, torch.Tensor],
    selected_clients: List[int],
    client_val_loaders: List[DataLoader],
    device: torch.device,
    loss_fn
) -> float:
    grad_list = []
    for cid in selected_clients:
        batch = get_one_batch(client_val_loaders[cid], device)
        if batch is None:
            continue
        g = grad_vector_at_batch(model_ctor, server_state_cpu, batch, device, loss_fn)
        if g.numel() == 0:
            continue
        grad_list.append(g)
    if len(grad_list) <= 1:
        return 0.0
    G = torch.stack(grad_list, dim=0)
    g_bar = torch.mean(G, dim=0)
    diffs = G - g_bar
    vals = torch.sum(diffs * diffs, dim=1)
    nu2 = float(torch.mean(vals).item())
    return nu2

# ------------------------- Federated loop (async + threads) -------------------------
async def federated_learning_rafl(
    clients_model_ctors: List[Callable[[], nn.Module]],
    server_model: nn.Module,
    clients_train_loaders: List[DataLoader],
    clients_val_loaders: List[DataLoader],
    test_loader: DataLoader,
    num_rounds=10, local_epochs=1, max_clients_per_round=3,
    max_parallel_clients=None,
    loss_fn=None,
    eta_client=1e-2,
    zeta0=1e-2,
    alpha_stale=0.1,
    beta_et=0.0,
    L_smooth=1.0,
    tau_max_rounds=3,
    trigger_eps=0.0,
    w_min=0.5, w_max=1.5,
    enable_byzantine=False, byz_frac=0.0, byz_mode="signflip", byz_scale=10.0,
    asb_trim=True,
    accumulation_steps=1, early_stopping_patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    I = local_epochs
    J_per_round = max_clients_per_round
    C_total = len(clients_model_ctors)

    if max_parallel_clients is None:
        max_parallel_clients = max_clients_per_round
    sem = asyncio.Semaphore(max_parallel_clients)

    server_model.to(device)
    history_len = tau_max_rounds + 1
    server_state_hist = deque(maxlen=history_len)
    server_state_hist.append(clone_state_dict_cpu(server_model.state_dict()))

    per_client_losses = [[] for _ in range(C_total)]
    server_losses = []
    selected_clients_log = []
    exec_times_by_round = []

    tau_bar = []
    p_supp = []
    zeta_ts, gamma_ts = [], []
    kappa_sq = max((w_max - 1.0) ** 2, (1.0 - w_min) ** 2)
    rho_sb_hat = []
    s_min_series = []
    D_T_accum = 0.0
    eps_lin_series = []
    nu2_series = []

    test_loss_series = []
    test_acc_series = []

    alpha_t = (eta_client ** 2) * (L_smooth ** 2) * I * (I - 1)
    if alpha_t >= 0.5:
        print(f"[WARN] η^2 L^2 I(I-1) = {alpha_t:.3f} ≥ 1/2. Consider smaller η and/or I to satisfy theory precondition.")

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

    for t in tqdm(range(num_rounds), desc="RAFL Rounds"):
        selected = random.sample(range(C_total), J_per_round)
        selected_clients_log.append(selected)

        tau_cap = min(t, tau_max_rounds)
        tau_c_map = {i: random.randint(0, tau_cap) for i in selected}

        tasks = []
        for i in selected:
            tau_i = tau_c_map[i]
            hist_idx = -1 - tau_i
            init_state_cpu = server_state_hist[hist_idx] if -history_len <= hist_idx < 0 else server_state_hist[0]
            tasks.append(launch_client(i, init_state_cpu))
        results = await asyncio.gather(*tasks)

        updates_J = []
        taus_J = []
        norms_J = []
        exec_times = []
        client_losses_round = []

        for i_client, local_state_cpu, delta_cpu, client_losses, num_samples, wall in results:
            exec_times.append(wall)
            client_losses_round.append(sum(client_losses) / max(1, len(client_losses)))
            updates_J.append(delta_cpu)
            taus_J.append(tau_c_map[i_client])
            norms_J.append(l2_norm_of_update(delta_cpu))
            per_client_losses[i_client].extend(client_losses)

        S_indices_in_J = [j for j, nrm in enumerate(norms_J) if nrm >= trigger_eps]
        p_t = 1.0 - (len(S_indices_in_J) / max(1, len(selected)))
        p_supp.append(p_t)
        server_losses.append(sum(client_losses_round) / max(1, len(client_losses_round)))
        exec_times_by_round.append(exec_times)

        wtilde_J = normalized_stale_weights(taus_J, alpha_stale, w_min, w_max)
        tau_bar_t = float(np.sum(wtilde_J * np.array(taus_J, dtype=np.float32)))
        tau_bar.append(tau_bar_t)

        A_t = (1.0 + alpha_stale * tau_bar_t) * (1.0 + beta_et * p_t)
        zeta_t = zeta0 / (np.sqrt(t + 1.0) * A_t)
        zeta_cap = 1.0 / (4.0 * L_smooth)
        if zeta_t > zeta_cap:
            zeta_t = zeta_cap
        zeta_ts.append(float(zeta_t))
        gamma_t = float(zeta_t / (eta_client * I))
        gamma_ts.append(gamma_t)

        D_T_accum += A_t

        updates_S = [updates_J[j] for j in S_indices_in_J]

        if enable_byzantine and len(updates_S) > 0:
            B_count = int(np.floor(byz_frac * len(updates_S)))
            byz_idx = set(random.sample(range(len(updates_S)), B_count)) if B_count > 0 else set()
            for idx in byz_idx:
                updates_S[idx] = byzantine_corrupt_update(updates_S[idx], mode=byz_mode, scale=byz_scale)

        trim_B = int(np.floor(byz_frac * len(updates_S))) if asb_trim else 0
        u_hat, rho_hat_sq = asb_aggregate(
            updates_S=updates_S,
            taus_J=taus_J,
            indices_S_in_J=S_indices_in_J,
            alpha_stale=alpha_stale,
            w_min=w_min, w_max=w_max,
            trim_B=trim_B
        )
        rho_sb_hat.append(rho_hat_sq)

        server_state_cpu_now = clone_state_dict_cpu(server_model.state_dict())
        nu2_t = heterogeneity_proxy_nu2(
            model_ctor=clients_model_ctors[0],
            server_state_cpu=server_state_cpu_now,
            selected_clients=selected,
            client_val_loaders=clients_val_loaders,
            device=device,
            loss_fn=loss_fn
        )
        nu2_series.append(nu2_t)

        if u_hat:
            with torch.no_grad():
                cur_sd = server_model.state_dict()
                for k, upd in u_hat.items():
                    if k not in cur_sd:
                        continue
                    if not cur_sd[k].dtype.is_floating_point:
                        continue
                    upd_fp = upd.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype)
                    cur_sd[k].add_(gamma_t * upd_fp)
                server_model.load_state_dict(cur_sd, strict=False)
            server_state_hist.append(clone_state_dict_cpu(server_model.state_dict()))
        else:
            server_state_hist.append(server_state_hist[-1])

        B_round = int(np.floor(byz_frac * J_per_round)) if enable_byzantine else 0
        s_min = w_min * (1.0 - float(B_round) / max(1, J_per_round))
        s_min_series.append(s_min)
        eps_lin_series.append(1.0 + rho_hat_sq)

        test_loss, test_acc = evaluate_model(server_model, test_loader, device, loss_fn)
        test_loss_series.append(test_loss)
        test_acc_series.append(test_acc)

    DT = D_T_accum / max(1, num_rounds)
    metrics = {
        "tau_bar": tau_bar,
        "p_t": p_supp,
        "zeta_t": zeta_ts,
        "gamma_t": gamma_ts,
        "kappa_sq": kappa_sq,
        "rho_sb_hat": rho_sb_hat,
        "s_min": s_min_series,
        "D_T": DT,
        "alpha_t": alpha_t,
        "nu2": nu2_series,
        "test_loss": test_loss_series,
        "test_acc": test_acc_series
    }
    return server_model, per_client_losses, server_losses, selected_clients_log, exec_times_by_round, metrics

# ------------------------- Main -------------------------
async def main():
    # ------------------ Hyperparameters cifar------------------
    num_clients = 10
    alpha_dirichlet = 0.5
    num_rounds = 200
    local_epochs = 10
    num_clients_per_round = 5
    max_parallel_clients = 5
    batch_size = 64

    eta_client = 5e-3
    zeta0 = 1e-2
    L_smooth = 1.0
    alpha_stale = 0.2
    beta_et = 0.5

    tau_max_rounds = 3
    trigger_eps = 0.05

    w_min, w_max = 0.5, 1.5

    enable_byzantine = False
    byz_frac = 0.3
    byz_mode = "signflip"
    byz_scale = 10.0
    asb_trim = True

    accumulation_steps = 1
    early_stopping_patience = 10

    # ------------------ Data ------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    client_indices = partition_non_iid(train_dataset, num_clients, alpha=alpha_dirichlet, num_classes=10)
    pin = torch.cuda.is_available()

    # CRITICAL: 0 workers for thread-safe iteration
    NUM_WORKERS_TRAIN = 0
    NUM_WORKERS_TEST = 0

    train_loaders, val_small_loaders = build_client_loaders(
        train_dataset, client_indices, batch_size=batch_size,
        val_size_per_client=256, num_workers=NUM_WORKERS_TRAIN, pin_memory=pin
    )

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=NUM_WORKERS_TEST, pin_memory=pin)

    def make_model():
        return ResNet(BasicBlock, [3, 3, 3], num_classes=10)

    clients_model_ctors = [make_model for _ in range(num_clients)]
    server_model = make_model()

    loss_fn = F.nll_loss

    results_dir = create_directory(
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        max_clients_per_round=num_clients_per_round,
        base_dir="results_rafl"
    )

    server_model, per_client_losses, server_losses, selected_clients, exec_times, metrics = await federated_learning_rafl(
        clients_model_ctors=clients_model_ctors,
        server_model=server_model,
        clients_train_loaders=train_loaders,
        clients_val_loaders=val_small_loaders,
        test_loader=test_loader,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        max_clients_per_round=num_clients_per_round,
        max_parallel_clients=max_parallel_clients,
        loss_fn=loss_fn,
        eta_client=eta_client,
        zeta0=zeta0,
        alpha_stale=alpha_stale,
        beta_et=beta_et,
        L_smooth=L_smooth,
        tau_max_rounds=tau_max_rounds,
        trigger_eps=trigger_eps,
        w_min=w_min, w_max=w_max,
        enable_byzantine=enable_byzantine,
        byz_frac=byz_frac,
        byz_mode=byz_mode,
        byz_scale=byz_scale,
        asb_trim=asb_trim,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience
    )

    # ------------------ Save outputs ------------------
    for i, cl in enumerate(per_client_losses):
        np.save(os.path.join(results_dir, f"client_{i}_losses.npy"), npar(cl))
        plot_series(cl, f"Client {i} Training Losses", os.path.join(results_dir, f"client_{i}_training_loss.png"),
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

    save_json(metrics, os.path.join(results_dir, "rafl_metrics.json"))
    plot_series(metrics["tau_bar"], "Weighted Average Staleness $\\bar{\\tau}_t$", os.path.join(results_dir, "tau_bar.png"))
    plot_series(metrics["p_t"], "Suppression Rate $p_t$", os.path.join(results_dir, "p_t.png"))
    plot_series(metrics["zeta_t"], "Composite Step Size $\\zeta_t$", os.path.join(results_dir, "zeta_t.png"))
    plot_series(metrics["gamma_t"], "Server Step Size $\\gamma_t$", os.path.join(results_dir, "gamma_t.png"))
    plot_series(metrics["rho_sb_hat"], "Proxy $\\widehat{\\rho}_{SB,t}^2$", os.path.join(results_dir, "rho_sb_hat.png"))
    plot_series(metrics["s_min"], "$s_{\\min}$", os.path.join(results_dir, "s_min.png"))

    np.save(os.path.join(results_dir, "nu2.npy"), npar(metrics["nu2"]))
    plot_series(metrics["nu2"], "Heterogeneity Proxy $\\hat{\\nu}_t^2$", os.path.join(results_dir, "nu2.png"))
    np.save(os.path.join(results_dir, "test_loss.npy"), npar(metrics["test_loss"]))
    np.save(os.path.join(results_dir, "test_acc.npy"), npar(metrics["test_acc"]))
    plot_series(metrics["test_loss"], "Test Loss per Round", os.path.join(results_dir, "test_loss.png"))
    plot_series(metrics["test_acc"], "Test Accuracy per Round", os.path.join(results_dir, "test_acc.png"),
                ylabel="Accuracy")

    print(f"[DONE] Results saved in: {results_dir}")

# ------------------------- Entry -------------------------
if __name__ == "__main__":
    asyncio.run(main())
