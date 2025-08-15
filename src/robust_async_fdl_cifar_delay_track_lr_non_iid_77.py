#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:55:27 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Asynchronous Federated Learning (RAFL) — CIFAR-10 (GPU-ready)

This version applies the performance-tuning we discussed:

• Stronger server step (uncapped):
    use_gamma_cap = False
    zeta0 = 5e-3  → with η=0.01 and I=10, γ0 = zeta0 / (η I) = 1.0
• Gentler staleness penalty and wider caps on stale weights:
    alpha_stale = 0.005, wmin_bound = 0.2, wmax_bound = 5.0
• Robustness neutral during tuning:
    epsilon_trigger = 0.0, beta_et = 0.0, trim_ratio = 0.0
• BatchNorm → GroupNorm to avoid stale running-stat issues in async training
• Clients use SGD(momentum, Nesterov) with η=0.01 and I_local=10
• Logs true server loss (train & test) each round + diagnostics

Switch USE_GROUPNORM=False if you want BatchNorm back.
"""

import os
import copy
import time
import math
import random
import asyncio
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# --------------------------
#   Model: Small ResNet-20
# --------------------------
USE_GROUPNORM = True  # <<< change to False to use BatchNorm instead

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = (nn.GroupNorm(groups, out_channels)
                      if USE_GROUPNORM else nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = (nn.GroupNorm(groups, out_channels)
                      if USE_GROUPNORM else nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                (nn.GroupNorm(groups, out_channels)
                 if USE_GROUPNORM else nn.BatchNorm2d(out_channels)),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(3, 3, 3), num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = (nn.GroupNorm(8, 16) if USE_GROUPNORM else nn.BatchNorm2d(16))
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# --------------------------
#   Data partition (non-IID)
# --------------------------
def partition_non_iid_cifar(dataset, num_clients, alpha=0.5, num_classes=10):
    data_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        data_by_class[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = data_by_class[c]
        np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha] * num_clients)
        counts = (props * len(idxs)).astype(int)
        start = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idxs[start:start + cnt])
            start += cnt
        leftovers = idxs[start:]
        for i, j in enumerate(leftovers):
            client_indices[i % num_clients].append(j)

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


# --------------------------
# Flatten / unflatten helpers
# --------------------------
def flatten_state_dict(sd):
    flats = []
    shapes = {}
    for k, v in sd.items():
        t = v.detach().view(-1).to(torch.float32)
        flats.append(t)
        shapes[k] = v.shape
    return torch.cat(flats), list(sd.keys()), shapes


def unflatten_to_state_dict(vec, keys, shapes, ref_state):
    out = {}
    offset = 0
    for k in keys:
        numel = int(np.prod(shapes[k]))
        chunk = vec[offset:offset + numel].view(shapes[k])
        out[k] = chunk.to(dtype=ref_state[k].dtype, device=ref_state[k].device)
        offset += numel
    return out


def apply_vector_update_to_state(server_state, update_vec, gamma_t, keys, shapes):
    base_vec, _, _ = flatten_state_dict(server_state)
    new_vec = base_vec + gamma_t * update_vec.to(base_vec.device)
    return unflatten_to_state_dict(new_vec, keys, shapes, ref_state=server_state)


# --------------------------
#  Weighting & Aggregation
# --------------------------
def _project_to_capped_simplex(v: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Project v (R^n) onto {w : sum w = 1, lower <= w_i <= upper}.
    """
    assert lower <= upper, "lower must be <= upper"
    lo = (v - upper).min().item()
    hi = (v - lower).max().item()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        w = torch.clamp(v - mid, min=lower, max=upper)
        if w.sum().item() > 1.0:
            lo = mid
        else:
            hi = mid
    w = torch.clamp(v - hi, min=lower, max=upper)
    s = w.sum()
    w = w / (s + 1e-12)
    return w


def compute_capped_normalized_stale_weights(taus, alpha_stale, J, wmin_bound=0.2, wmax_bound=5.0, device="cpu"):
    """
    Return \tilde w over J(t) with wmin/J <= \tilde w_c <= wmax/J and sum=1.
    """
    assert wmin_bound <= 1.0 <= wmax_bound, "Require wmin <= 1 <= wmax for feasibility."
    taus = torch.tensor(taus, dtype=torch.float32, device=device)
    raw = 1.0 / (1.0 + alpha_stale * taus)  # w_c
    v = raw / (raw.sum() + 1e-12)
    lower = float(wmin_bound) / float(J)
    upper = float(wmax_bound) / float(J)
    w_tilde = _project_to_capped_simplex(v, lower, upper)
    return w_tilde


def robust_trim_and_reweight(updates, base_weights, trim_ratio=0.0):
    """
    updates: list of 1D tensors (flattened) for S(t)
    base_weights: 1D tensor for those same clients (will be renormalized)
    Returns: keep_idx, weights_kept (sum=1), agg_update (1D tensor)
    """
    U = torch.stack(updates, dim=0)  # [n, P]
    if trim_ratio > 0:
        median = U.median(dim=0).values
        dists = torch.norm(U - median, dim=1)
        n = U.shape[0]
        keep_n = max(1, int(math.ceil((1.0 - trim_ratio) * n)))
        keep_idx = torch.topk(-dists, k=keep_n).indices
    else:
        keep_idx = torch.arange(U.shape[0], dtype=torch.long)

    wk = base_weights[keep_idx]
    wk = wk / (wk.sum() + 1e-12)
    agg_update = (wk.view(-1, 1) * U[keep_idx]).sum(dim=0)
    return keep_idx, wk, agg_update


# --------------------------
#   Client local training
# --------------------------
async def client_local_update(
    client_id,
    base_state,
    model_ctor,
    train_loader,
    device,
    I_local=10,
    eta_client=0.01,
    opt_name="sgd",          # "sgd" or "adam"
    momentum=0.9,
    nesterov=True,
    loss_fn=nn.CrossEntropyLoss(),
    delay_sim_max_s=0.0,
    accumulation_steps=1,
):
    if delay_sim_max_s > 0:
        await asyncio.sleep(random.uniform(0, delay_sim_max_s))

    model = model_ctor().to(device)
    model.load_state_dict(base_state, strict=True)
    model.train()

    if opt_name.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=eta_client, momentum=momentum, nesterov=nesterov)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=eta_client)

    total_loss, total_batches = 0.0, 0

    for _ in range(I_local):
        for b, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if (b + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
            total_batches += 1

    new_state = model.state_dict()
    base_vec, keys, shapes = flatten_state_dict(base_state)
    new_vec, _, _ = flatten_state_dict(new_state)
    update_vec = (new_vec - base_vec).detach().to("cpu")
    avg_epoch_loss = total_loss / max(1, total_batches)
    return update_vec, keys, shapes, avg_epoch_loss


def byzantine_attack(update_vec, mode="signflip", scale=5.0):
    if mode == "signflip":
        return -scale * update_vec
    elif mode == "gaussian":
        return update_vec + scale * torch.randn_like(update_vec)
    elif mode == "random":
        r = torch.randn_like(update_vec)
        r = scale * r / (r.norm() + 1e-12)
        return r
    else:
        return update_vec


# --------------------------
#     Evaluation helper
# --------------------------
@torch.no_grad()
def eval_server_loss(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        tot += float(loss.item()) * x.size(0)
        n += x.size(0)
    model.train()
    return tot / max(1, n)


# --------------------------
#        RAFL Runner
# --------------------------
async def run_rafl(
    # Data & model
    num_clients=10,
    alpha_dirichlet=0.5,
    batch_size=64,
    model_ctor=lambda: ResNet(BasicBlock, (3, 3, 3), num_classes=10),

    # FL process
    rounds=50,
    clients_per_round=6,
    I_local=10,
    eta_client=0.01,
    client_optimizer="sgd",
    momentum=0.9,
    nesterov=True,

    # Async & staleness
    tau_max=1,
    alpha_stale=0.005,
    delay_sim_max_s=0.0,

    # Event trigger (neutral)
    epsilon_trigger=0.0,
    beta_et=0.0,

    # Robust aggregation (neutral)
    trim_ratio=0.0,

    # Stepsize schedule (server)
    zeta0=5e-3,                 # γ0 = zeta0 / (η I) = 1.0
    wmin_bound=0.2,
    wmax_bound=5.0,

    # Smoothness cap (disabled for tuning)
    L_smooth_cap=250.0,
    use_gamma_cap=False,

    # Byzantines
    byz_frac=0.0,
    byz_mode="signflip",
    byz_scale=5.0,

    # Eval/logging
    eval_every=1,
    results_dir="results_rafl",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Full loaders for evaluation
    full_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    full_test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Non-IID partition for clients
    client_indices = partition_non_iid_cifar(train_dataset, num_clients, alpha=alpha_dirichlet)
    train_loaders = [
        DataLoader(Subset(train_dataset, idxs), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for idxs in client_indices
    ]

    # Server model & staleness history
    server_model = model_ctor().to(device)
    server_model.train()
    server_state = server_model.state_dict()
    server_history = [copy.deepcopy(server_state) for _ in range(tau_max + 1)]

    # Keys/shapes for vector apply (fixed layout)
    _, example_keys, example_shapes = flatten_state_dict(server_state)

    # Logs
    server_train_losses = []
    server_test_losses  = []
    client_loss_proxy   = []
    suppression_rates   = []
    tau_bar_logs        = []
    gamma_logs          = []
    zeta_logs           = []

    # Gamma cap (not used if disabled)
    denom = max(1e-12, eta_client) * max(1, I_local)
    gamma_cap = (1.0 / (4.0 * L_smooth_cap * denom)) if use_gamma_cap else float("inf")

    for t in tqdm(range(rounds), desc="RAFL Rounds"):
        # Sample clients
        selected = random.sample(range(num_clients), k=clients_per_round)
        J = len(selected)

        # Assign staleness for selected & get base states
        taus_selected = [random.randint(0, tau_max) for _ in selected]
        base_states = [copy.deepcopy(server_history[-(tau + 1)]) for tau in taus_selected]

        # Capped-normalized stale weights over ALL selected
        w_tilde_all = compute_capped_normalized_stale_weights(
            taus_selected, alpha_stale, J, wmin_bound=wmin_bound, wmax_bound=wmax_bound, device="cpu"
        )
        tau_bar = float((w_tilde_all * torch.tensor(taus_selected, dtype=torch.float32)).sum().item())

        # Byzantines this round
        n_byz = int(math.floor(byz_frac * clients_per_round))
        byz_set_local = set(random.sample(range(clients_per_round), k=n_byz)) if n_byz > 0 else set()

        # Launch local updates asynchronously
        async def train_one(j_in_round):
            c = selected[j_in_round]
            u_vec, keys, shapes, loss = await client_local_update(
                client_id=c,
                base_state=base_states[j_in_round],
                model_ctor=model_ctor,
                train_loader=train_loaders[c],
                device=device,
                I_local=I_local,
                eta_client=eta_client,
                opt_name=client_optimizer,
                momentum=momentum,
                nesterov=nesterov,
                loss_fn=nn.CrossEntropyLoss(),
                delay_sim_max_s=delay_sim_max_s,
            )
            return u_vec, keys, shapes, loss

        results = await asyncio.gather(*[train_one(j) for j in range(clients_per_round)])

        updates_raw, local_losses = [], []
        for j, (u_vec, keys, shapes, loss) in enumerate(results):
            if j in byz_set_local:
                u_vec = byzantine_attack(u_vec, mode=byz_mode, scale=byz_scale)
            updates_raw.append(u_vec)
            local_losses.append(loss)

        # Event-trigger S(t) (neutral → all)
        norms = [u.norm().item() for u in updates_raw]
        S_indices = [i for i, nrm in enumerate(norms) if nrm >= epsilon_trigger]
        p_t = 1.0 - (len(S_indices) / float(J))

        # Schedule & step (uncapped unless enabled)
        zeta_t = zeta0 / (math.sqrt(t + 1.0) * (1.0 + alpha_stale * tau_bar) * (1.0 + beta_et * p_t))
        gamma_t = zeta_t / denom
        if use_gamma_cap:
            gamma_t = min(gamma_t, gamma_cap)

        # Robust aggregation on S(t)
        if len(S_indices) > 0:
            updates_S = [updates_raw[i] for i in S_indices]
            base_w_S = w_tilde_all[torch.tensor(S_indices, dtype=torch.long)]
            _, _, agg_update = robust_trim_and_reweight(updates_S, base_w_S, trim_ratio=trim_ratio)
            server_state = apply_vector_update_to_state(server_state, agg_update, gamma_t, example_keys, example_shapes)

        # Update server model & staleness history
        server_model.load_state_dict(server_state, strict=True)
        server_history.append(copy.deepcopy(server_state))
        if len(server_history) > (tau_max + 1):
            server_history.pop(0)

        # Diagnostics
        client_loss_proxy.append(float(np.mean(local_losses)))
        suppression_rates.append(p_t)
        tau_bar_logs.append(tau_bar)
        gamma_logs.append(gamma_t)
        zeta_logs.append(zeta_t)

        # True server loss (train & test)
        if (t % eval_every) == 0:
            tr = eval_server_loss(server_model, full_train_loader, device)
            te = eval_server_loss(server_model, full_test_loader, device)
            server_train_losses.append(tr)
            server_test_losses.append(te)

    # Save logs
    torch.save(server_state, os.path.join(results_dir, "server_final_state.pt"))
    np.save(os.path.join(results_dir, "server_train_losses.npy"), np.array(server_train_losses))
    np.save(os.path.join(results_dir, "server_test_losses.npy"),  np.array(server_test_losses))
    np.save(os.path.join(results_dir, "client_loss_proxy.npy"),   np.array(client_loss_proxy))
    np.save(os.path.join(results_dir, "suppression_rates.npy"),   np.array(suppression_rates))
    np.save(os.path.join(results_dir, "tau_bar.npy"),             np.array(tau_bar_logs))
    np.save(os.path.join(results_dir, "gamma.npy"),               np.array(gamma_logs))
    np.save(os.path.join(results_dir, "zeta.npy"),                np.array(zeta_logs))

    return {
        "server_model": server_model,
        "server_state": server_state,
        "server_train_losses": server_train_losses,
        "server_test_losses": server_test_losses,
        "client_loss_proxy": client_loss_proxy,
        "suppression_rates": suppression_rates,
        "tau_bar": tau_bar_logs,
        "gamma": gamma_logs,
        "zeta": zeta_logs,
        "results_dir": results_dir,
    }


# --------------------------
# Helper to run coroutine
# --------------------------
def run_coro(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


# --------------------------
#      Entry point demo
# --------------------------
if __name__ == "__main__":
    cfg = dict(
        # Data / participation
        num_clients=10,
        clients_per_round=6,
        alpha_dirichlet=0.5,
        batch_size=64,

        # Rounds & local steps (stronger local training)
        rounds=50,
        I_local=10,
        eta_client=0.01,
        client_optimizer="sgd",
        momentum=0.9,
        nesterov=True,

        # Asynchrony & staleness
        tau_max=1,
        alpha_stale=0.005,
        delay_sim_max_s=0.0,

        # Neutral robustness (enable later)
        epsilon_trigger=0.0,
        beta_et=0.0,
        trim_ratio=0.0,

        # Stepsizes (γ0 ≈ 1.0)
        zeta0=5e-3,
        wmin_bound=0.2,
        wmax_bound=5.0,

        # Server step cap disabled for tuning
        L_smooth_cap=250.0,
        use_gamma_cap=False,

        # Byzantines (off for now)
        byz_frac=0.0,
        byz_mode="signflip",
        byz_scale=5.0,

        # Eval/log
        eval_every=1,
        results_dir=f"results_rafl/rafl_{time.strftime('%Y%m%d_%H%M%S')}",
        seed=42,
    )

    async def main():
        out = await run_rafl(**cfg)
        print(f"Results written to: {out['results_dir']}")

    run_coro(main())
