#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 07:54:19 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Asynchronous Federated Learning (RAFL) — CIFAR-10 Demo
Aligned to the PDF up to page 31 (Algorithm 1, definitions, and schedule).

Key changes vs your last script:
- Staleness weights: compute raw w_c = 1/(1+alpha_stale * tau_c),
  then project to the capped simplex so that wmin/J <= \tilde w_c <= wmax/J and sum=1.
- Compute tau_bar over ALL selected clients J(t) using those capped-normalized weights.
- Event-trigger S(t) for ||u_c|| >= epsilon; suppression p_t = 1 - |S|/J.
- Robust aggregation = trim-by-distance-to-median on S(t), then weighted mean
  using the SAME staleness weights restricted to S(t) and re-normalized on the kept set.
- Delay/ET-aware stepsize: zeta_t = zeta0 / (sqrt(t+1) * (1 + alpha_stale * tau_bar) * (1 + beta_et * p_t)).
  Server step: gamma_t = zeta_t / (eta_client * I_local), with optional cap gamma_t <= 1/(4 L eta I).
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
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(3, 3, 3), num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
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
#   Utils: partition, IO
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


# --------------------------
#  Weighting & Aggregation
# --------------------------
def _project_to_capped_simplex(v: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Project v (R^n) onto {w : sum w = 1, lower <= w_i <= upper}.
    Uses bisection on lambda such that w = clip(v - lambda, lower, upper) sums to 1.
    Requires n*lower <= 1 <= n*upper.
    """
    n = v.numel()
    lo = (v - upper).min().item()
    hi = (v - lower).max().item()

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        w = torch.clamp(v - mid, min=lower, max=upper)
        s = w.sum().item()
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    w = torch.clamp(v - hi, min=lower, max=upper)
    # tiny numerical correction
    s = w.sum()
    if s.abs() > 1e-12:
        w = w / (s + 1e-12)
    return w


def compute_capped_normalized_stale_weights(taus, alpha_stale, J, wmin_bound=0.5, wmax_bound=2.0, device="cpu"):
    """
    taus: list[int] or 1D tensor of length J (selected clients' staleness)
    Returns \tilde w over J(t) with wmin/J <= \tilde w_c <= wmax/J and sum=1.
    """
    taus = torch.tensor(taus, dtype=torch.float32, device=device)
    raw = 1.0 / (1.0 + alpha_stale * taus)  # w_c
    # Start from normalized raw weights, then project to box+simplex.
    v = raw / (raw.sum() + 1e-12)
    lower = wmin_bound / float(J)
    upper = wmax_bound / float(J)
    # Feasibility: need wmin_bound <= 1 <= wmax_bound
    assert wmin_bound <= 1.0 <= wmax_bound, "Choose bounds s.t. wmin <= 1 <= wmax."
    w_tilde = _project_to_capped_simplex(v, lower, upper)
    return w_tilde


def robust_trim_and_reweight(updates, base_weights, trim_ratio=0.2):
    """
    updates: list of 1D tensors (flattened updates) for the transmitting clients S(t)
    base_weights: 1D tensor of base weights for the same clients (sum may be <=1), will be renormalized on kept set
    Returns: keep_idx (tensor idx into S), weights_kept (sum=1 on kept set), agg_update (1D tensor)
    """
    U = torch.stack(updates, dim=0)  # [n, P]
    median = U.median(dim=0).values
    dists = torch.norm(U - median, dim=1)
    n = U.shape[0]
    keep_n = max(1, int(math.ceil((1.0 - trim_ratio) * n)))
    keep_idx = torch.topk(-dists, k=keep_n).indices
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
    I_local=5,
    eta_client=1e-3,
    loss_fn=nn.CrossEntropyLoss(),
    delay_sim_max_s=0.0,
    accumulation_steps=1,
):
    if delay_sim_max_s > 0:
        await asyncio.sleep(random.uniform(0, delay_sim_max_s))

    model = model_ctor().to(device)
    model.load_state_dict(base_state, strict=True)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=eta_client)

    total_loss = 0.0
    total_batches = 0

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


def apply_vector_update_to_state(server_state, update_vec, gamma_t, keys, shapes):
    base_vec, _, _ = flatten_state_dict(server_state)
    new_vec = base_vec + gamma_t * update_vec.to(base_vec.device)
    return unflatten_to_state_dict(new_vec, keys, shapes, ref_state=server_state)


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
#        RAFL Runner
# --------------------------
async def run_rafl(
    # Data & model
    num_clients=10,
    alpha_dirichlet=0.5,
    batch_size=64,
    model_ctor=lambda: ResNet(BasicBlock, (3, 3, 3), num_classes=10),
    # FL process
    rounds=200,
    clients_per_round=6,
    I_local=5,
    eta_client=1e-3,
    # Async & staleness
    tau_max=4,
    alpha_stale=0.01,
    delay_sim_max_s=0.0,
    # Event trigger
    epsilon_trigger=1e-3,
    beta_et=0.5,
    # Robust agg
    trim_ratio=0.2,
    # Stepsize schedule (server)
    zeta0=1.0,
    # Stale-weights bounds: wmin/J <= w_tilde_c <= wmax/J  (require wmin <= 1 <= wmax)
    wmin_bound=0.5,
    wmax_bound=2.0,
    # Smoothness cap for gamma_t <= 1 / (4 L eta I)
    L_smooth_cap=1.0,           # set to your estimate of L; 1.0 is a safe placeholder
    use_gamma_cap=True,
    # Byzantine
    byz_frac=0.0,
    byz_mode="signflip",
    byz_scale=5.0,
    # Logging
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
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Non-IID partition
    client_indices = partition_non_iid_cifar(train_dataset, num_clients, alpha=alpha_dirichlet)
    train_loaders = [DataLoader(Subset(train_dataset, idxs), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) for idxs in client_indices]
    client_sizes = [len(idxs) for idxs in client_indices]  # kept for reference; not used in weights (per Algorithm 1)

    # Server model & history for staleness
    server_model = model_ctor().to(device)
    server_model.train()
    server_state = server_model.state_dict()
    server_history = [copy.deepcopy(server_state) for _ in range(tau_max + 1)]

    # Keys/shapes for vector apply
    base_vec0, example_keys, example_shapes = flatten_state_dict(server_state)

    # Logs
    server_losses = []        # proxy: avg client local losses (selected set) per round
    suppression_rates = []
    tau_bar_logs = []         # tau_bar over J(t)
    gamma_logs = []
    zeta_logs = []

    # Gamma cap from smoothness constraint (gamma <= 1 / (4 L eta I))
    gamma_cap = 1.0 / (4.0 * L_smooth_cap * max(1e-12, eta_client) * max(1, I_local)) if use_gamma_cap else float("inf")

    for t in tqdm(range(rounds), desc="RAFL Rounds"):
        # Select clients without replacement
        selected = random.sample(range(num_clients), k=clients_per_round)
        J = len(selected)

        # Assign staleness for selected
        taus_selected = [random.randint(0, tau_max) for _ in selected]
        base_states = [copy.deepcopy(server_history[-(tau + 1)]) for tau in taus_selected]

        # Compute capped-normalized stale weights over ALL selected (Assumption 6)
        w_tilde_all = compute_capped_normalized_stale_weights(
            taus_selected, alpha_stale, J, wmin_bound=wmin_bound, wmax_bound=wmax_bound, device="cpu"
        )  # stays on CPU

        # Weighted average staleness over J(t)
        tau_bar = float((w_tilde_all * torch.tensor(taus_selected, dtype=torch.float32)).sum().item())

        # Determine Byzantines this round
        n_byz = int(math.floor(byz_frac * clients_per_round))
        byz_set_local = set(random.sample(range(clients_per_round), k=n_byz)) if n_byz > 0 else set()

        # Async launch of local updates
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
                loss_fn=nn.CrossEntropyLoss(),
                delay_sim_max_s=delay_sim_max_s,
            )
            return u_vec, keys, shapes, loss

        tasks = [train_one(j) for j in range(clients_per_round)]
        results = await asyncio.gather(*tasks)

        updates_raw = []
        local_losses = []
        for j, (u_vec, keys, shapes, loss) in enumerate(results):
            # optional Byzantine manipulation
            if j in byz_set_local:
                u_vec = byzantine_attack(u_vec, mode=byz_mode, scale=byz_scale)
            updates_raw.append(u_vec)
            local_losses.append(loss)

        # Event-trigger set S(t)
        norms = [u.norm().item() for u in updates_raw]
        S_indices = [i for i, nrm in enumerate(norms) if nrm >= epsilon_trigger]
        p_t = 1.0 - (len(S_indices) / float(J))

        # Stepsize schedule (delay- & ET-aware) uses tau_bar over ALL selected and p_t
        zeta_t = zeta0 / (math.sqrt(t + 1.0) * (1.0 + alpha_stale * tau_bar) * (1.0 + beta_et * p_t))
        gamma_t = zeta_t / max(1e-12, (eta_client * I_local))
        if use_gamma_cap:
            gamma_t = min(gamma_t, gamma_cap)

        # Logging (avg training loss of selected clients as a proxy)
        server_losses.append(float(np.mean(local_losses)))
        suppression_rates.append(p_t)
        tau_bar_logs.append(tau_bar)
        gamma_logs.append(gamma_t)
        zeta_logs.append(zeta_t)

        if len(S_indices) == 0:
            # No transmissions; update is skipped but schedule still progresses
            pass
        else:
            # Build the subset S(t)
            updates_S = [updates_raw[i] for i in S_indices]
            # Use the SAME stale weights, restricted to S(t) and re-normalized there
            base_w_S = w_tilde_all[torch.tensor(S_indices, dtype=torch.long)]
            # Robust trim & weighted mean
            keep_idx_S, weights_kept, agg_update = robust_trim_and_reweight(
                updates_S, base_w_S, trim_ratio=trim_ratio
            )
            # Apply server update: theta^{t+1} = theta^{t} + gamma_t * \hat u^{(t)}
            server_state = apply_vector_update_to_state(server_state, agg_update, gamma_t, example_keys, example_shapes)

        # Push new state into history buffer
        server_model.load_state_dict(server_state, strict=True)
        server_history.append(copy.deepcopy(server_state))
        if len(server_history) > (tau_max + 1):
            server_history.pop(0)

    # Save logs
    torch.save(server_state, os.path.join(results_dir, "server_final_state.pt"))
    np.save(os.path.join(results_dir, "server_losses.npy"), np.array(server_losses))
    np.save(os.path.join(results_dir, "suppression_rates.npy"), np.array(suppression_rates))
    np.save(os.path.join(results_dir, "tau_bar.npy"), np.array(tau_bar_logs))
    np.save(os.path.join(results_dir, "gamma.npy"), np.array(gamma_logs))
    np.save(os.path.join(results_dir, "zeta.npy"), np.array(zeta_logs))

    return {
        "server_model": server_model,
        "server_state": server_state,
        "server_losses": server_losses,
        "suppression_rates": suppression_rates,
        "tau_bar": tau_bar_logs,
        "gamma": gamma_logs,
        "zeta": zeta_logs,
    }


# --------------------------
# Helper to run coroutine
# --------------------------
def run_coro(coro):
    import asyncio
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
    num_clients=10,
    alpha_dirichlet=0.5,
    batch_size=64,
    rounds=50,                  # use 20 for a quick sanity run
    clients_per_round=6,
    I_local=10,
    eta_client=10e-3,

    # asynchrony & staleness
    tau_max=1,
    alpha_stale=0.01,
    delay_sim_max_s=0.0,

    # event trigger
    epsilon_trigger=1e-3,
    beta_et=0.5,

    # robust aggregation
    trim_ratio=0.2,

    # stepsizes (zeta0 chosen for gamma0_target ≈ 0.05 since eta*I = 0.001*10 = 0.01)
    zeta0=5e-4,                  # = 0.05 * 0.001 * 10

    # staleness-weight caps (require wmin <= 1 <= wmax)
    wmin_bound=0.5,
    wmax_bound=2.0,

    # theoretical cap: gamma_t <= 1/(4 L eta I)  -> with L≈250 gives cap ≈ 0.1
    L_smooth_cap=250.0,
    use_gamma_cap=True,

    # Byzantines (off by default)
    byz_frac=0.0,
    byz_mode="signflip",
    byz_scale=5.0,

    # logging/output
    results_dir=f"results_rafl/rafl_{time.strftime('%Y%m%d_%H%M%S')}",
    seed=42,)


    async def main():
        _ = await run_rafl(**cfg)
        print(f"Results written to: {cfg['results_dir']}")

    run_coro(main())
