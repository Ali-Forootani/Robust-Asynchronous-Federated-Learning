#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:39:57 2025

@author: forootan
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

# =============================================================
# Robust Asynchronous Federated Learning (RAFL) — CIFAR-10 Demo
# =============================================================
# Key features implemented:
# - Non-IID client partitioning via Dirichlet(alpha)
# - Asynchronous staleness: each selected client starts from a past server state
# - Staleness-aware weights: w_c = 1 / (1 + alpha_stale * tau_c)
# - Event-triggered communication: send update only if ||u_c|| >= epsilon
# - Robust aggregation: filter outliers via distance-to-median, then weighted mean
# - Delay/ET-aware stepsize: zeta_t = zeta0 / ( sqrt(t+1) * (1 + alpha_stale * \bar{tau}_t) * (1 + beta_et * p_t) )
# - Byzantine clients simulation (optional): fraction send adversarial updates
# =============================================================

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
    def __init__(self, block=BasicBlock, num_blocks=(3,3,3), num_classes=10):
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
        # Dirichlet proportions per class
        props = np.random.dirichlet([alpha] * num_clients)
        # Convert to counts
        counts = (props * len(idxs)).astype(int)
        # Distribute
        start = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idxs[start:start+cnt])
            start += cnt
        # any leftover (due to int truncation) -> assign arbitrarily
        leftovers = idxs[start:]
        for i, j in enumerate(leftovers):
            client_indices[i % num_clients].append(j)

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices

# Flatten / unflatten state dict helpers

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
        chunk = vec[offset:offset+numel].view(shapes[k])
        out[k] = chunk.to(dtype=ref_state[k].dtype, device=ref_state[k].device)
        offset += numel
    return out

# Robust filter: keep near-median updates

def robust_filter_and_weight(updates, taus, sizes, alpha_stale, trim_ratio=0.2, device="cpu"):
    """
    updates: list[Tensor] shaped (P,) flattened updates
    taus:    list[int] staleness per update
    sizes:   list[int] sample counts per client (for size-based weighting if desired)
    Returns: kept_indices, normalized_weights, bar_tau
    """
    if len(updates) == 0:
        return [], torch.tensor([]).to(device), 0.0

    U = torch.stack(updates, dim=0)  # [n, P]
    median = U.median(dim=0).values
    dists = torch.norm(U - median, dim=1)  # [n]

    n = U.shape[0]
    keep_n = max(1, int(math.ceil((1.0 - trim_ratio) * n)))
    keep_idx = torch.topk(-dists, k=keep_n).indices  # smallest distances

    # Staleness-aware raw weights
    taus_keep = torch.tensor([taus[i] for i in keep_idx.tolist()], dtype=torch.float32, device=device)
    w_raw = 1.0 / (1.0 + alpha_stale * taus_keep)
    # (Optional) multiply by local data fraction
    sizes_keep = torch.tensor([sizes[i] for i in keep_idx.tolist()], dtype=torch.float32, device=device)
    sizes_keep = sizes_keep / (sizes_keep.sum() + 1e-12)
    # combine (can toggle strategy): here: staleness * data fraction
    w_raw = w_raw * sizes_keep
    w = w_raw / (w_raw.sum() + 1e-12)

    bar_tau = float((w * taus_keep).sum().item())
    return keep_idx, w, bar_tau

# Client local training (asynchronous, from a stale snapshot)
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
    # simulate async wall-clock delay (does not change staleness index itself)
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

    # compute delta u_c = theta_local - base_state
    new_state = model.state_dict()
    base_vec, keys, shapes = flatten_state_dict(base_state)
    new_vec, _, _ = flatten_state_dict(new_state)
    update_vec = (new_vec - base_vec).detach().to("cpu")
    avg_epoch_loss = total_loss / max(1, total_batches)
    return update_vec, keys, shapes, avg_epoch_loss

# Apply a vector update to server params

def apply_vector_update_to_state(server_state, update_vec, gamma_t, keys, shapes):
    base_vec, _, _ = flatten_state_dict(server_state)
    new_vec = base_vec + gamma_t * update_vec.to(base_vec.device)
    return unflatten_to_state_dict(new_vec, keys, shapes, ref_state=server_state)

# Optional Byzantine manipulation

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
    model_ctor=lambda: ResNet(BasicBlock, (3,3,3), num_classes=10),
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
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Non-IID partition
    client_indices = partition_non_iid_cifar(train_dataset, num_clients, alpha=alpha_dirichlet)
    train_loaders = [DataLoader(Subset(train_dataset, idxs), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) for idxs in client_indices]
    client_sizes = [len(idxs) for idxs in client_indices]

    # Server model & history for staleness
    server_model = model_ctor().to(device)
    server_model.train()
    server_state = server_model.state_dict()
    server_history = [copy.deepcopy(server_state) for _ in range(tau_max + 1)]  # initialize with current state

    # For vector shape bookkeeping (set once on first client run)
    example_keys, example_shapes = None, None

    # Logs
    server_losses = []  # proxy: average of client local losses those who sent
    selected_clients_log = []
    suppression_rates = []
    bar_taus = []

    # Composite stepsize target: zeta_t = gamma_t * eta_client * I_local
    # => gamma_t computed each round from zeta_t

    for t in tqdm(range(rounds), desc="RAFL Rounds"):
        # Select clients without replacement
        selected = random.sample(range(num_clients), k=clients_per_round)
        selected_clients_log.append(selected)

        # Assign staleness to each selected client
        taus = [random.randint(0, tau_max) for _ in selected]
        base_states = [copy.deepcopy(server_history[-(tau + 1)]) for tau in taus]

        # Determine Byzantines in this round
        n_byz = int(math.floor(byz_frac * clients_per_round))
        byz_set = set(random.sample(range(clients_per_round), k=n_byz)) if n_byz > 0 else set()

        # Launch clients asynchronously
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
            if example_keys is None:
                example_keys, example_shapes = keys, shapes
            # Byzantine attack if chosen
            if j in byz_set:
                u_vec = byzantine_attack(u_vec, mode=byz_mode, scale=byz_scale)
            updates_raw.append(u_vec)
            local_losses.append(loss)

        # Event trigger: keep only updates with norm >= epsilon
        norms = [u.norm().item() for u in updates_raw]
        keep_mask = [norms[i] >= epsilon_trigger for i in range(len(updates_raw))]
        S_indices = [i for i, m in enumerate(keep_mask) if m]
        p_t = 1.0 - (len(S_indices) / float(clients_per_round))  # suppression rate

        if len(S_indices) == 0:
            # No transmissions — fall back to tiny step or skip
            suppression_rates.append(p_t)
            bar_taus.append(0.0)
            server_losses.append(float(np.mean(local_losses)))
            # still decay stepsize schedule (with bar_tau=0)
            zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * 0.0) * (1.0 + beta_et * p_t))
            gamma_t = zeta_t / max(1e-12, (eta_client * I_local))
            # no update applied due to no transmissions
        else:
            updates_kept = [updates_raw[i] for i in S_indices]
            taus_kept = [taus[i] for i in S_indices]
            sizes_kept = [client_sizes[selected[i]] for i in S_indices]

            # Robust filter + staleness weighting
            keep_idx_local, weights, bar_tau = robust_filter_and_weight(
                updates_kept, taus_kept, sizes_kept, alpha_stale=alpha_stale, trim_ratio=trim_ratio, device=updates_kept[0].device
            )
            if len(keep_idx_local) == 0:
                suppression_rates.append(p_t)
                bar_taus.append(0.0)
                server_losses.append(float(np.mean(local_losses)))
                zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * 0.0) * (1.0 + beta_et * p_t))
                gamma_t = zeta_t / max(1e-12, (eta_client * I_local))
            else:
                # Aggregate kept
                chosen_updates = [updates_kept[i] for i in keep_idx_local.tolist()]
                W = weights.view(-1, 1)
                U = torch.stack(chosen_updates, dim=0)
                agg_update = (W * U).sum(dim=0)

                # Stepsize schedule (delay & ET aware)
                suppression_rates.append(p_t)
                bar_taus.append(bar_tau)
                server_losses.append(float(np.mean(local_losses)))

                zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * bar_tau) * (1.0 + beta_et * p_t))
                gamma_t = zeta_t / max(1e-12, (eta_client * I_local))

                # Apply update to current server state
                server_state = apply_vector_update_to_state(server_state, agg_update, gamma_t, example_keys, example_shapes)

        # Load new state to model and push into history buffer
        server_model.load_state_dict(server_state, strict=True)
        server_history.append(copy.deepcopy(server_state))
        if len(server_history) > (tau_max + 1):
            server_history.pop(0)

    # Save basic logs
    torch.save(server_state, os.path.join(results_dir, "server_final_state.pt"))
    np.save(os.path.join(results_dir, "server_losses.npy"), np.array(server_losses))
    np.save(os.path.join(results_dir, "suppression_rates.npy"), np.array(suppression_rates))
    np.save(os.path.join(results_dir, "bar_taus.npy"), np.array(bar_taus))
    with open(os.path.join(results_dir, "selected_clients.csv"), "w") as f:
        f.write("round,selected_client_ids\n")
        for r, sel in enumerate(selected_clients_log, 1):
            f.write(f"{r}," + ",".join(map(str, sel)) + "\n")

    return {
        "server_model": server_model,
        "server_state": server_state,
        "server_losses": server_losses,
        "suppression_rates": suppression_rates,
        "bar_taus": bar_taus,
        "selected_clients": selected_clients_log,
    }



# --------------------------
# --------------------------
# --------------------------


def run_coro(coro):
    import asyncio
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Notebook/REPL case: allow nested event loops
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise




# --------------------------
#      Entry point demo
# --------------------------
if __name__ == "__main__":
    # Example configuration matching the paper-style assumptions
    cfg = dict(
        num_clients=10,
        alpha_dirichlet=0.5,   # non-IID severity
        batch_size=64,
        rounds=200,
        clients_per_round=6,
        I_local=10,
        eta_client=1e-3,
        tau_max=4,
        alpha_stale=0.01,
        delay_sim_max_s=0.0,   # wall-clock delay simulation (seconds)
        epsilon_trigger=1e-3,
        beta_et=0.5,
        trim_ratio=0.2,
        zeta0=1.0,
        byz_frac=0.0,          # set >0 to simulate Byzantine clients
        byz_mode="signflip",
        byz_scale=5.0,
        results_dir=f"results_rafl/rafl_{time.strftime('%Y%m%d_%H%M%S')}",
        seed=42,
    )

    async def main():
        _ = await run_rafl(**cfg)
        print(f"Results written to: {cfg['results_dir']}")

    run_coro(main())
