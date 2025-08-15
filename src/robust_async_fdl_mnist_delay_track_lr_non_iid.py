#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:26:14 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Asynchronous Federated Learning (RAFL) — MNIST Demo
Mirrors the CIFAR-10 RAFL architecture and pipeline.

Features:
- Non-IID client partitioning via Dirichlet(alpha)
- Asynchronous staleness: each selected client starts from a past server state
- Staleness-aware weights: w_c ∝ 1 / (1 + alpha_stale * tau_c)
- Event-triggered communication: send update only if ||u_c|| >= epsilon
- Robust aggregation: filter outliers via distance-to-median, then weighted mean
- Delay/ET-aware stepsize: zeta_t = zeta0 / ( sqrt(t+1) * (1 + alpha_stale * \bar{tau}_t) * (1 + beta_et * p_t) )
- Optional Byzantine clients
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
#   Model: Small CNN (MNIST)
# --------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._init()

    def _init(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.zero_()
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MNISTNet(nn.Module):
    """
    Lightweight CNN roughly analogous in depth/role to the CIFAR ResNet used in the other script.
    """
    def __init__(self, num_classes=10, stem_channels=32, depth=3):
        super().__init__()
        c = stem_channels
        self.blocks = nn.ModuleList()
        self.blocks.append(ConvLayer(1, c))
        for _ in range(depth - 1):
            self.blocks.append(ConvLayer(c, c))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(c, num_classes)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --------------------------
#   Utils: partition, IO
# --------------------------
def partition_non_iid_mnist(dataset, num_clients, alpha=0.5, num_classes=10):
    """
    Dirichlet-based non-IID split, consistent with CIFAR splitter structure.
    """
    data_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        data_by_class[int(label)].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = data_by_class[c]
        np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha] * num_clients)
        counts = (props * len(idxs)).astype(int)
        start = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idxs[start:start+cnt])
            start += cnt
        leftovers = idxs[start:]
        for i, j in enumerate(leftovers):
            client_indices[i % num_clients].append(j)

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices

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

def robust_filter_and_weight(updates, taus, sizes, alpha_stale, trim_ratio=0.2, device="cpu"):
    """
    Distance-to-median filter then staleness- & size-aware weighting.
    """
    if len(updates) == 0:
        return [], torch.tensor([]).to(device), 0.0

    U = torch.stack(updates, dim=0)  # [n, P]
    median = U.median(dim=0).values
    dists = torch.norm(U - median, dim=1)  # [n]

    n = U.shape[0]
    keep_n = max(1, int(math.ceil((1.0 - trim_ratio) * n)))
    keep_idx = torch.topk(-dists, k=keep_n).indices  # smallest distances (via negative)

    taus_keep = torch.tensor([taus[i] for i in keep_idx.tolist()], dtype=torch.float32, device=device)
    sizes_keep = torch.tensor([sizes[i] for i in keep_idx.tolist()], dtype=torch.float32, device=device)

    # staleness-aware (downweight stale)
    w_raw = 1.0 / (1.0 + alpha_stale * taus_keep)
    # combine with data fraction
    sizes_keep = sizes_keep / (sizes_keep.sum() + 1e-12)
    w_raw = w_raw * sizes_keep
    w = w_raw / (w_raw.sum() + 1e-12)

    bar_tau = float((w * taus_keep).sum().item())
    return keep_idx, w, bar_tau

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
    # Optional wall-clock delay simulation
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

# --------------------------
#        RAFL Runner
# --------------------------
async def run_rafl_mnist(
    # Data & model
    num_clients=10,
    alpha_dirichlet=0.5,
    batch_size=64,
    model_ctor=lambda: MNISTNet(num_classes=10, stem_channels=32, depth=3),
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
    results_dir="results_rafl_mnist",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets (MNIST grayscale)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)  # not used here, but handy

    # Non-IID partition
    client_indices = partition_non_iid_mnist(train_dataset, num_clients, alpha=alpha_dirichlet)
    train_loaders = [
        DataLoader(Subset(train_dataset, idxs), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for idxs in client_indices
    ]
    client_sizes = [len(idxs) for idxs in client_indices]

    # Server model & history for staleness
    server_model = model_ctor().to(device)
    server_model.train()
    server_state = server_model.state_dict()
    server_history = [copy.deepcopy(server_state) for _ in range(tau_max + 1)]

    # Bookkeeping for vector shapes
    example_keys, example_shapes = None, None

    # Logs
    server_losses = []
    selected_clients_log = []
    suppression_rates = []
    bar_taus = []

    for t in tqdm(range(rounds), desc="RAFL MNIST Rounds"):
        # Select clients
        selected = random.sample(range(num_clients), k=clients_per_round)
        selected_clients_log.append(selected)

        # Assign staleness
        taus = [random.randint(0, tau_max) for _ in selected]
        base_states = [copy.deepcopy(server_history[-(tau + 1)]) for tau in taus]

        # Byzantines this round
        n_byz = int(math.floor(byz_frac * clients_per_round))
        byz_set = set(random.sample(range(clients_per_round), k=n_byz)) if n_byz > 0 else set()

        # Launch async clients
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
            # Byzantine manipulation
            if j in byz_set:
                u_vec = byzantine_attack(u_vec, mode=byz_mode, scale=byz_scale)
            updates_raw.append(u_vec)
            local_losses.append(loss)

        # Event trigger
        norms = [u.norm().item() for u in updates_raw]
        keep_mask = [norms[i] >= epsilon_trigger for i in range(len(updates_raw))]
        S_indices = [i for i, m in enumerate(keep_mask) if m]
        p_t = 1.0 - (len(S_indices) / float(clients_per_round))  # suppression rate

        if len(S_indices) == 0:
            # No transmissions this round
            suppression_rates.append(p_t)
            bar_taus.append(0.0)
            server_losses.append(float(np.mean(local_losses)))
            # stepsize still decays with p_t (bar_tau = 0)
            zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * 0.0) * (1.0 + beta_et * p_t))
            gamma_t = zeta_t / max(1e-12, (eta_client * I_local))
            # no update applied
        else:
            updates_kept = [updates_raw[i] for i in S_indices]
            taus_kept = [taus[i] for i in S_indices]
            sizes_kept = [client_sizes[selected[i]] for i in S_indices]

            # Robust filter + staleness weighting
            keep_idx_local, weights, bar_tau = robust_filter_and_weight(
                updates_kept, taus_kept, sizes_kept,
                alpha_stale=alpha_stale, trim_ratio=trim_ratio, device=updates_kept[0].device
            )
            if len(keep_idx_local) == 0:
                suppression_rates.append(p_t)
                bar_taus.append(0.0)
                server_losses.append(float(np.mean(local_losses)))
                zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * 0.0) * (1.0 + beta_et * p_t))
                gamma_t = zeta_t / max(1e-12, (eta_client * I_local))
            else:
                chosen_updates = [updates_kept[i] for i in keep_idx_local.tolist()]
                W = weights.view(-1, 1)
                U = torch.stack(chosen_updates, dim=0)
                agg_update = (W * U).sum(dim=0)

                suppression_rates.append(p_t)
                bar_taus.append(bar_tau)
                server_losses.append(float(np.mean(local_losses)))

                zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * bar_tau) * (1.0 + beta_et * p_t))
                gamma_t = zeta_t / max(1e-12, (eta_client * I_local))

                # Apply update
                server_state = apply_vector_update_to_state(server_state, agg_update, gamma_t, example_keys, example_shapes)

        # Push to model & maintain history buffer
        server_model.load_state_dict(server_state, strict=True)
        server_history.append(copy.deepcopy(server_state))
        if len(server_history) > (tau_max + 1):
            server_history.pop(0)

    # Save logs
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
#   Async runner wrapper
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
        num_clients=10,
        alpha_dirichlet=0.5,   # non-IID severity (lower -> more skew)
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
        results_dir=f"results_rafl_mnist/rafl_mnist_{time.strftime('%Y%m%d_%H%M%S')}",
        seed=42,
    )

    async def main():
        _ = await run_rafl_mnist(**cfg)
        print(f"MNIST RAFL results written to: {cfg['results_dir']}")

    run_coro(main())
