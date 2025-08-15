#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:35:26 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Asynchronous Federated Learning (RAFL) — CIFAR-10 Demo

Implements Algorithm (RAFL) exactly as specified in the user's formulation:
- Notation, indices, client sampling without replacement
- Local SGD with I steps from a stale snapshot theta^{(t - tau_c)}
- Staleness-aware weighting w_c = 1 / (1 + alpha * tau_c), normalized to \tilde w_c
- Event-triggered communication: send only if ||u_c|| >= eps
- Robust aggregation: trim by distance-to-median, then weighted mean
- Server update: theta^{(t+1)} = theta^{(t)} + gamma_t * \hat u^{(t)}
- Composite stepsize: zeta_t = gamma_t * eta * I = zeta0 / (sqrt(t+1) (1 + alpha * bar_tau_t) (1 + beta * p_t))

Notes:
* To keep gamma_t in a stable range, we set by default zeta0 = gamma0 * eta * I with gamma0=0.1.
  You can override zeta0 explicitly if you want full control.

Author: forootan (adapted & cleaned)
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

# ------------------------------------------------------------
#  Small ResNet-20 for CIFAR-10
# ------------------------------------------------------------
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

# ------------------------------------------------------------
#  Utilities
# ------------------------------------------------------------
def partition_non_iid_cifar(dataset, num_clients, alpha=0.5, num_classes=10):
    """Dirichlet(alpha) class-proportions per client (non-IID)."""
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
    flats, shapes, keys = [], {}, []
    for k, v in sd.items():
        t = v.detach().view(-1).to(torch.float32)
        flats.append(t)
        shapes[k] = v.shape
        keys.append(k)
    return torch.cat(flats), keys, shapes

def unflatten_to_state_dict(vec, keys, shapes, ref_state):
    out, offset = {}, 0
    for k in keys:
        numel = int(np.prod(shapes[k]))
        chunk = vec[offset:offset+numel].view(shapes[k])
        out[k] = chunk.to(dtype=ref_state[k].dtype, device=ref_state[k].device)
        offset += numel
    return out

# ------------------------------------------------------------
#  Robust filter: trim by distance-to-median then weighted mean
# ------------------------------------------------------------
def robust_filter_and_weight(updates, taus, sizes, alpha_stale, trim_ratio=0.2, wmin_over_J=None, wmax_over_J=None, device="cpu"):
    """
    updates: list[Tensor] flattened updates u_c
    taus: staleness list
    sizes: local dataset sizes (for size-aware weighting)
    Returns: kept_indices, normalized_weights, bar_tau
    """
    if len(updates) == 0:
        return [], torch.tensor([], device=device), 0.0

    U = torch.stack(updates, dim=0)         # [n, P]
    median = U.median(dim=0).values
    dists = torch.norm(U - median, dim=1)   # [n]

    n = U.shape[0]
    keep_n = max(1, int(math.ceil((1.0 - trim_ratio) * n)))
    keep_idx = torch.topk(-dists, k=keep_n).indices  # smallest distances

    taus_keep = torch.tensor([taus[i] for i in keep_idx.tolist()], dtype=torch.float32, device=device)
    sizes_keep = torch.tensor([sizes[i] for i in keep_idx.tolist()], dtype=torch.float32, device=device)
    sizes_keep = sizes_keep / (sizes_keep.sum() + 1e-12)

    # staleness-aware * data fraction
    w_raw = (1.0 / (1.0 + alpha_stale * taus_keep)) * sizes_keep

    # Optional clipping to [wmin/J, wmax/J] AFTER normalization (approximate)
    w = w_raw / (w_raw.sum() + 1e-12)
    if (wmin_over_J is not None) or (wmax_over_J is not None):
        if wmin_over_J is None: wmin_over_J = 0.0
        if wmax_over_J is None: wmax_over_J = 1.0
        w = torch.clamp(w, min=wmin_over_J, max=wmax_over_J)
        w = w / (w.sum() + 1e-12)

    bar_tau = float((w * taus_keep).sum().item())
    return keep_idx, w, bar_tau

# ------------------------------------------------------------
#  Client local training (asynchronous, from stale snapshot)
#  Also returns Xi contribution: sum_i ||theta_{c,i} - theta^{(t)}||^2
# ------------------------------------------------------------
async def client_local_update(
    client_id,
    base_state,                  # theta^{(t - tau_c)}
    server_theta_t_vec,          # flattened theta^{(t)} for Xi_t proxy
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

    total_loss, total_batches = 0.0, 0
    Xi_contrib = 0.0

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

            # Xi contribution at this inner iterate
            # (||theta_{c,i} - theta^{(t)}||^2)
            c_state = model.state_dict()
            c_vec, _, _ = flatten_state_dict(c_state)
            Xi_contrib += float(torch.norm(c_vec.detach().cpu() - server_theta_t_vec).item() ** 2)

    # compute delta u_c = theta_local - base_state
    new_state = model.state_dict()
    base_vec, keys, shapes = flatten_state_dict(base_state)
    new_vec, _, _ = flatten_state_dict(new_state)
    update_vec = (new_vec - base_vec).detach().to("cpu")
    avg_epoch_loss = total_loss / max(1, total_batches)
    return update_vec, keys, shapes, avg_epoch_loss, Xi_contrib

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

# ------------------------------------------------------------
#  Evaluation on test set
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        tot_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        tot_correct += int((pred == y).sum().item())
        tot += x.size(0)
    return tot_loss / tot, tot_correct / tot

# ------------------------------------------------------------
#  RAFL Runner (Algorithm lines match comments)
# ------------------------------------------------------------
async def run_rafl(
    # Data & model
    num_clients=10,
    alpha_dirichlet=0.5,
    batch_size=64,
    model_ctor=lambda: ResNet(BasicBlock, (3,3,3), num_classes=10),
    # FL process
    rounds=200,
    clients_per_round=6,
    I_local=10,
    eta_client=1e-3,
    # Async & staleness
    tau_max=4,
    alpha_stale=0.01,         # alpha in weights and schedule
    delay_sim_max_s=0.0,
    # Event trigger
    epsilon_trigger=1e-3,
    beta_et=0.5,
    # Robust agg
    trim_ratio=0.2,
    wmin=0.5,                 # bounds for normalized weights (per-J scaling applied below)
    wmax=1.5,
    # Stepsizes
    zeta0=None,               # composite stepsize base; if None, set to eta*I*gamma0
    gamma0=0.1,               # used only if zeta0 is None
    gamma_cap=1.0,            # safety cap
    # Trust region & per-update clipping (stability)
    trust_region_frac=0.02,   # cap ||agg_update|| <= trust_region_frac * ||theta||
    clip_updates=True,
    clip_multiplier=2.5,      # cap each client's ||u|| <= 2.5 * median(||u||)
    # Byzantine
    byz_frac=0.0, byz_mode="signflip", byz_scale=5.0,
    # Logging
    results_dir="results_rafl",
    eval_every=5,
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
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    client_sizes = [len(idxs) for idxs in client_indices]

    # Server model & history for staleness
    server_model = model_ctor().to(device)
    server_state = server_model.state_dict()
    server_history = [copy.deepcopy(server_state) for _ in range(tau_max + 1)]  # circular buffer

    # For vector shape bookkeeping
    example_keys, example_shapes = None, None

    # Logs
    server_losses = []     # proxy: avg local losses of transmitters
    test_losses, test_accs = [], []
    suppression_rates = []
    bar_taus = []
    Xi_list = []

    # zeta0 (composite) default
    if zeta0 is None:
        zeta0 = eta_client * I_local * gamma0  # ensures gamma_t starts near gamma0

    for t in tqdm(range(rounds), desc="RAFL Rounds"):
        # ---------- Algorithm lines 1–2: client sampling ----------
        selected = random.sample(range(num_clients), k=clients_per_round)
        J = clients_per_round

        # Bounded normalized weight interval (per-J scaling)
        wmin_over_J = (wmin / J) if wmin is not None else None
        wmax_over_J = (wmax / J) if wmax is not None else None

        # Assign staleness to selected clients and collect base states
        taus = [random.randint(0, tau_max) for _ in selected]
        base_states = [copy.deepcopy(server_history[-(tau + 1)]) for tau in taus]

        # Choose Byzantines this round
        n_byz = int(math.floor(byz_frac * J))
        byz_set = set(random.sample(range(J), k=n_byz)) if n_byz > 0 else set()

        # Flatten current theta^{(t)} for Xi proxy
        theta_t_vec, _, _ = flatten_state_dict(server_state)
        theta_t_vec = theta_t_vec.detach().cpu()

        # ---------- Parallel local training ----------
        async def train_one(j_in_round):
            c = selected[j_in_round]
            u_vec, keys, shapes, loss, Xi_contrib = await client_local_update(
                client_id=c,
                base_state=base_states[j_in_round],
                server_theta_t_vec=theta_t_vec,
                model_ctor=model_ctor,
                train_loader=train_loaders[c],
                device=device,
                I_local=I_local,
                eta_client=eta_client,
                loss_fn=nn.CrossEntropyLoss(),
                delay_sim_max_s=delay_sim_max_s,
            )
            return u_vec, keys, shapes, loss, Xi_contrib

        tasks = [train_one(j) for j in range(J)]
        results = await asyncio.gather(*tasks)

        updates_raw, local_losses, Xi_contribs = [], [], []
        for j, (u_vec, keys, shapes, loss, Xi_contrib) in enumerate(results):
            if example_keys is None:
                example_keys, example_shapes = keys, shapes
            if j in byz_set:  # Byzantine manipulation
                u_vec = byzantine_attack(u_vec, mode=byz_mode, scale=byz_scale)
            updates_raw.append(u_vec)
            local_losses.append(loss)
            Xi_contribs.append(Xi_contrib)

        # Optional per-update clipping to median norm (stability)
        if clip_updates and len(updates_raw) > 0:
            norms = torch.tensor([u.norm() for u in updates_raw], dtype=torch.float32)
            med = norms.median().item() if norms.numel() > 0 else 0.0
            cap = clip_multiplier * (med + 1e-12)
            updates_raw = [u * min(1.0, cap / (u.norm() + 1e-12)) for u in updates_raw]

        # ---------- Event-trigger: S^{(t)} ----------
        norms = [u.norm().item() for u in updates_raw]
        keep_mask = [norms[i] >= epsilon_trigger for i in range(len(updates_raw))]
        S_indices = [i for i, m in enumerate(keep_mask) if m]
        p_t = 1.0 - (len(S_indices) / float(J))   # suppression rate
        suppression_rates.append(p_t)

        # Default logs for the round
        Xi_t = float(np.sum(Xi_contribs))
        Xi_list.append(Xi_t)
        server_losses.append(float(np.mean(local_losses)))

        # If no transmissions, just decay stepsize schedule (bar_tau=0)
        if len(S_indices) == 0:
            bar_tau = 0.0
            bar_taus.append(bar_tau)
            # zeta_t schedule AND gamma_t = zeta_t / (eta * I)
            zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * bar_tau) * (1.0 + beta_et * p_t))
            gamma_t = min(zeta_t / max(eta_client * I_local, 1e-12), gamma_cap)
            # No update applied
        else:
            # ---------- Robust aggregation ----------
            updates_kept = [updates_raw[i] for i in S_indices]
            taus_kept    = [taus[i] for i in S_indices]
            sizes_kept   = [client_sizes[selected[i]] for i in S_indices]

            keep_idx_local, weights, bar_tau = robust_filter_and_weight(
                updates_kept, taus_kept, sizes_kept,
                alpha_stale=alpha_stale,
                trim_ratio=trim_ratio,
                wmin_over_J=wmin_over_J,
                wmax_over_J=wmax_over_J,
                device=updates_kept[0].device
            )
            if len(keep_idx_local) == 0:
                bar_tau = 0.0
                bar_taus.append(bar_tau)
                zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * bar_tau) * (1.0 + beta_et * p_t))
                gamma_t = min(zeta_t / max(eta_client * I_local, 1e-12), gamma_cap)
                # No update
            else:
                chosen_updates = [updates_kept[i] for i in keep_idx_local.tolist()]
                W = weights.view(-1, 1)
                U = torch.stack(chosen_updates, dim=0)
                agg_update = (W * U).sum(dim=0)

                bar_taus.append(bar_tau)

                # ---------- Stepsize: composite schedule -> gamma_t ----------
                zeta_t = zeta0 / (math.sqrt(t + 1) * (1.0 + alpha_stale * bar_tau) * (1.0 + beta_et * p_t))
                gamma_t = min(zeta_t / max(eta_client * I_local, 1e-12), gamma_cap)

                # ---------- Trust region on aggregated update (stability) ----------
                if trust_region_frac is not None and trust_region_frac > 0.0:
                    server_vec, _, _ = flatten_state_dict(server_state)
                    max_step = trust_region_frac * (server_vec.norm() + 1e-12)
                    u_norm = agg_update.norm() + 1e-12
                    if u_norm > max_step:
                        agg_update = agg_update * (max_step / u_norm)

                # ---------- Server update ----------
                server_state = apply_vector_update_to_state(server_state, agg_update, gamma_t, example_keys, example_shapes)

        # Push new state into history buffer and into model
        server_model.load_state_dict(server_state, strict=True)
        server_history.append(copy.deepcopy(server_state))
        if len(server_history) > (tau_max + 1):
            server_history.pop(0)

        # Periodic eval on test set (global model)
        if (t + 1) % eval_every == 0:
            loss_te, acc_te = evaluate(server_model, test_loader, device)
            test_losses.append(loss_te)
            test_accs.append(acc_te)

    # Save logs
    np.save(os.path.join(results_dir, "server_losses.npy"), np.array(server_losses))
    np.save(os.path.join(results_dir, "test_losses.npy"),   np.array(test_losses))
    np.save(os.path.join(results_dir, "test_accs.npy"),     np.array(test_accs))
    np.save(os.path.join(results_dir, "suppression_rates.npy"), np.array(suppression_rates))
    np.save(os.path.join(results_dir, "bar_taus.npy"), np.array(bar_taus))
    np.save(os.path.join(results_dir, "Xi_proxy.npy"), np.array(Xi_list))
    torch.save(server_state, os.path.join(results_dir, "server_final_state.pt"))

    return {
        "server_model": server_model,
        "server_state": server_state,
        "server_losses": server_losses,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "suppression_rates": suppression_rates,
        "bar_taus": bar_taus,
        "Xi_proxy": Xi_list,
    }

# ------------------------------------------------------------
#  Helper to run coroutine safely in notebooks/scripts
# ------------------------------------------------------------
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

# ------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = dict(
        num_clients=10,
        alpha_dirichlet=0.5,
        batch_size=64,
        rounds=200,
        clients_per_round=6,
        I_local=10,
        eta_client=1e-3,
        tau_max=4,
        alpha_stale=0.01,
        delay_sim_max_s=0.0,
        epsilon_trigger=1e-3,
        beta_et=0.5,
        trim_ratio=0.2,
        wmin=0.5,
        wmax=1.5,
        zeta0=None,          # will be set to eta*I*gamma0
        gamma0=0.1,          # initial server mixing factor (approx)
        gamma_cap=1.0,
        trust_region_frac=0.02,
        clip_updates=True,
        clip_multiplier=2.5,
        byz_frac=0.0,
        byz_mode="signflip",
        byz_scale=5.0,
        results_dir=f"results_rafl/rafl_{time.strftime('%Y%m%d_%H%M%S')}",
        eval_every=5,
        seed=42,
    )

    async def main():
        os.makedirs(cfg["results_dir"], exist_ok=True)
        out = await run_rafl(**cfg)
        print(f"Results written to: {cfg['results_dir']}")
        print(f"Final test acc (last eval): {out['test_accs'][-1] if len(out['test_accs'])>0 else 'N/A'}")

    run_coro(main())
