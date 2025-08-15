#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:44:48 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAFL — CIFAR-10 (GPU-ready, matches PDF up to p.31)
Train-only logging: save server training loss each round (length == rounds).

Changes vs previous:
- REMOVED: server_losses_proxy (proxy), all test set & test evals.
- ADDED: per-round evaluation of server model on training set.
- SAVES: server_train_losses.npy (length == rounds).

RAFL — CIFAR-10 (GPU-ready, matches PDF up to p.31)

Implements:
- Client sampling without replacement
- Local SGD (I steps) from stale snapshot θ^{(t-τ_c)}
- Event trigger: send u_c only if ||u_c|| >= ε
- Staleness-aware weights: w_c ∝ 1/(1+α τ_c), normalized with bounds [wmin/J, wmax/J]
- Robust aggregation: trim by distance-to-median, then weighted mean over kept set
- Server update: θ^{(t+1)} = θ^{(t)} + γ_t \hat u^{(t)}
- Composite stepsize: ζ_t = ζ0 / ( sqrt(t+1) (1 + α \bar τ_t) (1 + β p_t) )
  with admissible server stepsize cap γ_t ≤ 1/(4 L η I)

Device policy:
- All training tensors on `device = torch.device("cuda" if available else "cpu")`
- Everything saved on CPU

"""

import os, copy, time, math, random, asyncio
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ----------------------- CIFAR-10 Mini-ResNet20 -----------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = (nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if (stride != 1 or in_channels != out_channels) else nn.Identity())

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(identity)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(3,3,3), num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3,16,3,1,1,bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_ch, n_blocks, stride):
        layers = [block(self.in_channels, out_ch, stride)]
        self.in_channels = out_ch
        for _ in range(1, n_blocks):
            layers.append(block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), -1)
        return self.fc(x)

# ------------------------------ Utils --------------------------------
def partition_non_iid_cifar(dataset, num_clients, alpha=0.5, num_classes=10):
    from collections import defaultdict as dd
    data_by_class = dd(list)
    for idx, (_, y) in enumerate(dataset):
        data_by_class[int(y)].append(idx)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = data_by_class[c]; np.random.shuffle(idxs)
        props = np.random.dirichlet([alpha]*num_clients)
        counts = (props * len(idxs)).astype(int)
        start = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idxs[start:start+cnt]); start += cnt
        leftovers = idxs[start:]
        for i, j in enumerate(leftovers):
            client_indices[i % num_clients].append(j)
    for i in range(num_clients): np.random.shuffle(client_indices[i])
    return client_indices
# ------------------------------ Utils (patched) --------------------------------
def flatten_state_dict(sd, *, exclude_bn_buffers=True):
    """
    Flattens *only* the entries we want to aggregate.
    If exclude_bn_buffers=True, skip BatchNorm buffers:
      - running_mean, running_var, num_batches_tracked
    """
    flats, shapes, keys = [], {}, []
    for k, v in sd.items():
        if exclude_bn_buffers:
            if ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k):
                continue
        # Only float tensors (skip ints like num_batches_tracked even if not caught above)
        if not torch.is_floating_point(v):
            continue
        t = v.detach().view(-1).to(torch.float32)
        flats.append(t); shapes[k] = v.shape; keys.append(k)
    return torch.cat(flats), keys, shapes

def unflatten_to_state_dict(vec, keys, shapes, ref_state):
    out, offset = {}, 0
    # Start by copying ref_state so untouched buffers (e.g., BN) pass through unchanged
    out.update({k: v.clone() for k, v in ref_state.items()})
    for k in keys:
        n = int(np.prod(shapes[k]))
        out[k] = vec[offset:offset+n].view(shapes[k]).to(
            dtype=ref_state[k].dtype,
            device=ref_state[k].device
        )
        offset += n
    return out


# robust trimming + weighted mean
def robust_filter_and_weight(updates, taus, sizes, alpha_stale, trim_ratio=0.2,
                             wmin_over_J=None, wmax_over_J=None, device="cpu"):
    if len(updates) == 0:
        return [], torch.tensor([], device=device), 0.0
    U = torch.stack(updates, 0).to(device)  # [n,P]
    median = U.median(dim=0).values
    dists = torch.norm(U - median, dim=1)
    n = U.shape[0]; keep_n = max(1, int(math.ceil((1.0 - trim_ratio) * n)))
    keep_idx = torch.topk(-dists, k=keep_n).indices

    taus_keep  = torch.tensor([taus[i]  for i in keep_idx.tolist()], dtype=torch.float32, device=device)
    sizes_keep = torch.tensor([sizes[i] for i in keep_idx.tolist()], dtype=torch.float32, device=device)
    sizes_keep = sizes_keep / (sizes_keep.sum() + 1e-12)

    w_raw = (1.0 / (1.0 + alpha_stale * taus_keep)) * sizes_keep
    w = w_raw / (w_raw.sum() + 1e-12)
    # enforce normalized bounds AFTER normalization
    if (wmin_over_J is not None) or (wmax_over_J is not None):
        wmin_over_J = 0.0 if wmin_over_J is None else wmin_over_J
        wmax_over_J = 1.0 if wmax_over_J is None else wmax_over_J
        w = torch.clamp(w, min=wmin_over_J, max=wmax_over_J)
        w = w / (w.sum() + 1e-12)

    bar_tau = float((w * taus_keep).sum().item())
    return keep_idx, w, bar_tau

async def client_local_update(
    client_id, base_state, server_theta_t_vec, model_ctor, train_loader, device,
    I_local=5, eta_client=1e-2, loss_fn=nn.CrossEntropyLoss(), delay_sim_max_s=0.0,
    accumulation_steps=1,
):
    if delay_sim_max_s > 0:
        await asyncio.sleep(random.uniform(0, delay_sim_max_s))

    model = model_ctor().to(device)
    model.load_state_dict(base_state, strict=True)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=eta_client, momentum=0.9, weight_decay=5e-4)

    total_loss, total_batches = 0.0, 0
    Xi_contrib = 0.0

    for _ in range(I_local):
        for b, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if (b + 1) % accumulation_steps == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            total_loss += float(loss.item()); total_batches += 1

            # Xi proxy term
            c_vec, _, _ = flatten_state_dict(model.state_dict())
            Xi_contrib += float(torch.norm(c_vec - server_theta_t_vec).item() ** 2)

    new_state = model.state_dict()
    base_vec, keys, shapes = flatten_state_dict(base_state)
    new_vec, _, _ = flatten_state_dict(new_state)
    update_vec = (new_vec - base_vec).detach().to(device)
    avg_loss = total_loss / max(1, total_batches)
    return update_vec, keys, shapes, avg_loss, Xi_contrib

def apply_vector_update_to_state(server_state, update_vec, gamma_t, keys, shapes):
    base_vec, _, _ = flatten_state_dict(server_state)
    new_vec = base_vec + gamma_t * update_vec.to(base_vec.device)
    return unflatten_to_state_dict(new_vec, keys, shapes, ref_state=server_state)

def byzantine_attack(update_vec, mode="signflip", scale=5.0):
    if mode == "signflip":
        return -scale * update_vec
    if mode == "gaussian":
        return update_vec + scale * torch.randn_like(update_vec, device=update_vec.device)
    if mode == "random":
        r = torch.randn_like(update_vec, device=update_vec.device); r = scale * r / (r.norm() + 1e-12); return r
    return update_vec

@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, tot = 0.0, 0
    for x, y in data_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x); loss = loss_fn(logits, y)
        bs = x.size(0)
        tot_loss += float(loss.item()) * bs
        tot += bs
    return tot_loss / max(1, tot)

# ------------------------------ RAFL ------------------------------
async def run_rafl(
    num_clients=10, alpha_dirichlet=0.5, batch_size=64,
    model_ctor=lambda: ResNet(BasicBlock, (3,3,3), num_classes=10),
    rounds=200, clients_per_round=6, I_local=10, eta_client=5e-2,
    tau_max=4, alpha_stale=0.01, delay_sim_max_s=0.0,
    epsilon_trigger=1e-6, beta_et=0.5,
    trim_ratio=0.05, wmin=0.5, wmax=1.5,
    zeta0=None, gamma0=0.2,  # if zeta0 is None, set from eta*I*gamma0
    L_smooth_est=1.0,        # conservative estimate for admissible cap
    trust_region_frac=0.02, clip_updates=True, clip_multiplier=2.5,
    byz_frac=0.0, byz_mode="signflip", byz_scale=5.0,
    results_dir="results_rafl", seed=42,
    train_eval_subset=None,   # None => full train set; else int number of samples
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)

    # Client partitions for local training
    client_indices = partition_non_iid_cifar(train_dataset, num_clients, alpha=alpha_dirichlet)
    pin = (device.type == "cuda")
    train_loaders = [DataLoader(Subset(train_dataset, idxs), batch_size=batch_size, shuffle=True,
                                num_workers=1, pin_memory=pin) for idxs in client_indices]
    client_sizes  = [len(idxs) for idxs in client_indices]

    # Train-eval loader (server training loss)
    if train_eval_subset is not None and train_eval_subset > 0:
        full_indices = np.arange(len(train_dataset))
        np.random.shuffle(full_indices)
        eval_indices = full_indices[:train_eval_subset].tolist()
        train_eval_loader = DataLoader(Subset(train_dataset, eval_indices),
                                       batch_size=256, shuffle=False, num_workers=1, pin_memory=pin)
    else:
        train_eval_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=1, pin_memory=pin)

    server_model = model_ctor().to(device)
    server_state = server_model.state_dict()
    server_history = [copy.deepcopy(server_state) for _ in range(tau_max+1)]

    example_keys, example_shapes = None, None

    # Logs
    server_train_losses = []  # length == rounds
    suppression_rates, bar_taus, Xi_list = [], [], []

    # zeta0 default per PDF (use gamma0 as initial mixing)
    if zeta0 is None:
        zeta0 = eta_client * I_local * gamma0

    # admissible cap from Lemma 2: γ_t ≤ 1 / (4 L η I)
    gamma_cap_theory = 1.0 / max(4.0 * L_smooth_est * eta_client * I_local, 1e-12)

    for t in tqdm(range(rounds), desc="RAFL Rounds"):
        selected = random.sample(range(num_clients), k=clients_per_round)
        J = clients_per_round
        wmin_over_J = (wmin / J) if wmin is not None else None
        wmax_over_J = (wmax / J) if wmax is not None else None

        taus = [random.randint(0, tau_max) for _ in selected]
        base_states = [copy.deepcopy(server_history[-(tau+1)]) for tau in taus]

        n_byz = int(math.floor(byz_frac * J))
        byz_set = set(random.sample(range(J), k=n_byz)) if n_byz > 0 else set()

        theta_t_vec, _, _ = flatten_state_dict(server_state)

        async def train_one(j):
            c = selected[j]
            return await client_local_update(
                client_id=c,
                base_state=base_states[j],
                server_theta_t_vec=theta_t_vec,
                model_ctor=model_ctor,
                train_loader=train_loaders[c],
                device=device,
                I_local=I_local,
                eta_client=eta_client,
                loss_fn=nn.CrossEntropyLoss(),
                delay_sim_max_s=delay_sim_max_s,
            )

        results = await asyncio.gather(*[train_one(j) for j in range(J)])

        updates_raw, local_losses, Xi_contribs = [], [], []
        for j, (u_vec, keys, shapes, loss, Xi_contrib) in enumerate(results):
            if example_keys is None:
                example_keys, example_shapes = keys, shapes
            if j in byz_set:
                u_vec = byzantine_attack(u_vec, mode=byz_mode, scale=byz_scale)
            updates_raw.append(u_vec.to(device))
            local_losses.append(float(loss))
            Xi_contribs.append(float(Xi_contrib))

        # median-norm clipping (stability)
        if clip_updates and len(updates_raw) > 0:
            norms = torch.stack([u.norm() for u in updates_raw])
            med = norms.median().item() if norms.numel() > 0 else 0.0
            cap = clip_multiplier * (med + 1e-12)
            updates_raw = [u * min(1.0, cap / (u.norm().item() + 1e-12)) for u in updates_raw]

        # event-triggered subset S(t)
        norms = [u.norm().item() for u in updates_raw]
        S_indices = [i for i, nrm in enumerate(norms) if nrm >= epsilon_trigger]
        p_t = 1.0 - (len(S_indices) / float(J))
        suppression_rates.append(float(p_t))

        Xi_t = float(np.sum(Xi_contribs)); Xi_list.append(Xi_t)

        # compute ζ_t and γ_t for this round
        def compute_gamma(bar_tau_val, p_val):
            # small 0.1*t smoothing kept from your note
            zeta_t = zeta0 / (math.sqrt(0.1*t + 1.0) * (1.0 + alpha_stale * bar_tau_val) * (1.0 + beta_et * p_val))
            gamma_t = min(zeta_t / max(eta_client * I_local, 1e-12), gamma_cap_theory)
            return zeta_t, gamma_t

        if len(S_indices) == 0:
            bar_tau = 0.0
            bar_taus.append(bar_tau)
            _ = compute_gamma(bar_tau, p_t)  # computed but no update
        else:
            updates_kept = [updates_raw[i] for i in S_indices]
            taus_kept    = [taus[i]        for i in S_indices]
            sizes_kept   = [client_sizes[selected[i]] for i in S_indices]

            keep_idx_local, weights, bar_tau = robust_filter_and_weight(
                updates_kept, taus_kept, sizes_kept,
                alpha_stale=alpha_stale, trim_ratio=trim_ratio,
                wmin_over_J=wmin_over_J, wmax_over_J=wmax_over_J, device=device
            )
            if len(keep_idx_local) == 0:
                bar_tau = 0.0
                bar_taus.append(bar_tau)
                _ = compute_gamma(bar_tau, p_t)  # no update
            else:
                chosen_updates = [updates_kept[i] for i in keep_idx_local.tolist()]
                W = weights.view(-1, 1)
                U = torch.stack(chosen_updates, 0)
                agg_update = (W * U).sum(0)
                bar_taus.append(float(bar_tau))

                _, gamma_t = compute_gamma(bar_tau, p_t)

                # trust-region safeguard
                if trust_region_frac is not None and trust_region_frac > 0.0:
                    server_vec, _, _ = flatten_state_dict(server_state)
                    max_step = float(trust_region_frac) * (float(server_vec.norm().item()) + 1e-12)
                    u_norm   = float(agg_update.norm().item()) + 1e-12
                    if u_norm > max_step:
                        agg_update = agg_update * (max_step / u_norm)

                # server update
                server_state = apply_vector_update_to_state(server_state, agg_update, gamma_t,
                                                            example_keys, example_shapes)

        # rotate history buffer and set model state
        server_model.load_state_dict(server_state, strict=True)
        server_history.append(copy.deepcopy(server_state))
        if len(server_history) > (tau_max + 1): server_history.pop(0)

        # --------- Evaluate server loss on TRAINING set each round ---------
        train_loss = evaluate_loss(server_model, train_eval_loader, device)
        server_train_losses.append(float(train_loss))

    # --------------- Save logs on CPU ---------------
    np.save(os.path.join(results_dir, "server_train_losses.npy"), np.array(server_train_losses, dtype=np.float64))
    np.save(os.path.join(results_dir, "suppression_rates.npy"), np.array(suppression_rates, dtype=np.float64))
    np.save(os.path.join(results_dir, "bar_taus.npy"),          np.array(bar_taus, dtype=np.float64))
    np.save(os.path.join(results_dir, "Xi_proxy.npy"),          np.array(Xi_list, dtype=np.float64))

    cpu_state = {k: v.detach().cpu() for k, v in server_state.items()}
    torch.save(cpu_state, os.path.join(results_dir, "server_final_state.pt"))

    return dict(server_model=server_model, server_state=server_state,
                server_train_losses=server_train_losses,
                suppression_rates=suppression_rates, bar_taus=bar_taus, Xi_proxy=Xi_list)

# ------------------------- Runner helper -------------------------
def run_coro(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            import nest_asyncio; nest_asyncio.apply()
            loop = asyncio.get_event_loop(); return loop.run_until_complete(coro)
        raise

# --------------------------- Main --------------------------------
if __name__ == "__main__":
    cfg = dict(
        num_clients=10, alpha_dirichlet=0.5, batch_size=64,
        rounds=20, clients_per_round=6, I_local=1, eta_client=5e-2,
        tau_max=4, alpha_stale=0.01, delay_sim_max_s=0.0,
        epsilon_trigger=1e-6, beta_et=0.5,
        trim_ratio=0.05, wmin=0.5, wmax=1.5,
        zeta0=None, gamma0=0.5, L_smooth_est=1.0,
        trust_region_frac=0.02, clip_updates=True, clip_multiplier=2.5,
        byz_frac=0.0, byz_mode="signflip", byz_scale=5.0,
        results_dir=f"results_rafl/rafl_{time.strftime('%Y%m%d_%H%M%S')}",
        seed=42,
        model_ctor=lambda: ResNet(BasicBlock, (3,3,3), num_classes=10),
        train_eval_subset=None,   # set e.g. 10000 for faster per-round eval
    )

    async def main():
        os.makedirs(cfg["results_dir"], exist_ok=True)
        out = await run_rafl(**cfg)
        print(f"Results written to: {cfg['results_dir']}")
        print(f"Saved: server_train_losses.npy (len={len(out['server_train_losses'])})")
    run_coro(main())
