#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate *_updated runs and render figures matching the LaTeX includes.

Creates:
  images/test_loss_{ds}.png
  images/test_acc_{ds}.png
  images/server_training_loss_{ds}.png
  images/zeta_t_{ds}.png           (if available, typically for RAFL)
  images/gamma_t_{ds}.png          (if available)
  images/tau_bar_{ds}.png          (if available)
  images/nonconvex_{ds}_server_losses_comparison_with_rafl.png
"""

import os, re, json, glob, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.expanduser(
    "~/Documents/ReSTEP/Robust_AFL_non_convex/Robust_AFL_non_convex_non_iid-main/src/results_rafl"
)
IMG_DIR = os.path.join(os.path.dirname(BASE_DIR), "images")
os.makedirs(IMG_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
DATASET_ALIASES = {
    "mnist": ["mnist"],
    "fashion_mnist": ["fashion_mnist", "fmnist", "fashion-mnist", "fashionmnist"],
    "cifar": ["cifar", "cifar10", "cifar-10"],
}

def infer_method(run_name: str) -> Optional[str]:
    rn = run_name.lower()
    if rn.startswith("brafl"):
        return "Baseline RAFL"
    if rn.startswith("async"):
        return "AFL"
    if rn.startswith("rafl"):
        return "RAFL"
    # sometimes synchronous baselines are named like 'cifar_clients_...'
    if rn.startswith("cifar") or rn.startswith("mnist") or rn.startswith("fashion") or rn.startswith("fmnist"):
        return "Synchronous FL"
    return None

def infer_dataset(run_name: str) -> Optional[str]:
    rn = run_name.lower()
    for canonical, toks in DATASET_ALIASES.items():
        for t in toks:
            if re.search(rf"(?:^|[_\-]){re.escape(t)}(?:[_\-]|$)", rn):
                return canonical
    return None

def list_updated_dirs(base_dir: str) -> List[str]:
    all_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    return sorted([d for d in all_dirs if os.path.basename(d).endswith("_updated") or "_updated_" in os.path.basename(d)])

def load_npy(path: str) -> Optional[np.ndarray]:
    try:
        if os.path.isfile(path):
            arr = np.load(path, allow_pickle=False)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                return arr.astype(float)
    except Exception as e:
        warnings.warn(f"Failed loading {path}: {e}")
    return None

def load_json(path: str) -> Optional[dict]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed reading {path}: {e}")
    return None

def align_stack(series_list: List[np.ndarray]) -> Optional[np.ndarray]:
    series_list = [s for s in series_list if s is not None and len(s) > 0]
    if not series_list:
        return None
    T = min(len(s) for s in series_list)
    if T == 0:
        return None
    return np.stack([s[:T] for s in series_list], axis=0)  # (N, T)

def mean_ci(a2d: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.nanmean(a2d, axis=0)
    n = a2d.shape[0]
    if n > 1:
        s = np.nanstd(a2d, axis=0, ddof=1)
        half = z * s / np.sqrt(n)
        lo, hi = m - half, m + half
    else:
        lo = hi = m.copy()
    return m, lo, hi

def plot_metric(ax, x, stats_by_method: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], title: str, ylabel: str):
    for method, (m, lo, hi) in stats_by_method.items():
        ax.plot(x, m, label=method)
        if hi is not None and lo is not None:
            ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Rounds")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

# -----------------------------
# Collect runs
# -----------------------------
run_dirs = list_updated_dirs(BASE_DIR)
if not run_dirs:
    print(f"No *_updated runs found in {BASE_DIR}")
    raise SystemExit(0)

runs = []
for rd in run_dirs:
    name = os.path.basename(rd)
    dataset = infer_dataset(name) or "unknown"
    method = infer_method(name) or "unknown"
    test_loss = load_npy(os.path.join(rd, "test_loss.npy"))
    test_acc  = load_npy(os.path.join(rd, "test_acc.npy"))
    server_losses = load_npy(os.path.join(rd, "server_losses.npy"))  # used as "server loss proxy"
    rjson = load_json(os.path.join(rd, "rafl_metrics.json"))  # optional

    zeta = rjson.get("zeta_t") if isinstance(rjson, dict) else None
    gamma = rjson.get("gamma_t") if isinstance(rjson, dict) else None
    taubar = rjson.get("tau_bar") if isinstance(rjson, dict) else None
    # convert lists → np arrays if present
    zeta  = np.asarray(zeta, dtype=float) if isinstance(zeta, (list, tuple)) else None
    gamma = np.asarray(gamma, dtype=float) if isinstance(gamma, (list, tuple)) else None
    taubar= np.asarray(taubar, dtype=float) if isinstance(taubar, (list, tuple)) else None

    runs.append({
        "name": name,
        "dir": rd,
        "dataset": dataset,
        "method": method,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "server_losses": server_losses,
        "zeta_t": zeta,
        "gamma_t": gamma,
        "tau_bar": taubar,
    })

# -----------------------------
# Aggregate per (dataset, method)
# -----------------------------
by_ds_method: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
for r in runs:
    key = (r["dataset"], r["method"])
    by_ds_method.setdefault(key, {"test_loss": [], "test_acc": [], "server_losses": [], "zeta_t": [], "gamma_t": [], "tau_bar": []})
    for k in ["test_loss", "test_acc", "server_losses", "zeta_t", "gamma_t", "tau_bar"]:
        if r[k] is not None:
            by_ds_method[key][k].append(r[k])

datasets = sorted({r["dataset"] for r in runs if r["dataset"] != "unknown"})

# -----------------------------
# Render per-dataset figures (Fig. 1 layout)
# -----------------------------
for ds in datasets:
    # 1) test_loss
    stats = {}
    T_min = None
    for method in sorted({k[1] for k in by_ds_method.keys() if k[0] == ds}):
        a2d = align_stack(by_ds_method.get((ds, method), {}).get("test_loss", []))
        if a2d is not None:
            m, lo, hi = mean_ci(a2d)
            T_min = len(m) if T_min is None else min(T_min, len(m))
            stats[method] = (m, lo, hi)
    if stats:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, list(stats.values())[0][0].shape[0] + 1)
        plot_metric(ax, x, stats, f"{ds.upper()} — Test loss per round", "Test loss")
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f"test_loss_{ds}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 2) test_acc
    stats = {}
    for method in sorted({k[1] for k in by_ds_method.keys() if k[0] == ds}):
        a2d = align_stack(by_ds_method.get((ds, method), {}).get("test_acc", []))
        if a2d is not None:
            m, lo, hi = mean_ci(a2d)
            stats[method] = (m, lo, hi)
    if stats:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, list(stats.values())[0][0].shape[0] + 1)
        plot_metric(ax, x, stats, f"{ds.upper()} — Test accuracy per round", "Test accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f"test_acc_{ds}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3) server loss proxy (use server_losses.npy)
    stats = {}
    for method in sorted({k[1] for k in by_ds_method.keys() if k[0] == ds}):
        a2d = align_stack(by_ds_method.get((ds, method), {}).get("server_losses", []))
        if a2d is not None:
            m, lo, hi = mean_ci(a2d)
            stats[method] = (m, lo, hi)
    if stats:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, list(stats.values())[0][0].shape[0] + 1)
        plot_metric(ax, x, stats, f"{ds.upper()} — Server loss proxy", "Server loss (proxy)")
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f"server_training_loss_{ds}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 4–6) RAFL extras (if present) — we average ONLY over RAFL runs
    for key, label in [("zeta_t", "Composite step $\\zeta_t$"),
                       ("gamma_t", "Server step $\\gamma_t$"),
                       ("tau_bar", "Weighted staleness $\\bar\\tau_t$")]:
        a2d = align_stack(by_ds_method.get((ds, "RAFL"), {}).get(key, []))
        if a2d is None:
            continue
        m, lo, hi = mean_ci(a2d)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, label="RAFL")
        ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_title(f"{ds.upper()} — {label}")
        ax.set_xlabel("Rounds")
        ax.set_ylabel(label.replace("$", ""))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f"{key}_{ds}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # -------------------------
    # Comparison figure (Fig. 2 style): server loss across methods with log-x
    # -------------------------
    stats_cmp = {}
    for method in sorted({k[1] for k in by_ds_method.keys() if k[0] == ds}):
        a2d = align_stack(by_ds_method.get((ds, method), {}).get("server_losses", []))
        if a2d is not None:
            m, lo, hi = mean_ci(a2d)
            stats_cmp[method] = (m, lo, hi)
    if stats_cmp:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        # use 1..T to support log-x
        T = list(stats_cmp.values())[0][0].shape[0]
        x = np.arange(1, T + 1)
        for method, (m, lo, hi) in stats_cmp.items():
            ax.plot(x, m, label=method)
            ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("Rounds (log scale)")
        ax.set_ylabel("Server loss (proxy)")
        ax.set_title(f"{ds.upper()} — Server loss comparison (Synchronous, AFL, RAFL)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f"nonconvex_{ds}_server_losses_comparison_with_rafl.png"),
                    dpi=220, bbox_inches="tight")
        plt.close(fig)

print(f"Done. Figures saved under: {IMG_DIR}")
