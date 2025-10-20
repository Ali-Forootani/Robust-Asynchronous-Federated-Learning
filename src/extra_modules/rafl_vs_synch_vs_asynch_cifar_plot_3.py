#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:46:12 2025

@author: forootan
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE_SRC = "/home/forootan/Documents/ReSTEP/Robust_AFL_non_convex/Robust_AFL_non_convex_non_iid-main/src"
RESULTS_DIR = os.path.join(BASE_SRC, "results")
RESULTS_RAFL_DIR = os.path.join(BASE_SRC, "results_rafl")

# === Optional manual overrides (leave as None to auto-detect) ===
MANUAL_ASYNC_DIR = None  # e.g. "/.../src/results/cifar_clients_10_rounds_200_epochs_10_clients_per_round_8_20250221_091038"
MANUAL_SYNC_DIR  = None  # e.g. "/.../src/results/synch_cifar_20250216_145508"
MANUAL_RAFL_DIR  = None  # e.g. "/.../src/results_rafl/rafl_cifar_clients_10_rounds_200_epochs_3_clients_per_round_5_20250912_080213"
# ================================================================

def list_candidates(root, filename, include_keys=None, exclude_keys=None):
    include_keys = include_keys or []
    exclude_keys = exclude_keys or []
    patterns = [os.path.join(root, "**", filename)]
    hits = []
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            dirpath = os.path.dirname(path)
            name = os.path.basename(dirpath)
            if include_keys and not any(k in name for k in include_keys):
                continue
            if exclude_keys and any(k in name for k in exclude_keys):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            hits.append((path, mtime))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits

def pick_latest(root, filename_options, include_keys=None, exclude_keys=None):
    for fname in filename_options:
        cand = list_candidates(root, fname, include_keys, exclude_keys)
        if cand:
            return cand[0][0]  # latest by mtime
    return None

def load_losses(np_path):
    print(f"[INFO] Using file: {np_path}")
    return np.load(np_path)

def compare_three(la, ls, lr, log_x=True, max_points=None,
                  save_basename="nonconvex_cifar_server_losses_comparison_with_rafl"):
    L = min(len(la), len(ls), len(lr))
    if max_points is not None:
        L = min(L, max_points)
    x = np.arange(L)

    plt.figure(figsize=(10, 6))
    plt.plot(x, la[:L], label="Asynchronous FL — Server Loss", linestyle='-', marker='o', markevery=max(1, L//20))
    plt.plot(x, ls[:L], label="Synchronous FL — Server Loss",   linestyle='--', marker='s', markevery=max(1, L//20))
    plt.plot(x, lr[:L], label="RAFL — Server Loss",             linestyle='-.', marker='^', markevery=max(1, L//20))
    plt.xlabel("Rounds", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    if log_x:
        plt.xscale("log")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(True)

    out_dir = RESULTS_DIR  # save alongside other plots
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{save_basename}.png")
    pdf_path = os.path.join(out_dir, f"{save_basename}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"[OK] Saved: {png_path}\n[OK] Saved: {pdf_path}")
    plt.show()

if __name__ == "__main__":
    # --- ASYNC (baseline non-RAFL) ---
    if MANUAL_ASYNC_DIR:
        async_file = os.path.join(MANUAL_ASYNC_DIR, "server_losses.npy")
    else:
        # Prefer CIFAR async runs; avoid sync/mnist folders
        async_file = pick_latest(
            RESULTS_DIR,
            filename_options=["server_losses.npy"],
            include_keys=["cifar_clients_"],
            exclude_keys=["synch_", "mnist_"]
        )
    if not async_file or not os.path.exists(async_file):
        raise FileNotFoundError("[ERROR] Could not locate ASYNC server_losses.npy. "
                                "Set MANUAL_ASYNC_DIR or check that the run produced server_losses.npy.")

    # --- SYNC ---
    if MANUAL_SYNC_DIR:
        # try specific sync filename first, fall back to generic
        sync_file = (os.path.join(MANUAL_SYNC_DIR, "synch_cifar_server_losses.npy")
                     if os.path.exists(os.path.join(MANUAL_SYNC_DIR, "synch_cifar_server_losses.npy"))
                     else os.path.join(MANUAL_SYNC_DIR, "server_losses.npy"))
    else:
        # Try the special sync filename, then generic
        sync_file = pick_latest(
            RESULTS_DIR,
            filename_options=["synch_cifar_server_losses.npy", "server_losses.npy"],
            include_keys=["synch_cifar_"],
            exclude_keys=[]
        )
    if not sync_file or not os.path.exists(sync_file):
        raise FileNotFoundError("[ERROR] Could not locate SYNC server losses file. "
                                "Set MANUAL_SYNC_DIR or verify the sync run output file name.")

    # --- RAFL ---
    if MANUAL_RAFL_DIR:
        rafl_file = os.path.join(MANUAL_RAFL_DIR, "server_losses.npy")
    else:
        # Prefer CIFAR-tagged RAFL if present; otherwise any RAFL run
        rafl_file = pick_latest(
            RESULTS_RAFL_DIR,
            filename_options=["server_losses.npy"],
            include_keys=["rafl_cifar", "rafl_"],
            exclude_keys=[]
        )
    if not rafl_file or not os.path.exists(rafl_file):
        raise FileNotFoundError("[ERROR] Could not locate RAFL server_losses.npy in results_rafl. "
                                "Set MANUAL_RAFL_DIR or verify the RAFL run directory.")

    print("\n[SELECTION]")
    print(f"ASYNC: {async_file}")
    print(f"SYNC : {sync_file}")
    print(f"RAFL : {rafl_file}\n")

    losses_async = load_losses(async_file)
    losses_sync  = load_losses(sync_file)
    losses_rafl  = load_losses(rafl_file)

    compare_three(losses_async, losses_sync, losses_rafl,
                  log_x=True, max_points=200,
                  save_basename="nonconvex_cifar_server_losses_comparison_with_rafl")
