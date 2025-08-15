#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 17:36:20 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# EDIT THESE ONLY IF YOUR FOLDERS/FILES ARE DIFFERENT
# =============================================================================
# Project base (the 'src' folder)
BASE = "/home/forootan/Documents/ReSTEP/Robust_AFL_non_convex/Robust_AFL_non_convex_non_iid-main/src"

# Async run (YOUR pwd listing shows this exact folder)
ASYNC_DIR = os.path.join(
    BASE,
    "results",
    "cifar_clients_10_rounds_200_epochs_10_clients_per_round_5_20250106_125808"
)
ASYNC_FILE = "server_losses.npy"

# Sync run
SYNC_DIR  = os.path.join(BASE, "results", "synch_cifar_20250216_145508")
SYNC_FILE = "synch_cifar_server_losses.npy"   # change to "server_losses.npy" if that's your filename

# RAFL run
RAFL_DIR  = os.path.join(BASE, "results_rafl", "rafl_20250815_212420")
RAFL_FILE = "server_test_losses.npy"

# Output
SAVE_BASENAME = "nonconvex_cifar_server_losses_comparison_with_rafl"
MAX_POINTS = 200
LOG_X = True
LOG_Y = True
# =============================================================================


def find_latest_with_file(root, subdir, filename, pattern="*"):
    """Find the most recent directory under root/subdir matching pattern that contains filename."""
    search_root = os.path.join(root, subdir)
    candidates = glob.glob(os.path.join(search_root, pattern))
    candidates = [d for d in candidates if os.path.isdir(d) and os.path.exists(os.path.join(d, filename))]
    if not candidates:
        return None
    candidates.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return candidates[0]


def load_server_losses(results_dir, filename="server_losses.npy"):
    path = os.path.join(results_dir, filename)
    print(f"[INFO] Trying: {path}")
    if os.path.exists(path):
        try:
            arr = np.load(path)
            print(f"[OK] Loaded shape: {arr.shape}")
            return arr
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return None
    print(f"[WARN] Not found: {path}")
    return None


def compare_three_losses(results_dir_async, file_async,
                         results_dir_sync,  file_sync,
                         results_dir_rafl,  file_rafl,
                         max_points=None, log_x=True,
                         save_basename="nonconvex_cifar_server_losses_comparison_with_rafl",
                         base_folder_for_output=None):
    # Auto-discovery fallbacks if any path is None
    if results_dir_async is None:
        results_dir_async = find_latest_with_file(BASE, "results", file_async, pattern="cifar_*")
        print(f"[AUTO] async dir -> {results_dir_async}")
    if results_dir_sync is None:
        results_dir_sync = find_latest_with_file(BASE, "results", file_sync, pattern="synch_*")
        print(f"[AUTO] sync dir  -> {results_dir_sync}")
    if results_dir_rafl is None:
        results_dir_rafl = find_latest_with_file(BASE, "results_rafl", file_rafl, pattern="rafl_*")
        print(f"[AUTO] RAFL dir  -> {results_dir_rafl}")

    losses_async = load_server_losses(results_dir_async, file_async) if results_dir_async else None
    losses_sync  = load_server_losses(results_dir_sync,  file_sync)  if results_dir_sync  else None
    losses_rafl  = load_server_losses(results_dir_rafl,  file_rafl)  if results_dir_rafl  else None

    if any(x is None for x in [losses_async, losses_sync, losses_rafl]):
        print("\n[ERROR] One or more loss files missing. Check the three paths and filenames above.\n"
              "Hints:\n"
              f" - ASYNC_DIR exists? {os.path.isdir(results_dir_async) if results_dir_async else False}\n"
              f" - SYNC_DIR  exists? {os.path.isdir(results_dir_sync)  if results_dir_sync  else False}\n"
              f" - RAFL_DIR  exists? {os.path.isdir(results_dir_rafl)  if results_dir_rafl  else False}\n"
              " - Filenames match exactly (including 'synch_' prefix)?\n")
        return

    # Align lengths
    L = min(len(losses_async), len(losses_sync), len(losses_rafl))
    if max_points is not None:
        L = min(L, max_points)

    x  = np.arange(L)
    la = losses_async[:L]
    ls = losses_sync[:L]
    lr = losses_rafl[:L]
    
    print(lr)
    print(ls)

    plt.figure(figsize=(10, 6))
    markevery = max(1, L // 20)
    plt.plot(x, la, label="Asynchronous FL — Server Loss", linestyle='-',  marker='o', markevery=markevery)
    plt.plot(x, ls, label="Synchronous FL — Server Loss",   linestyle='--', marker='s', markevery=markevery)
    plt.plot(x, lr, label="RAFL — Server Loss",             linestyle='-.', marker='^', markevery=markevery)

    plt.xlabel("Rounds", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    if log_x:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(True)

    # Save under src/results/
    out_dir = os.path.join(BASE if base_folder_for_output is None else base_folder_for_output, "results")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{save_basename}.png")
    pdf_path = os.path.join(out_dir, f"{save_basename}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"[OK] Saved: {png_path}\n[OK] Saved: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    compare_three_losses(
        ASYNC_DIR, ASYNC_FILE,
        SYNC_DIR,  SYNC_FILE,
        RAFL_DIR,  RAFL_FILE,
        max_points=MAX_POINTS,
        log_x=LOG_X,
        save_basename=SAVE_BASENAME
    )
