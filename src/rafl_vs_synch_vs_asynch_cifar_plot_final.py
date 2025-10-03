#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:49:29 2025

@author: forootan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===========
# BASE PATH
# ===========
BASE = Path("/home/forootan/Documents/ReSTEP/Robust_AFL_non_convex/Robust_AFL_non_convex_non_iid-main/src")

# =======================
# MANUAL RELATIVE PATHS
# =======================
ASYNC_LOSSES = BASE / "results_rafl" / "async_cifar_clients_10_rounds_200_epochs_10_clients_per_round_5_20250915_163225_updated" / "server_losses.npy"
SYNC_LOSSES  = BASE / "results_rafl" / "brafl_cifar_clients_10_rounds_200_epochs_10_cpr_5_agg_stale_mean_20250916_171631_updated" / "server_losses.npy"
RAFL_LOSSES  = BASE / "results_rafl" / "rafl_cifar_clients_10_rounds_200_epochs_10_clients_per_round_5_20250914_224650_updated" / "server_losses.npy"

# Where to save the figure
OUT_DIR = BASE / "results"
SAVE_BASENAME = "nonconvex_cifar_server_losses_comparison_with_rafl"

def must_exist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Not found: {path}")
    print(f"[OK] Found: {path}")

def main(max_points=200, log_x=True):
    # Hard checks (manual mode)
    must_exist(ASYNC_LOSSES)
    #must_exist(SYNC_LOSSES)
    must_exist(RAFL_LOSSES)

    la = np.load(ASYNC_LOSSES)
    ls = np.load(SYNC_LOSSES)
    lr = np.load(RAFL_LOSSES)

    L = min(len(la),
            #len(ls),
            len(lr))
    if max_points is not None:
        L = min(L, max_points)

    x = np.arange(L)

    plt.figure(figsize=(10, 6))
    plt.plot(x, la[:L], label="AFL", linestyle='-', marker='o', markevery=max(1, L//20))
    plt.plot(x, ls[:L], label="Baseline Robust AFL",   linestyle='--', marker='s', markevery=max(1, L//20))
    plt.plot(x, lr[:L], label="Robust AFL",    linestyle='-.', marker='^', markevery=max(1, L//20))
    plt.xlabel("Rounds", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title("CIFAR Server Loss", fontsize=18)
    #plt.yscale("log")
    if log_x:
        plt.xscale("log")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(True)

    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, f"{SAVE_BASENAME}.png")
    pdf_path = os.path.join(OUT_DIR, f"{SAVE_BASENAME}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"[SAVED] {png_path}\n[SAVED] {pdf_path}")
    plt.show()

if __name__ == "__main__":
    main()
