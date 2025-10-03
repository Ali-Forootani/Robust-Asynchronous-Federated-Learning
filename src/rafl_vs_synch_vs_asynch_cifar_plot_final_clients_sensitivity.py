#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:32:21 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

# ============ CONFIG ============
BASE = Path("/home/forootan/Documents/ReSTEP/Robust_AFL_non_convex/Robust_AFL_non_convex_non_iid-main/src")
RESULTS_DIR = BASE / "results_rafl"
OUT_DIR = BASE / "results"
SAVE_BASENAME = "cifar_server_losses_rafl_cpr_5_6_7_comparison"
LOG_X = True           # set True if you want x-axis log scale
MAX_POINTS = None       # e.g. 1000 to cap points, or None for full length
LINEWIDTH = 2.0         # your preferred line width
MARK_EVERY_FRACTION = 20  # put a marker roughly every 1/N of the points

# Folder name must start with this prefix
PREFIX = "rafl_cifar_clients_10_rounds_200_epochs_10_clients_per_round_"

# Regex to extract: clients_per_round (cpr) and a trailing timestamp-ish bit
# Example folder: rafl_cifar_clients_10_rounds_200_epochs_10_clients_per_round_7_20250917_120050_updated
PATTERN = re.compile(
    r"^rafl_cifar_clients_10_rounds_200_epochs_10_clients_per_round_(\d+)_(.+)$"
)

def find_matching_runs(results_dir: Path):
    """Return list of (path, cpr:int, stamp:str) for matching runs with server_losses.npy present."""
    runs = []
    for entry in results_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith(PREFIX):
            continue
        m = PATTERN.match(name)
        if not m:
            continue
        cpr = int(m.group(1))
        stamp = m.group(2)
        losses_path = entry / "test_acc.npy"
        if losses_path.exists():
            runs.append((entry, cpr, stamp, losses_path))
    return runs

def load_losses(paths):
    """Load npy arrays from list of file paths."""
    series = []
    for (run_dir, cpr, stamp, losses_path) in paths:
        try:
            arr = np.load(losses_path)
            series.append((run_dir.name, cpr, stamp, arr))
        except Exception as e:
            print(f"[WARN] Failed to load {losses_path}: {e}")
    return series

def trim_series(series, max_points=None):
    """Trim all arrays to the minimum common length (and <= max_points if set)."""
    if not series:
        return series, 0
    min_len = min(len(arr) for (_, _, _, arr) in series)
    if max_points is not None:
        min_len = min(min_len, max_points)
    trimmed = [(name, cpr, stamp, arr[:min_len]) for (name, cpr, stamp, arr) in series]
    return trimmed, min_len

def main():
    runs = find_matching_runs(RESULTS_DIR)

    # Keep only CPR in {5,6,7} as requested
    runs = [r for r in runs if r[1] in {5, 6, 7}]
    if not runs:
        raise FileNotFoundError("No matching RAF L CIFAR runs with CPR in {5,6,7} found.")

    # Sort by CPR then timestamp (alphabetical on the tail is fine)
    runs.sort(key=lambda x: (x[1], x[2]))

    series = load_losses(runs)
    series, L = trim_series(series, MAX_POINTS)
    if L == 0:
        raise RuntimeError("Loaded series are empty after trimming.")

    x = np.arange(L)

    plt.figure(figsize=(10, 6))
    for (name, cpr, stamp, arr) in series:
        markevery = max(1, L // MARK_EVERY_FRACTION)
        plt.plot(
            x, arr,
            label=f"RAFL (CPR={cpr}, {stamp})",
            linestyle='-',
            marker='o',
            markevery=markevery,
            linewidth=LINEWIDTH
        )

    plt.xlabel("Rounds", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title("CIFAR-10 • RAFL • Server Loss (CPR=5,6,7)", fontsize=18)
    if LOG_X:
        plt.xscale("log")
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{SAVE_BASENAME}.png"
    pdf_path = OUT_DIR / f"{SAVE_BASENAME}.pdf"
    plt.savefig(png_path.as_posix(), dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path.as_posix(), bbox_inches='tight')
    print(f"[SAVED] {png_path}\n[SAVED] {pdf_path}")
    plt.show()

if __name__ == "__main__":
    main()
