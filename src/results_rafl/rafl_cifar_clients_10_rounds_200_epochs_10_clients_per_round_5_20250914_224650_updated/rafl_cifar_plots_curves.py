#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:06:01 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAFL metrics plotting — matches the style of your comparison script.

Generates (and overwrites if present):
  - test_loss.png / .pdf
  - test_acc.png / .pdf
  - server_training_loss.png / .pdf
  - zeta_t.png / .pdf
  - gamma_t.png / .pdf
  - tau_bar.png / .pdf
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =======================
# MANUAL: set your folder
# =======================
#RESULTS_DIR = Path(" ")


from pathlib import Path

# Use the script's directory if run as a file; fall back to current working dir (e.g., Jupyter)
try:
    RESULTS_DIR = Path(__file__).parent.resolve()
except NameError:
    RESULTS_DIR = Path.cwd()

print(f"[INFO] RESULTS_DIR = {RESULTS_DIR}")



# ---------------
# Helper routines
# ---------------
def must_exist(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Not found: {path}")
    print(f"[OK] Found: {path}")

"""
def save_lineplot(x, y, xlabel, ylabel, title, out_png: Path, out_pdf: Path,
                  log_x=False, marker=None):
    plt.figure(figsize=(10, 6))
    if marker is not None:
        # sparse markers similar to your comparison plot
        markevery = max(1, len(x)//20)
        plt.plot(x, y, linestyle='-', marker=marker, markevery=markevery)
    else:
        plt.plot(x, y, linestyle='-')
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title)
    if log_x:
        plt.xscale("log")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix(), dpi=600, bbox_inches='tight')
    plt.savefig(out_pdf.as_posix(), bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_png}\n[SAVED] {out_pdf}")
"""

def save_lineplot(x, y, xlabel, ylabel, title, out_png: Path, out_pdf: Path,
                  log_x=False, marker=None, linewidth: float = 3.0):
    plt.figure(figsize=(10, 6))
    if marker is not None:
        # sparse markers similar to your comparison plot
        markevery = max(1, len(x)//20)
        plt.plot(x, y, linestyle='-', marker=marker, markevery=markevery,
                 linewidth=linewidth)
    else:
        plt.plot(x, y, linestyle='-', linewidth=linewidth)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title)
    if log_x:
        plt.xscale("log")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix(), dpi=600, bbox_inches='tight')
    plt.savefig(out_pdf.as_posix(), bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_png}\n[SAVED] {out_pdf}")





def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def main(log_x=False):
    # ---- inputs
    test_loss_npy = RESULTS_DIR / "test_loss.npy"
    test_acc_npy  = RESULTS_DIR / "test_acc.npy"
    server_losses = RESULTS_DIR / "server_losses.npy"
    metrics_json  = RESULTS_DIR / "rafl_metrics.json"

    # existence checks
    for p in [test_loss_npy, test_acc_npy, server_losses, metrics_json]:
        must_exist(p)

    # load arrays
    test_loss = np.load(test_loss_npy)
    test_acc  = np.load(test_acc_npy)
    serv_loss = np.load(server_losses)
    metrics   = load_json(metrics_json)

    zeta_t  = np.asarray(metrics.get("zeta_t", []), dtype=float)
    gamma_t = np.asarray(metrics.get("gamma_t", []), dtype=float)
    tau_bar = np.asarray(metrics.get("tau_bar", []), dtype=float)

    # x-axes
    T = min(len(test_loss), len(test_acc), len(serv_loss),
            len(zeta_t), len(gamma_t), len(tau_bar))
    x_rounds = np.arange(T)

    # ---- plots (PNG + PDF), titles match your LaTeX captions
    save_lineplot(
        x_rounds, test_loss[:T],
        xlabel="Rounds", ylabel="Value",
        title="Test Loss per Round",
        out_png=RESULTS_DIR / "test_loss_cifar.png",
        out_pdf=RESULTS_DIR / "test_loss_cifar.pdf",
        log_x=log_x
    )

    save_lineplot(
        x_rounds, test_acc[:T],
        xlabel="Rounds", ylabel="Accuracy",
        title="Test Accuracy per Round",
        out_png=RESULTS_DIR / "test_acc_cifar.png",
        out_pdf=RESULTS_DIR / "test_acc_cifar.pdf",
        log_x=log_x
    )

    save_lineplot(
        x_rounds, serv_loss[:T],
        xlabel="Rounds", ylabel="Value",
        title="Server Loss Proxy (avg client loss per round)",
        out_png=RESULTS_DIR / "server_training_loss_cifar.png",
        out_pdf=RESULTS_DIR / "server_training_loss_cifar.pdf",
        log_x=log_x
    )

    save_lineplot(
        x_rounds, zeta_t[:T],
        xlabel="Rounds", ylabel="Value",
        title=r"Composite Step Size $\zeta_t$",
        out_png=RESULTS_DIR / "zeta_t_cifar.png",
        out_pdf=RESULTS_DIR / "zeta_t_cifar.pdf",
        log_x=log_x
    )

    save_lineplot(
        x_rounds, gamma_t[:T],
        xlabel="Rounds", ylabel="Value",
        title=r"Server Step Size $\gamma_t$",
        out_png=RESULTS_DIR / "gamma_t_cifar.png",
        out_pdf=RESULTS_DIR / "gamma_t_cifar.pdf",
        log_x=log_x
    )

    save_lineplot(
        x_rounds, tau_bar[:T],
        xlabel="Rounds", ylabel="Value",
        title=r"Weighted Average Staleness $\bar{\tau}_t$",
        out_png=RESULTS_DIR / "tau_bar_cifar.png",
        out_pdf=RESULTS_DIR / "tau_bar_cifar.pdf",
        log_x=log_x
    )

if __name__ == "__main__":
    # Set log_x=True if you prefer the same log-x look as in your comparison figure.
    main(log_x=False)
