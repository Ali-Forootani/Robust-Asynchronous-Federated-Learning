import os
import numpy as np
import matplotlib.pyplot as plt

def load_server_losses(results_dir, filename="server_losses.npy"):
    path = os.path.join(results_dir, filename)
    print(f"[INFO] Trying: {path}")
    if os.path.exists(path):
        return np.load(path)
    print(f"[WARN] Not found: {path}")
    return None

def compare_three_losses(results_dir_async, file_async,
                         results_dir_sync, file_sync,
                         results_dir_rafl, file_rafl,
                         max_points=None, log_x=True,
                         save_basename="nonconvex_cifar_server_losses_comparison_with_rafl"):
    losses_async = load_server_losses(results_dir_async, file_async)
    losses_sync  = load_server_losses(results_dir_sync,  file_sync)
    losses_rafl  = load_server_losses(results_dir_rafl,  file_rafl)

    if any(x is None for x in [losses_async, losses_sync, losses_rafl]):
        print("[ERROR] One or more loss files missing. Aborting plot.")
        return

    L = min(len(losses_async), len(losses_sync), len(losses_rafl))
    if max_points is not None:
        L = min(L, max_points)

    x  = np.arange(L)
    la = losses_async[:L]
    ls = losses_sync[:L]
    lr = losses_rafl[:L]

    plt.figure(figsize=(10, 6))
    plt.plot(x, la, label="Asynchronous FL — Server Loss", linestyle='-', marker='o', markevery=max(1, L//20))
    plt.plot(x, ls, label="Synchronous FL — Server Loss",   linestyle='--', marker='s', markevery=max(1, L//20))
    plt.plot(x, lr, label="RAFL — Server Loss",             linestyle='-.', marker='^', markevery=max(1, L//20))
    plt.xlabel("Rounds", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    if log_x:
        plt.xscale("log")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(True)

    # Save to ./results/
    root_dir = os.path.abspath(os.getcwd())
    out_dir = os.path.join(root_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{save_basename}.png")
    pdf_path = os.path.join(out_dir, f"{save_basename}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"[OK] Saved: {png_path}\n[OK] Saved: {pdf_path}")
    plt.show()

# ---- Correct paths ----
# Async (use the folder you’re currently in)
results_dir_async = os.getcwd()  # your pwd is the async run with 'cifar_' prefix
file_async = "server_losses.npy"

# Sync
# adjust if needed; this example keeps your earlier name
results_dir_sync = "/home/forootan/Documents/ReSTEP/Robust_AFL_non_convex/Robust_AFL_non_convex_non_iid-main/src/results/synch_cifar_20250216_145508"
file_sync = "synch_cifar_server_losses.npy"

# RAFL
results_dir_rafl = "/home/forootan/Documents/ReSTEP/Robust_AFL_non_convex/Robust_AFL_non_convex_non_iid-main/src/results_rafl/rafl_20250812_114220"
file_rafl = "server_losses.npy"

compare_three_losses(results_dir_async, file_async,
                     results_dir_sync,  file_sync,
                     results_dir_rafl,  file_rafl,
                     max_points=200, log_x=True)
