import numpy as np
import matplotlib.pyplot as plt
import os

def load_server_losses(results_dir, filename):
    """Load server losses from the given directory and filename."""
    server_losses_file = os.path.join(results_dir, filename)
    if os.path.exists(server_losses_file):
        return np.load(server_losses_file)
    else:
        print(f"Server losses file not found: {server_losses_file}")
        return None

def compare_server_losses(results_dir_async, label_async, file_async, results_dir_sync, label_sync, file_sync):
    """Load and compare server losses from asynchronous and synchronous experiments."""
    server_losses_async = load_server_losses(results_dir_async, file_async)
    server_losses_sync = load_server_losses(results_dir_sync, file_sync)

    if server_losses_async is not None and server_losses_sync is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(server_losses_async, label=f"{label_async} - Server Loss", linestyle='dashed')
        plt.plot(server_losses_sync, label=f"{label_sync} - Server Loss")
        plt.xlabel("Rounds")
        plt.xscale("log")
        plt.ylabel("Loss")
        plt.title("Comparison of Server Loss Across Rounds")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("One of the server losses files was not found.")

# Define result directories and corresponding filenames
root_dir = os.path.abspath(os.getcwd())
results_dir_async = os.path.join(root_dir, "results/clients_10_rounds_200_epochs_10_clients_per_round_5_20250106_125808")  # Update path
results_dir_sync = os.path.join(root_dir, "results/synch_cifar_20250216_145508")  # Update path

file_async = "server_losses.npy"
file_sync = "synch_cifar_server_losses.npy"

compare_server_losses(results_dir_async, "Asynchronous", file_async, results_dir_sync, "Synchronous", file_sync)
