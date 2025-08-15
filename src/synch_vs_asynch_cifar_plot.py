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
        plt.figure(figsize=(10, 6))
        plt.plot(server_losses_async[:200], label="Asynchronous FL Server Loss", linestyle='-', marker='o')
        plt.plot(server_losses_sync[:200], label="Synchronous FL Server Loss", linestyle='--', marker='s')
        plt.xlabel("Rounds",  fontsize=18)
        plt.xscale("log")
        plt.ylabel("Loss", fontsize=18)
        #plt.title("Comparison of Server Losses in CIFAR dataset (Asynchronous vs. Synchronous)",  fontsize=18)
        plt.legend()
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)  # Set x-tick font size
        plt.yticks(fontsize=16)  # Set y-tick font size
        plt.grid(True)
        
        plt.savefig(root_dir + "/results/" +"nonconvex_cifar_server_losses_comparison.png", dpi=300, bbox_inches='tight')  # High-quality PNG
        plt.savefig(root_dir + "/results/"+ "nonconvex_cifar_server_losses_comparison.pdf", bbox_inches='tight')  # PDF format

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
