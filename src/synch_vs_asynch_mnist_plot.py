#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:43:03 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare server losses from two different runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

root_dir = setting_directory(0)

def load_server_losses(results_dir, server_losses_filename):
    """
    Load server losses from a given results directory.
    """
    server_losses_file = os.path.join(results_dir, server_losses_filename)
    
    if os.path.exists(server_losses_file):
        return np.load(server_losses_file)
    else:
        print(f"Server losses file not found in: {server_losses_file}")
        return None

# Define results directories
results_dir_1 = root_dir + "/results/mnist_clients_10_rounds_1000_epochs_10_clients_per_round_5_20250205_072223"
results_dir_2 = root_dir + "/results/synch_clients_10_rounds_1000_epochs_10_clients_per_round_5_20250216_132617"

# Load server losses
server_losses_1 = load_server_losses(results_dir_1, "server_losses.npy")
server_losses_2 = load_server_losses(results_dir_2, "synch_server_losses.npy")

# Plot the server losses
plt.figure(figsize=(10, 6))

if server_losses_1 is not None:
    plt.plot(server_losses_1, label="Asynchronous FL Server Loss", linestyle='-', marker='o')

if server_losses_2 is not None:
    plt.plot(server_losses_2, label="Synchronous FL Server Loss", linestyle='--', marker='s')

plt.xlabel("Rounds", fontsize=18,)
plt.xscale("log")
plt.ylabel("Loss", fontsize=18,)
#plt.title("Comparison of Server Losses in MNIST dataset (Asynchronous vs. Synchronous)",  fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)  # Set x-tick font size
plt.yticks(fontsize=16)  # Set y-tick font size
plt.grid(True)



plt.savefig(root_dir + "/results/" +"nonconvex_mnist_server_losses_comparison.png", dpi=300, bbox_inches='tight')  # High-quality PNG
plt.savefig(root_dir + "/results/"+ "nonconvex_mnist_server_losses_comparison.pdf", bbox_inches='tight')  # PDF format

plt.show()
###############################################
###############################################
###############################################




















