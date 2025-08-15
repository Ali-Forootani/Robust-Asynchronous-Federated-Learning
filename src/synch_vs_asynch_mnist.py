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
results_dir_1 = root_dir + "/results/clients_10_rounds_200_epochs_10_clients_per_round_5_20250106_125808"
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

plt.xlabel("Rounds")
plt.xscale("log")
plt.ylabel("Loss")
plt.title("Comparison of Server Losses (Asynchronous vs. Synchronous)")
plt.legend()
plt.grid(True)
plt.show()
