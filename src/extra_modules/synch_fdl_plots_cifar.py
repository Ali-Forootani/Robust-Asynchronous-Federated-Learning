#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:23:20 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:29:21 2025

@author: forootan
"""

import numpy as np
import matplotlib.pyplot as plt
import os



import numpy as np
import sys
import os


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir


root_dir = setting_directory(0)




def load_and_plot_saved_losses(results_dir):
    
    """
    Load saved NumPy arrays and plot the training losses.
    """
    
    client_losses_files = [f for f in os.listdir(results_dir) if f.startswith("client_") and f.endswith("_losses.npy")]
    server_losses_file = os.path.join(results_dir, "synch_cifar_server_losses.npy")

    # Load and plot client losses
    plt.figure(figsize=(10, 6))
    for client_file in client_losses_files:
        client_index = int(client_file.split("_")[1])
        client_losses = np.load(os.path.join(results_dir, client_file))
        plt.plot(client_losses, label=f"Client {client_index}")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.yscale("log")
    #plt.xscale("log")
    plt.title("Client Training Losses")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Load and plot server losses
    if os.path.exists(server_losses_file):
        server_losses = np.load(server_losses_file)
        plt.figure(figsize=(10, 6))
        plt.plot(server_losses, label="Server Loss")
        plt.xlabel("Rounds")
        plt.xscale("log")
        plt.ylabel("Loss")
        #plt.yscale("log")
        plt.title("Server Loss Across Rounds")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Server losses file not found.")


# Example usage
results_dir = root_dir + "/results/synch_cifar_20250216_145508"  # Replace with your actual results directory
load_and_plot_saved_losses(results_dir)
