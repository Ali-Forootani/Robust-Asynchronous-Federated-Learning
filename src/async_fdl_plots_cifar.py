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
    server_losses_file = os.path.join(results_dir, "server_losses.npy")

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
results_dir = root_dir + "/results/cifar_clients_10_rounds_200_epochs_10_clients_per_round_6_20250220_143519"  # Replace with your actual results directory
load_and_plot_saved_losses(results_dir)








#####################################################




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:29:21 2025

@author: forootan
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

def load_and_plot_saved_losses(results_dir):
    """
    Load saved NumPy arrays and plot the training losses.
    """
    client_losses_files = [f for f in os.listdir(results_dir) if f.startswith("client_") and f.endswith("_losses.npy")]
    server_losses_file = os.path.join(results_dir, "server_losses.npy")

    # Load and plot client losses
    plt.figure(figsize=(10, 6))
    for client_file in client_losses_files:
        client_index = int(client_file.split("_")[1])
        client_losses = np.load(os.path.join(results_dir, client_file))
        plt.plot(client_losses, label=f"Client {client_index}")

    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title("Client Training Losses", fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.show()

    # Load and plot server losses
    if os.path.exists(server_losses_file):
        server_losses = np.load(server_losses_file)
        plt.figure(figsize=(10, 6))
        plt.plot(server_losses, label="Asynchronous Federated Learning", linestyle='-', marker='o')
        plt.xlabel("Rounds", fontsize=18)
        plt.xscale("log")
        plt.ylabel("Loss", fontsize=18)
        plt.title("Server Loss Across Rounds in CIFAR dataset with 50% clients participation", fontsize=18)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.savefig(root_dir + "/results/" +"nonconvex_cifar_server_losses.png", dpi=300, bbox_inches='tight')  # High-quality PNG
        plt.savefig(root_dir + "/results/"+ "nonconvex_cifar_server_losses.pdf", bbox_inches='tight')  # PDF format

        plt.show()
    else:
        print("Server losses file not found.")

# Example usage
results_dir = root_dir + "/results/cifar_clients_10_rounds_200_epochs_10_clients_per_round_6_20250220_143519"  # Replace with your actual results directory
load_and_plot_saved_losses(results_dir)






import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = root_dir + "/results/cifar_clients_10_rounds_200_epochs_10_clients_per_round_6_20250220_143519"+"/selected_clients.csv"  # Update if needed
df = pd.read_csv(file_path)

# Check the structure of the CSV
print(df.head())

# Assuming the column with selected client IDs is named 'client_id'
# Modify accordingly if the actual column name differs
column_name = df.columns[0]  # Using the first column as default
selected_clients = df[column_name]






import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 6))
plt.hist(selected_clients, bins=50, edgecolor='black', alpha=0.7)

# Set font sizes
plt.xlabel("Client IDs", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.title("Selected Clients for CIFAR with 50% client participation", fontsize=18)

# Set x-ticks with a jump of 1
plt.xticks(np.arange(min(selected_clients), max(selected_clients) + 1, 1), fontsize=16)
plt.yticks(fontsize=16)  # Set y-tick font size

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(root_dir + "/results/" +"histogram_afl_cifar_clients_5.png", dpi=300, bbox_inches='tight')  # High-quality PNG
plt.savefig(root_dir + "/results/"+ "histogram_afl_cifar_clients_5.pdf", bbox_inches='tight')  # PDF format
plt.show()


plt.show()






import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Replace this with your actual selected_clients data
#selected_clients = np.random.randint(0, 10, size=100)  # Simulating client selections

# Define bins from 0 to 9 (10 bins covering range 0-9)
bins = np.arange(11)  # Creates bins at [0,1,2,...,9,10]

# Compute histogram data
counts, edges = np.histogram(selected_clients, bins=bins)

# Compute bin centers (midpoints between bin edges)
bin_centers = (edges[:-1] + edges[1:]) / 2
bar_width = 0.3  # Set thinner bar width

# Use a color palette for bars
colors = sns.color_palette("Greys", len(bin_centers))

plt.figure(figsize=(8, 6))
plt.bar(bin_centers, counts, width=bar_width, edgecolor='black', alpha=0.9, color=colors)

# Set font sizes
plt.xlabel("Client IDs", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.title("Selected Clients for CIFAR with 50% client participation", fontsize=16)

# Set x-ticks at bin centers
plt.xticks(bin_centers, labels=np.arange(10), fontsize=16)  # Labels 0 to 9
plt.yticks(fontsize=16)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig(root_dir + "/results/" +"histogram_afl_cifar_clients_5.png", dpi=300, bbox_inches='tight')  # High-quality PNG
plt.savefig(root_dir + "/results/"+ "histogram_afl_cifar_clients_5.pdf", bbox_inches='tight')  # PDF format

# Show plot
plt.show()

























