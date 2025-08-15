#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:38:40 2025

@author: forootan
"""

import os
import numpy as np
import matplotlib.pyplot as plt



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


root_dir = setting_directory(0) + "/results/"


# Get the current directory
current_dir = os.getcwd()

# Find all directories that start with "cifar"
cifar_dirs = [d for d in os.listdir(root_dir) if d.startswith("cifar") and os.path.isdir(os.path.join(root_dir , d))]

# Initialize a dictionary to store server losses
server_losses_dict = {}

# Load server losses from each directory
for cifar_dir in cifar_dirs:
    server_losses_path = os.path.join(root_dir, cifar_dir, "server_losses.npy")
    
    if os.path.exists(server_losses_path):
        server_losses = np.load(server_losses_path)
        server_losses_dict[cifar_dir] = server_losses

# Plot server losses for each directory
plt.figure(figsize=(10, 6))
for cifar_dir, server_losses in server_losses_dict.items():
    plt.plot(server_losses, label=cifar_dir)

plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Server Loss Across Rounds for Different cifar Experiments")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.show()


#########################################
#########################################


import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for _ in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

root_dir = setting_directory(0) + "/results/"

# Get the current directory
current_dir = os.getcwd()

# Find all directories that start with "cifar"
cifar_dirs = [d for d in os.listdir(root_dir) if d.startswith("cifar") and os.path.isdir(os.path.join(root_dir, d))]

# Initialize a dictionary to store server losses
server_losses_dict = {}

# Load server losses from each directory
for cifar_dir in cifar_dirs:
    server_losses_path = os.path.join(root_dir, cifar_dir, "server_losses.npy")
    
    if os.path.exists(server_losses_path):
        server_losses = np.load(server_losses_path, allow_pickle=True)  # Allow pickled objects
        
        # Ensure server_losses is a valid 1D NumPy array
        if isinstance(server_losses, np.ndarray):
            try:
                server_losses = np.array(server_losses, dtype=np.float32).flatten()  # Convert to float32 and flatten
                server_losses_dict[cifar_dir] = server_losses
            except ValueError:
                print(f"Skipping {cifar_dir}: Unable to convert to uniform NumPy array.")
        else:
            print(f"Skipping {cifar_dir}: Inconsistent type {type(server_losses)}")

# Debugging: Print directory names and extracted parts
for cifar_dir in server_losses_dict.keys():
    parts = cifar_dir.split("_")
    print(f"Directory: {cifar_dir}, Split Parts: {parts}")


# Define standard Matplotlib line styles
line_styles = ['-', '--', '-.', ':', '-', '--']

plt.figure(figsize=(10, 6))

for idx, (cifar_dir, server_losses) in enumerate(server_losses_dict.items()):
    parts = cifar_dir.split("_")

    try:
        clients_index = parts.index("round") + 1  # Extract number of clients per round dynamically
        clients_per_round = parts[clients_index]
    except (IndexError, ValueError):
        print(f"Skipping {cifar_dir}: Unexpected format")
        continue

    label = f"Clients per round {clients_per_round}"

    # Explicitly ensure correct NumPy format
    server_losses = np.asarray(server_losses, dtype=np.float32)

    if server_losses.ndim == 1:
        plt.plot(server_losses, line_styles[idx % len(line_styles)], label=label)
    else:
        print(f"Skipping {cifar_dir}: Unexpected shape {server_losses.shape}")

plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Server Loss Across Rounds for Different cifar Experiments")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()



####################################################
####################################################





import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for _ in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

root_dir = setting_directory(0) + "/results/"

# Get the current directory
current_dir = os.getcwd()

# Find all directories that start with "cifar"
cifar_dirs = [d for d in os.listdir(root_dir) if d.startswith("cifar") and os.path.isdir(os.path.join(root_dir, d))]

# Initialize a dictionary to store server losses
server_losses_dict = {}

# Load server losses from each directory
for cifar_dir in cifar_dirs:
    server_losses_path = os.path.join(root_dir, cifar_dir, "server_losses.npy")
    
    if os.path.exists(server_losses_path):
        server_losses = np.load(server_losses_path, allow_pickle=True)  # Allow pickled objects
        
        # Ensure server_losses is a valid 1D NumPy array
        if isinstance(server_losses, np.ndarray):
            try:
                server_losses = np.array(server_losses, dtype=np.float32).flatten()  # Convert to float32 and flatten
                server_losses_dict[cifar_dir] = server_losses
            except ValueError:
                print(f"Skipping {cifar_dir}: Unable to convert to uniform NumPy array.")
        else:
            print(f"Skipping {cifar_dir}: Inconsistent type {type(server_losses)}")

# Debugging: Print directory names and extracted parts
for cifar_dir in server_losses_dict.keys():
    parts = cifar_dir.split("_")
    print(f"Directory: {cifar_dir}, Split Parts: {parts}")

# Define standard Matplotlib line styles and markers
line_styles = ['-', '--', '-.', ':', '-', '--']
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'x', '+']  # Different markers for each curve

plt.figure(figsize=(10, 6))

for idx, (cifar_dir, server_losses) in enumerate(server_losses_dict.items()):
    parts = cifar_dir.split("_")

    try:
        clients_index = parts.index("round") + 1  # Extract number of clients per round dynamically
        clients_per_round = parts[clients_index]
    except (IndexError, ValueError):
        print(f"Skipping {cifar_dir}: Unexpected format")
        continue

    label = f"Clients per round {clients_per_round}"

    # Explicitly ensure correct NumPy format
    server_losses = np.asarray(server_losses, dtype=np.float32)

    if server_losses.ndim == 1:
        plt.plot(server_losses, line_styles[idx % len(line_styles)], marker=markers[idx % len(markers)], label=label)
    else:
        print(f"Skipping {cifar_dir}: Unexpected shape {server_losses.shape}")


plt.xlabel("Rounds", fontsize=18,)
plt.xscale("log")
plt.ylabel("Loss", fontsize=18,)
plt.title("Server Loss in AFL for Different CIFAR Experiments", fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)  # Set x-tick font size
plt.yticks(fontsize=16)  # Set y-tick font size
plt.legend()
plt.grid(True)


plt.savefig(root_dir + "/pictures" + "/nonconvex_cifar_server_losses_clients_participation.png", dpi=300, bbox_inches='tight')  # High-quality PNG
plt.savefig(root_dir + "/pictures" + "/nonconvex_cifar_server_losses_client_participation.pdf", bbox_inches='tight')  # PDF format

plt.show()






