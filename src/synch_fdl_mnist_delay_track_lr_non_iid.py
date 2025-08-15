#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:01:14 2025

@author: forootan
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict

# CNN Model for MNIST
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.fill_(0)

    def forward(self, x):
        return self.relu(self.conv(x))


class CNNMnistModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, hidden_channels=32, num_layers=3):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ConvLayer(input_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.conv_layers.append(ConvLayer(hidden_channels, hidden_channels))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Function to create directory for saving results
def create_directory(num_clients, num_rounds, local_epochs, max_clients_per_round, base_dir="results"):
    dir_name = f"{base_dir}/synch_clients_{num_clients}_rounds_{num_rounds}_epochs_{local_epochs}_clients_per_round_{max_clients_per_round}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


# Function to save training losses as a NumPy array
def save_losses_as_numpy(losses, filename):
    np.save(filename, np.array(losses))


# Function to save selected clients
def save_selected_clients(selected_clients, filename):
    with open(filename, "w") as f:
        f.write("Round, Selected Clients\n")
        for round_num, clients in enumerate(selected_clients):
            f.write(f"{round_num}, {', '.join(map(str, clients))}\n")


# Function to plot losses
def plot_losses(losses, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Loss")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# Function to plot selected clients per round
def plot_selected_clients(selected_clients, save_path):
    plt.figure(figsize=(10, 6))
    num_rounds = len(selected_clients)
    for round_num, clients in enumerate(selected_clients):
        plt.scatter([round_num] * len(clients), clients, label=f"Round {round_num + 1}", marker="x", color="b")
    plt.xlabel("Round")
    plt.ylabel("Client Index")
    plt.title("Clients Selected Per Round")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# Non-IID Partitioning using Dirichlet distribution
def partition_non_iid(dataset, num_clients, alpha=0.5):
    num_classes = 10
    data_by_class = defaultdict(list)

    for idx, (image, label) in enumerate(dataset):
        data_by_class[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        np.random.shuffle(data_by_class[c])
        class_indices = data_by_class[c]
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)

        for i, proportion in enumerate(proportions):
            client_indices[i].extend(class_indices[:proportion])
            class_indices = class_indices[proportion:]

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


# Synchronous client training
def train_client(client_model, train_loader, device, local_epochs, loss_fn, gamma_0,
                 alpha, delay_t, accumulation_steps, early_stopping_patience):
    client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []
    
    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0
        gamma_t = gamma_0 / (torch.sqrt(torch.tensor(epoch + 1, dtype=torch.float32)) * (1 + alpha * delay_t))
        optimizer = torch.optim.Adam(client_model.parameters(), lr=gamma_t.item())
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        client_losses.append(avg_epoch_loss)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    return client_model.state_dict(), client_losses


# Federated Learning Process (Synchronous)
def federated_learning(clients_models, server_model, clients_train_loaders, num_rounds,
                       local_epochs, max_clients_per_round, loss_fn,
                       gamma_0, alpha, delay_t, accumulation_steps, early_stopping_patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses = [[] for _ in range(len(clients_models))]
    server_losses = []
    selected_clients_per_round = []

    results_dir = create_directory(len(clients_models), num_rounds, local_epochs, max_clients_per_round)

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        selected_clients = random.sample(range(len(clients_models)), max_clients_per_round)
        selected_clients_per_round.append(selected_clients)
        client_weights = []
        round_loss = 0
        
        for i in selected_clients:
            state_dict, client_loss = train_client(
                clients_models[i], clients_train_loaders[i], device, local_epochs,
                loss_fn, gamma_0, alpha, delay_t, accumulation_steps, early_stopping_patience
            )
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            round_loss += sum(client_loss) / len(client_loss)
        
        server_losses.append(round_loss / len(selected_clients))
        
        total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
        new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}
        
        for key in new_server_state_dict:
            for i, client_weight in zip(selected_clients, client_weights):
                weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                new_server_state_dict[key] += client_weight[key] * weight_factor
        
        server_model.load_state_dict(new_server_state_dict)

    # Save and plot training results
    for i, client_losses in enumerate(training_losses):
        save_losses_as_numpy(client_losses, os.path.join(results_dir, f"synch_client_{i}_losses.npy"))
        plot_losses(client_losses, title=f"Synch Client {i} Training Losses",
                    save_path=os.path.join(results_dir, f"synch_client_{i}_training_loss.png"))

    save_losses_as_numpy(server_losses, os.path.join(results_dir, "synch_server_losses.npy"))
    plot_losses(server_losses, title="Synch Server Loss Across Rounds",
                save_path=os.path.join(results_dir, "synch_server_training_loss.png"))

    save_selected_clients(selected_clients_per_round, os.path.join(results_dir, "synch_selected_clients.csv"))
    plot_selected_clients(selected_clients_per_round, os.path.join(results_dir, "synch_selected_clients_plot.png"))

    print(f"Results saved in {results_dir}")

    return server_model, training_losses, server_losses, selected_clients_per_round




# Dataset Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Non-IID partitioning
num_clients = 10
alpha = 0.5  # Controls how non-IID the distribution is (lower alpha -> more imbalanced)
client_data_indices = partition_non_iid(train_dataset, num_clients, alpha=alpha)

batch_size = 64
train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True) for indices in client_data_indices]

# Initialize models
clients_models = [CNNMnistModel(input_channels=1, num_classes=10, hidden_channels=32, num_layers=3) for _ in range(num_clients)]
server_model = CNNMnistModel(input_channels=1, num_classes=10, hidden_channels=32, num_layers=3)

num_clients_per_round = 5


num_rounds=1000
local_epochs=10



server_model, training_losses, server_losses, selected_clients_per_round = federated_learning(
    clients_models=clients_models,
    server_model=server_model,
    clients_train_loaders=train_loaders,
    num_rounds= num_rounds,  # Same as asynch
    local_epochs= local_epochs,  # Same as asynch
    max_clients_per_round= num_clients_per_round,  # Same as asynch
    loss_fn=F.nll_loss,  # Same as asynch
    gamma_0=1e-3,  # Same as asynch
    alpha=0.01,  # Same as asynch
    delay_t=0.02,  # Same as asynch
    accumulation_steps=1,  # Same as asynch
    early_stopping_patience=10  # Same as asynch
)






