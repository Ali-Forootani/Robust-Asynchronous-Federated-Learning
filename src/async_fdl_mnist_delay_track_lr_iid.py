#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 5 07:17:54 2025
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
import asyncio
import nest_asyncio


# Allow nested event loops for asyncio
nest_asyncio.apply()

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
    dir_name = f"{base_dir}/clients_{num_clients}_rounds_{num_rounds}_epochs_{local_epochs}_clients_per_round_{max_clients_per_round}_{time.strftime('%Y%m%d_%H%M%S')}"
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


# Function to save execution times
def save_execution_times(execution_times, filename):
    with open(filename, "w") as f:
        f.write("Round, Client Index, Execution Time (s)\n")
        for round_num, times in enumerate(execution_times):
            for client_idx, exec_time in enumerate(times):
                f.write(f"{round_num}, {client_idx}, {exec_time:.4f}\n")


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


# Function to simulate client training with delays
async def train_random_clients_with_classification_objective_delay_4(
    client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t=0, accumulation_steps=1, early_stopping_patience=10
):
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []
    delay_simulation = random.uniform(0, delay_t)  # Simulate network delay
    await asyncio.sleep(delay_simulation)  # Simulate asynchronous delay

    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0
        gamma_t = gamma_0 / (torch.sqrt(torch.tensor(epoch + 1, dtype=torch.float32)) * (1 + alpha * delay_t))  # Delay-aware LR
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


async def federated_learning_with_mnist_cnn(
    clients_models, server_model, clients_train_loaders, num_rounds=10, local_epochs=1, max_clients_per_round=3,
    loss_fn=None, gamma_0=1e-3, alpha=0.1, delay_t=2, accumulation_steps=1, early_stopping_patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses = [[] for _ in range(len(clients_models))]
    server_losses = []  # Track server loss over rounds
    selected_clients_per_round = []  # Track selected clients for each round
    execution_times_by_round = []  # Track execution times for each round

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        selected_clients = random.sample(range(len(clients_models)), max_clients_per_round)
        selected_clients_per_round.append(selected_clients)
        client_weights = []

        async def train_client_task(i):
            start_time = time.time()  # Start timing
            state_dict, client_losses = await train_random_clients_with_classification_objective_delay_4(
                clients_models[i], clients_train_loaders[i], device, local_epochs, loss_fn, gamma_0, alpha,
                delay_t=delay_t,  # Apply delay
                accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            end_time = time.time()
            execution_time = end_time - start_time  # Calculate execution time
            return state_dict, client_losses, execution_time

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        round_loss = 0  # Track the server's loss for this round
        execution_times = []  # Track execution times for this round

        for i, (state_dict, client_loss, execution_time) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            round_loss += sum(client_loss) / len(client_loss)
            execution_times.append(execution_time)

        server_losses.append(round_loss / len(selected_clients))  # Average loss over selected clients
        execution_times_by_round.append(execution_times)  # Store execution times for this round

        # Aggregate weights
        total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
        new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}

        for key in new_server_state_dict:
            for i, client_weight in zip(selected_clients, client_weights):
                weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)

    return server_model, training_losses, server_losses, selected_clients_per_round, execution_times_by_round


# Dataset Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

reduced_train_size = 60000
train_subset = Subset(train_dataset, range(reduced_train_size))

# Partitioning IID data for clients
num_clients = 10
batch_size = 64
client_data_indices = np.array_split(np.arange(len(train_subset)), num_clients)
train_loaders = [DataLoader(Subset(train_subset, indices), batch_size=batch_size, shuffle=True) for indices in client_data_indices]

# Initialize models
clients_models = [CNNMnistModel(input_channels=1, num_classes=10, hidden_channels=32, num_layers=3) for _ in range(num_clients)]
server_model = CNNMnistModel(input_channels=1, num_classes=10, hidden_channels=32, num_layers=3)

num_clients_per_round = 5

num_rounds=50
local_epochs=10


# Async main function
async def main():
    loss_fn = F.nll_loss
    global server_model

    # User-defined number of clients per round
    #num_clients_per_round = int(input("Enter the number of clients to select per round (1 to 10): "))
    assert 1 <= num_clients_per_round <= num_clients

    # Create results directory based on the settings
    results_dir = create_directory(num_clients=num_clients, num_rounds=num_rounds, local_epochs=local_epochs, max_clients_per_round=num_clients_per_round)

    server_model, training_losses, server_losses, selected_clients_per_round, execution_times_by_round = await federated_learning_with_mnist_cnn(
        clients_models=clients_models,
        server_model=server_model,
        clients_train_loaders=train_loaders,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        max_clients_per_round=num_clients_per_round,
        loss_fn=loss_fn,
        gamma_0=1e-3,
        alpha=0.01,
        delay_t=2,
        accumulation_steps=1,
        early_stopping_patience=10
    )

    # Save and plot losses for each client
    for i, client_losses in enumerate(training_losses):
        client_numpy_filename = os.path.join(results_dir, f"client_{i}_losses.npy")
        save_losses_as_numpy(client_losses, client_numpy_filename)  # Save as NumPy array
        plot_losses(
            client_losses,
            title=f"Client {i} Training Losses",
            save_path=os.path.join(results_dir, f"client_{i}_training_loss.png")
        )

    # Save and plot server loss
    server_numpy_filename = os.path.join(results_dir, "server_losses.npy")
    save_losses_as_numpy(server_losses, server_numpy_filename)
    plot_losses(
        server_losses,
        title="Server Loss Across Rounds",
        save_path=os.path.join(results_dir, "server_training_loss.png")
    )

    # Save and plot selected clients
    selected_clients_filename = os.path.join(results_dir, "selected_clients.csv")
    save_selected_clients(selected_clients_per_round, selected_clients_filename)
    plot_selected_clients(selected_clients_per_round, save_path=os.path.join(results_dir, "selected_clients_plot.png"))

    # Save execution times
    execution_times_filename = os.path.join(results_dir, "execution_times.csv")
    save_execution_times(execution_times_by_round, execution_times_filename)

    print(f"Results saved in {results_dir}")


# Run async main function
asyncio.run(main())
