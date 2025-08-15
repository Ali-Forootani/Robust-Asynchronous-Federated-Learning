#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:39:02 2025

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
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import asyncio
import nest_asyncio

# Allow nested event loops for asyncio
nest_asyncio.apply()

# LSTM Model for Regression
class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.fill_(0)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc.bias.fill_(0)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        return self.fc(last_hidden_state)


def create_directory(num_clients, num_rounds, local_epochs, max_clients_per_round, base_dir="results"):
    dir_name = f"{base_dir}/clients_{num_clients}_rounds_{num_rounds}_epochs_{local_epochs}_clients_per_round_{max_clients_per_round}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def save_losses_as_numpy(losses, filename):
    np.save(filename, np.array(losses))


def save_selected_clients(selected_clients, filename):
    with open(filename, "w") as f:
        f.write("Round, Selected Clients\n")
        for round_num, clients in enumerate(selected_clients):
            f.write(f"{round_num}, {', '.join(map(str, clients))}\n")


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


def plot_selected_clients(selected_clients, save_path):
    plt.figure(figsize=(10, 6))
    for round_num, clients in enumerate(selected_clients):
        plt.scatter([round_num] * len(clients), clients, label=f"Round {round_num + 1}", marker="x", color="b")
    plt.xlabel("Round")
    plt.ylabel("Client Index")
    plt.title("Clients Selected Per Round")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def partition_non_iid(dataset, num_clients, alpha=0.5):
    num_samples = len(dataset)
    data_by_index = list(range(num_samples))
    np.random.shuffle(data_by_index)
    proportions = np.random.dirichlet([alpha] * num_clients)
    proportions = (proportions * num_samples).astype(int)

    for i in range(len(proportions)):
        if proportions[i] == 0:
            proportions[i] = 1

    while sum(proportions) > num_samples:
        proportions[np.argmax(proportions)] -= 1
    while sum(proportions) < num_samples:
        proportions[np.argmin(proportions)] += 1

    client_indices = [data_by_index[sum(proportions[:i]):sum(proportions[:i + 1])] for i in range(num_clients)]

    return client_indices


async def train_random_clients_with_regression_objective_delay(
    client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t=0, accumulation_steps=1, early_stopping_patience=10
):
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []
    delay_simulation = random.uniform(0, delay_t)
    await asyncio.sleep(delay_simulation)

    optimizer = torch.optim.Adam(client_model.parameters(), lr=gamma_0)
    for epoch in range(local_epochs):
        client_model.train()
        epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)
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


async def federated_learning_with_lstm(
    clients_models, server_model, clients_train_loaders, num_rounds=10, local_epochs=1, max_clients_per_round=3,
    loss_fn=None, gamma_0=1e-3, alpha=0.1, delay_t=2, accumulation_steps=1, early_stopping_patience=10, val_loader=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses = [[] for _ in range(len(clients_models))]
    server_losses = []
    selected_clients_per_round = []

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        selected_clients = random.sample(range(len(clients_models)), max_clients_per_round)
        selected_clients_per_round.append(selected_clients)
        client_weights = []

        async def train_client_task(i):
            state_dict, client_losses = await train_random_clients_with_regression_objective_delay(
                clients_models[i], clients_train_loaders[i], device, local_epochs, loss_fn, gamma_0, alpha,
                delay_t=delay_t, accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            num_samples = len(clients_train_loaders[i].dataset)
            return state_dict, client_losses, num_samples

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        total_samples = 0
        round_loss = 0

        for i, (state_dict, client_loss, num_samples) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            total_samples += num_samples
            round_loss += num_samples * np.mean(client_loss)

        weighted_server_loss = round_loss / total_samples
        server_losses.append(weighted_server_loss)

        new_server_state_dict = {key: torch.zeros_like(value) for key, value in client_weights[0].items()}
        for key in new_server_state_dict:
            for i, client_weight in zip(selected_clients, client_weights):
                weight_factor = len(clients_train_loaders[i].dataset) / total_samples
                new_server_state_dict[key] += client_weight[key] * weight_factor

        server_model.load_state_dict(new_server_state_dict)

    return server_model, training_losses, selected_clients_per_round, server_losses


num_samples = 10000
time_steps = 10
input_features = 1
x_data = torch.randn(num_samples, time_steps, input_features)
y_data = torch.sum(x_data, dim=1, keepdim=True)

train_dataset = TensorDataset(x_data, y_data)
num_clients = 10
alpha = 0.5
client_data_indices = partition_non_iid(train_dataset, num_clients, alpha=alpha)

batch_size = 64
train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True) for indices in client_data_indices]

val_indices = random.sample(range(len(train_dataset)), 1000)
val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=False)

clients_models = [LSTMLayer(input_size=input_features, hidden_size=32, num_layers=3) for _ in range(num_clients)]
server_model = LSTMLayer(input_size=input_features, hidden_size=32, num_layers=3)

num_clients_per_round = 5
num_rounds = 50
local_epochs = 10

async def main():
    loss_fn = F.mse_loss
    global server_model

    results_dir = create_directory(num_clients=num_clients, num_rounds=num_rounds, local_epochs=local_epochs, max_clients_per_round=num_clients_per_round)

    server_model, training_losses, selected_clients_per_round, server_losses = await federated_learning_with_lstm(
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
        val_loader=val_loader
    )

    for i, client_losses in enumerate(training_losses):
        save_losses_as_numpy(client_losses, os.path.join(results_dir, f"client_{i}_losses.npy"))
        plot_losses(client_losses, title=f"Client {i} Training Losses", save_path=os.path.join(results_dir, f"client_{i}_training_loss.png"))

    save_selected_clients(selected_clients_per_round, os.path.join(results_dir, "selected_clients.csv"))
    plot_selected_clients(selected_clients_per_round, save_path=os.path.join(results_dir, "selected_clients_plot.png"))

    save_losses_as_numpy(server_losses, os.path.join(results_dir, "server_losses.npy"))
    plot_losses(server_losses, title="Global Server Loss", save_path=os.path.join(results_dir, "server_loss_plot.png"))

    print(f"Results saved in {results_dir}")

asyncio.run(main())
