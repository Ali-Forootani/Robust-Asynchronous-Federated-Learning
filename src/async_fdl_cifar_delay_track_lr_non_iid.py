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
from collections import defaultdict
import csv

# Allow nested event loops for asyncio
nest_asyncio.apply()

# ResNet Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # Fixed
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# ResNet CNN
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return F.log_softmax(self.fc(out), dim=1)

# Helper function to create result directories
def create_directory(num_clients, num_rounds, local_epochs, max_clients_per_round, base_dir="results"):
    dir_name = f"{base_dir}/cifar_clients_{num_clients}_rounds_{num_rounds}_epochs_{local_epochs}_clients_per_round_{max_clients_per_round}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

# Save losses as NumPy array
def save_losses_as_numpy(losses, filename):
    np.save(filename, np.array(losses))

# Save selected clients as CSV
def save_selected_clients(selected_clients_per_round, filename):
    with open(filename, "w") as f:
        f.write("Round, Selected Clients\n")
        for round_num, clients in enumerate(selected_clients_per_round):
            f.write(f"{round_num + 1}, {', '.join(map(str, clients))}\n")

# Save execution times as CSV
def save_execution_times(execution_times, filename):
    with open(filename, "w") as f:
        f.write("Round, Client Index, Execution Time (s)\n")
        for round_num, times in enumerate(execution_times):
            for client_idx, exec_time in enumerate(times):
                f.write(f"{round_num + 1}, {client_idx}, {exec_time:.4f}\n")

# Plot and save losses
def plot_losses(losses, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Plot and save selected clients per round
def plot_selected_clients(selected_clients_per_round, save_path):
    plt.figure(figsize=(10, 6))
    for round_num, clients in enumerate(selected_clients_per_round):
        plt.scatter([round_num + 1] * len(clients), clients, label=f"Round {round_num + 1}", marker="x", color="b")
    plt.xlabel("Round")
    plt.ylabel("Client Index")
    plt.title("Clients Selected Per Round")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Non-IID Partitioning using Dirichlet distribution
def partition_non_iid(dataset, num_clients, alpha=0.5):
    num_classes = 10  # Number of classes in CIFAR-10
    data_by_class = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
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

# Function to simulate client training with delays
async def train_random_clients_with_classification_objective_delay(
    client_model, train_loader, device, local_epochs, loss_fn, gamma_0, alpha, delay_t=0, accumulation_steps=1, early_stopping_patience=10
):
    client_model = client_model.to(device)
    patience_counter = 0
    best_loss = float("inf")
    client_losses = []
    delay_simulation = random.uniform(0, delay_t)
    await asyncio.sleep(delay_simulation)

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

async def federated_learning_with_resnet(
    clients_models, server_model, clients_train_loaders, num_rounds=10, local_epochs=1, max_clients_per_round=3,
    loss_fn=None, gamma_0=1e-3, alpha=0.1, delay_t=2, accumulation_steps=1, early_stopping_patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses = [[] for _ in range(len(clients_models))]
    server_losses = []
    selected_clients_per_round = []
    execution_times_by_round = []

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        selected_clients = random.sample(range(len(clients_models)), max_clients_per_round)
        selected_clients_per_round.append(selected_clients)
        client_weights = []

        async def train_client_task(i):
            start_time = time.time()
            state_dict, client_losses = await train_random_clients_with_classification_objective_delay(
                clients_models[i], clients_train_loaders[i], device, local_epochs, loss_fn, gamma_0, alpha,
                delay_t=delay_t, accumulation_steps=accumulation_steps, early_stopping_patience=early_stopping_patience
            )
            end_time = time.time()
            return state_dict, client_losses, end_time - start_time

        tasks = [train_client_task(i) for i in selected_clients]
        results = await asyncio.gather(*tasks)

        round_loss = 0
        execution_times = []

        for i, (state_dict, client_loss, execution_time) in zip(selected_clients, results):
            client_weights.append(state_dict)
            training_losses[i].extend(client_loss)
            round_loss += sum(client_loss) / len(client_loss)
            execution_times.append(execution_time)

        server_losses.append(round_loss / len(selected_clients))
        execution_times_by_round.append(execution_times)

        total_samples = sum(len(clients_train_loaders[i].dataset) for i in selected_clients)
        new_server_state_dict = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in client_weights[0].items()}

        for key in new_server_state_dict:
            for i, client_weight in zip(selected_clients, client_weights):
                weight_factor = torch.tensor(len(clients_train_loaders[i].dataset) / total_samples, dtype=torch.float32)
                new_server_state_dict[key] += client_weight[key].float() * weight_factor

        server_model.load_state_dict(new_server_state_dict)

    return server_model, training_losses, server_losses, selected_clients_per_round, execution_times_by_round

# Dataset Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

num_clients = 10
alpha = 0.5
num_rounds = 200
local_epochs = 10
num_clients_per_round = 6

client_data_indices = partition_non_iid(train_dataset, num_clients, alpha=alpha)
batch_size = 64
train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True) for indices in client_data_indices]

clients_models = [ResNet(BasicBlock, [3, 3, 3], num_classes=10) for _ in range(num_clients)]
server_model = ResNet(BasicBlock, [3, 3, 3], num_classes=10)

# Async main function
async def main():
    loss_fn = F.nll_loss
    global server_model

    results_dir = create_directory(num_clients=num_clients, num_rounds=num_rounds, local_epochs=local_epochs, max_clients_per_round=num_clients_per_round)

    server_model, training_losses, server_losses, selected_clients_per_round, execution_times_by_round = await federated_learning_with_resnet(
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
        save_losses_as_numpy(client_losses, client_numpy_filename)
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
