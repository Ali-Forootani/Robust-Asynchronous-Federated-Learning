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
        layers = [block(self.in_channels, out_channels, stride)]
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

# Create directory for results
def create_directory(prefix="synch_cifar", base_dir="results"):
    dir_name = f"{base_dir}/{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

# Save and plot loss
def save_losses(losses, filename):
    np.save(filename, np.array(losses))

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

# Partition dataset non-IID using Dirichlet distribution
def partition_non_iid(dataset, num_clients, alpha=0.5):
    num_classes = 10  # CIFAR-10
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

# Simulated client training
def train_client(model, train_loader, device, local_epochs, loss_fn, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    client_losses = []

    for epoch in range(local_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        client_losses.append(avg_loss)

    return model.state_dict(), client_losses

# Federated learning loop
def federated_learning(clients_models, server_model, clients_train_loaders, num_rounds=10, local_epochs=1, max_clients_per_round=3, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses = [[] for _ in range(len(clients_models))]
    server_losses = []
    selected_clients_per_round = []
    execution_times_by_round = []

    for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
        selected_clients = random.sample(range(len(clients_models)), max_clients_per_round)
        selected_clients_per_round.append(selected_clients)
        client_weights = []
        execution_times = []
        round_loss = 0

        for i in selected_clients:
            start_time = time.time()
            state_dict, client_losses = train_client(
                clients_models[i], clients_train_loaders[i], device, local_epochs, loss_fn, lr=1e-3
            )
            end_time = time.time()
            client_weights.append(state_dict)
            training_losses[i].extend(client_losses)
            round_loss += sum(client_losses) / len(client_losses)
            execution_times.append(end_time - start_time)

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
num_clients = 10
alpha = 0.5
num_rounds = 1000
local_epochs = 10
num_clients_per_round = 5
batch_size = 64

client_data_indices = partition_non_iid(train_dataset, num_clients, alpha=alpha)
train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True) for indices in client_data_indices]

clients_models = [ResNet(BasicBlock, [3, 3, 3], num_classes=10) for _ in range(num_clients)]
server_model = ResNet(BasicBlock, [3, 3, 3], num_classes=10)

# Main execution
results_dir = create_directory()
server_model, training_losses, server_losses, selected_clients_per_round, execution_times_by_round = federated_learning(
    clients_models, server_model, train_loaders, num_rounds, local_epochs, num_clients_per_round, loss_fn=F.nll_loss
)

for i, losses in enumerate(training_losses):
    save_losses(losses, os.path.join(results_dir, f"synch_cifar_client_{i}_losses.npy"))

save_losses(server_losses, os.path.join(results_dir, "synch_cifar_server_losses.npy"))

print(f"Results saved in {results_dir}")
