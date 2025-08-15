#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:32:14 2024

@author: forootan
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.fill_(0)

    def forward(self, x):
        return self.relu(self.conv(x))

class CNNMnistModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, hidden_channels=32, num_layers=3, learning_rate=1e-3):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ConvLayer(self.input_channels, self.hidden_channels))

        for _ in range(self.num_layers - 1):
            self.conv_layers.append(ConvLayer(self.hidden_channels, self.hidden_channels))

        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to [batch_size, channels, 1, 1]
        self.fc = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.pool(x)  # Pool across height and width
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, channels]
        x = self.fc(x)

        return x

    def optimizer_func(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def scheduler_setting(self):
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer_func(),
            step_size=10,
            gamma=0.1
        )

    def run(self):
        model = self
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()

        return model, optimizer, scheduler

