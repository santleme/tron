"""Pinned experimental spatial-policy inference for Kaggle tournaments."""

from __future__ import annotations

def model_class():
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )

        def forward(self, value):
            return torch.relu(value + self.layers(value))

    class SpatialPolicy(nn.Module):
        def __init__(self, width: int, height: int, hidden_size: int,
                     channels: int, blocks: int):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(channels, hidden_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_size), nn.ReLU(inplace=True),
            )
            self.residual = nn.Sequential(*(ResidualBlock(hidden_size) for _ in range(blocks)))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True),
                nn.Linear(hidden_size, 4),
            )

        def forward(self, observation):
            value = observation.permute(0, 3, 1, 2).contiguous()
            return self.head(self.residual(self.stem(value)))

    return SpatialPolicy

