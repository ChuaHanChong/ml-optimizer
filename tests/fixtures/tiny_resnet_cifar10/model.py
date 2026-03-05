"""Tiny ResNet model for CIFAR-10 (~75-100K parameters)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TinyResNet(nn.Module):
    """Tiny ResNet: 3 residual blocks (16→32→64 channels), ~75-100K params."""

    def __init__(self, num_classes: int = 10, base_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, base_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        c = base_channels
        self.block1 = ResidualBlock(c, c, stride=1)
        self.block2 = ResidualBlock(c, c * 2, stride=2)
        self.block3 = ResidualBlock(c * 2, c * 4, stride=2)

        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model(num_classes: int = 10, base_channels: int = 16) -> TinyResNet:
    """Factory function to create a TinyResNet."""
    return TinyResNet(num_classes=num_classes, base_channels=base_channels)
