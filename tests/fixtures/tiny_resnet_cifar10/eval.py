#!/usr/bin/env python3
"""Evaluate a TinyResNet checkpoint on CIFAR-10 test subset."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TinyResNet on CIFAR-10")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--subset_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform,
    )

    g = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(test_dataset), generator=g)[:args.subset_size].tolist()
    loader = DataLoader(Subset(test_dataset, indices), batch_size=64, shuffle=False, num_workers=0)

    model = get_model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    eval_loss = total_loss / total if total > 0 else 0.0
    eval_accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"eval_loss={eval_loss:.4f} eval_accuracy={eval_accuracy:.1f}")


if __name__ == "__main__":
    main()
