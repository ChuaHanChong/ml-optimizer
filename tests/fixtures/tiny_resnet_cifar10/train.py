#!/usr/bin/env python3
"""Self-contained CIFAR-10 training script for TinyResNet.

Log format: kv lines matching parse_logs.py expectations.
Example: epoch=1 step=50 loss=2.1234 lr=0.001 accuracy=25.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Allow importing model.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyResNet on CIFAR-10")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--config", type=str, default=None, help="YAML config file (overrides CLI args)")
    parser.add_argument("--subset_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config, with fallback to simple parsing if PyYAML unavailable."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Simple YAML parser for flat/nested key-value pairs
        config = {}
        current_section = None
        with open(config_path) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    current_section = line[:-1].strip()
                    config[current_section] = {}
                elif current_section and ":" in line:
                    key, val = line.strip().split(":", 1)
                    val = val.strip()
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    config[current_section][key.strip()] = val
        return config


def apply_config(args, config: dict):
    """Apply YAML config values to args, config takes precedence."""
    training = config.get("training", {})
    data = config.get("data", {})
    for key in ["lr", "batch_size", "epochs", "weight_decay", "scheduler"]:
        if key in training:
            setattr(args, key, training[key])
    for key in ["subset_size", "seed"]:
        if key in data:
            setattr(args, key, data[key])


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(data_dir: str, subset_size: int, batch_size: int, seed: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test,
    )

    # Deterministic subset selection
    g = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(len(train_dataset), generator=g)[:subset_size].tolist()
    test_indices = torch.randperm(len(test_dataset), generator=g)[:subset_size].tolist()

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
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
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data(
        args.data_dir, args.subset_size, args.batch_size, args.seed,
    )

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            global_step += 1

        scheduler.step()
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = 100.0 * correct / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"epoch={epoch} step={global_step} loss={epoch_loss:.4f} lr={current_lr:.6f} accuracy={epoch_acc:.1f}")
        sys.stdout.flush()

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"final loss={test_loss:.4f} accuracy={test_acc:.1f}")

    # Save checkpoint
    checkpoint_path = output_dir / "model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }, checkpoint_path)

    return test_loss, test_acc


def main():
    args = parse_args()
    if args.config:
        config = load_yaml_config(args.config)
        apply_config(args, config)
    train(args)


if __name__ == "__main__":
    main()
