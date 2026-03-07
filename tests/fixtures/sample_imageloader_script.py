"""Minimal script using ImageFolder for dataset format detection tests."""
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

dataset = ImageFolder(root="./data/train", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
