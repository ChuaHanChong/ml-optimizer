"""Minimal script using pandas CSV loading for dataset format detection tests."""
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="./data/train.csv")
parser.add_argument("--val_path", type=str, default="./data/val.csv")

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float32)
