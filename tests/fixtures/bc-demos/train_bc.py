#!/usr/bin/env python3
"""Tiny behavior cloning on a robomimic-style HDF5 demo file (h5py + torch).

kv-format logging that the plugin's parse_logs.py reads natively — one line
per epoch:

    epoch=1 loss=0.4321 val_loss=0.4400 transitions=1000

Usage:
    python3 train_bc.py --demos <path.hdf5> [--epochs 50] [--lr 1e-2] [--seed 0]
"""

import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn

DEMO_FORMAT = "robomimic"  # data/demo_N/{obs/state,actions}, attrs num_samples


def load_robomimic_hdf5(path):
    """Load all transitions from a robomimic-layout HDF5 file."""
    obs_list, act_list = [], []
    with h5py.File(path, "r") as f:
        for demo_name in sorted(f["data"].keys()):
            demo = f["data"][demo_name]
            obs_list.append(demo["obs/state"][:])
            act_list.append(demo["actions"][:])
    return np.concatenate(obs_list), np.concatenate(act_list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs, actions = load_robomimic_hdf5(args.demos)
    n = len(obs)
    perm = np.random.permutation(n)
    split = int(n * 0.9)
    tr, va = perm[:split], perm[split:]
    x = torch.as_tensor(obs, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, y.shape[1]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x[tr]), y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x[va]), y[va])
        print(
            f"epoch={epoch} loss={loss.item():.4f} "
            f"val_loss={val_loss.item():.4f} transitions={n}",
            flush=True,
        )


if __name__ == "__main__":
    main()
