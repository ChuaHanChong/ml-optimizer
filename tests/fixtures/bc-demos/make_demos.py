#!/usr/bin/env python3
"""Generate a synthetic robomimic-style HDF5 demonstration file.

Layout (robomimic conventions):
    data/                       group, attrs: total (int), env_args (JSON str)
    data/demo_<i>/              group, attrs: num_samples (int)
    data/demo_<i>/obs/state     float32 [T, OBS_DIM]
    data/demo_<i>/actions       float32 [T, ACT_DIM]
    data/demo_<i>/rewards       float32 [T]
    data/demo_<i>/dones         int64   [T]

Actions are a fixed linear function of observations plus noise, so behavior
cloning has a learnable target.

Usage:
    python3 make_demos.py <out.hdf5> [--demos 20] [--steps 50] [--seed 0]
"""

import argparse
import json

import h5py
import numpy as np

OBS_DIM = 10
ACT_DIM = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--demos", type=int, default=20)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    expert_w = (0.5 * rng.standard_normal((OBS_DIM, ACT_DIM))).astype(np.float32)

    with h5py.File(args.out, "w") as f:
        data = f.create_group("data")
        total = 0
        for i in range(args.demos):
            obs = rng.standard_normal((args.steps, OBS_DIM)).astype(np.float32)
            noise = 0.05 * rng.standard_normal((args.steps, ACT_DIM))
            actions = (obs @ expert_w + noise).astype(np.float32)
            demo = data.create_group(f"demo_{i}")
            demo.create_dataset("obs/state", data=obs)
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("rewards", data=np.zeros(args.steps, dtype=np.float32))
            demo.create_dataset("dones", data=np.zeros(args.steps, dtype=np.int64))
            demo.attrs["num_samples"] = args.steps
            total += args.steps
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps({"env_name": "SyntheticReach", "type": 1})

    print(f"wrote {args.out}: demos={args.demos} transitions={total}")


if __name__ == "__main__":
    main()
