#!/usr/bin/env python3
"""Minimal PPO on CartPole-v1 (gymnasium + torch). CPU-only, minutes.

Test fixture for the ml-optimizer RL path. Logs kv-format lines that the
plugin's parse_logs.py reads natively — one line per update:

    update=3 global_step=1536 episode_return=24.60 policy_loss=-0.0123 value_loss=54.2101 entropy=0.6812

and a final deterministic evaluation (same protocol the baseline skill uses
for RL: fixed episode count, argmax policy, mean+std — never best-over-curve):

    final_eval_mean=142.30 final_eval_std=51.20 eval_episodes=10

With --eval_tasks pole_short,pole_long the same line also carries one key per
task, which the plugin aggregates into a mean and a worst-task value:

    final_eval_mean_pole_short=98.40 final_eval_std_pole_short=31.10 ...

Flags map to the plugin's RL conventions:
    --seed N              framework seed (recorded as random_seed)
    --total_timesteps N   environment-step budget (fixed_step_budget)
    --lr F                Adam learning rate
    --eval_tasks a,b      comma-separated CartPole variants for multi-task eval

Usage:
    python3 train_ppo.py --seed 1 --total_timesteps 8192 --lr 2.5e-4
"""

import argparse
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 4
MINIBATCH = 128
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.policy = nn.Linear(64, n_actions)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total_timesteps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--num_steps", type=int, default=512, help="rollout length per update")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument(
        "--eval_tasks", type=str, default="",
        help="comma-separated CartPole variants to evaluate on "
             "(pole_short/pole_long/heavy); empty = single-task",
    )
    return p.parse_args()


# CartPole variants used as distinct "tasks" — each perturbs the environment's
# physics, so a policy that only solves the default pole scores worse here.
TASK_VARIANTS = {
    "pole_short": {"length": 0.25},
    "pole_long": {"length": 1.0},
    "heavy": {"masspole": 0.5},
}


def evaluate(model, env, seed: int, episodes: int) -> tuple[float, float]:
    """Deterministic argmax rollout; returns (mean, std) of episode returns."""
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done = False
        ret = 0.0
        while not done:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, terminated, truncated, _ = env.step(int(logits.argmax()))
            ret += float(reward)
            done = terminated or truncated
        returns.append(ret)
    mean = sum(returns) / len(returns)
    std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5
    return mean, std


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    recent_returns = deque(maxlen=10)
    ep_return = 0.0
    global_step = 0
    update = 0

    while global_step < args.total_timesteps:
        update += 1
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []

        for _ in range(args.num_steps):
            t_obs = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, value = model(t_obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            obs_buf.append(t_obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(float(reward))
            done_buf.append(done)
            ep_return += float(reward)
            global_step += 1
            obs = next_obs
            if done:
                recent_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            _, next_value = model(torch.as_tensor(obs, dtype=torch.float32))

        advantages = torch.zeros(args.num_steps)
        lastgaelam = 0.0
        for t in reversed(range(args.num_steps)):
            nonterminal = 0.0 if done_buf[t] else 1.0
            nv = next_value if t == args.num_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + GAMMA * nv * nonterminal - val_buf[t]
            lastgaelam = delta + GAMMA * GAE_LAMBDA * nonterminal * lastgaelam
            advantages[t] = lastgaelam

        b_obs = torch.stack(obs_buf)
        b_act = torch.stack(act_buf)
        b_logp = torch.stack(logp_buf)
        b_val = torch.stack(val_buf)
        b_ret = advantages + b_val
        b_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.arange(args.num_steps)
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, args.num_steps, MINIBATCH):
                mb = idx[start:start + MINIBATCH]
                logits, value = model(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = ((value - b_ret[mb]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

        mean_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
        print(
            f"update={update} global_step={global_step} "
            f"episode_return={mean_return:.2f} policy_loss={policy_loss.item():.4f} "
            f"value_loss={value_loss.item():.4f} entropy={entropy.item():.4f}",
            flush=True,
        )

    # Final deterministic eval — mirrors the baseline skill's RL Evaluation
    # protocol (fixed episode count, deterministic policy, mean+std).
    mean, std = evaluate(model, env, args.seed, args.eval_episodes)
    parts = [
        f"final_eval_mean={mean:.2f}",
        f"final_eval_std={std:.2f}",
        f"eval_episodes={args.eval_episodes}",
    ]

    # Multi-task eval: one <metric>_<task> key per variant, flat and numeric so
    # the plugin's aggregator reads them without any schema change.
    tasks = [t for t in args.eval_tasks.split(",") if t]
    for task in tasks:
        if task not in TASK_VARIANTS:
            raise SystemExit(f"unknown eval task {task!r}; known: {sorted(TASK_VARIANTS)}")
        task_env = gym.make("CartPole-v1")
        for attr, value in TASK_VARIANTS[task].items():
            setattr(task_env.unwrapped, attr, value)
        # Gymnasium caches derived constants at init; recompute them after attribute changes
        # so physics equations use the correct values (e.g., total_mass divides into cart accel).
        u = task_env.unwrapped
        u.total_mass = u.masspole + u.masscart
        u.polemass_length = u.masspole * u.length
        t_mean, t_std = evaluate(model, task_env, args.seed, args.eval_episodes)
        task_env.close()
        parts.append(f"final_eval_mean_{task}={t_mean:.2f}")
        parts.append(f"final_eval_std_{task}={t_std:.2f}")

    print(" ".join(parts), flush=True)
    env.close()


if __name__ == "__main__":
    main()
