# Experiment Script Templates

All templates launch the training command **in the background** with output redirected to the log, record the training PID (`$!` — NOT the wrapper's `$$`, which the monitor cannot use to kill training), `wait` for it, and propagate the **real exit code**. Exit code 124 means the `timeout` wrapper fired — success ONLY in the fixed-time-budget template.

## Basic Training Script (with PID tracking)
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}

echo "Starting experiment {exp_id} on GPU {gpu_id}"
echo "Config: {config_summary}"
echo "Started at: $(date)"

{train_command} > <exp_root>/logs/{round_dir}/{exp_id}/train.log 2>&1 &
echo $! > <exp_root>/logs/{round_dir}/{exp_id}/pid
EXIT_CODE=0
wait $! || EXIT_CODE=$?

echo "Experiment {exp_id} finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
```

## Training with Fixed Time Budget (124 = success ONLY here)
```bash
#!/bin/bash
# Experiment: {exp_id} (fixed_time_budget={time_budget}s)
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}

timeout --signal=SIGTERM --kill-after=60 {time_budget} {train_command} > <exp_root>/logs/{round_dir}/{exp_id}/train.log 2>&1 &
echo $! > <exp_root>/logs/{round_dir}/{exp_id}/pid
EXIT_CODE=0
wait $! || EXIT_CODE=$?

# 124 = time budget reached — success ONLY in this fixed-time-budget branch
if [ $EXIT_CODE -eq 124 ]; then
    echo "Time budget reached ({time_budget}s) — stopped normally"
    EXIT_CODE=0
fi
exit $EXIT_CODE
```

## Training with Config Override (PyTorch/YAML)
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}

python train.py \
  --config {base_config} \
  --lr {lr} \
  --batch_size {batch_size} \
  --weight_decay {weight_decay} \
  --epochs {epochs} \
  --output_dir <exp_root>/logs/{round_dir}/{exp_id} \
  > <exp_root>/logs/{round_dir}/{exp_id}/train.log 2>&1 &
echo $! > <exp_root>/logs/{round_dir}/{exp_id}/pid
EXIT_CODE=0
wait $! || EXIT_CODE=$?
exit $EXIT_CODE
```

## Training with Eval at End
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}

# Training (background + real PID + real exit code)
{train_command} > <exp_root>/logs/{round_dir}/{exp_id}/train.log 2>&1 &
echo $! > <exp_root>/logs/{round_dir}/{exp_id}/pid
EXIT_CODE=0
wait $! || EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code $EXIT_CODE — skipping eval"
    exit $EXIT_CODE
fi

# Evaluation (foreground — short, and an eval failure must fail the script)
{eval_command} > <exp_root>/logs/{round_dir}/{exp_id}/eval.log 2>&1

echo "Experiment {exp_id} completed"
```

## Training with Code Changes (Git Worktree)
```bash
#!/bin/bash
# Experiment: {exp_id}
# Code changes: {change_description}
# Code branch: {code_branch}
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}
mkdir -p <exp_root>/artifacts/{round_dir}/{exp_id}

# Isolated worktree for the code branch, OUTSIDE <exp_root>/ (a worktree under
# <exp_root>/ can be wiped by a parallel-cleanup race). --detach lets multiple
# parallel experiments share {code_branch} (plain add fails "already checked out").
PROJECT_HASH=$(echo "<project_root>" | sha1sum | cut -c1-8)
WORKTREE_ROOT="/tmp/ml-opt-worktrees-${PROJECT_HASH}"; mkdir -p "$WORKTREE_ROOT"
WORKTREE_PATH="$WORKTREE_ROOT/{exp_id}"
git worktree add --detach "$WORKTREE_PATH" {code_branch}
cd "$WORKTREE_PATH"

# Training (absolute log path — the worktree is outside <exp_root>): background
# launch, training PID ($!) to the pid file, wait, capture the real exit code.
{train_command} > <exp_root>/logs/{round_dir}/{exp_id}/train.log 2>&1 &
echo $! > <exp_root>/logs/{round_dir}/{exp_id}/pid
EXIT_CODE=0
wait $! || EXIT_CODE=$?

# Copy artifacts out of the worktree before cleanup — checkpoints may be nested
# in framework output dirs, so search up to 4 levels deep (not a root-only glob).
find . -maxdepth 4 \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.h5' \
  -o -name '*.pkl' -o -name '*.safetensors' \) \
  -exec cp {} <exp_root>/artifacts/{round_dir}/{exp_id}/ \; 2>/dev/null || true

# Evaluation — MUST run inside the worktree before cleanup (skip if training failed)
if [ $EXIT_CODE -eq 0 ]; then
    {eval_command} > <exp_root>/logs/{round_dir}/{exp_id}/eval.log 2>&1
fi

# Cleanup worktree (validate registration first, never rm -rf)
cd - >/dev/null
if git worktree list --porcelain | grep -q "worktree $WORKTREE_PATH$"; then
    git worktree remove "$WORKTREE_PATH"
else
    git worktree prune
fi
exit $EXIT_CODE
```
