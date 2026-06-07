# Experiment Script Templates

## Basic Training Script (with PID tracking)
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}
echo $$ > <exp_root>/logs/{round_dir}/{exp_id}/pid

echo "Starting experiment {exp_id} on GPU {gpu_id}"
echo "Config: {config_summary}"
echo "Started at: $(date)"

{train_command} 2>&1 | tee <exp_root>/logs/{round_dir}/{exp_id}/train.log

echo "Experiment {exp_id} completed at: $(date)"
```

## Training with Config Override (PyTorch/YAML)
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}
echo $$ > <exp_root>/logs/{round_dir}/{exp_id}/pid

python train.py \
  --config {base_config} \
  --lr {lr} \
  --batch_size {batch_size} \
  --weight_decay {weight_decay} \
  --epochs {epochs} \
  --output_dir <exp_root>/logs/{round_dir}/{exp_id} \
  2>&1 | tee <exp_root>/logs/{round_dir}/{exp_id}/train.log
```

## Training with Eval at End
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}
echo $$ > <exp_root>/logs/{round_dir}/{exp_id}/pid

# Training
{train_command} 2>&1 | tee <exp_root>/logs/{round_dir}/{exp_id}/train.log

# Evaluation
{eval_command} 2>&1 | tee <exp_root>/logs/{round_dir}/{exp_id}/eval.log

echo "Experiment {exp_id} completed"
```

## Training with Code Changes (Git Worktree)
```bash
#!/bin/bash
# Experiment: {exp_id}
# Code changes: {change_description}
# Code branch: {code_branch}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}
mkdir -p <exp_root>/artifacts/{round_dir}/{exp_id}
echo $$ > <exp_root>/logs/{round_dir}/{exp_id}/pid

# Isolated worktree for the code branch, OUTSIDE <exp_root>/ (a worktree under
# <exp_root>/ can be wiped by a parallel-cleanup race). --detach lets multiple
# parallel experiments share {code_branch} (plain add fails "already checked out").
PROJECT_HASH=$(echo "<project_root>" | sha1sum | cut -c1-8)
WORKTREE_ROOT="/tmp/ml-opt-worktrees-${PROJECT_HASH}"; mkdir -p "$WORKTREE_ROOT"
WORKTREE_PATH="$WORKTREE_ROOT/{exp_id}"
git worktree add --detach "$WORKTREE_PATH" {code_branch}
cd "$WORKTREE_PATH"

# Training (absolute log path — the worktree is outside <exp_root>)
{train_command} 2>&1 | tee <exp_root>/logs/{round_dir}/{exp_id}/train.log

# Copy artifacts out of the worktree before cleanup
cp -r *.pt *.pth *.ckpt *.h5 *.pkl *.safetensors <exp_root>/artifacts/{round_dir}/{exp_id}/ 2>/dev/null || true

# Evaluation — MUST run inside the worktree before cleanup
{eval_command} 2>&1 | tee <exp_root>/logs/{round_dir}/{exp_id}/eval.log

# Cleanup worktree (validate registration first, never rm -rf)
cd - >/dev/null
if git worktree list --porcelain | grep -q "worktree $WORKTREE_PATH$"; then
    git worktree remove "$WORKTREE_PATH"
else
    git worktree prune
fi
```

## Background Training with PID Tracking
```bash
#!/bin/bash
# Experiment: {exp_id} (background)
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p <exp_root>/logs/{round_dir}/{exp_id}

{train_command} > <exp_root>/logs/{round_dir}/{exp_id}/train.log 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > <exp_root>/logs/{round_dir}/{exp_id}/pid

echo "Experiment {exp_id} running in background (PID: $TRAIN_PID)"
wait $TRAIN_PID
EXIT_CODE=$?

echo "Experiment {exp_id} finished with exit code $EXIT_CODE"
exit $EXIT_CODE
```
