# Experiment Script Templates

## Basic Training Script (with PID tracking)
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p experiments/logs/{exp_id}
echo $$ > experiments/logs/{exp_id}/pid

echo "Starting experiment {exp_id} on GPU {gpu_id}"
echo "Config: {config_summary}"
echo "Started at: $(date)"

{train_command} 2>&1 | tee experiments/logs/{exp_id}/train.log

echo "Experiment {exp_id} completed at: $(date)"
```

## Training with Config Override (PyTorch/YAML)
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p experiments/logs/{exp_id}
echo $$ > experiments/logs/{exp_id}/pid

python train.py \
  --config {base_config} \
  --lr {lr} \
  --batch_size {batch_size} \
  --weight_decay {weight_decay} \
  --epochs {epochs} \
  --output_dir experiments/logs/{exp_id} \
  2>&1 | tee experiments/logs/{exp_id}/train.log
```

## Training with Eval at End
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p experiments/logs/{exp_id}
echo $$ > experiments/logs/{exp_id}/pid

# Training
{train_command} 2>&1 | tee experiments/logs/{exp_id}/train.log

# Evaluation
{eval_command} 2>&1 | tee experiments/logs/{exp_id}/eval.log

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

mkdir -p experiments/logs/{exp_id}
mkdir -p experiments/artifacts/{exp_id}
echo $$ > experiments/logs/{exp_id}/pid

# Set up isolated worktree for code branch
git worktree add experiments/worktrees/{exp_id} {code_branch}
cd experiments/worktrees/{exp_id}

# Training
{train_command} 2>&1 | tee ../../logs/{exp_id}/train.log

# Copy artifacts out of worktree before cleanup
cp -r *.pt *.pth *.ckpt *.h5 *.pkl *.safetensors ../../artifacts/{exp_id}/ 2>/dev/null || true

# Evaluation — MUST run inside worktree before cleanup
{eval_command} 2>&1 | tee ../../logs/{exp_id}/eval.log

# Cleanup worktree
cd -
git worktree remove experiments/worktrees/{exp_id}
```

## Background Training with PID Tracking
```bash
#!/bin/bash
# Experiment: {exp_id} (background)
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p experiments/logs/{exp_id}

{train_command} > experiments/logs/{exp_id}/train.log 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > experiments/logs/{exp_id}/pid

echo "Experiment {exp_id} running in background (PID: $TRAIN_PID)"
wait $TRAIN_PID
EXIT_CODE=$?

echo "Experiment {exp_id} finished with exit code $EXIT_CODE"
exit $EXIT_CODE
```
