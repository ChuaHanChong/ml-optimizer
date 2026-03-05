# Experiment Script Templates

## Basic Training Script
```bash
#!/bin/bash
# Experiment: {exp_id}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

mkdir -p experiments/logs/{exp_id}

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

# Training
{train_command} 2>&1 | tee experiments/logs/{exp_id}/train.log

# Evaluation
{eval_command} 2>&1 | tee experiments/logs/{exp_id}/eval.log

echo "Experiment {exp_id} completed"
```

## Training with Code Changes (Branch-Based)
```bash
#!/bin/bash
# Experiment: {exp_id}
# Code changes: {change_description}
set -e

export CUDA_VISIBLE_DEVICES={gpu_id}

# Apply code changes (already done by experiment skill before script generation)
mkdir -p experiments/logs/{exp_id}

{train_command} 2>&1 | tee experiments/logs/{exp_id}/train.log
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
