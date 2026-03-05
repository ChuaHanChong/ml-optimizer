# Implementation Patterns for ML Code Changes

Patterns for applying research proposals to ML codebases. Each category covers: where to look, what to read first, how to make the minimal reversible change, and common pitfalls.

> **Note:** These patterns use PyTorch syntax. Adapt for JAX, TensorFlow, etc.

---

## 1. Loss Function Changes

### Where to look
- Training step method (`training_step`, `forward`, or the main training loop)
- Existing loss computation (search for `F.l1_loss`, `F.mse_loss`, `nn.CrossEntropyLoss`, etc.)
- Config files for loss weights

### What to read first
- The current loss function and how it's computed
- How the loss value flows into `backward()`
- Whether multiple loss terms are already combined

### Minimal change pattern
```python
# 1. Add the new loss module to __init__
self.perceptual_loss = VGGPerceptualLoss().eval()
for p in self.perceptual_loss.parameters():
    p.requires_grad = False

# 2. Compute in forward/training_step
loss_l1 = F.l1_loss(pred, target)
loss_perceptual = self.perceptual_loss(pred, target)
loss = loss_l1 + self.config.perceptual_weight * loss_perceptual
```

### Pitfalls
- **Device mismatch:** New loss modules must be on the same device as model tensors. Use `.to(device)` or register as a submodule.
- **Gradient leaking:** Freeze feature extractors with `eval()` and `requires_grad = False`.
- **Scale mismatch:** Different losses may have very different magnitudes. Start with a small weight (0.01-0.1) and tune.
- **Input range:** VGG expects [0,1] or ImageNet-normalized inputs. Check your model's output range.

---

## 2. Architecture Changes

### Where to look
- Model class definition (inherits `nn.Module`, `LightningModule`, etc.)
- Layer definitions in `__init__`
- Forward pass method

### What to read first
- The full `__init__` and `forward` to understand the data flow
- Input/output tensor shapes at each stage
- Skip connections or residual connections

### Minimal change pattern
```python
# 1. Add new module in __init__ alongside existing
self.new_block = NewBlock(channels=self.mid_channels)

# 2. Insert in forward pass at the right location
x = self.existing_block(x)
x = self.new_block(x)  # Insert here
x = self.next_block(x)
```

### Pitfalls
- **Shape mismatch:** Always verify that the new block preserves spatial dimensions and channel counts. Print shapes before and after.
- **Broken skip connections:** If modifying a U-Net, ensure encoder-decoder skip connections still align.
- **Weight initialization:** New layers use random weights. Consider specific initialization if needed.
- **Checkpoint compatibility:** Adding layers breaks loading from existing checkpoints. Handle with `strict=False` or key mapping.

---

## 3. Data Augmentation Changes

### Where to look
- Dataset class (`__getitem__` method)
- Transform/augmentation pipeline (often in dataset init or a separate transforms file)
- Dataloader construction

### What to read first
- Current transforms applied to training data
- Whether augmentations differ between training and validation
- Image format and range (PIL, numpy, tensor; [0,255] vs [0,1])

### Minimal change pattern
```python
# In dataset __getitem__ or transform pipeline
if self.is_training:
    if random.random() < self.cutmix_prob:
        img, target = cutmix(img, target, other_img, other_target)
```

### Pitfalls
- **Train-only guard:** Augmentations must NOT apply during validation/testing. Always check the split/mode flag.
- **Paired data:** For image restoration, augmentations must be applied consistently to both input and target (e.g., same crop, same flip).
- **Data type:** Some augmentations expect PIL images, others expect tensors. Apply in the right order.
- **Reproducibility:** Set random seeds for augmentations if reproducibility matters.

---

## 4. Training Strategy Changes

### Where to look
- Optimizer construction (search for `Adam`, `SGD`, `AdamW`)
- Learning rate scheduler (search for `lr_scheduler`, `CosineAnnealing`, `StepLR`)
- Training loop (epoch/step iteration)

### What to read first
- Current optimizer and its hyperparameters
- Current scheduler and when it steps
- Gradient accumulation or clipping if present

### Minimal change pattern
```python
# Swap optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay,
)

# Swap scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=config.restart_period, T_mult=2,
)
```

### Pitfalls
- **State mismatch on resume:** Changing optimizer/scheduler breaks checkpoint resume. Clear optimizer state or start fresh.
- **Scheduler step timing:** Some schedulers step per-epoch, others per-batch. Match the existing pattern.
- **Warmup conflicts:** If adding warmup, ensure it doesn't conflict with an existing scheduler.
- **Weight decay + bias:** AdamW applies weight decay to all params by default. Consider excluding bias and normalization parameters.

---

## 5. Regularization Changes

### Where to look
- Model `__init__` for existing dropout/normalization layers
- Training config for weight decay
- Forward pass for any regularization applied during training

### What to read first
- Current regularization (dropout rate, weight decay, batch norm, etc.)
- Whether the model uses `model.train()`/`model.eval()` correctly
- Overfitting signals in training logs (train loss << val loss)

### Minimal change pattern
```python
# Add dropout to specific layers
self.attn_dropout = nn.Dropout(p=0.1)
# In forward:
x = self.attn_dropout(self.attention(x))

# Or add stochastic depth
from torchvision.ops import StochasticDepth
self.stoch_depth = StochasticDepth(p=0.1, mode="row")
```

### Pitfalls
- **Train/eval mode:** Dropout and batch norm behave differently in train vs eval. Ensure `model.train()` and `model.eval()` are called correctly.
- **Too aggressive:** High dropout (>0.3) in generative models usually hurts. Start conservative.
- **Label smoothing + other losses:** Label smoothing changes the loss landscape. May need to adjust learning rate.
- **Stacking regularizers:** Combining multiple regularization techniques can over-regularize. Add one at a time.

---

## 6. Mixed Precision Training

### Where to look
- Training loop (the main training step)
- Model forward pass and loss computation
- Gradient scaling logic

### What to read first
- Whether the project already uses AMP (`torch.cuda.amp`)
- What dtype the model and inputs use
- Whether custom loss functions handle mixed precision correctly

### Minimal change pattern
```python
# Add scaler and autocast to training loop
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(input)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Pitfalls
- **Custom losses with reduce ops:** Some custom losses break in float16. Use `torch.cuda.amp.autocast(enabled=False)` for problematic operations.
- **Gradient overflow:** GradScaler handles this, but very high LR + AMP can cause more frequent skipped steps.
- **BN statistics:** Batch norm accumulates in float32 automatically, but verify if using custom normalization.

---

## 7. Distributed Training (DDP/FSDP)

### Where to look
- Model initialization (wrapping with DDP)
- Dataloader (DistributedSampler)
- Training loop (gradient synchronization)

### What to read first
- Current single-GPU training loop structure
- How model is created and moved to device
- Whether there are any `model.module` accesses

### Minimal change pattern (DDP)
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Wrap model
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

### Pitfalls
- **Saving/loading:** Save `model.module.state_dict()` not `model.state_dict()` to avoid DDP wrapper keys.
- **Metrics logging:** Only log from rank 0 to avoid duplicate entries.
- **Random seeds:** Set different seeds per rank for data augmentation, same seed for model init.
- **Batch size:** Per-GPU batch size × num_GPUs = effective batch size. Adjust LR accordingly.
