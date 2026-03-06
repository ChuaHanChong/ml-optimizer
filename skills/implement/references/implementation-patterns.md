# Implementation Patterns for ML Code Changes

Patterns for applying research proposals to ML codebases. Each category covers: where to look, what to read first, how to make the minimal reversible change, and common pitfalls.

## Framework Detection

Before applying any pattern, determine the project's ML framework by scanning imports:

| Framework | Detection Pattern | Key Differences |
|-----------|------------------|-----------------|
| **PyTorch** | `import torch`, `from torch` | `nn.Module`, manual training loop |
| **TensorFlow/Keras** | `import tensorflow`, `from keras` | `tf.keras.Model`, `model.fit()` |
| **JAX/Flax** | `import jax`, `from flax` | Functional style, explicit state |
| **PyTorch Lightning** | `import lightning`, `import pytorch_lightning` | `LightningModule`, callbacks |
| **HuggingFace Transformers** | `from transformers`, `Trainer` | `Trainer` API, `compute_metrics` |

Use the matching framework's syntax when applying patterns below. The examples use PyTorch; see **Framework Adaptations** subsections for alternatives.

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

### Framework Adaptations
- **TF/Keras:** Add loss in `model.compile(loss=...)` or compute in `train_step()`. Use `tf.stop_gradient()` instead of `requires_grad=False`.
- **JAX/Flax:** Losses are pure functions. Pass as argument to the training step function. Use `jax.lax.stop_gradient()`.
- **Lightning:** Override `training_step()` return value. Log via `self.log('loss_perceptual', ...)`.
- **HF Trainer:** Override `compute_loss()` in a custom `Trainer` subclass.

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

### Framework Adaptations
- **TF/Keras:** Subclass `tf.keras.Model` or use `tf.keras.layers.Layer`. Use `model.summary()` to verify shapes.
- **JAX/Flax:** Define new `nn.Module` in Flax. Initialize params explicitly with `module.init()`.
- **Lightning:** Same as PyTorch but changes go in the `LightningModule`. Use `self.example_input_array` for shape verification.
- **HF Transformers:** Modify the model config class and `forward()`. Use `AutoModel.from_config()` for testing.

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

### Framework Adaptations
- **TF/Keras:** Use `tf.image` ops or `tf.keras.layers` preprocessing layers. Apply in `tf.data.Dataset.map()`.
- **JAX:** Use `jax.random` for stochastic augmentations. Pass PRNG keys explicitly.
- **Lightning:** Use `LightningDataModule` to centralize augmentation logic.
- **HF Transformers:** Use `datasets.map()` with custom transform functions.

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

### Framework Adaptations
- **TF/Keras:** Configure optimizer in `model.compile()`. Use `tf.keras.optimizers.schedules` for schedulers.
- **JAX/Optax:** Use `optax.chain()` to compose optimizer + scheduler. State is explicit — pass and return it.
- **Lightning:** Override `configure_optimizers()` to return optimizer and scheduler dicts.
- **HF Trainer:** Set optimizer/scheduler via `TrainingArguments` or override `create_optimizer()`.

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

### Framework Adaptations
- **TF/Keras:** Use `tf.keras.layers.Dropout`. Keras handles train/eval mode automatically.
- **JAX/Flax:** Use `nn.Dropout` with `deterministic` flag. Pass `rngs={'dropout': key}` during training.
- **Lightning:** Same as PyTorch. Lightning handles `model.train()`/`model.eval()` automatically.

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

### Framework Adaptations
- **TF/Keras:** Use `tf.keras.mixed_precision.set_global_policy('mixed_float16')`. No manual scaler needed.
- **JAX:** Use `jax.default_matmul_precision('float16')` or explicit `jnp.float16` casts. Use `jmp` library for policies.
- **Lightning:** Set `Trainer(precision='16-mixed')`. No code changes needed in the model.
- **HF Trainer:** Set `TrainingArguments(fp16=True)` or `bf16=True`.

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

### Framework Adaptations
- **TF/Keras:** Use `tf.distribute.MirroredStrategy()`. Wrap model creation in `strategy.scope()`.
- **JAX:** Use `jax.pmap()` for data parallelism. Shard data across devices with `jax.device_put_sharded()`.
- **Lightning:** Set `Trainer(strategy='ddp', devices=N)`. No model code changes needed.
- **HF Trainer:** Set `TrainingArguments(...)` and launch with `torchrun` or `accelerate launch`.

---

## 8. From-Scratch Implementation (Paper-Based)

When a research proposal has `implementation_strategy: from_scratch`, implement directly from the paper's method description, pseudocode, and algorithm.

### Prerequisites
- Paper URL or method description available
- Algorithm description, pseudocode, or equations from the paper
- Target framework understood (use Framework Detection table above)

### Process

1. **Extract algorithm:** Read the paper's method section, pseudocode, and equations. Map mathematical notation to code constructs (e.g., summation → loop or `torch.sum`, element-wise product → `*`).

2. **Map to project structure:** Determine which pattern category (1-7) the change falls into. Identify where in the existing codebase to insert the new code.

3. **Implement incrementally:**
   - Core computation first (the algorithm itself as a function or module)
   - Wire into existing code (integrate at the identified insertion point)
   - Add config parameters (expose tunable values through the project's config system)

4. **Handle ambiguity:** When the paper is unclear:
   - Prefer the simpler interpretation
   - Add comments: `# [ml-opt] Paper ambiguous on <detail>, using <chosen_approach>`
   - Flag in the implementation manifest notes

### Pitfalls
- **Notation mismatch:** Paper uses math notation that doesn't directly translate. Map carefully: subscripts → indexing, superscripts → powers, Greek letters → descriptive variable names.
- **Missing details:** Papers often omit initialization, normalization, or edge cases. Use standard defaults from the framework.
- **Scale differences:** Paper may test on different data scales. Verify that constants (learning rates, thresholds) are appropriate for the target dataset.
- **Framework mismatch:** Paper's pseudocode may assume different tensor layout (channels-first vs channels-last). Adapt accordingly.

### When to Escalate
- No pseudocode AND ambiguous method section — flag as `implementation_error`
- Required operations unavailable in the target framework
- Method requires fundamentally different training paradigm not described in the proposal

---

## 9. From-Reference-Repo Implementation (Code Adaptation)

When a research proposal has `implementation_strategy: from_reference`, clone the paper's reference repository and adapt relevant code into the user's project.

### Prerequisites
- Reference repo URL verified and accessible
- Relevant files identified (from research proposal's `reference_files` field)
- Both the reference framework and target framework understood

### Process

1. **Clone and analyze:**
   ```bash
   python3 ~/.claude/plugins/ml-optimizer/scripts/implement_utils.py clone <repo_url> <dest_dir>
   python3 ~/.claude/plugins/ml-optimizer/scripts/implement_utils.py analyze <dest_dir>
   ```
   Review the analysis output: framework, relevant files, dependencies.

2. **Understand reference code:** Read the relevant files identified by the research agent. Identify:
   - Core implementation (the algorithm/module to extract)
   - Internal dependencies (other files in the repo that the core code imports)
   - External dependencies (pip packages not in the target project)

3. **Assess adaptation complexity:**
   - **Direct copy:** Same framework, minimal dependencies → copy and adjust imports
   - **Translation required:** Different framework → rewrite using equivalent APIs
   - **Extraction required:** Code deeply entangled with repo infrastructure → extract logic, reimplement wrapper

4. **Adapt:** Extract only the relevant functions/classes. For each:
   - Adapt import statements to the target project
   - Translate framework-specific calls if needed (e.g., `tf.nn.relu` → `F.relu`)
   - Match tensor conventions (shape ordering, dtype, device handling)
   - Preserve numerical behavior (same initialization, same constants)

5. **Track provenance:** Add comments to all adapted code:
   ```python
   # [ml-opt] Adapted from <repo_url>, file: <original_path>
   # [ml-opt] License: <license_type>
   ```

6. **Cleanup:** Remove the cloned repo after extraction:
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, '$HOME/.claude/plugins/ml-optimizer/scripts')
   from implement_utils import cleanup_reference_repo
   cleanup_reference_repo('<dest_dir>')
   "
   ```

### Pitfalls
- **License issues:** Always check the LICENSE file before adapting code. Flag repos with no license or restrictive licenses (GPL, proprietary) in the manifest as `license_warning`.
- **Dependency bloat:** Reference code may import heavy packages not needed for the core algorithm. Extract only what's necessary.
- **Version mismatch:** Reference code may use older API versions (e.g., deprecated PyTorch ops). Update to current equivalents.
- **Hidden state/registries:** Some frameworks use global registries or module-level state. Ensure adapted code doesn't depend on repo-specific initialization.

### When to Escalate
- No license file found — flag `license_warning` and inform user
- Extraction would require rewriting >50% of the reference code
- Framework translation is infeasible (e.g., JAX functional style → TF eager with heavy Keras integration)
- Core implementation has unresolvable internal dependencies (imports 10+ repo-specific modules)
