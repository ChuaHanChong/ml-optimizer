# Progressive Validation Checklist

Fail-fast validation for implemented code changes. Run checks in order — stop at the first failure and fix before continuing.

---

## Level 1: Syntax (Mandatory)

Run `py_compile` on every modified file.

```bash
python3 ~/.claude/plugins/ml-optimizer/scripts/implement_utils.py validate_syntax <file1> <file2> ...
```

Or in Python:
```python
from implement_utils import validate_syntax
results = validate_syntax(modified_files)
```

**Pass criteria:** All files compile without errors.
**Common failures:** Missing colons, unmatched parentheses, bad indentation after edit.

---

## Level 2: Imports Resolve (Mandatory)

Attempt to import each modified module.

```python
from implement_utils import validate_imports
result = validate_imports(module_path, project_root)
```

**Pass criteria:** Module loads without ImportError or ModuleNotFoundError.
**Common failures:** New dependency not installed, wrong import path, circular import introduced.

**If new dependencies are needed:** Flag them — do NOT install automatically. Report to user for confirmation.

---

## Level 3: Model Instantiates (Recommended)

Verify the model can be created with the existing config after code changes.

```python
# Load the project's config
# Instantiate the model class
# Check it creates without errors
model = ModelClass(**config_params)
```

**Pass criteria:** Model object is created without runtime errors.
**Common failures:** Shape mismatch in layer definitions, missing config keys for new parameters, device issues.

---

## Level 4: Forward Pass Shape (Recommended)

Run a dummy forward pass and check output shape matches expected.

```python
import torch
dummy_input = torch.randn(1, C, H, W)  # Match expected input shape
with torch.no_grad():
    output = model(dummy_input)
assert output.shape == expected_shape
```

**Pass criteria:** Output tensor has the expected shape.
**Common failures:** Channel count changed, spatial dimensions altered by new layers, wrong reshape/permute.

---

## Level 5: Loss Computes Without NaN (Recommended)

Compute the loss with dummy data and check for NaN/Inf.

```python
pred = model(dummy_input)
loss = loss_fn(pred, dummy_target)
assert not torch.isnan(loss), "Loss is NaN"
assert not torch.isinf(loss), "Loss is Inf"
assert loss.item() > 0, "Loss is non-positive (unexpected)"
```

**Pass criteria:** Loss is a finite, reasonable number.
**Common failures:** Division by zero in new loss, log of negative values, mismatched input ranges.

---

## Level 6: Gradients Flow (Optional)

Verify gradients propagate through new layers.

```python
loss.backward()
for name, param in model.named_parameters():
    if param.requires_grad:
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
```

**Pass criteria:** All trainable parameters have non-NaN gradients.
**Common failures:** Detached tensors breaking the computation graph, frozen layers that should be trainable, in-place operations.

---

## Level 7: GPU Memory Within Bounds (Optional)

Check that the modified model fits in GPU memory at the expected batch size.

```python
import torch
torch.cuda.reset_peak_memory_stats()
model = model.cuda()
dummy_input = torch.randn(batch_size, C, H, W).cuda()
with torch.no_grad():
    output = model(dummy_input)
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"Peak GPU memory: {peak_mb:.0f} MiB")
```

**Pass criteria:** Peak memory is within available GPU memory with headroom for gradients (~2-3x forward pass).
**Common failures:** New layers significantly increase memory, large feature extractors (VGG) not accounted for.

---

## When to Run Which Levels

| Context | Levels to run |
|---------|--------------|
| Quick syntax check after edit | 1-2 |
| Standard validation before commit | 1-4 |
| Full validation before experiment | 1-6 |
| Memory-sensitive changes | 1-7 |

Levels 1-2 are mandatory and automated via `implement_utils.py`. Levels 3-7 require project-specific setup and are run as bash commands by the implement agent.
