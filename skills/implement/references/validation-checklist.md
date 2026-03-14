# Progressive Validation Checklist

Fail-fast validation for implemented code changes. Run checks in order — stop at the first failure and fix before continuing.

---

## Level 1: Syntax (Mandatory)

Run `py_compile` on every modified file.

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

**Addendum for adapted reference code (`from_reference`):** If an import fails, distinguish between:
- **Missing module from reference repo** (adaptation incomplete): The adapted code still references a module from the original repo. Fix by extracting the missing dependency or reimplementing the needed functionality.
- **External package not installed** (new dependency): The reference code depends on a pip package not in the target project. Add to `new_dependencies` in the manifest.

---

## Level 3: Model Instantiates (Recommended)

Verify the model can be created with the existing config after code changes.

```python
# PyTorch / Lightning
model = ModelClass(**config_params)

# TF/Keras
model = tf.keras.Model(...)  # or model = create_model(config)

# JAX/Flax
variables = model.init(jax.random.PRNGKey(0), dummy_input)

# HuggingFace
model = AutoModel.from_config(config)
```

**Pass criteria:** Model object is created without runtime errors.
**Common failures:** Shape mismatch in layer definitions, missing config keys for new parameters, device issues.

---

## Level 4: Forward Pass Shape (Recommended)

Run a dummy forward pass and check output shape matches expected.

```python
# PyTorch / Lightning
import torch
dummy_input = torch.randn(1, C, H, W)
with torch.no_grad():
    output = model(dummy_input)
assert output.shape == expected_shape

# TF/Keras
import numpy as np
dummy_input = np.random.randn(1, H, W, C).astype('float32')  # Note: channels-last
output = model(dummy_input)

# JAX/Flax
dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, H, W, C))
output = model.apply(variables, dummy_input)
```

**Pass criteria:** Output tensor has the expected shape.
**Common failures:** Channel count changed, spatial dimensions altered by new layers, wrong reshape/permute.

---

## Level 5: Unit Tests (Recommended)

Write and run a focused unit test for the implemented change.

**Test location:** `experiments/tests/test_<slug>.py`

Test the specific functionality introduced by the proposal — not the entire model.

```python
# Example: testing a new CutMix augmentation
import torch
from <module> import cutmix_data

def test_cutmix_output_shape():
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    x_mixed, y_a, y_b, lam = cutmix_data(x, y)
    assert x_mixed.shape == x.shape
    assert 0.0 <= lam <= 1.0

def test_cutmix_no_nan():
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    x_mixed, _, _, _ = cutmix_data(x, y)
    assert not torch.isnan(x_mixed).any()
```

Run:
```bash
python3 -m pytest experiments/tests/test_<slug>.py -v --timeout=30
```

**Pass criteria:** All tests pass within timeout.
**Failure handling:** Log as warning, do not block. Tests are informational — they catch issues that static validation misses.
**Common test patterns:**
- Shape preservation: `output.shape == expected_shape`
- No NaN/Inf: `assert not torch.isnan(output).any()`
- Value range: `assert output.min() >= 0` (for ReLU outputs, probabilities, etc.)
- Gradient flow: `loss.backward(); assert param.grad is not None`

---

## Level 6: Loss Computes Without NaN (Recommended)

Compute the loss with dummy data and check for NaN/Inf.

```python
# PyTorch / Lightning
pred = model(dummy_input)
loss = loss_fn(pred, dummy_target)
assert not torch.isnan(loss), "Loss is NaN"
assert not torch.isinf(loss), "Loss is Inf"

# TF/Keras
loss = loss_fn(dummy_target, pred)
assert not tf.math.is_nan(loss), "Loss is NaN"
assert not tf.math.is_inf(loss), "Loss is Inf"

# JAX
import jnp
assert not jnp.isnan(loss), "Loss is NaN"
assert not jnp.isinf(loss), "Loss is Inf"
```

**Pass criteria:** Loss is a finite, reasonable number.
**Common failures:** Division by zero in new loss, log of negative values, mismatched input ranges.

---

## Level 7: Gradients Flow (Optional)

Verify gradients propagate through new layers.

```python
# PyTorch / Lightning
loss.backward()
for name, param in model.named_parameters():
    if param.requires_grad:
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

# JAX
grads = jax.grad(loss_fn)(params, dummy_input, dummy_target)
jax.tree_util.tree_map(lambda g: assert not jnp.isnan(g).any(), grads)

# TF/Keras (GradientTape)
with tf.GradientTape() as tape:
    loss = loss_fn(dummy_target, model(dummy_input))
grads = tape.gradient(loss, model.trainable_variables)
for g, v in zip(grads, model.trainable_variables):
    assert g is not None, f"No gradient for {v.name}"
```

**Pass criteria:** All trainable parameters have non-NaN gradients.
**Common failures:** Detached tensors breaking the computation graph, frozen layers that should be trainable, in-place operations.

---

## Level 8: GPU Memory Within Bounds (Optional)

Check that the modified model fits in GPU memory at the expected batch size.

```python
# PyTorch / Lightning
import torch
torch.cuda.reset_peak_memory_stats()
model = model.cuda()
dummy_input = torch.randn(batch_size, C, H, W).cuda()
with torch.no_grad():
    output = model(dummy_input)
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"Peak GPU memory: {peak_mb:.0f} MiB")

# TF/Keras — use nvidia-smi or tf.config.experimental.get_memory_info()
# JAX — use jax.local_devices()[0].memory_stats()
```

**Pass criteria:** Peak memory is within available GPU memory with headroom for gradients (~2-3x forward pass).
**Common failures:** New layers significantly increase memory, large feature extractors (VGG) not accounted for.

---

## When to Run Which Levels

| Context | Levels to run |
|---------|--------------|
| Quick syntax check after edit | 1-2 |
| Standard validation before commit | 1-5 |
| Full validation before experiment | 1-7 |
| Memory-sensitive changes | 1-8 |

Levels 1-2 are mandatory and automated via `implement_utils.py`. Level 5 (unit tests) is written and run by the implement agent. Levels 3-4 and 6-8 require project-specific setup and are run as bash commands by the implement agent.
