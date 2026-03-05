# Final Optimization Report Template

```markdown
# ML Optimization Report

**Date:** <date>
**Model:** <model description>
**Task:** <task description>
**Primary Metric:** <metric name> (<lower/higher> is better)
**GPUs Used:** <count> × <GPU name> (<memory_total> MiB each)
**Total Experiments:** <N> (<completed> completed, <diverged> diverged, <failed> failed)
**Total GPU Time:** <sum of duration_seconds across experiments, formatted>

---

## Executive Summary

<2-3 sentences summarizing the optimization effort and key result>

**Best result:** <exp_id> achieved <metric>=<value> (<X% improvement> over baseline)

---

## Baseline

| Metric | Value |
|--------|-------|
| <metric1> | <value> |
| <metric2> | <value> |

**Configuration:**
```json
<baseline config>
```

**Profiling:**
| Measure | Value |
|---------|-------|
| GPU Memory Used | <gpu_memory_used_mib> MiB / <gpu_memory_total_mib> MiB |
| Throughput | <throughput_samples_per_sec> samples/sec |
| Est. Max Batch Size | <estimated_max_batch_size> |
| Training Duration | <duration_seconds>s |

---

## Search Space

| Parameter | Range | Priority |
|-----------|-------|----------|
| <param>   | <range> | <High/Medium/Low> |

**Tuning iterations:** <N> batches of <M> experiments each
**Strategy:** <exploration → exploitation progression summary>

---

## Research & Code Changes

*(Include this section only if implementation manifest exists)*

| Proposal | Branch | Complexity | Status | Files Modified |
|----------|--------|------------|--------|----------------|
| <name>   | <branch> | <Low/Med/High> | validated/failed | <file list> |

**Conflicts:** <none or list of overlapping files>
**New dependencies:** <none or list>

---

## Optimization Journey

### Phase 1: <description>
<what was tried and why>

### Phase 2: <description>
<what was tried and why>

---

## All Experiments

| # | Exp ID | Status | Code Branch | Key Config Changes | <Metric1> | <Metric2> | vs Baseline | Duration | GPU |
|---|--------|--------|-------------|--------------------|-----------|-----------|-------------|----------|-----|
| 1 | baseline | completed | baseline code | - | <val> | <val> | - | <dur> | <gpu> |
| 2 | exp-001 | completed | baseline code | lr=X | <val> | <val> | +X% | <dur> | <gpu> |
| 3 | exp-002 | diverged | <branch> | lr=Y | - | - | diverged | <dur> | <gpu> |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## Best Configuration

**Experiment:** <exp_id>

| Parameter | Baseline | Best | Change |
|-----------|----------|------|--------|
| lr | <val> | <val> | <delta> |
| batch_size | <val> | <val> | <delta> |
| ... | ... | ... | ... |

**Metrics:**
| Metric | Baseline | Best | Improvement |
|--------|----------|------|-------------|
| <metric1> | <val> | <val> | +X% |
| <metric2> | <val> | <val> | +Y% |

---

## Hyperparameter Sensitivity

Analysis based on <N> completed experiments:

| Parameter | Best Performers Avg | Worst Performers Avg | Direction |
|-----------|--------------------|--------------------|-----------|
| <param>   | <top_avg>          | <bottom_avg>       | lower is better |

---

## Visualizations

### Metric Comparison
```
<output of plot_results.py comparison chart>
```

### Improvement Timeline
```
<output of plot_results.py timeline chart>
```

### HP Sensitivity
```
<output of plot_results.py sensitivity chart for highest-impact HP>
```

---

## Key Findings

1. **<Finding 1 title>:** <detail>
2. **<Finding 2 title>:** <detail>
3. **<Finding 3 title>:** <detail>

## What Worked
- <technique/HP that helped>

## What Didn't Work
- <technique/HP that didn't help or made things worse>

---

## Recommendations

### To reproduce the best result:
```bash
<exact command to run>
```

### For further improvement:
- <suggestion 1>
- <suggestion 2>

---

## Appendix

### Reproduction Commands
**Training:**
```bash
<train_command from baseline.json>
```
**Evaluation:**
```bash
<eval_command from baseline.json>
```

### Experiment Artifacts
| Exp ID | Log File | Script | Result JSON |
|--------|----------|--------|-------------|
| <id>   | <log_file path> | <script_file path> | <result path> |

### Research Findings
<Summary or link to `experiments/reports/research-findings.md`>

### Batch Analyses
<List of `experiments/reports/batch-*-analysis.md` with 1-line summary each>

### Dev Notes
See `experiments/dev_notes.md` for full session log.
```
