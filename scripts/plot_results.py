#!/usr/bin/env python3
"""ASCII and matplotlib visualization of experiment results."""

import math
import sys
from pathlib import Path

# Allow importing sibling modules when run directly
sys.path.insert(0, str(Path(__file__).parent))

from result_analyzer import load_results, rank_by_metric

# Optional matplotlib for progress chart
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for file output
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def ascii_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    width: int = 50,
) -> str:
    """Render a horizontal bar chart using ASCII characters.

    Each bar is formatted as:  label | ████████ value
    Bars are proportional to the maximum value.

    Returns the chart as a multi-line string.
    """
    if not labels or not values:
        return ""

    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return ""

    max_val = max(abs(v) for v in finite)
    if max_val == 0:
        max_val = 1

    # Determine label column width
    label_width = max(len(l) for l in labels)

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * (label_width + 3 + width + 10))

    for label, value in zip(labels, values):
        if math.isfinite(value):
            bar_len = int(abs(value) / max_val * width)
        else:
            bar_len = 0
        bar = "\u2588" * bar_len
        lines.append(f"{label:>{label_width}} | {bar} {value:.4g}")

    return "\n".join(lines)


def ascii_line_chart(
    values: list[float],
    title: str = "",
    width: int = 60,
    height: int = 15,
) -> str:
    """Render a simple ASCII line chart.

    Y-axis on the left with scaled values, X-axis on the bottom with
    index numbers.  Points are marked with '*'.

    Returns the chart as a multi-line string.
    """
    if not values:
        return ""

    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return ""

    min_val = min(finite)
    max_val = max(finite)
    val_range = max_val - min_val
    if val_range == 0:
        val_range = 1

    # Map each value to a row (0 = bottom, height-1 = top)
    n = len(values)
    # Determine how many data points to show (resample if more than width)
    if n > width:
        step = n / width
        sampled = [values[int(i * step)] for i in range(width)]
        sampled[-1] = values[-1]
    else:
        sampled = list(values)

    num_cols = len(sampled)

    # Y-axis label width (format to 4 significant figures)
    y_labels = []
    for r in range(height):
        frac = r / (height - 1) if height > 1 else 0
        y_val = min_val + frac * val_range
        y_labels.append(f"{y_val:.4g}")
    y_label_width = max(len(l) for l in y_labels)

    # Build the grid
    grid = [[" "] * num_cols for _ in range(height)]

    for col_idx, v in enumerate(sampled):
        if not math.isfinite(v):
            continue
        row = int((v - min_val) / val_range * (height - 1))
        row = min(row, height - 1)
        grid[row][col_idx] = "*"

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("-" * (y_label_width + 3 + num_cols))

    # Print rows top-to-bottom (highest value first)
    for r in range(height - 1, -1, -1):
        row_label = y_labels[r].rjust(y_label_width)
        row_chars = "".join(grid[r])
        lines.append(f"{row_label} | {row_chars}")

    # X-axis separator
    lines.append(" " * y_label_width + " +" + "-" * num_cols)

    # X-axis labels (show first, middle, last index)
    if n > 1:
        x_axis = " " * (y_label_width + 2)
        x_axis += "0"
        mid = (num_cols - 1) // 2
        last = num_cols - 1
        # Build a sparse x-axis label line
        x_line = [" "] * num_cols
        start_label = "0"
        for i, ch in enumerate(start_label):
            if i < num_cols:
                x_line[i] = ch
        mid_label = str((n - 1) // 2)
        mid_start = max(0, mid - len(mid_label) // 2)
        for i, ch in enumerate(mid_label):
            pos = mid_start + i
            if 0 <= pos < num_cols:
                x_line[pos] = ch
        end_label = str(n - 1)
        end_start = max(0, last - len(end_label) + 1)
        for i, ch in enumerate(end_label):
            pos = end_start + i
            if 0 <= pos < num_cols:
                x_line[pos] = ch
        lines.append(" " * (y_label_width + 2) + "".join(x_line))

    return "\n".join(lines)


def plot_metric_comparison(
    results_dir: str,
    metric: str,
    lower_is_better: bool = True,
) -> str:
    """Generate a bar chart comparing a metric across experiments.

    Loads results via result_analyzer.load_results, ranks them by
    *metric*, and renders an ascii_bar_chart.  The baseline experiment
    (if present) is marked with a '[B]' suffix.

    Returns the chart string.
    """
    results = load_results(results_dir)
    if not results:
        return "No results found."

    ranked = rank_by_metric(results, metric, lower_is_better)
    if not ranked:
        return f"Metric '{metric}' not found in any experiment."

    labels: list[str] = []
    values: list[float] = []
    for entry in ranked:
        exp_id = entry["exp_id"]
        if exp_id.lower() == "baseline":
            exp_id = exp_id + " [B]"
        labels.append(exp_id)
        values.append(entry["value"])

    direction = "lower is better" if lower_is_better else "higher is better"
    title = f"Metric comparison: {metric} ({direction})"
    return ascii_bar_chart(labels, values, title=title)


def plot_improvement_timeline(
    results_dir: str,
    metric: str,
    lower_is_better: bool = True,
) -> str:
    """Generate a line chart of best-so-far metric over experiments.

    Experiments are sorted by exp_id (chronological proxy).
    At each step the best metric value seen so far is recorded.

    Returns the chart string.
    """
    results = load_results(results_dir)
    if not results:
        return "No results found."

    ranked = rank_by_metric(results, metric, lower_is_better)
    if not ranked:
        return f"Metric '{metric}' not found in any experiment."

    # Sort by exp_id for chronological order
    ranked.sort(key=lambda x: x["exp_id"])

    best_so_far: list[float] = []
    current_best = None
    for entry in ranked:
        val = entry["value"]
        if current_best is None:
            current_best = val
        else:
            if lower_is_better:
                current_best = min(current_best, val)
            else:
                current_best = max(current_best, val)
        best_so_far.append(current_best)

    title = f"Best-so-far: {metric}"
    return ascii_line_chart(best_so_far, title=title)


def plot_hp_sensitivity(
    results_dir: str,
    metric: str,
    hp_name: str,
) -> str:
    """Generate an ASCII scatter of metric vs a numeric hyperparameter.

    Extracts (hp_value, metric_value) pairs from results, sorts by
    hp_value, and renders an ascii_line_chart as a pseudo-scatter.

    Returns the chart string.
    """
    results = load_results(results_dir)
    if not results:
        return "No results found."

    pairs: list[tuple[float, float]] = []
    for _exp_id, data in results.items():
        config = data.get("config", {})
        metrics = data.get("metrics", data)
        if hp_name in config and metric in metrics:
            try:
                hp_val = float(config[hp_name])
                met_val = float(metrics[metric])
                pairs.append((hp_val, met_val))
            except (ValueError, TypeError):
                continue

    if not pairs:
        return f"No numeric data for hp='{hp_name}' and metric='{metric}'."

    pairs.sort(key=lambda p: p[0])
    metric_values = [p[1] for p in pairs]

    title = f"Sensitivity: {metric} vs {hp_name}"
    return ascii_line_chart(metric_values, title=title)


def plot_progress_chart(
    results_dir: str,
    metric: str,
    lower_is_better: bool = True,
    output_path: str | None = None,
) -> str | None:
    """Generate a matplotlib progress chart showing optimization progress.

    Plots each experiment as a dot — green if it set a new running best,
    gray otherwise.  A blue step line tracks the running best frontier.
    Kept experiments are annotated with their exp_id.

    Returns the output file path, or ``None`` if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    results = load_results(results_dir)
    if not results:
        return None

    ranked = rank_by_metric(results, metric, lower_is_better)
    if not ranked:
        return None

    # Sort chronologically by exp_id
    ranked.sort(key=lambda x: x["exp_id"])

    indices: list[int] = []
    values: list[float] = []
    exp_ids: list[str] = []
    is_new_best: list[bool] = []
    best_so_far: list[float] = []
    current_best = None

    for i, entry in enumerate(ranked):
        val = entry["value"]
        exp_id = entry["exp_id"]
        indices.append(i)
        values.append(val)
        exp_ids.append(exp_id)

        if current_best is None:
            current_best = val
            is_new_best.append(True)
        else:
            improved = val < current_best if lower_is_better else val > current_best
            if improved:
                current_best = val
                is_new_best.append(True)
            else:
                is_new_best.append(False)
        best_so_far.append(current_best)

    # Separate kept and discarded
    kept_idx = [i for i, b in zip(indices, is_new_best) if b]
    kept_val = [v for v, b in zip(values, is_new_best) if b]
    disc_idx = [i for i, b in zip(indices, is_new_best) if not b]
    disc_val = [v for v, b in zip(values, is_new_best) if not b]

    fig, ax = plt.subplots(figsize=(max(8, len(ranked) * 0.4), 5))

    # Discarded experiments (gray)
    if disc_idx:
        ax.scatter(disc_idx, disc_val, c="gray", s=50, alpha=0.5,
                   label="Discarded", zorder=2)

    # Kept experiments (green)
    if kept_idx:
        ax.scatter(kept_idx, kept_val, c="green", s=80, edgecolors="darkgreen",
                   label="New best", zorder=3)

    # Running best frontier (blue step line)
    ax.step(indices, best_so_far, where="post", color="royalblue",
            linewidth=1.5, alpha=0.7, label="Running best", zorder=1)

    # Annotate kept experiments
    for i, b in zip(indices, is_new_best):
        if b:
            ax.annotate(
                exp_ids[i], (i, values[i]),
                textcoords="offset points", xytext=(5, 8),
                fontsize=7, rotation=30, ha="left",
            )

    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"Optimization Progress: {metric} ({direction})")
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Determine output path
    if output_path is None:
        results_path = Path(results_dir)
        reports_dir = results_path.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(reports_dir / "progress_chart.png")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: plot_results.py <results_dir> <metric> "
            "[comparison|timeline|sensitivity <hp_name>|progress]"
        )
        sys.exit(1)

    results_dir = sys.argv[1]
    metric = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "comparison"
    lower_is_better = "--higher-is-better" not in sys.argv

    if mode == "comparison":
        print(plot_metric_comparison(results_dir, metric, lower_is_better))
    elif mode == "timeline":
        print(plot_improvement_timeline(results_dir, metric, lower_is_better))
    elif mode == "sensitivity":
        if len(sys.argv) < 5:
            print("sensitivity mode requires <hp_name> argument")
            sys.exit(1)
        hp_name = sys.argv[4]
        print(plot_hp_sensitivity(results_dir, metric, hp_name))
    elif mode == "progress":
        path = plot_progress_chart(results_dir, metric, lower_is_better)
        if path:
            print(f"Progress chart saved to: {path}")
        else:
            print("matplotlib not available — cannot generate progress chart")
            sys.exit(1)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
