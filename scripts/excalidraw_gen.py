#!/usr/bin/env python3
"""Generate Excalidraw-compatible JSON diagrams for ML optimization results.

Produces .excalidraw files that can be opened in excalidraw.com or the
desktop app.  Stdlib-only — no external dependencies.

Usage:
    python3 excalidraw_gen.py <exp_root> pipeline <primary_metric>
    python3 excalidraw_gen.py <exp_root> comparison <exp_id_1> <exp_id_2>
    python3 excalidraw_gen.py <exp_root> hp-landscape <hp_name> <metric>
    python3 excalidraw_gen.py <exp_root> architecture <proposal_name>
"""

import json
import math
import sys
import uuid
from pathlib import Path

# Allow importing sibling modules when run directly
sys.path.insert(0, str(Path(__file__).parent))

from result_analyzer import load_results, rank_by_metric

# ---------------------------------------------------------------------------
# Excalidraw element helpers
# ---------------------------------------------------------------------------

# Color palette
COLORS = {
    "green": "#2f9e44",
    "red": "#e03131",
    "blue": "#1971c2",
    "gray": "#868e96",
    "yellow": "#f08c00",
    "bg_green": "#b2f2bb",
    "bg_red": "#ffc9c9",
    "bg_blue": "#a5d8ff",
    "bg_gray": "#dee2e6",
    "bg_yellow": "#ffec99",
    "white": "#ffffff",
}


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _rect(x, y, w, h, *, label="", stroke="#000000", bg="#ffffff",
          font_size=16, **kw):
    """Create a rectangle element, optionally with a centered label."""
    elements = []
    rid = _uid()
    elements.append({
        "id": rid,
        "type": "rectangle",
        "x": x, "y": y,
        "width": w, "height": h,
        "strokeColor": stroke,
        "backgroundColor": bg,
        "fillStyle": "solid",
        "roundness": {"type": 3},
        "seed": hash(rid) & 0x7FFFFFFF,
        "version": 1,
    })
    if label:
        tid = _uid()
        elements.append({
            "id": tid,
            "type": "text",
            "x": x + 10, "y": y + (h - font_size) / 2,
            "width": w - 20, "height": font_size + 4,
            "text": label,
            "fontSize": font_size,
            "fontFamily": 3,
            "textAlign": "center",
            "verticalAlign": "middle",
            "strokeColor": stroke,
            "seed": hash(tid) & 0x7FFFFFFF,
            "version": 1,
            "containerId": rid,
        })
    return elements


def _text(x, y, text, *, font_size=16, color="#000000"):
    """Create a standalone text element."""
    tid = _uid()
    return {
        "id": tid,
        "type": "text",
        "x": x, "y": y,
        "width": len(text) * font_size * 0.6,
        "height": font_size + 4,
        "text": text,
        "fontSize": font_size,
        "fontFamily": 3,
        "strokeColor": color,
        "seed": hash(tid) & 0x7FFFFFFF,
        "version": 1,
    }


def _arrow(x1, y1, x2, y2, *, color="#000000"):
    """Create an arrow from (x1,y1) to (x2,y2)."""
    aid = _uid()
    return {
        "id": aid,
        "type": "arrow",
        "x": x1, "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "strokeColor": color,
        "endArrowhead": "arrow",
        "seed": hash(aid) & 0x7FFFFFFF,
        "version": 1,
    }


def _write_excalidraw(path: Path, elements: list) -> str:
    """Write an Excalidraw JSON file."""
    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "ml-optimizer-plugin",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return str(path)


# ---------------------------------------------------------------------------
# Diagram: Pipeline overview
# ---------------------------------------------------------------------------

def generate_pipeline_diagram(exp_root: str, metric: str) -> str:
    """Generate an optimization journey flowchart."""
    results_dir = Path(exp_root) / "results"
    results = load_results(str(results_dir))

    if not results:
        elements = [_text(50, 50, "No experiment results found", font_size=20)]
        out = Path(exp_root) / "artifacts" / "pipeline-overview.excalidraw"
        return _write_excalidraw(out, elements)

    # Get baseline
    baseline_val = None
    baseline = results.get("baseline", {})
    if baseline:
        baseline_val = baseline.get("metrics", {}).get(metric)

    # Rank experiments
    ranked = rank_by_metric(results, metric, lower_is_better=True)

    elements = []
    # Title
    elements.append(_text(50, 20, f"Optimization Pipeline — {metric}", font_size=24, color=COLORS["blue"]))

    # Baseline box
    bl_label = f"Baseline\n{metric}={baseline_val}" if baseline_val is not None else "Baseline"
    elements.extend(_rect(50, 70, 200, 60, label=bl_label, bg=COLORS["bg_blue"], stroke=COLORS["blue"]))

    # Arrow down
    elements.append(_arrow(150, 130, 150, 160, color=COLORS["gray"]))

    # Experiment boxes
    y = 170
    exp_entries = [r for r in ranked if r.get("exp_id", "").startswith("exp-")]
    for i, entry in enumerate(exp_entries[:12]):  # Cap at 12 for readability
        exp_id = entry.get("exp_id", "?")
        val = entry.get("metric_value")
        status = entry.get("status", "?")

        if status == "completed" and baseline_val is not None and val is not None:
            try:
                if val < baseline_val:
                    bg, stroke = COLORS["bg_green"], COLORS["green"]
                else:
                    bg, stroke = COLORS["bg_gray"], COLORS["gray"]
            except TypeError:
                bg, stroke = COLORS["bg_gray"], COLORS["gray"]
        elif status in ("failed", "diverged"):
            bg, stroke = COLORS["bg_red"], COLORS["red"]
        else:
            bg, stroke = COLORS["bg_gray"], COLORS["gray"]

        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        label = f"{exp_id}\n{metric}={val_str}"
        elements.extend(_rect(50, y, 200, 50, label=label, bg=bg, stroke=stroke, font_size=12))

        if i < len(exp_entries) - 1 and i < 11:
            elements.append(_arrow(150, y + 50, 150, y + 60, color=COLORS["gray"]))
        y += 60

    out = Path(exp_root) / "artifacts" / "pipeline-overview.excalidraw"
    return _write_excalidraw(out, elements)


# ---------------------------------------------------------------------------
# Diagram: Experiment comparison
# ---------------------------------------------------------------------------

def generate_comparison_diagram(exp_root: str, exp_id_1: str, exp_id_2: str) -> str:
    """Generate a side-by-side comparison of two experiments."""
    results_dir = Path(exp_root) / "results"
    results = load_results(str(results_dir))

    data1 = results.get(exp_id_1, {})
    data2 = results.get(exp_id_2, {})

    elements = []
    elements.append(_text(50, 20, f"Comparison: {exp_id_1} vs {exp_id_2}", font_size=24, color=COLORS["blue"]))

    # Left box
    config1 = data1.get("config", {})
    metrics1 = data1.get("metrics", {})
    left_lines = [exp_id_1, f"Status: {data1.get('status', '?')}"]
    for k, v in sorted(metrics1.items()):
        left_lines.append(f"{k}: {v}")
    left_lines.append("--- Config ---")
    for k, v in sorted(config1.items()):
        left_lines.append(f"{k}: {v}")
    elements.extend(_rect(50, 70, 250, max(200, len(left_lines) * 20 + 20),
                          label="\n".join(left_lines), bg=COLORS["bg_blue"],
                          stroke=COLORS["blue"], font_size=12))

    # Right box
    config2 = data2.get("config", {})
    metrics2 = data2.get("metrics", {})
    right_lines = [exp_id_2, f"Status: {data2.get('status', '?')}"]
    for k, v in sorted(metrics2.items()):
        right_lines.append(f"{k}: {v}")
    right_lines.append("--- Config ---")
    for k, v in sorted(config2.items()):
        right_lines.append(f"{k}: {v}")
    elements.extend(_rect(350, 70, 250, max(200, len(right_lines) * 20 + 20),
                          label="\n".join(right_lines), bg=COLORS["bg_green"],
                          stroke=COLORS["green"], font_size=12))

    # Diff arrow
    elements.append(_arrow(300, 170, 350, 170, color=COLORS["yellow"]))

    out = Path(exp_root) / "artifacts" / f"comparison-{exp_id_1}-vs-{exp_id_2}.excalidraw"
    return _write_excalidraw(out, elements)


# ---------------------------------------------------------------------------
# Diagram: HP landscape
# ---------------------------------------------------------------------------

def generate_hp_landscape(exp_root: str, hp_name: str, metric: str) -> str:
    """Visualize HP search space as a scatter of tried configs."""
    results_dir = Path(exp_root) / "results"
    results = load_results(str(results_dir))

    baseline = results.get("baseline", {})
    baseline_val = baseline.get("metrics", {}).get(metric)

    elements = []
    elements.append(_text(50, 20, f"HP Landscape: {hp_name} vs {metric}", font_size=24, color=COLORS["blue"]))

    # Collect data points
    points = []
    for exp_id, data in results.items():
        if not exp_id.startswith("exp-"):
            continue
        config = data.get("config", {})
        metrics = data.get("metrics", {})
        hp_val = config.get(hp_name)
        metric_val = metrics.get(metric)
        if hp_val is not None and metric_val is not None:
            try:
                points.append((float(hp_val), float(metric_val), exp_id, data.get("status", "?")))
            except (ValueError, TypeError):
                continue

    if not points:
        elements.append(_text(50, 70, f"No experiments found with HP '{hp_name}'", font_size=16))
        out = Path(exp_root) / "artifacts" / f"hp-landscape-{hp_name}.excalidraw"
        return _write_excalidraw(out, elements)

    # Normalize to chart coordinates
    hp_vals = [p[0] for p in points]
    metric_vals = [p[1] for p in points]
    hp_min, hp_max = min(hp_vals), max(hp_vals)
    m_min, m_max = min(metric_vals), max(metric_vals)

    chart_x, chart_y = 80, 80
    chart_w, chart_h = 500, 300

    # Axes
    elements.extend(_rect(chart_x, chart_y, chart_w, chart_h,
                          bg=COLORS["white"], stroke=COLORS["gray"]))
    elements.append(_text(chart_x + chart_w / 2 - 40, chart_y + chart_h + 10,
                          hp_name, font_size=14, color=COLORS["gray"]))
    elements.append(_text(chart_x - 60, chart_y + chart_h / 2,
                          metric, font_size=14, color=COLORS["gray"]))

    # Plot points
    hp_range = hp_max - hp_min if hp_max != hp_min else 1
    m_range = m_max - m_min if m_max != m_min else 1

    for hp_v, m_v, eid, status in points:
        px = chart_x + 20 + (hp_v - hp_min) / hp_range * (chart_w - 40)
        py = chart_y + chart_h - 20 - (m_v - m_min) / m_range * (chart_h - 40)

        if baseline_val is not None and m_v < baseline_val:
            color = COLORS["green"]
        elif status in ("failed", "diverged"):
            color = COLORS["red"]
        else:
            color = COLORS["gray"]

        dot = _rect(px - 6, py - 6, 12, 12, bg=color, stroke=color)
        elements.extend(dot)
        elements.append(_text(px + 10, py - 8, eid, font_size=10, color=color))

    # Baseline line
    if baseline_val is not None and m_min <= baseline_val <= m_max:
        bl_y = chart_y + chart_h - 20 - (baseline_val - m_min) / m_range * (chart_h - 40)
        elements.append(_text(chart_x + chart_w + 5, bl_y - 8,
                              f"baseline={baseline_val:.4f}", font_size=10, color=COLORS["blue"]))

    out = Path(exp_root) / "artifacts" / f"hp-landscape-{hp_name}.excalidraw"
    return _write_excalidraw(out, elements)


# ---------------------------------------------------------------------------
# Diagram: Architecture change
# ---------------------------------------------------------------------------

def generate_architecture_diagram(exp_root: str, proposal_name: str) -> str:
    """Generate a before/after architecture diagram for a proposal."""
    manifest_path = Path(exp_root) / "results" / "implementation-manifest.json"

    proposal = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            for p in manifest.get("proposals", []):
                if p.get("name", "").lower() == proposal_name.lower() or \
                   p.get("slug", "").lower() == proposal_name.lower():
                    proposal = p
                    break
        except (json.JSONDecodeError, OSError):
            pass

    elements = []
    elements.append(_text(50, 20, f"Architecture: {proposal_name}", font_size=24, color=COLORS["blue"]))

    if proposal is None:
        elements.append(_text(50, 70, f"Proposal '{proposal_name}' not found in manifest", font_size=16))
        out = Path(exp_root) / "artifacts" / f"architecture-{proposal_name}.excalidraw"
        return _write_excalidraw(out, elements)

    # Before section
    elements.append(_text(50, 70, "BEFORE", font_size=18, color=COLORS["red"]))
    elements.extend(_rect(50, 100, 250, 120,
                          label=f"Original Code\n\nFiles:\n{chr(10).join(proposal.get('files_modified', ['?'])[:5])}",
                          bg=COLORS["bg_red"], stroke=COLORS["red"], font_size=12))

    # Arrow
    elements.append(_arrow(175, 220, 175, 260, color=COLORS["yellow"]))

    # After section
    elements.append(_text(50, 270, "AFTER", font_size=18, color=COLORS["green"]))
    status = proposal.get("status", "?")
    strategy = proposal.get("implementation_strategy", "from_scratch")
    branch = proposal.get("branch", "?")
    after_lines = [
        f"Branch: {branch}",
        f"Strategy: {strategy}",
        f"Status: {status}",
        f"Complexity: {proposal.get('complexity', '?')}",
    ]
    files_created = proposal.get("files_created", [])
    if files_created:
        after_lines.append(f"New files: {len(files_created)}")
    elements.extend(_rect(50, 300, 250, 120,
                          label="\n".join(after_lines),
                          bg=COLORS["bg_green"], stroke=COLORS["green"], font_size=12))

    # Notes
    notes = proposal.get("notes", "")
    if notes:
        elements.append(_text(350, 100, f"Notes: {notes[:100]}", font_size=12, color=COLORS["gray"]))

    out = Path(exp_root) / "artifacts" / f"architecture-{proposal_name}.excalidraw"
    return _write_excalidraw(out, elements)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    if len(sys.argv) < 4:
        print("Usage: excalidraw_gen.py <exp_root> <mode> [args...]", file=sys.stderr)
        print("Modes: pipeline <metric>, comparison <id1> <id2>, hp-landscape <hp> <metric>, architecture <proposal>", file=sys.stderr)
        sys.exit(1)

    exp_root = sys.argv[1]
    mode = sys.argv[2]

    if mode == "pipeline":
        metric = sys.argv[3]
        path = generate_pipeline_diagram(exp_root, metric)
        print(json.dumps({"generated": True, "path": path}))

    elif mode == "comparison":
        if len(sys.argv) < 5:
            print("Usage: excalidraw_gen.py <exp_root> comparison <id1> <id2>", file=sys.stderr)
            sys.exit(1)
        path = generate_comparison_diagram(exp_root, sys.argv[3], sys.argv[4])
        print(json.dumps({"generated": True, "path": path}))

    elif mode == "hp-landscape":
        if len(sys.argv) < 5:
            print("Usage: excalidraw_gen.py <exp_root> hp-landscape <hp_name> <metric>", file=sys.stderr)
            sys.exit(1)
        path = generate_hp_landscape(exp_root, sys.argv[3], sys.argv[4])
        print(json.dumps({"generated": True, "path": path}))

    elif mode == "architecture":
        proposal = sys.argv[3]
        path = generate_architecture_diagram(exp_root, proposal)
        print(json.dumps({"generated": True, "path": path}))

    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
