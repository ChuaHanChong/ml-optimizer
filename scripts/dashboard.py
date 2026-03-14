#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard for ML optimization progress.

Stdlib-only — no Streamlit, Flask, or external JS libraries.

Usage:
    python3 dashboard.py <exp_root>                  # Generate HTML file
    python3 dashboard.py <exp_root> --serve --port 8080  # Serve via HTTP
"""

import html as html_mod
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from string import Template

# Allow importing sibling modules when run directly
sys.path.insert(0, str(Path(__file__).parent))

from result_analyzer import load_results, rank_by_metric, identify_correlations


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dashboard_data(exp_root: str) -> dict:
    """Load all data needed for the dashboard."""
    root = Path(exp_root)
    data = {
        "experiments": [],
        "baseline": None,
        "pipeline_state": None,
        "error_summary": None,
        "dead_ends": [],
        "agenda": [],
        "correlations": [],
        "proposals": [],
        "running_experiments": [],
        "primary_metric": None,
        "lower_is_better": True,
        "is_running": False,
    }

    # Load results
    results_dir = root / "results"
    if results_dir.is_dir():
        results = load_results(str(results_dir))
        baseline = results.get("baseline")
        if baseline:
            data["baseline"] = baseline

        # Load pipeline state for metric info
        state_path = root / "pipeline-state.json"
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text())
                data["pipeline_state"] = state
                uc = state.get("user_choices", {})
                data["primary_metric"] = uc.get("primary_metric")
                data["lower_is_better"] = uc.get("lower_is_better", True)
                data["running_experiments"] = state.get("running_experiments", [])
                data["is_running"] = state.get("status") == "running"
            except (json.JSONDecodeError, OSError):
                pass

        # Load implementation manifest for method explanations
        manifest_path = results_dir / "implementation-manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                data["proposals"] = manifest.get("proposals", [])
            except (json.JSONDecodeError, OSError):
                pass

        metric = data["primary_metric"] or "loss"
        lower = data["lower_is_better"]

        # Rank experiments
        ranked = rank_by_metric(results, metric, lower_is_better=lower)
        data["experiments"] = ranked

        # Correlations
        try:
            corr_result = identify_correlations(results, metric, lower_is_better=lower)
            data["correlations"] = corr_result.get("correlations", [])
        except Exception:
            pass

    # Error summary
    error_log_path = root / "reports" / "error-log.json"
    if error_log_path.is_file():
        try:
            log = json.loads(error_log_path.read_text())
            data["error_summary"] = log.get("summary", {})
        except (json.JSONDecodeError, OSError):
            pass

    # Dead ends
    dead_path = root / "reports" / "dead-ends.json"
    if dead_path.is_file():
        try:
            de = json.loads(dead_path.read_text())
            data["dead_ends"] = de.get("dead_ends", [])
        except (json.JSONDecodeError, OSError):
            pass

    # Agenda
    agenda_path = root / "reports" / "research-agenda.json"
    if agenda_path.is_file():
        try:
            ag = json.loads(agenda_path.read_text())
            data["agenda"] = ag.get("ideas", [])
        except (json.JSONDecodeError, OSError):
            pass

    return data


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
${auto_refresh}
<title>ML Optimizer Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#f8f9fa;color:#212529;padding:20px}
h1{font-size:1.5rem;margin-bottom:4px;color:#1971c2}
.subtitle{color:#868e96;margin-bottom:20px;font-size:0.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:24px}
.card{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}
.card h2{font-size:1.1rem;margin-bottom:12px;color:#495057;border-bottom:1px solid #e9ecef;padding-bottom:6px}
table{width:100%;border-collapse:collapse;font-size:0.85rem}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid #e9ecef}
th{background:#f1f3f5;font-weight:600;cursor:pointer}
th:hover{background:#dee2e6}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600}
.badge-green{background:#d3f9d8;color:#2b8a3e}
.badge-red{background:#ffe3e3;color:#c92a2a}
.badge-gray{background:#e9ecef;color:#495057}
.badge-blue{background:#d0ebff;color:#1864ab}
.badge-yellow{background:#fff3bf;color:#e67700}
.metric{font-size:1.5rem;font-weight:700;margin:4px 0}
.metric-label{font-size:0.8rem;color:#868e96}
.bar{height:14px;border-radius:3px;margin:2px 0}
.progress-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 1px}
svg{max-width:100%}
.timeline{overflow-x:auto}
</style>
</head>
<body>
<h1>ML Optimizer Dashboard</h1>
<p class="subtitle">${subtitle}</p>

<div class="grid">
  <div class="card">
    <h2>Overview</h2>
    <div class="metric">${total_experiments}</div>
    <div class="metric-label">Total Experiments</div>
    <div style="margin-top:12px">
      <span class="badge badge-green">${completed} completed</span>
      <span class="badge badge-red">${failed} failed</span>
      <span class="badge badge-gray">${diverged} diverged</span>
    </div>
  </div>
  <div class="card">
    <h2>Best Result</h2>
    <div class="metric">${best_value}</div>
    <div class="metric-label">${metric_name} (${direction})</div>
    <div style="margin-top:8px;font-size:0.85rem;color:#495057">${best_exp_id}</div>
    <div style="font-size:0.85rem;color:#2f9e44">${improvement}</div>
  </div>
  <div class="card">
    <h2>Pipeline State</h2>
    <div style="font-size:0.9rem">
      <div>Phase: <strong>${phase}</strong></div>
      <div>Iteration: <strong>${iteration}</strong></div>
      <div>Budget mode: <strong>${budget_mode}</strong></div>
    </div>
  </div>
</div>

${running_section}

<div class="grid">
  <div class="card" style="grid-column:1/-1">
    <h2>Progress Timeline</h2>
    <div class="timeline">${timeline_svg}</div>
  </div>
</div>

<div class="grid">
  <div class="card" style="grid-column:1/-1">
    <h2>Results Table</h2>
    <table id="results-table">
      <thead><tr>
        <th onclick="sortTable(0)">Exp ID</th>
        <th onclick="sortTable(1)">Status</th>
        <th onclick="sortTable(2)">${metric_name}</th>
        <th onclick="sortTable(3)">vs Baseline</th>
        <th onclick="sortTable(4)">Branch</th>
        <th onclick="sortTable(5)">Iteration</th>
      </tr></thead>
      <tbody>${results_rows}</tbody>
    </table>
  </div>
</div>

${hp_section}

${agenda_section}

${errors_section}

${methods_section}

<script>
function sortTable(n){
  var t=document.getElementById("results-table"),switching=true,dir="asc",switchcount=0;
  while(switching){switching=false;var rows=t.rows;
  for(var i=1;i<rows.length-1;i++){var shouldSwitch=false;
  var x=rows[i].getElementsByTagName("TD")[n],y=rows[i+1].getElementsByTagName("TD")[n];
  var xv=parseFloat(x.dataset.val)||x.innerText.toLowerCase();
  var yv=parseFloat(y.dataset.val)||y.innerText.toLowerCase();
  if(dir=="asc"?xv>yv:xv<yv){shouldSwitch=true;break}}
  if(shouldSwitch){rows[i].parentNode.insertBefore(rows[i+1],rows[i]);switching=true;switchcount++}
  else if(switchcount==0&&dir=="asc"){dir="desc";switching=true}}
}
</script>
</body>
</html>""")


def _format_value(v, precision=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _status_badge(status):
    classes = {
        "completed": "badge-green", "failed": "badge-red",
        "diverged": "badge-red", "timeout": "badge-yellow",
        "running": "badge-blue",
    }
    cls = classes.get(status, "badge-gray")
    return f'<span class="badge {cls}">{status}</span>'


def _generate_timeline_svg(experiments, metric, baseline_val, lower_is_better):
    """Generate an inline SVG timeline chart."""
    completed = [e for e in experiments if e.get("status") == "completed"
                 and e.get("value") is not None
                 and e.get("exp_id", "").startswith("exp-")]
    if not completed:
        return "<p style='color:#868e96'>No completed experiments to chart</p>"

    vals = [e["value"] for e in completed]
    min_v, max_v = min(vals), max(vals)
    if baseline_val is not None:
        min_v = min(min_v, baseline_val)
        max_v = max(max_v, baseline_val)
    v_range = max_v - min_v if max_v != min_v else 1

    w, h = 700, 200
    pad_x, pad_y = 60, 20
    chart_w = w - pad_x - 20
    chart_h = h - 2 * pad_y

    lines = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    # Background
    lines.append(f'<rect x="{pad_x}" y="{pad_y}" width="{chart_w}" height="{chart_h}" fill="#f8f9fa" stroke="#dee2e6"/>')

    # Baseline line
    if baseline_val is not None:
        by = pad_y + chart_h - (baseline_val - min_v) / v_range * chart_h
        lines.append(f'<line x1="{pad_x}" y1="{by:.1f}" x2="{pad_x+chart_w}" y2="{by:.1f}" stroke="#1971c2" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{pad_x-5}" y="{by+4:.1f}" text-anchor="end" font-size="10" fill="#1971c2">baseline</text>')

    # Data points
    n = len(completed)
    for i, exp in enumerate(completed):
        v = exp["value"]
        x = pad_x + (i / max(n - 1, 1)) * chart_w
        y = pad_y + chart_h - (v - min_v) / v_range * chart_h

        is_better = (baseline_val is not None and
                     ((lower_is_better and v < baseline_val) or
                      (not lower_is_better and v > baseline_val)))
        color = "#2f9e44" if is_better else "#868e96"
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        # Connect with line
        if i > 0:
            prev = completed[i-1]
            prev_v = prev["value"]
            px = pad_x + ((i-1) / max(n - 1, 1)) * chart_w
            py = pad_y + chart_h - (prev_v - min_v) / v_range * chart_h
            lines.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#dee2e6"/>')

    # Y-axis labels
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        v = min_v + frac * v_range
        y = pad_y + chart_h - frac * chart_h
        lines.append(f'<text x="{pad_x-5}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#868e96">{v:.3f}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def generate_dashboard(exp_root: str, *, live: bool = False) -> str:
    """Generate the dashboard HTML file. Returns the output path.

    When *live* is True (or the pipeline is still running), the HTML
    includes ``<meta http-equiv="refresh" content="30">`` so the browser
    auto-reloads every 30 seconds.
    """
    data = _load_dashboard_data(exp_root)
    metric = data["primary_metric"] or "loss"
    lower = data["lower_is_better"]
    exps = data["experiments"]
    baseline = data["baseline"]
    baseline_val = baseline.get("metrics", {}).get(metric) if baseline else None

    completed = [e for e in exps if e.get("status") == "completed"]
    failed = [e for e in exps if e.get("status") == "failed"]
    diverged = [e for e in exps if e.get("status") == "diverged"]

    # Best result
    best = completed[0] if completed else None
    best_value = _format_value(best.get("value") if best else None)
    best_id = best.get("exp_id", "—") if best else "—"
    improvement = ""
    if best and baseline_val is not None:
        bv = best.get("value")
        if bv is not None and baseline_val != 0:
            delta = baseline_val - bv if lower else bv - baseline_val
            pct = delta / abs(baseline_val) * 100
            improvement = f"{pct:+.2f}% vs baseline"

    # Pipeline state
    ps = data["pipeline_state"] or {}
    uc = ps.get("user_choices", {})

    # Results rows
    rows = []
    for exp in exps:
        eid = exp.get("exp_id", "?")
        status = exp.get("status", "?")
        val = exp.get("value")
        branch = exp.get("code_branch") or "—"
        iteration = exp.get("iteration", "—")
        delta = ""
        if val is not None and baseline_val is not None and baseline_val != 0:
            d = baseline_val - val if lower else val - baseline_val
            pct = d / abs(baseline_val) * 100
            delta = f"{pct:+.2f}%"
        rows.append(
            f'<tr><td>{eid}</td><td>{_status_badge(status)}</td>'
            f'<td data-val="{val if val is not None else ""}">{_format_value(val)}</td>'
            f'<td>{delta}</td><td>{branch}</td><td>{iteration}</td></tr>'
        )

    # HP sensitivity section
    hp_section = ""
    corrs = data.get("correlations", [])
    if corrs:
        hp_rows = []
        for c in corrs[:8]:
            hp = c.get("hp", "?")
            rho = c.get("correlation", 0)
            bar_w = abs(rho) * 200
            color = "#2f9e44" if rho > 0 else "#e03131"
            hp_rows.append(
                f'<tr><td>{hp}</td><td>{rho:+.3f}</td>'
                f'<td><div class="bar" style="width:{bar_w:.0f}px;background:{color}"></div></td></tr>'
            )
        hp_section = f"""<div class="grid"><div class="card" style="grid-column:1/-1">
<h2>HP Sensitivity (Spearman ρ)</h2>
<table><thead><tr><th>Parameter</th><th>ρ</th><th>Impact</th></tr></thead>
<tbody>{"".join(hp_rows)}</tbody></table></div></div>"""

    # Agenda section
    agenda_section = ""
    agenda = data.get("agenda", [])
    if agenda:
        agenda_rows = []
        for idea in sorted(agenda, key=lambda x: x.get("priority", 0), reverse=True):
            status_cls = {
                "untried": "badge-blue", "tried": "badge-gray",
                "improved": "badge-green", "dead-end": "badge-red",
            }.get(idea.get("status", ""), "badge-gray")
            agenda_rows.append(
                f'<tr><td>{idea.get("name","?")}</td>'
                f'<td><span class="badge {status_cls}">{idea.get("status","?")}</span></td>'
                f'<td>{idea.get("priority","?")}</td>'
                f'<td>{idea.get("source","?")}</td></tr>'
            )
        agenda_section = f"""<div class="grid"><div class="card" style="grid-column:1/-1">
<h2>Research Agenda</h2>
<table><thead><tr><th>Idea</th><th>Status</th><th>Priority</th><th>Source</th></tr></thead>
<tbody>{"".join(agenda_rows)}</tbody></table></div></div>"""

    # Errors section
    errors_section = ""
    err = data.get("error_summary")
    dead = data.get("dead_ends", [])
    if err or dead:
        parts = []
        if err:
            cat_items = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                               for k, v in err.get("by_category", {}).items())
            parts.append(f"""<div class="card"><h2>Error Summary</h2>
<div class="metric">{err.get('total_events',0)}</div><div class="metric-label">Total Events</div>
<table style="margin-top:8px"><thead><tr><th>Category</th><th>Count</th></tr></thead>
<tbody>{cat_items}</tbody></table></div>""")
        if dead:
            de_items = "".join(f"<tr><td>{d.get('technique','?')}</td><td>{d.get('reason','?')}</td></tr>"
                               for d in dead)
            parts.append(f"""<div class="card"><h2>Dead Ends ({len(dead)})</h2>
<table><thead><tr><th>Technique</th><th>Reason</th></tr></thead>
<tbody>{de_items}</tbody></table></div>""")
        errors_section = f'<div class="grid">{"".join(parts)}</div>'

    # Timeline SVG
    timeline_svg = _generate_timeline_svg(exps, metric, baseline_val, lower)

    # Auto-refresh (live mode)
    is_live = live or data.get("is_running", False)
    auto_refresh = '<meta http-equiv="refresh" content="30">' if is_live else ""

    # Running experiments section
    running_section = ""
    running = data.get("running_experiments", [])
    if running:
        run_items = "".join(f'<span class="badge badge-blue" style="margin:2px">{eid}</span>' for eid in running)
        running_section = f"""<div class="grid"><div class="card" style="grid-column:1/-1">
<h2>Running Experiments ({len(running)})</h2>
<div>{run_items}</div>
<div style="margin-top:8px;font-size:0.8rem;color:#868e96">Dashboard auto-refreshes every 30 seconds</div>
</div></div>"""

    # Method explanations section (only for code-change proposals)
    methods_section = ""
    proposals = data.get("proposals", [])
    code_proposals = [p for p in proposals if p.get("status") == "validated"
                      and p.get("explanation")]
    if code_proposals:
        method_cards = []
        for p in code_proposals:
            name = p.get("name", "?")
            explanation = p.get("explanation", "")
            strategy = p.get("implementation_strategy", "from_scratch")
            files = p.get("files_modified", [])
            diff = p.get("diff_summary", {})
            source = p.get("proposal_source") or "unknown"

            files_str = ", ".join(f"<code>{f}</code>" for f in files[:5])
            if len(files) > 5:
                files_str += f" (+{len(files)-5} more)"

            diff_info = ""
            if isinstance(diff, dict) and diff.get("files_changed"):
                diff_info = (f'<div style="font-size:0.8rem;color:#868e96;margin-top:4px">'
                            f'+{diff.get("lines_added",0)} / -{diff.get("lines_removed",0)} lines '
                            f'across {diff.get("files_changed",0)} files</div>')
                funcs = diff.get("changed_functions", [])
                if funcs:
                    diff_info += f'<div style="font-size:0.8rem;color:#495057">Functions: {", ".join(funcs[:5])}</div>'

            source_badge = f'<span class="badge badge-{"green" if source == "paper" else "yellow"}">{source}</span>'

            method_cards.append(f"""<div class="card">
<h2>{html_mod.escape(name)} {source_badge}</h2>
<p style="margin:8px 0">{html_mod.escape(explanation)}</p>
<div style="font-size:0.85rem"><strong>Strategy:</strong> {strategy}</div>
<div style="font-size:0.85rem"><strong>Files:</strong> {files_str}</div>
{diff_info}
</div>""")

        methods_section = f"""<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(350px,1fr))">
{"".join(method_cards)}</div>"""
        methods_section = f'<h2 style="margin:16px 0 8px;color:#495057">Method Implementation Details</h2>\n{methods_section}'

    html = _HTML_TEMPLATE.substitute(
        auto_refresh=auto_refresh,
        subtitle=f"Metric: {metric} ({'lower' if lower else 'higher'} is better)",
        total_experiments=str(len(exps)),
        completed=str(len(completed)),
        failed=str(len(failed)),
        diverged=str(len(diverged)),
        best_value=best_value,
        metric_name=metric,
        direction="lower is better" if lower else "higher is better",
        best_exp_id=best_id,
        improvement=improvement,
        phase=str(ps.get("phase", "—")),
        iteration=str(ps.get("iteration", "—")),
        budget_mode=uc.get("budget_mode", "—"),
        running_section=running_section,
        timeline_svg=timeline_svg,
        results_rows="\n".join(rows),
        hp_section=hp_section,
        agenda_section=agenda_section,
        errors_section=errors_section,
        methods_section=methods_section,
    )

    out_path = Path(exp_root) / "reports" / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    if len(sys.argv) < 2:
        print("Usage: dashboard.py <exp_root> [--serve --port PORT]", file=sys.stderr)
        sys.exit(1)

    exp_root = sys.argv[1]
    live = "--live" in sys.argv
    path = generate_dashboard(exp_root, live=live)
    print(json.dumps({"generated": True, "path": path, "live": live}))

    if "--serve" in sys.argv:
        port = 8080
        port_idx = sys.argv.index("--port") if "--port" in sys.argv else -1
        if port_idx >= 0 and port_idx + 1 < len(sys.argv):
            port = int(sys.argv[port_idx + 1])

        import os
        os.chdir(str(Path(path).parent))
        server = HTTPServer(("", port), SimpleHTTPRequestHandler)
        print(f"Serving dashboard at http://localhost:{port}/dashboard.html", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()


if __name__ == "__main__":
    _cli_main()
