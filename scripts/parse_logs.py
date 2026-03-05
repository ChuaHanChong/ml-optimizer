#!/usr/bin/env python3
"""Parse training log files for metrics."""

import json
import re
import sys
from pathlib import Path


def parse_kv_line(line: str) -> dict:
    """Parse key=value pairs from a log line."""
    metrics = {}
    # Match patterns like: loss=0.123, lr=1e-4, epoch=5
    for match in re.finditer(r'(\w+)\s*[=:]\s*([0-9eE.+\-]+(?:nan|inf)?|[+-]?(?:nan|inf))', line, re.IGNORECASE):
        key, value = match.group(1), match.group(2)
        try:
            metrics[key] = float(value)
        except ValueError:
            if value.lower() == "nan":
                metrics[key] = float("nan")
            elif value.lower() == "inf":
                metrics[key] = float("inf")
    return metrics


def parse_json_line(line: str) -> dict:
    """Parse a JSON line for metrics."""
    try:
        data = json.loads(line.strip())
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, (int, float))}
    except json.JSONDecodeError:
        pass
    return {}


def parse_csv_lines(lines: list[str]) -> list[dict]:
    """Parse CSV-formatted lines (first line is header)."""
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    results = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(headers):
            continue
        row = {}
        for h, v in zip(headers, values):
            try:
                row[h] = float(v)
            except ValueError:
                continue
        if row:
            results.append(row)
    return results


def detect_format(lines: list[str]) -> str:
    """Auto-detect log format: 'json', 'csv', or 'kv'."""
    for line in lines[:5]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass
    # Check if first non-empty line looks like CSV header
    for line in lines[:3]:
        stripped = line.strip()
        if stripped and "," in stripped and not re.search(r'[=:]', stripped):
            return "csv"
    return "kv"


def parse_log(filepath: str, fmt: str | None = None) -> list[dict]:
    """Parse a training log file and return list of metric dicts per step."""
    path = Path(filepath)
    if not path.exists():
        return []

    lines = path.read_text().strip().split("\n")
    if not lines:
        return []

    if fmt is None:
        fmt = detect_format(lines)

    if fmt == "json":
        return [m for line in lines if (m := parse_json_line(line))]
    elif fmt == "csv":
        return parse_csv_lines(lines)
    else:
        return [m for line in lines if (m := parse_kv_line(line))]


def extract_metric_trajectory(records: list[dict], metric: str) -> list[float]:
    """Extract a single metric's values over time."""
    return [r[metric] for r in records if metric in r]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_logs.py <logfile> [format]")
        sys.exit(1)
    filepath = sys.argv[1]
    fmt = sys.argv[2] if len(sys.argv) > 2 else None
    records = parse_log(filepath, fmt)
    print(json.dumps(records, indent=2))
