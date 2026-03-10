#!/usr/bin/env python3
"""Parse training log files for metrics."""

import json
import re
import sys
import warnings
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


def parse_python_logging_line(line: str) -> dict:
    """Parse a Python logging format line for metrics.

    Matches lines like: 2024-01-15 10:30:45,123 INFO epoch=5 loss=0.234 accuracy=87.5
    Extracts key=value or key: value metrics from the message part,
    plus a wall_time field with the timestamp.
    """
    m = re.match(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.\d]*)(?:\s+(?:(?!DEBUG\b|INFO\b|WARNING\b|WARN\b|ERROR\b|CRITICAL\b|FATAL\b|TRACE\b)[A-Z]{2,5}|[+-]\d{4}))?\s+\S+\s+(.*)', line)
    if not m:
        return {}
    timestamp, message = m.group(1), m.group(2)
    metrics = parse_kv_line(message)
    if metrics:
        metrics['wall_time'] = timestamp
    return metrics


def parse_tqdm_line(line: str) -> dict:
    """Parse a tqdm progress bar line for metrics.

    Matches lines like: 100%|████████| 50/50 [00:30<00:00, 1.67it/s, loss=0.5, acc=85.2]
    Extracts key=value metrics from the trailing bracket section.
    """
    # Find trailing bracket content that contains key=value pairs
    m = re.search(r'\[([^\]]*,\s*\w+\s*=\s*[^\]]+)\]\s*$', line)
    if not m:
        return {}
    bracket_content = m.group(1)
    metrics = {}
    for kv_match in re.finditer(r'(\w+)\s*=\s*([0-9eE.+\-]+)', bracket_content):
        key, value = kv_match.group(1), kv_match.group(2)
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def parse_xgboost_line(line: str) -> dict:
    """Parse XGBoost/LightGBM bracket-prefixed log lines.

    Matches: [10]\\ttrain-auc:0.85\\tval-auc:0.80
    Also: [100]\\tvalidation_0-logloss:0.345
    Also LightGBM: [1]\\ttraining's binary_logloss:0.6800\\tvalid_1's binary_logloss:0.6850
    """
    m = re.match(r'^\[(\d+)\]\s+(.+)', line.strip())
    if not m:
        return {}
    iteration = int(m.group(1))
    rest = m.group(2)
    metrics = {"iteration": float(iteration)}
    # Split by tab to handle LightGBM keys with spaces/apostrophes
    segments = re.split(r'\t+', rest)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        # Match "key:value" — key is everything before the last colon+number
        kv_match = re.match(r'(.+?)\s*:\s*([0-9eE.+\-]+)$', segment)
        if kv_match:
            key = kv_match.group(1).strip()
            # Normalize LightGBM possessive keys: "training's X" → "training_X"
            key = re.sub(r"'s\s+", "_", key)
            key = re.sub(r"\s+", "_", key)
            try:
                metrics[key] = float(kv_match.group(2))
            except ValueError:
                continue
        else:
            # Fallback: try original regex for space-separated entries
            for kv in re.finditer(r'([\w][\w.-]*)\s*:\s*([0-9eE.+\-]+)', segment):
                try:
                    metrics[kv.group(1)] = float(kv.group(2))
                except ValueError:
                    continue
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


def parse_hf_trainer_line(line: str) -> dict:
    """Parse HuggingFace Trainer log line: {'loss': 0.5, 'learning_rate': 5e-5, 'epoch': 1.0}"""
    stripped = line.strip()
    m = re.match(r"^\{.*'.*':.*\}$", stripped)
    if not m:
        return {}
    try:
        converted = stripped.replace("'", '"')
        data = json.loads(converted)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError):
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
    """Auto-detect log format: 'json', 'csv', 'logging', 'tqdm', or 'kv'."""
    for line in lines[:5]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass
    # Check for Python logging format (lines starting with datetime)
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', stripped):
            return "logging"
    # Check for tqdm progress bar format
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and re.search(r'\d+%\|', stripped):
            return "tqdm"
    # Check for XGBoost/LightGBM bracket format
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and re.match(r'^\[\d+\]\s+\S+.*?:', stripped):
            return "xgboost"
    # Check for HuggingFace Trainer format (single-quote Python dicts)
    hf_count = 0
    for line in lines:
        if parse_hf_trainer_line(line):
            hf_count += 1
    if hf_count >= 2:
        return "hf_trainer"
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

    lines = path.read_text(errors="replace").strip().split("\n")
    if not lines:
        return []

    auto_detected = fmt is None
    if fmt is None:
        fmt = detect_format(lines)

    if fmt == "json":
        return [m for line in lines if (m := parse_json_line(line))]
    elif fmt == "csv":
        return parse_csv_lines(lines)
    elif fmt == "logging":
        return [m for line in lines if (m := parse_python_logging_line(line))]
    elif fmt == "tqdm":
        return [m for line in lines if (m := parse_tqdm_line(line))]
    elif fmt == "xgboost":
        return [m for line in lines if (m := parse_xgboost_line(line))]
    elif fmt == "hf_trainer":
        results = []
        for line in lines:
            parsed = parse_hf_trainer_line(line)
            if parsed:
                results.append(parsed)
        return results
    else:
        result = [m for line in lines if (m := parse_kv_line(line))]
        if auto_detected and not result and lines:
            warnings.warn(
                f"No metrics found in {filepath}; auto-detected format 'kv' may be incorrect",
                stacklevel=2,
            )
        return result


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
