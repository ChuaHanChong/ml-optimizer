"""Tests for parse_logs.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from parse_logs import parse_kv_line, parse_json_line, parse_csv_lines, detect_format, parse_log, extract_metric_trajectory

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_kv_line():
    line = "epoch=1 step=100 loss=2.3456 lr=0.001 accuracy=35.2"
    metrics = parse_kv_line(line)
    assert metrics["epoch"] == 1.0
    assert metrics["step"] == 100.0
    assert abs(metrics["loss"] - 2.3456) < 1e-6
    assert metrics["lr"] == 0.001
    assert metrics["accuracy"] == 35.2


def test_parse_kv_line_with_nan():
    import math
    metrics = parse_kv_line("loss=nan lr=0.001")
    assert "loss" in metrics
    assert math.isnan(metrics["loss"])
    assert metrics["lr"] == 0.001


def test_parse_kv_line_with_inf():
    import math
    metrics = parse_kv_line("loss=inf lr=0.001")
    assert "loss" in metrics
    assert math.isinf(metrics["loss"])
    assert metrics["loss"] > 0


def test_parse_kv_line_with_neg_inf():
    import math
    metrics = parse_kv_line("loss=-inf lr=0.001")
    assert "loss" in metrics
    assert math.isinf(metrics["loss"])
    assert metrics["loss"] < 0


def test_parse_json_line():
    line = '{"loss": 0.5, "lr": 0.001, "epoch": 1, "name": "test"}'
    metrics = parse_json_line(line)
    assert metrics["loss"] == 0.5
    assert metrics["lr"] == 0.001
    assert metrics["epoch"] == 1
    assert "name" not in metrics  # non-numeric excluded


def test_parse_json_line_invalid():
    assert parse_json_line("not json") == {}


def test_parse_csv_lines():
    lines = ["loss,lr,epoch", "0.5,0.001,1", "0.4,0.001,2"]
    results = parse_csv_lines(lines)
    assert len(results) == 2
    assert results[0]["loss"] == 0.5
    assert results[1]["epoch"] == 2.0


def test_detect_format_kv():
    lines = ["epoch=1 loss=0.5", "epoch=2 loss=0.4"]
    assert detect_format(lines) == "kv"


def test_detect_format_json():
    lines = ['{"loss": 0.5}', '{"loss": 0.4}']
    assert detect_format(lines) == "json"


def test_detect_format_csv():
    lines = ["loss,lr,epoch", "0.5,0.001,1"]
    assert detect_format(lines) == "csv"


def test_parse_log_sample(tmp_path):
    records = parse_log(str(FIXTURES / "sample_train_log.txt"))
    assert len(records) == 8
    assert records[0]["loss"] == 2.3456
    assert records[-1]["accuracy"] == 82.5


def test_parse_log_nonexistent():
    assert parse_log("/nonexistent/file.txt") == []


def test_parse_log_divergent_captures_nan():
    import math
    records = parse_log(str(FIXTURES / "divergent_log.txt"))
    losses = extract_metric_trajectory(records, "loss")
    # Line 11 has loss=nan — must be captured
    assert any(math.isnan(v) for v in losses), "NaN loss should be captured from divergent log"


def test_extract_metric_trajectory():
    records = parse_log(str(FIXTURES / "sample_train_log.txt"))
    losses = extract_metric_trajectory(records, "loss")
    assert len(losses) == 8
    assert losses[0] > losses[-1]  # loss should decrease
