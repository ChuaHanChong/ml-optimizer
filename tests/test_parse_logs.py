"""Tests for parse_logs.py."""

import json
from pathlib import Path

from parse_logs import parse_kv_line, parse_json_line, parse_csv_lines, parse_python_logging_line, parse_tqdm_line, detect_format, parse_log, extract_metric_trajectory

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


def test_parse_python_logging_line():
    line = "2024-01-15 10:30:45,123 INFO epoch=5 loss=0.234 accuracy=87.5"
    metrics = parse_python_logging_line(line)
    assert metrics["epoch"] == 5.0
    assert abs(metrics["loss"] - 0.234) < 1e-6
    assert metrics["accuracy"] == 87.5
    assert metrics["wall_time"] == "2024-01-15 10:30:45,123"


def test_parse_tqdm_line():
    line = "100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 50/50 [00:30<00:00, 1.67it/s, loss=0.5, acc=85.2]"
    metrics = parse_tqdm_line(line)
    assert metrics["loss"] == 0.5
    assert metrics["acc"] == 85.2


def test_detect_format_logging():
    lines = [
        "2024-01-15 10:30:45,123 INFO epoch=1 loss=2.345",
        "2024-01-15 10:30:46,456 INFO epoch=1 loss=2.100",
    ]
    assert detect_format(lines) == "logging"


def test_detect_format_tqdm():
    lines = [
        "100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 50/50 [00:30<00:00, 1.67it/s, loss=0.5, acc=85.2]",
    ]
    assert detect_format(lines) == "tqdm"


def test_parse_kv_line_negative_value():
    line = "delta=-0.5 reward=-1.2"
    metrics = parse_kv_line(line)
    assert metrics["delta"] == -0.5
    assert metrics["reward"] == -1.2


# --- Error / edge cases ---


def test_parse_kv_line_unparseable_value():
    """Non-numeric, non-nan/inf kv values are skipped."""
    line = "loss=abc lr=0.001"
    metrics = parse_kv_line(line)
    assert "loss" not in metrics
    assert metrics.get("lr") == 0.001


def test_parse_huggingface_trainer_log(tmp_path):
    """HuggingFace Trainer JSON logs are parsed correctly."""
    f = tmp_path / "trainer.log"
    f.write_text(
        '{"loss": 2.345, "learning_rate": 5e-05, "epoch": 0.5, "step": 100}\n'
        '{"loss": 1.876, "learning_rate": 4.5e-05, "epoch": 1.0, "step": 200}\n'
        '{"eval_loss": 1.5, "eval_accuracy": 0.82, "epoch": 1.0, "step": 200}\n'
    )
    records = parse_log(str(f))
    assert len(records) == 3
    assert records[0]["loss"] == 2.345
    assert records[0]["learning_rate"] == 5e-05
    assert records[2]["eval_accuracy"] == 0.82


def test_parse_tqdm_line_non_numeric():
    """Tqdm line with non-numeric value skips that key."""
    line = "100%|████| 50/50 [00:30<00:00, 1.67it/s, loss=abc, acc=85.2]"
    metrics = parse_tqdm_line(line)
    assert "loss" not in metrics
    assert metrics.get("acc") == 85.2


def test_parse_csv_lines_non_numeric():
    """CSV with non-float cell skips that field."""
    lines = ["loss,lr,status", "0.5,0.001,ok", "0.4,0.002,ok"]
    results = parse_csv_lines(lines)
    assert len(results) == 2
    assert "status" not in results[0]
    assert results[0]["loss"] == 0.5


def test_parse_csv_lines_single_line():
    """Header-only CSV returns empty list."""
    lines = ["loss,lr,epoch"]
    results = parse_csv_lines(lines)
    assert results == []


def test_parse_csv_lines_mismatched_cols():
    """CSV row with wrong column count is skipped."""
    lines = ["loss,lr", "0.5,0.001", "0.4"]
    results = parse_csv_lines(lines)
    assert len(results) == 1
    assert results[0]["loss"] == 0.5


# --- parse_log format dispatch ---


def test_parse_log_json_format(tmp_path):
    """parse_log dispatches to JSON parser."""
    f = tmp_path / "log.jsonl"
    f.write_text('{"loss": 0.5, "epoch": 1}\n{"loss": 0.4, "epoch": 2}\n')
    records = parse_log(str(f))
    assert len(records) == 2
    assert records[0]["loss"] == 0.5


def test_parse_log_csv_format(tmp_path):
    """parse_log dispatches to CSV parser."""
    f = tmp_path / "log.csv"
    f.write_text("loss,lr\n0.5,0.001\n0.4,0.002\n")
    records = parse_log(str(f))
    assert len(records) == 2
    assert records[0]["loss"] == 0.5


def test_parse_log_logging_format(tmp_path):
    """parse_log dispatches to Python logging parser."""
    f = tmp_path / "log.txt"
    f.write_text("2024-01-15 10:30:45,123 INFO epoch=1 loss=2.345\n2024-01-15 10:30:46,456 INFO epoch=2 loss=2.100\n")
    records = parse_log(str(f))
    assert len(records) == 2
    assert records[0]["epoch"] == 1.0


def test_parse_log_tqdm_format(tmp_path):
    """parse_log dispatches to tqdm parser."""
    f = tmp_path / "log.txt"
    f.write_text("100%|████████| 50/50 [00:30<00:00, 1.67it/s, loss=0.5, acc=85.2]\n")
    records = parse_log(str(f))
    assert len(records) == 1
    assert records[0]["loss"] == 0.5


# --- CLI tests ---


def test_parse_log_kv_fallback_warns_on_empty(tmp_path):
    """Auto-detected 'kv' format with no parseable metrics should warn."""
    import warnings
    f = tmp_path / "weird.log"
    f.write_text("This is just a plain text log\nNo metrics here\nJust words\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        records = parse_log(str(f))
        assert records == []
        assert len(w) == 1
        assert "kv" in str(w[0].message).lower()


def test_cli_parse_log(run_main):
    """CLI parses sample log file."""
    r = run_main("parse_logs.py", str(FIXTURES / "sample_train_log.txt"))
    assert r.returncode == 0
    output = json.loads(r.stdout)
    assert len(output) == 8


def test_parse_log_non_utf8(tmp_path):
    """Non-UTF-8 log files are handled gracefully (no crash)."""
    f = tmp_path / "binary.log"
    f.write_bytes(b"epoch=1 loss=0.5\nloss=\xff\xfe 0.3\nepoch=3 loss=0.2\n")
    records = parse_log(str(f))
    # Should extract at least the parseable lines without raising
    assert isinstance(records, list)
    assert len(records) >= 1


def test_parse_log_tqdm_fixture():
    """Parse the tqdm fixture file and verify metric extraction."""
    fixture = Path(__file__).parent / "fixtures" / "tqdm_log.txt"
    records = parse_log(str(fixture))
    assert len(records) > 0
    # Lines with loss= and acc= in tqdm format should be parsed
    assert any("loss" in r for r in records)
    assert any("acc" in r for r in records)


def test_parse_log_python_logging_fixture():
    """Parse the Python logging fixture file and verify metric extraction."""
    fixture = Path(__file__).parent / "fixtures" / "python_logging_log.txt"
    records = parse_log(str(fixture))
    assert len(records) > 0
    # Lines with epoch=, loss=, accuracy= should be parsed
    assert any("loss" in r for r in records)
    assert any("accuracy" in r for r in records)


def test_cli_no_args(run_main):
    """CLI with no args prints usage and exits 1."""
    r = run_main("parse_logs.py")
    assert r.returncode == 1
    assert "Usage" in r.stdout
