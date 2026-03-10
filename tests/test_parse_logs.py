"""Tests for parse_logs.py."""

import json
import math

import pytest

from conftest import FIXTURES
from parse_logs import parse_kv_line, parse_json_line, parse_hf_trainer_line, parse_csv_lines, parse_python_logging_line, parse_tqdm_line, parse_xgboost_line, detect_format, parse_log, extract_metric_trajectory


def test_parse_kv_line():
    line = "epoch=1 step=100 loss=2.3456 lr=0.001 accuracy=35.2"
    metrics = parse_kv_line(line)
    assert metrics["epoch"] == 1.0
    assert metrics["step"] == 100.0
    assert abs(metrics["loss"] - 2.3456) < 1e-6
    assert metrics["lr"] == 0.001
    assert metrics["accuracy"] == 35.2


@pytest.mark.parametrize("input_str,check", [
    ("loss=nan lr=0.001", lambda v: math.isnan(v)),
    ("loss=inf lr=0.001", lambda v: math.isinf(v) and v > 0),
    ("loss=-inf lr=0.001", lambda v: math.isinf(v) and v < 0),
])
def test_parse_kv_line_special_values(input_str, check):
    """parse_kv_line handles NaN, Inf, and -Inf values."""
    metrics = parse_kv_line(input_str)
    assert "loss" in metrics
    assert check(metrics["loss"])
    assert metrics["lr"] == 0.001


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


@pytest.mark.parametrize("lines,expected_format", [
    (["epoch=1 loss=0.5", "epoch=2 loss=0.4"], "kv"),
    (['{"loss": 0.5}', '{"loss": 0.4}'], "json"),
    (["loss,lr,epoch", "0.5,0.001,1"], "csv"),
])
def test_detect_format_basic(lines, expected_format):
    """detect_format correctly identifies kv, json, and csv formats."""
    assert detect_format(lines) == expected_format


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


@pytest.mark.parametrize("lines,expected_format", [
    (["2024-01-15 10:30:45,123 INFO epoch=1 loss=2.345", "2024-01-15 10:30:46,456 INFO epoch=1 loss=2.100"], "logging"),
    (["100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 50/50 [00:30<00:00, 1.67it/s, loss=0.5, acc=85.2]"], "tqdm"),
])
def test_detect_format_structured(lines, expected_format):
    """detect_format correctly identifies logging and tqdm formats."""
    assert detect_format(lines) == expected_format


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
    fixture = FIXTURES / "tqdm_log.txt"
    records = parse_log(str(fixture))
    assert len(records) > 0
    # Lines with loss= and acc= in tqdm format should be parsed
    assert any("loss" in r for r in records)
    assert any("acc" in r for r in records)


def test_parse_log_python_logging_fixture():
    """Parse the Python logging fixture file and verify metric extraction."""
    fixture = FIXTURES / "python_logging_log.txt"
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


# --- XGBoost/LightGBM format tests ---

def test_parse_xgboost_line_basic():
    """Parse XGBoost bracket format with hyphenated keys."""
    line = "[10]\ttrain-auc:0.85\tval-auc:0.80"
    m = parse_xgboost_line(line)
    assert m["iteration"] == 10.0
    assert m["train-auc"] == 0.85
    assert m["val-auc"] == 0.80


def test_parse_xgboost_line_logloss():
    """Parse LightGBM format with underscores and hyphens."""
    line = "[100]\tvalidation_0-logloss:0.345\ttrain-logloss:0.321"
    m = parse_xgboost_line(line)
    assert m["iteration"] == 100.0
    assert m["validation_0-logloss"] == 0.345
    assert m["train-logloss"] == 0.321


def test_parse_xgboost_line_no_match():
    """Non-bracket line returns empty dict."""
    assert parse_xgboost_line("epoch=1 loss=0.5") == {}


def test_detect_format_xgboost():
    """XGBoost bracket lines detected as 'xgboost' format."""
    lines = ["[0]\ttrain-auc:0.50\tval-auc:0.48", "[1]\ttrain-auc:0.55\tval-auc:0.52"]
    assert detect_format(lines) == "xgboost"


def test_parse_log_xgboost_file(tmp_path):
    """Full file parsing with XGBoost format."""
    log = tmp_path / "xgb.log"
    log.write_text("[0]\ttrain-logloss:0.693\tval-logloss:0.695\n[1]\ttrain-logloss:0.650\tval-logloss:0.660\n[2]\ttrain-logloss:0.600\tval-logloss:0.620\n")
    records = parse_log(str(log))
    assert len(records) == 3
    assert records[0]["iteration"] == 0.0
    assert records[2]["train-logloss"] == 0.600


# --- Timezone-suffixed Python logging format ---


def test_parse_python_logging_line_with_utc_timezone():
    """Python logging format with UTC timezone suffix."""
    line = "2024-01-15 10:30:45,123 UTC INFO epoch=5 loss=0.234"
    metrics = parse_python_logging_line(line)
    assert metrics["epoch"] == 5.0
    assert abs(metrics["loss"] - 0.234) < 1e-6
    assert "wall_time" in metrics


def test_parse_python_logging_line_with_offset_timezone():
    """Python logging format with +0800 offset timezone."""
    line = "2024-01-15 10:30:45,123 +0800 INFO epoch=3 loss=0.5"
    metrics = parse_python_logging_line(line)
    assert metrics["epoch"] == 3.0
    assert abs(metrics["loss"] - 0.5) < 1e-6


def test_parse_python_logging_line_with_est_timezone():
    """Python logging format with EST timezone abbreviation."""
    line = "2024-01-15 10:30:45.123 EST WARNING lr=0.001 batch_loss=0.123"
    metrics = parse_python_logging_line(line)
    assert abs(metrics["lr"] - 0.001) < 1e-6
    assert abs(metrics["batch_loss"] - 0.123) < 1e-6


# --- XGBoost/LightGBM session fixture tests ---


def test_parse_xgboost_session_fixture():
    """Parse the XGBoost session fixture: 20 iterations of train-auc and val-auc."""
    records = parse_log(str(FIXTURES / "xgboost_session_log.txt"))
    assert len(records) == 20
    # First iteration
    assert records[0]["iteration"] == 0.0
    assert records[0]["train-auc"] == 0.5
    assert records[0]["val-auc"] == 0.498
    # Last iteration
    assert records[-1]["iteration"] == 19.0
    assert records[-1]["val-auc"] == 0.8052
    # Verify trajectory extraction works
    val_auc = extract_metric_trajectory(records, "val-auc")
    assert len(val_auc) == 20
    assert val_auc[-1] > val_auc[0]  # AUC should increase


def test_parse_lightgbm_session_fixture():
    """Parse the LightGBM session fixture: possessive key format with apostrophes."""
    records = parse_log(str(FIXTURES / "lightgbm_session_log.txt"))
    assert len(records) == 20
    # LightGBM keys normalized: "training's binary_logloss" → "training_binary_logloss"
    assert "training_binary_logloss" in records[0]
    assert "valid_1_binary_logloss" in records[0]
    # First iteration
    assert records[0]["iteration"] == 1.0
    assert records[0]["training_binary_logloss"] == 0.68
    assert records[0]["valid_1_binary_logloss"] == 0.685
    # Last iteration
    assert records[-1]["iteration"] == 20.0
    assert records[-1]["training_binary_logloss"] == 0.4
    # Trajectory extraction
    val_loss = extract_metric_trajectory(records, "valid_1_binary_logloss")
    assert len(val_loss) == 20
    assert val_loss[-1] < val_loss[0]  # logloss should decrease


def test_xgboost_fixture_divergence_detection():
    """XGBoost AUC metrics work with divergence detection (higher-is-better)."""
    from detect_divergence import check_divergence
    records = parse_log(str(FIXTURES / "xgboost_session_log.txt"))
    val_auc = extract_metric_trajectory(records, "val-auc")
    result = check_divergence(val_auc, lower_is_better=False)
    assert result["diverged"] is False


def test_lightgbm_fixture_divergence_detection():
    """LightGBM logloss metrics work with divergence detection (lower-is-better)."""
    from detect_divergence import check_divergence
    records = parse_log(str(FIXTURES / "lightgbm_session_log.txt"))
    val_loss = extract_metric_trajectory(records, "valid_1_binary_logloss")
    result = check_divergence(val_loss, lower_is_better=True)
    assert result["diverged"] is False


def test_parse_xgboost_line_lightgbm_possessive():
    """LightGBM possessive format: training's X → training_X."""
    line = "[5]\ttraining's binary_logloss:0.4500\tvalid_1's binary_logloss:0.5100"
    m = parse_xgboost_line(line)
    assert m["iteration"] == 5.0
    assert m["training_binary_logloss"] == 0.45
    assert m["valid_1_binary_logloss"] == 0.51


def test_detect_format_lightgbm():
    """LightGBM bracket lines with possessive keys detected as 'xgboost' format."""
    lines = [
        "[1]\ttraining's binary_logloss:0.6800\tvalid_1's binary_logloss:0.6850",
        "[2]\ttraining's binary_logloss:0.6650\tvalid_1's binary_logloss:0.6720",
    ]
    assert detect_format(lines) == "xgboost"


class TestHuggingFaceTrainerFormat:
    """Tests for HuggingFace Trainer log format parsing (Task 3.6)."""

    def test_parse_hf_trainer_line_basic(self):
        line = "{'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}"
        result = parse_hf_trainer_line(line)
        assert result == {'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}

    def test_parse_hf_trainer_line_with_whitespace(self):
        line = "  {'loss': 0.3214, 'grad_norm': 1.5, 'epoch': 2.0}  "
        result = parse_hf_trainer_line(line)
        assert 'loss' in result
        assert result['loss'] == 0.3214

    def test_parse_hf_trainer_line_non_numeric_filtered(self):
        line = "{'loss': 0.5, 'some_string': 'hello', 'epoch': 1.0}"
        result = parse_hf_trainer_line(line)
        assert 'loss' in result
        assert 'epoch' in result
        assert 'some_string' not in result

    def test_parse_hf_trainer_line_not_matching(self):
        line = "loss: 0.5"
        result = parse_hf_trainer_line(line)
        assert result == {}

    def test_parse_hf_trainer_line_empty(self):
        result = parse_hf_trainer_line("")
        assert result == {}

    def test_detect_format_hf_trainer(self):
        lines = [
            "{'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}",
            "{'loss': 0.4, 'learning_rate': 4e-05, 'epoch': 2.0}",
            "{'loss': 0.3, 'learning_rate': 3e-05, 'epoch': 3.0}",
        ]
        fmt = detect_format(lines)
        assert fmt == "hf_trainer"

    def test_parse_log_hf_trainer_integration(self, tmp_path):
        log_content = "\n".join([
            "Some header text",
            "{'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}",
            "{'loss': 0.4, 'learning_rate': 4e-05, 'epoch': 2.0}",
            "{'loss': 0.3, 'learning_rate': 3e-05, 'epoch': 3.0}",
        ])
        log_file = tmp_path / "hf_train.log"
        log_file.write_text(log_content)
        results = parse_log(str(log_file))
        assert len(results) == 3
        assert results[0]['loss'] == 0.5
        assert results[2]['epoch'] == 3.0


class TestEmptyInputEdgeCases:
    """Edge case tests for empty inputs (Task 3.5)."""

    def test_parse_csv_lines_empty(self):
        result = parse_csv_lines([])
        assert result == []

    def test_detect_format_empty(self):
        result = detect_format([])
        assert isinstance(result, str)  # should return a default format, not crash

    def test_parse_log_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.log"
        empty_file.write_text("")
        result = parse_log(str(empty_file))
        assert result == []

    def test_parse_log_nonexistent_file(self):
        result = parse_log("/nonexistent/path/to/file.log")
        assert result == []


class TestUnicodeLogFile:
    """Test Unicode handling in log files (Task 3.7)."""

    def test_parse_log_unicode_metric_names(self, tmp_path):
        log_content = "époque: 1, perte: 0.5, précision: 0.8\népoque: 2, perte: 0.3, précision: 0.9\n"
        log_file = tmp_path / "unicode.log"
        log_file.write_text(log_content, encoding="utf-8")
        results = parse_log(str(log_file))
        assert len(results) >= 0  # should not crash on Unicode

    def test_parse_log_unicode_comments(self, tmp_path):
        log_content = "# 训练日志 - Training log\nloss: 0.5, accuracy: 0.8\nloss: 0.3, accuracy: 0.9\n"
        log_file = tmp_path / "unicode_comments.log"
        log_file.write_text(log_content, encoding="utf-8")
        results = parse_log(str(log_file))
        assert len(results) == 2
        assert results[0]['loss'] == 0.5
