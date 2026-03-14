"""Consolidated tests for parse_logs.py and detect_divergence.py.

Covers: log parsing (kv, json, csv, hf_trainer, xgboost, lightgbm, tqdm,
python_logging), format auto-detection, divergence detection (NaN, Inf,
explosion, plateau, drift), model-category thresholds, edge cases, and CLI.
"""

import json
import math
import random
import warnings

import pytest

from conftest import FIXTURES
from detect_divergence import (
    MODEL_CATEGORY_DEFAULTS,
    check_divergence,
    detect_explosion,
    detect_gradual_drift,
    detect_nan_inf,
    detect_plateau,
    get_thresholds_for_category,
)
from parse_logs import (
    detect_format,
    extract_metric_trajectory,
    parse_csv_lines,
    parse_hf_trainer_line,
    parse_json_line,
    parse_kv_line,
    parse_log,
    parse_python_logging_line,
    parse_tqdm_line,
    parse_xgboost_line,
)


# ---------------------------------------------------------------------------
# TestLogParsing — individual format parsers
# ---------------------------------------------------------------------------


class TestLogParsing:
    """Tests for each log format parser."""

    # -- kv format (includes special values) --

    @pytest.mark.parametrize("line,expected", [
        ("epoch=1 step=100 loss=2.3456 lr=0.001 accuracy=35.2",
         {"epoch": 1.0, "step": 100.0, "loss": 2.3456, "lr": 0.001, "accuracy": 35.2}),
        ("delta=-0.5 reward=-1.2", {"delta": -0.5, "reward": -1.2}),
        ("loss=abc lr=0.001", {"lr": 0.001}),  # non-numeric skipped
    ])
    def test_parse_kv_line(self, line, expected):
        metrics = parse_kv_line(line)
        for k, v in expected.items():
            assert abs(metrics[k] - v) < 1e-6

    def test_parse_kv_special_values(self):
        """NaN, Inf, -Inf parsed correctly."""
        for line, check in [
            ("loss=nan lr=0.001", lambda v: math.isnan(v)),
            ("loss=inf lr=0.001", lambda v: math.isinf(v) and v > 0),
            ("loss=-inf lr=0.001", lambda v: math.isinf(v) and v < 0),
        ]:
            metrics = parse_kv_line(line)
            assert check(metrics["loss"])
            assert metrics["lr"] == 0.001

    # -- json format --

    @pytest.mark.parametrize("line,expected_keys,excluded", [
        ('{"loss": 0.5, "lr": 0.001, "epoch": 1, "name": "test"}',
         {"loss": 0.5, "lr": 0.001, "epoch": 1}, ["name"]),
        ("not json", {}, []),
    ])
    def test_parse_json_line(self, line, expected_keys, excluded):
        metrics = parse_json_line(line)
        for k, v in expected_keys.items():
            assert metrics[k] == v
        for k in excluded:
            assert k not in metrics

    # -- csv format --

    @pytest.mark.parametrize("lines,exp_len,checks", [
        (["loss,lr,epoch", "0.5,0.001,1", "0.4,0.001,2"],
         2, [("loss", 0, 0.5), ("epoch", 1, 2.0)]),
        (["loss,lr,status", "0.5,0.001,ok", "0.4,0.002,ok"],
         2, [("loss", 0, 0.5)]),  # non-float skipped
        ([], 0, []),  # empty
    ])
    def test_parse_csv_lines(self, lines, exp_len, checks):
        results = parse_csv_lines(lines)
        assert len(results) == exp_len
        for key, idx, val in checks:
            assert results[idx][key] == val

    # -- hf_trainer format --

    @pytest.mark.parametrize("line,expected", [
        ("{'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}",
         {"loss": 0.5, "learning_rate": 5e-05, "epoch": 1.0}),
        ("{'loss': 0.5, 'some_string': 'hello', 'epoch': 1.0}",
         {"loss": 0.5, "epoch": 1.0}),  # non-numeric filtered
        ("loss: 0.5", {}),  # not hf_trainer format
    ])
    def test_parse_hf_trainer_line(self, line, expected):
        assert parse_hf_trainer_line(line) == expected

    # -- xgboost/lightgbm format --

    @pytest.mark.parametrize("line,expected", [
        ("[10]\ttrain-auc:0.85\tval-auc:0.80",
         {"iteration": 10.0, "train-auc": 0.85, "val-auc": 0.80}),
        ("[5]\ttraining's binary_logloss:0.4500\tvalid_1's binary_logloss:0.5100",
         {"iteration": 5.0, "training_binary_logloss": 0.45, "valid_1_binary_logloss": 0.51}),
        ("epoch=1 loss=0.5", {}),  # non-bracket
    ])
    def test_parse_xgboost_line(self, line, expected):
        assert parse_xgboost_line(line) == expected

    # -- tqdm format --

    @pytest.mark.parametrize("line,expected_keys,excluded", [
        ("100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 50/50 [00:30<00:00, 1.67it/s, loss=0.5, acc=85.2]",
         {"loss": 0.5, "acc": 85.2}, []),
        ("100%|\u2588\u2588\u2588\u2588| 50/50 [00:30<00:00, 1.67it/s, loss=abc, acc=85.2]",
         {"acc": 85.2}, ["loss"]),
    ])
    def test_parse_tqdm_line(self, line, expected_keys, excluded):
        metrics = parse_tqdm_line(line)
        for k, v in expected_keys.items():
            assert metrics[k] == v
        for k in excluded:
            assert k not in metrics

    # -- python_logging format --

    @pytest.mark.parametrize("line,expected_keys,has_wall_time", [
        ("2024-01-15 10:30:45,123 INFO epoch=5 loss=0.234 accuracy=87.5",
         {"epoch": 5.0, "loss": 0.234, "accuracy": 87.5}, True),
        ("2024-01-15 10:30:45.123 EST WARNING lr=0.001 batch_loss=0.123",
         {"lr": 0.001, "batch_loss": 0.123}, True),
        ("Just some random text", {}, False),
    ])
    def test_parse_python_logging_line(self, line, expected_keys, has_wall_time):
        metrics = parse_python_logging_line(line)
        for k, v in expected_keys.items():
            assert abs(metrics[k] - v) < 1e-6
        if has_wall_time and expected_keys:
            assert "wall_time" in metrics

    # -- full file parsing via parse_log --

    @pytest.mark.parametrize("write_content,exp_len,checks", [
        ('{"loss": 0.5, "epoch": 1}\n{"loss": 0.4, "epoch": 2}\n',
         2, [("loss", 0, 0.5)]),
        ("loss,lr\n0.5,0.001\n0.4,0.002\n",
         2, [("loss", 0, 0.5)]),
        ("2024-01-15 10:30:45,123 INFO epoch=1 loss=2.345\n2024-01-15 10:30:46,456 INFO epoch=2 loss=2.100\n",
         2, [("epoch", 0, 1.0)]),
        ("100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 50/50 [00:30<00:00, 1.67it/s, loss=0.5, acc=85.2]\n",
         1, [("loss", 0, 0.5)]),
        # hf_trainer integration
        ("Some header text\n"
         "{'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}\n"
         "{'loss': 0.4, 'learning_rate': 4e-05, 'epoch': 2.0}\n"
         "{'loss': 0.3, 'learning_rate': 3e-05, 'epoch': 3.0}\n",
         3, [("loss", 0, 0.5), ("epoch", 2, 3.0)]),
        # xgboost file
        ("[0]\ttrain-logloss:0.693\tval-logloss:0.695\n"
         "[1]\ttrain-logloss:0.650\tval-logloss:0.660\n"
         "[2]\ttrain-logloss:0.600\tval-logloss:0.620\n",
         3, [("iteration", 0, 0.0), ("train-logloss", 2, 0.600)]),
    ])
    def test_parse_log_formats(self, tmp_path, write_content, exp_len, checks):
        f = tmp_path / "log.txt"
        f.write_text(write_content)
        records = parse_log(str(f))
        assert len(records) == exp_len
        for key, idx, val in checks:
            assert records[idx][key] == val

    # -- fixture files (parametrized over all fixtures) --

    @pytest.mark.parametrize("fixture_file", [
        "sample_train_log.txt",
        "divergent_log.txt",
        "tqdm_log.txt",
        "python_logging_log.txt",
        "xgboost_session_log.txt",
        "lightgbm_session_log.txt",
        "noisy_train_log.txt",
        "partial_log.txt",
        "oom_log.txt",
    ])
    def test_fixture_files_parse(self, fixture_file):
        """All fixture files parse without error and produce non-empty results."""
        records = parse_log(str(FIXTURES / fixture_file))
        assert len(records) > 0
        assert all(isinstance(r, dict) for r in records)

    def test_fixture_specific_assertions(self):
        """Spot-check key fixture properties (sample_train, xgboost, lightgbm, divergent)."""
        # sample_train_log
        records = parse_log(str(FIXTURES / "sample_train_log.txt"))
        assert len(records) == 8
        assert records[0]["loss"] == 2.3456
        assert records[-1]["accuracy"] == 82.5
        losses = extract_metric_trajectory(records, "loss")
        assert len(losses) == 8 and losses[0] > losses[-1]

        # xgboost
        xgb = parse_log(str(FIXTURES / "xgboost_session_log.txt"))
        assert len(xgb) == 20
        assert xgb[0]["iteration"] == 0.0
        val_auc = extract_metric_trajectory(xgb, "val-auc")
        assert val_auc[-1] > val_auc[0]

        # lightgbm
        lgb = parse_log(str(FIXTURES / "lightgbm_session_log.txt"))
        assert len(lgb) == 20
        assert "training_binary_logloss" in lgb[0]
        val_loss = extract_metric_trajectory(lgb, "valid_1_binary_logloss")
        assert val_loss[-1] < val_loss[0]

        # divergent
        div = parse_log(str(FIXTURES / "divergent_log.txt"))
        div_losses = extract_metric_trajectory(div, "loss")
        assert any(math.isnan(v) for v in div_losses)


# ---------------------------------------------------------------------------
# TestFormatDetection — auto-detect format from log content
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """Tests for detect_format() auto-detection."""

    @pytest.mark.parametrize("lines,expected_format", [
        (["epoch=1 loss=0.5", "epoch=2 loss=0.4"], "kv"),
        (['{"loss": 0.5}', '{"loss": 0.4}'], "json"),
        (["loss,lr,epoch", "0.5,0.001,1"], "csv"),
        (["{'loss': 0.5, 'learning_rate': 5e-05, 'epoch': 1.0}",
          "{'loss': 0.4, 'learning_rate': 4e-05, 'epoch': 2.0}",
          "{'loss': 0.3, 'learning_rate': 3e-05, 'epoch': 3.0}"], "hf_trainer"),
        (["[0]\ttrain-auc:0.50\tval-auc:0.48",
          "[1]\ttrain-auc:0.55\tval-auc:0.52"], "xgboost"),
    ])
    def test_detect_format(self, lines, expected_format):
        assert detect_format(lines) == expected_format

    def test_detect_format_empty(self):
        result = detect_format([])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestDivergenceDetection — NaN, Inf, explosion, plateau, drift
# ---------------------------------------------------------------------------


class TestDivergenceDetection:
    """Tests for divergence detection functions."""

    # -- NaN/Inf --

    @pytest.mark.parametrize("values,expected_reason,expected_step", [
        ([1.0, 0.9, 0.8, float("nan"), 0.6], "NaN detected", 3),
        ([1.0, 0.9, float("inf"), 0.7], "Inf detected", 2),
    ])
    def test_detect_nan_inf(self, values, expected_reason, expected_step):
        result = detect_nan_inf(values)
        assert result is not None
        assert result["diverged"] is True
        assert result["reason"] == expected_reason
        assert result["step"] == expected_step

    def test_detect_nan_inf_clean_and_empty(self):
        assert detect_nan_inf([1.0, 0.9, 0.8, 0.7]) is None
        assert detect_nan_inf([]) is None

    # -- explosion --

    @pytest.mark.parametrize("values,window,threshold,lower,should_explode", [
        ([1.0] * 15 + [50.0], 10, 5.0, True, True),  # spike
        ([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
         5, 5.0, True, False),  # healthy
        ([1.0, 0.9], 10, 5.0, True, False),  # too short
        ([1.0] * 15 + [float("inf")], 10, 5.0, True, False),  # inf skipped
        ([80] * 15 + [5], 10, 5.0, False, True),  # crash (higher-is-better)
        ([0.01] * 15 + [1.0], 10, 5.0, True, True),  # real explosion
    ])
    def test_detect_explosion(self, values, window, threshold, lower, should_explode):
        result = detect_explosion(values, window=window, threshold=threshold,
                                  lower_is_better=lower)
        if should_explode:
            assert result is not None and result["diverged"] is True
        else:
            assert result is None

    def test_detect_explosion_boundary_window(self):
        window = 10
        values = [1.0] * window + [50.0]
        assert detect_explosion(values, window=window, threshold=5.0) is not None
        values_short = [1.0] * (window - 1) + [50.0]
        assert detect_explosion(values_short, window=window, threshold=5.0) is None

    # -- plateau --

    @pytest.mark.parametrize("values,patience,lower,should_plateau", [
        ([1.0, 0.9, 0.8] + [0.8] * 25, 20, True, True),
        ([1.0 - i * 0.01 for i in range(50)], 20, True, False),  # improving
        ([1.0, 0.9, 0.9], 20, True, False),  # too short
        ([50, 55, 60, 65, 70, 75, 80] + [80] * 25, 20, False, True),  # flat HIB
    ])
    def test_detect_plateau(self, values, patience, lower, should_plateau):
        result = detect_plateau(values, patience=patience, lower_is_better=lower)
        if should_plateau:
            assert result is not None and result["diverged"] is True
            assert "plateau" in result["reason"].lower()
        else:
            assert result is None

    def test_detect_plateau_boundary_patience(self):
        patience = 20
        values = [1.0] + [1.0] * patience
        assert detect_plateau(values, patience=patience, min_delta=1e-6) is not None
        values_short = [1.0] + [1.0] * (patience - 1)
        assert detect_plateau(values_short, patience=patience, min_delta=1e-6) is None

    # -- gradual drift --

    @pytest.mark.parametrize("values,lower,should_drift", [
        ([0.5 + i * 0.005 for i in range(60)], True, True),  # increasing loss
        ([1.0 - i * 0.005 for i in range(60)], True, False),  # decreasing loss
        ([80 - i * 0.3 for i in range(60)], False, True),  # decreasing accuracy
    ])
    def test_detect_gradual_drift(self, values, lower, should_drift):
        result = detect_gradual_drift(values, window=50, min_slope_ratio=0.1,
                                      lower_is_better=lower)
        if should_drift:
            assert result is not None and result["diverged"] is True
            assert "drift" in result["reason"].lower()
        else:
            assert result is None

    def test_detect_gradual_drift_edge_cases(self):
        """Noisy stable, oscillating, and too-short sequences don't trigger drift."""
        rng = random.Random(42)
        # noisy stable
        assert detect_gradual_drift(
            [0.5 + rng.gauss(0, 0.1) for _ in range(100)],
            window=50, min_slope_ratio=0.1) is None
        # oscillating
        assert detect_gradual_drift(
            [0.5 + 0.1 * math.sin(i * 0.3) for i in range(100)],
            window=50, min_slope_ratio=0.1) is None
        # too short
        assert detect_gradual_drift(
            [0.5 + i * 0.01 for i in range(10)], window=50) is None
        # identical values (ss_tot=0)
        assert detect_gradual_drift(
            [0.5] * 60, window=50, min_slope_ratio=0.1,
            lower_is_better=True) is None
        # noisy with real trend still triggers
        rng2 = random.Random(42)
        result = detect_gradual_drift(
            [0.5 + i * 0.005 + rng2.gauss(0, 0.02) for i in range(60)],
            window=50, min_slope_ratio=0.1)
        assert result is not None and "drift" in result["reason"].lower()

    # -- check_divergence composite --

    @pytest.mark.parametrize("values,kwargs,expected_diverged,reason_contains", [
        ([1.0 - i * 0.01 for i in range(50)], {}, False, "healthy"),
        ([1.0, 0.9, float("nan"), 0.7], {}, True, "nan"),
        ([], {}, False, "no data"),
        ([1.0] * 15 + [100.0],
         {"explosion_window": 10, "explosion_threshold": 5.0}, True, "explosion"),
        ([0.5 + i * 0.005 for i in range(60)],
         {"gradual_drift_window": 50}, True, "drift"),
        ([0.5] * 25, {"plateau_patience": 20}, True, "plateau"),
    ])
    def test_check_divergence(self, values, kwargs, expected_diverged, reason_contains):
        result = check_divergence(values, **kwargs)
        assert result["diverged"] is expected_diverged
        assert reason_contains.lower() in result["reason"].lower()

    def test_divergence_priority_order(self):
        """NaN > explosion > drift > plateau priority ordering."""
        # NaN before explosion
        values = [1.0] * 5 + [float("nan")] + [1.0] * 9 + [50.0]
        result = check_divergence(values, explosion_window=10, explosion_threshold=5.0)
        assert result["diverged"] is True and "NaN" in result["reason"]

        # Explosion over plateau
        values = [1.0] * 30 + [50.0]
        result = check_divergence(values, explosion_window=10,
                                  explosion_threshold=5.0, plateau_patience=20)
        assert "explosion" in result["reason"].lower()

    # -- xgboost/lightgbm fixture divergence --

    @pytest.mark.parametrize("fixture,metric,lower", [
        ("xgboost_session_log.txt", "val-auc", False),
        ("lightgbm_session_log.txt", "valid_1_binary_logloss", True),
    ])
    def test_tree_fixture_no_divergence(self, fixture, metric, lower):
        records = parse_log(str(FIXTURES / fixture))
        values = extract_metric_trajectory(records, metric)
        assert check_divergence(values, lower_is_better=lower)["diverged"] is False


# ---------------------------------------------------------------------------
# TestModelCategoryThresholds — rl/generative/supervised overrides
# ---------------------------------------------------------------------------


class TestModelCategoryThresholds:
    """Tests for per-model-category divergence threshold overrides."""

    @pytest.mark.parametrize("category,expected", [
        ("rl", {"explosion_threshold": 20.0, "plateau_patience": 50}),
        ("generative", {"explosion_threshold": 10.0, "plateau_patience": 40}),
        (None, {}),
    ])
    def test_get_thresholds(self, category, expected):
        t = get_thresholds_for_category(category)
        for k, v in expected.items():
            assert t[k] == v
        if not expected:
            assert t == {}

    def test_rl_prevents_false_positive(self):
        """Reward crash triggers with default threshold but not RL threshold.
        Also verifies immutability of MODEL_CATEGORY_DEFAULTS."""
        values_crash = [8.0] * 15 + [1.0]
        default = check_divergence(values_crash, explosion_threshold=5.0,
                                   lower_is_better=False)
        assert default["diverged"] is True
        rl_kwargs = get_thresholds_for_category("rl")
        rl_kwargs_copy = dict(rl_kwargs)
        rl_result = check_divergence(values_crash, lower_is_better=False, **rl_kwargs)
        assert rl_result["diverged"] is False
        # immutability
        rl_kwargs_copy["explosion_threshold"] = 999
        assert MODEL_CATEGORY_DEFAULTS["rl"]["explosion_threshold"] == 20.0

    def test_generative_longer_patience(self):
        """25 flat values: default patience triggers, generative does not."""
        values = [0.5] + [0.5] * 25
        assert check_divergence(values, plateau_patience=20)["diverged"] is True
        gen_kwargs = get_thresholds_for_category("generative")
        assert check_divergence(values, **gen_kwargs)["diverged"] is False


# ---------------------------------------------------------------------------
# TestEdgeCases — empty, partial, corrupted, OOM, short sequences
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty logs, partial logs, corrupted data, OOM, short sequences."""

    def test_parse_log_nonexistent_and_empty(self, tmp_path):
        """Non-existent file and empty file both return []."""
        assert parse_log("/nonexistent/file.txt") == []
        f = tmp_path / "empty.log"
        f.write_text("")
        assert parse_log(str(f)) == []

    def test_parse_log_binary_and_unicode(self, tmp_path):
        """Non-UTF8 binary data and unicode content handled gracefully."""
        # binary
        f_bin = tmp_path / "binary.log"
        f_bin.write_bytes(b"epoch=1 loss=0.5\nloss=\xff\xfe 0.3\nepoch=3 loss=0.2\n")
        records = parse_log(str(f_bin))
        assert isinstance(records, list) and len(records) >= 1

        # unicode comments
        f_uni = tmp_path / "unicode.log"
        f_uni.write_text("# \u8bad\u7ec3\u65e5\u5fd7 - Training log\nloss: 0.5, accuracy: 0.8\nloss: 0.3, accuracy: 0.9\n",
                         encoding="utf-8")
        results = parse_log(str(f_uni))
        assert len(results) == 2 and results[0]["loss"] == 0.5

        # unicode metric names (must not crash)
        f_met = tmp_path / "unicode_metrics.log"
        f_met.write_text("\u00e9poque: 1, perte: 0.5\n\u00e9poque: 2, perte: 0.3\n",
                         encoding="utf-8")
        parse_log(str(f_met))

    def test_kv_fallback_warns_on_empty(self, tmp_path):
        f = tmp_path / "weird.log"
        f.write_text("This is just a plain text log\nNo metrics here\nJust words\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            records = parse_log(str(f))
            assert records == []
            assert len(w) == 1
            assert "kv" in str(w[0].message).lower()

    def test_oom_and_partial_fixture(self):
        """OOM and partial logs parse with extractable metrics."""
        oom = parse_log(str(FIXTURES / "oom_log.txt"))
        assert len(extract_metric_trajectory(oom, "loss")) >= 3
        partial = parse_log(str(FIXTURES / "partial_log.txt"))
        assert len(partial) >= 3

    @pytest.mark.parametrize("values,expected_diverged,reason_contains", [
        ([0.5], False, "insufficient"),
        ([1.0, 0.9, 0.8], False, "insufficient"),
        ([0.5, 0.4, 0.3, 0.2, 0.1], False, "healthy"),  # 5 values runs checks
        ([1.0, float("nan"), 0.8], True, "nan"),  # short with NaN
        ([1.0, float("inf")], True, "inf"),  # short with Inf
        ([1.0] * 30, True, "plateau"),  # constant values
    ])
    def test_divergence_edge_cases(self, values, expected_diverged, reason_contains):
        result = check_divergence(values, plateau_patience=20)
        assert result["diverged"] is expected_diverged
        assert reason_contains.lower() in result["reason"].lower()


# ---------------------------------------------------------------------------
# TestCLI — CLI tests for parse_logs.py and detect_divergence.py
# ---------------------------------------------------------------------------


class TestCLI:
    """CLI integration tests for parse_logs.py and detect_divergence.py."""

    def test_parse_logs_cli(self, run_main):
        """Basic parse and missing-args error."""
        r = run_main("parse_logs.py", str(FIXTURES / "sample_train_log.txt"))
        assert r.returncode == 0
        assert len(json.loads(r.stdout)) == 8
        r2 = run_main("parse_logs.py")
        assert r2.returncode == 1 and "Usage" in r2.stdout

    @pytest.mark.parametrize("args,expected_diverged", [
        (["[0.5, 0.4, 0.3, 0.2]"], False),
        (["[50, 60, 70, 80]", "--higher-is-better"], False),
        (["[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5]",
          "--model-category", "rl", "--explosion-threshold", "3.0"], True),
    ])
    def test_detect_divergence_cli(self, run_main, args, expected_diverged):
        r = run_main("detect_divergence.py", *args)
        assert r.returncode == 0
        assert json.loads(r.stdout)["diverged"] is expected_diverged

    def test_detect_divergence_cli_errors(self, run_main):
        """Invalid JSON, no args, and flag-only all produce errors."""
        for args in [("not-json",), (), ("--higher-is-better",)]:
            r = run_main("detect_divergence.py", *args)
            assert r.returncode == 1
