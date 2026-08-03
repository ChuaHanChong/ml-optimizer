"""Smoke tests for the CartPole PPO fixture — proves the RL path live.

Covers: kv parsing of episodic-return logs, the sb3 pipe-table parser (B7),
environment-step budget (--total_timesteps), rollout eval symmetry (B9 shape),
and healthy divergence verdicts on real noisy returns including a
negative-to-positive shifted curve (B1 shift-invariance).

Needs torch + gymnasium to run the fixture; those tests skip cleanly without.
"""

import math
import subprocess
import sys

import pytest

from conftest import FIXTURES

from detect_divergence import check_divergence, get_thresholds_for_category
from parse_logs import detect_format, extract_metric_trajectory, parse_log

RL_FIXTURE = FIXTURES / "rl-cartpole"
TRAIN_PPO = RL_FIXTURE / "train_ppo.py"
SB3_LOG = RL_FIXTURE / "sb3_train_log.txt"

TOTAL_TIMESTEPS = 8192  # 16 updates of 512 steps — well under a minute on CPU
NUM_UPDATES = 16


def test_fixture_exists_with_budget_flags():
    """The fixture exposes the canonical RL flags (--seed / --total_timesteps / --lr)."""
    assert TRAIN_PPO.is_file(), f"missing fixture: {TRAIN_PPO}"
    text = TRAIN_PPO.read_text()
    for flag in ("--seed", "--total_timesteps", "--lr"):
        assert flag in text, f"train_ppo.py missing {flag} flag"


@pytest.fixture(scope="module")
def cartpole_log(tmp_path_factory):
    """Run the fixture once per module; return the captured kv train log."""
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    result = subprocess.run(
        [sys.executable, str(TRAIN_PPO), "--seed", "1",
         "--total_timesteps", str(TOTAL_TIMESTEPS), "--lr", "2.5e-4"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"train_ppo failed: {result.stderr}"
    log = tmp_path_factory.mktemp("rl") / "train.log"
    log.write_text(result.stdout)
    return log


@pytest.mark.slow
class TestCartPoleRun:
    """Live run of the fixture: budget, kv logging, eval symmetry, divergence."""

    def test_runs_n_updates_and_logs_kv(self, cartpole_log):
        lines = cartpole_log.read_text().strip().split("\n")
        assert detect_format(lines) == "kv"
        records = parse_log(str(cartpole_log))
        updates = [r for r in records if "update" in r]
        assert len(updates) == NUM_UPDATES
        # environment-step budget honored exactly (fixed_step_budget semantics)
        assert updates[-1]["global_step"] == float(TOTAL_TIMESTEPS)

    def test_episode_return_trajectory_extracted(self, cartpole_log):
        records = parse_log(str(cartpole_log))
        returns = extract_metric_trajectory(records, "episode_return")
        assert len(returns) == NUM_UPDATES
        assert all(math.isfinite(v) for v in returns)

    def test_final_eval_symmetry(self, cartpole_log):
        """B9 shape: final deterministic eval reports mean+std over a fixed episode count."""
        records = parse_log(str(cartpole_log))
        final = records[-1]
        assert "final_eval_mean" in final and "final_eval_std" in final
        assert final["eval_episodes"] == 10.0
        assert math.isfinite(final["final_eval_mean"])

    def test_divergence_healthy_on_noisy_returns(self, cartpole_log):
        records = parse_log(str(cartpole_log))
        returns = extract_metric_trajectory(records, "episode_return")
        verdict = check_divergence(returns, lower_is_better=False,
                                   **get_thresholds_for_category("rl"))
        assert verdict["diverged"] is False, verdict

    def test_divergence_shift_invariant_negative_to_positive(self, cartpole_log):
        """B1: midrange-shifting real noisy returns below zero must not read as a crash."""
        records = parse_log(str(cartpole_log))
        returns = extract_metric_trajectory(records, "episode_return")
        assert max(returns) > min(returns)
        mid = (max(returns) + min(returns)) / 2
        shifted = [v - mid for v in returns]
        assert min(shifted) < 0 < max(shifted)  # genuinely negative-to-positive
        verdict = check_divergence(shifted, lower_is_better=False,
                                   **get_thresholds_for_category("rl"))
        assert verdict["diverged"] is False, verdict


MULTI_TASK_NAMES = ["pole_short", "pole_long"]


@pytest.fixture(scope="module")
def cartpole_multitask_log(tmp_path_factory):
    """Run the fixture in multi-task eval mode; return the captured kv log."""
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    result = subprocess.run(
        [sys.executable, str(TRAIN_PPO), "--seed", "1",
         "--total_timesteps", str(TOTAL_TIMESTEPS), "--lr", "2.5e-4",
         "--eval_tasks", ",".join(MULTI_TASK_NAMES)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"train_ppo failed: {result.stderr}"
    log = tmp_path_factory.mktemp("rl-mt") / "train.log"
    log.write_text(result.stdout)
    return log


def test_fixture_exposes_eval_tasks_flag():
    """The fixture accepts --eval_tasks (no torch/gym needed to check the source)."""
    assert "--eval_tasks" in TRAIN_PPO.read_text()


@pytest.mark.slow
class TestMultiTaskEval:
    """The parse -> aggregate seam on real eval stdout."""

    def test_per_task_keys_parsed_from_real_output(self, cartpole_multitask_log):
        records = parse_log(str(cartpole_multitask_log))
        final = records[-1]
        for task in MULTI_TASK_NAMES:
            key = f"final_eval_mean_{task}"
            assert key in final, f"missing {key} in {sorted(final)}"
            assert math.isfinite(final[key])
        # the whole point of using distinct physics variants: confirm they
        # actually produce different behavior, not silently-identical numbers
        values = [final[f"final_eval_mean_{t}"] for t in MULTI_TASK_NAMES]
        assert len(set(values)) > 1, f"task variants produced identical values: {values}"

    def test_aggregates_computed_from_parsed_metrics(self, cartpole_multitask_log):
        from result_analyzer import aggregate_task_metrics
        final = parse_log(str(cartpole_multitask_log))[-1]
        out = aggregate_task_metrics(final, "final_eval_mean", MULTI_TASK_NAMES)
        assert out["warnings"] == [], out["warnings"]
        vals = [final[f"final_eval_mean_{t}"] for t in MULTI_TASK_NAMES]
        assert out["aggregates"]["final_eval_mean"] == pytest.approx(sum(vals) / len(vals))
        assert out["aggregates"]["final_eval_mean_worst"] == pytest.approx(min(vals))

    def test_undeclared_task_detected_on_real_output(self, cartpole_multitask_log):
        """Declaring a task the eval never reported is caught, not silently averaged."""
        from result_analyzer import aggregate_task_metrics
        final = parse_log(str(cartpole_multitask_log))[-1]
        out = aggregate_task_metrics(final, "final_eval_mean", MULTI_TASK_NAMES + ["never_ran"])
        assert any("never_ran" in w for w in out["warnings"])

    def test_single_task_mode_still_emits_flat_key(self, cartpole_log):
        """Default (no --eval_tasks) output is unchanged — no per-task keys."""
        final = parse_log(str(cartpole_log))[-1]
        assert "final_eval_mean" in final
        assert not any(k.startswith("final_eval_mean_") for k in final)


class TestSB3Parsing:
    """B7: the sb3 pipe-table format parses and auto-detects (static sample, no deps)."""

    def test_sb3_forced_parse(self):
        records = parse_log(str(SB3_LOG), "sb3")
        rew = extract_metric_trajectory(records, "ep_rew_mean")
        assert rew == [22.5, 41.2, 87.6]
        steps = extract_metric_trajectory(records, "total_timesteps")
        assert steps == [2048.0, 4096.0, 6144.0]
        # section prefixes (rollout/, train/) stripped from keys
        assert any("value_loss" in r for r in records)
        assert not any(any("/" in k for k in r) for r in records)

    def test_sb3_autodetect(self):
        lines = SB3_LOG.read_text().strip().split("\n")
        assert detect_format(lines) == "sb3"
