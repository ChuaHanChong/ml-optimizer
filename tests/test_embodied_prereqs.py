"""Batch D tests — prerequisites + VLA data (embodied generality plan).

Covers: no-dataset RL path contracts (Task 18), robot demonstration formats
(Task 19), install safety (Task 20), env-aware import validation + headless
sim env forwarding (Task 21).
"""

import json
import sys
from pathlib import Path

import pytest

from conftest import SKILLS_DIR, REFERENCES_DIR

from prerequisites_check import (
    bulk_install_command,
    check_missing_packages,
    detect_dataset_format,
    pip_name,
    validate_data_path,
)
from implement_utils import validate_imports


def _read(path):
    return Path(path).read_text()


# ===========================================================================
# Task 18 — TestNoDatasetRlPath (D1)
# ===========================================================================


class TestNoDatasetRlPath:
    """D1 — Phase 0 Q11 no-dataset option, model_category threading, prereq short-circuit."""

    def test_phase0_q11_no_dataset_option(self):
        content = _read(REFERENCES_DIR / "phase-0-discovery.md")
        assert "no dataset — training interacts with a simulator/RL environment" in content

    def test_phase0_summary_confirms_model_category(self):
        content = _read(REFERENCES_DIR / "phase-0-discovery.md")
        assert "Model category: {model_category}" in content

    def test_prerequisites_skill_rl_short_circuit(self):
        content = _read(SKILLS_DIR / "prerequisites" / "SKILL.md")
        assert "rl_environment" in content
        assert "Skip Steps 2, 3, and 4" in content

    def test_prerequisites_skill_q_renumbering(self):
        content = _read(SKILLS_DIR / "prerequisites" / "SKILL.md")
        assert "Phase 0 Q11: `train_data_path`" in content
        assert "Phase 0 Q12: `env_manager`" in content
        assert "Q10" not in content

    def test_phase1_rl_import_trigger_list(self):
        content = _read(REFERENCES_DIR / "phase-1-understand.md")
        for mod in ("mujoco", "dm_control", "habitat", "isaaclab", "omni.isaac.lab",
                    "rsl_rl", "skrl", "brax", "sample_factory", "robosuite"):
            assert f"`{mod}`" in content, f"missing RL import trigger: {mod}"

    def test_phase2_dispatch_gains_exp_root_and_model_category(self):
        content = _read(REFERENCES_DIR / "phase-2-prerequisites.md")
        assert "exp_root: {exp_root}" in content
        assert "model_category: {model_category}" in content


# ===========================================================================
# Task 19 — TestDemoFormatDetection / TestSmallDatasetShiftFix (D2, D3)
# ===========================================================================


class TestDemoFormatDetection:
    """D2 — robot demonstration format patterns + zarr store validation."""

    @pytest.mark.parametrize("source,expected", [
        ("from lerobot.common.datasets.lerobot_dataset import LeRobotDataset\n"
         "ds = LeRobotDataset('lerobot/pusht')\n", "lerobot"),
        ("import tensorflow_datasets as tfds\n"
         "builder = tfds.builder_from_directory('/data/rlds_task')\n", "rlds"),
        ("import zarr\nroot = zarr.open('/data/buffer.zarr', mode='r')\n", "zarr"),
        ("from robomimic.utils.dataset import SequenceDataset\n"
         "ds = SequenceDataset(hdf5_path='demo.hdf5')\n", "robomimic"),
    ])
    def test_demo_format_detection(self, tmp_path, source, expected):
        script = tmp_path / "train.py"
        script.write_text(source)
        assert detect_dataset_format(str(script))["format"] == expected

    def test_validate_zarr_top_level_marker(self, tmp_path):
        store = tmp_path / "replay_buffer.zarr"
        store.mkdir()
        (store / ".zgroup").write_text('{"zarr_format": 2}')
        assert validate_data_path(str(store), "zarr")["format_matches"] is True

    def test_validate_zarr_nested_marker(self, tmp_path):
        store = tmp_path / "buffer.zarr"
        (store / "data").mkdir(parents=True)
        (store / "data" / ".zarray").write_text('{"zarr_format": 2}')
        assert validate_data_path(str(store), "zarr")["format_matches"] is True

    def test_validate_zarr_no_marker(self, tmp_path):
        d = tmp_path / "not_zarr"
        d.mkdir()
        (d / "file.txt").write_text("x")
        result = validate_data_path(str(d), "zarr")
        assert result["format_matches"] is False
        assert result["errors"]

    def test_validate_robomimic_hdf5_file(self, tmp_path):
        f = tmp_path / "demo.hdf5"
        f.write_bytes(b"\x89HDF\r\n")
        assert validate_data_path(str(f), "robomimic")["format_matches"] is True

    def test_dataset_formats_reference_documents_demo_formats(self):
        content = _read(SKILLS_DIR / "prerequisites" / "references" / "dataset-formats.md")
        assert "## Robot Demonstration Formats" in content
        for term in ("LeRobot", "RLDS", "robomimic", "zarr", ".zgroup"):
            assert term in content, f"missing demo format doc: {term}"


class TestSmallDatasetShiftFix:
    """D3 — transitions counting + IL low-data techniques in research SKILL."""

    def test_research_counts_transitions_not_episodes(self):
        content = _read(SKILLS_DIR / "research" / "SKILL.md")
        assert "count **transitions**" in content
        assert "Skip this check for online RL" in content

    def test_research_il_low_data_techniques(self):
        content = _read(SKILLS_DIR / "research" / "SKILL.md")
        for term in ("action chunking", "observation augmentation",
                     "pretrained visual encoders", "co-training"):
            assert term in content, f"missing IL low-data technique: {term}"


# ===========================================================================
# Task 20 — TestInstallSafety (D4, D5)
# ===========================================================================


class TestInstallSafety:
    """D4/D5 — --prune removal, NEVER_AUTO_INSTALL, pip-name fixes, dry-run contracts."""

    def test_conda_env_update_has_no_prune(self, tmp_path):
        (tmp_path / "environment.yml").write_text("name: x\ndependencies:\n  - numpy\n")
        cmd = bulk_install_command(str(tmp_path), "conda")["install_command"]
        assert "--prune" not in cmd
        assert "conda env update" in cmd

    def test_conda_env_update_yaml_variant_no_prune(self, tmp_path):
        (tmp_path / "environment.yaml").write_text("name: x\ndependencies:\n  - numpy\n")
        cmd = bulk_install_command(str(tmp_path), "conda")["install_command"]
        assert "--prune" not in cmd

    def test_never_auto_install_set_complete(self):
        from prerequisites_check import NEVER_AUTO_INSTALL
        for name in ("omni", "carb", "pxr", "isaacsim", "isaaclab",
                     "habitat", "habitat_sim", "rclpy", "rospy", "warp"):
            assert name in NEVER_AUTO_INSTALL, f"missing NEVER_AUTO_INSTALL entry: {name}"
            assert NEVER_AUTO_INSTALL[name], f"empty guidance for {name}"

    def test_check_packages_routes_manual_install_required(self):
        result = check_missing_packages(
            ["omni", "definitely_not_a_real_module_xyz"],
            python_executable=sys.executable,
        )
        assert result["manual_install_required"].get("omni")
        assert "omni" not in result["missing"]
        assert "definitely_not_a_real_module_xyz" in result["missing"]

    @pytest.mark.parametrize("import_name,expected_pip", [
        ("stable_baselines3", "stable-baselines3"),
        ("sample_factory", "sample-factory"),
        ("rsl_rl", "rsl-rl-lib"),
        ("tensorflow_datasets", "tensorflow-datasets"),
    ])
    def test_import_to_package_corrections(self, import_name, expected_pip):
        assert pip_name(import_name) == expected_pip

    def test_prereq_skill_manual_install_and_dry_run_contracts(self):
        content = _read(SKILLS_DIR / "prerequisites" / "SKILL.md")
        assert "manual_install_required" in content
        assert "Exit code 124 is a dry-run PASS" in content
        assert "300-600s" in content
        assert "do NOT auto-create" in content
        assert "only ever done behind an explicit AskUserQuestion" in content


# ===========================================================================
# Task 21 — TestEnvAwareImportValidation / TestHeadlessSimEnvForwarding (D6, D7)
# ===========================================================================


class TestEnvAwareImportValidation:
    """D6 — validate_imports project-env python + simulator allow-list."""

    def test_sim_runtime_import_skipped_env_dependent(self, tmp_path):
        mod = tmp_path / "isaac_train.py"
        mod.write_text("import omni.isaac.core\n")
        result = validate_imports(str(mod), str(tmp_path))
        assert result["passed"] is True
        assert result["status"] == "skipped_env_dependent"
        assert result["error"]  # the ModuleNotFoundError line is preserved

    @pytest.mark.parametrize("sim_import", ["isaacgym", "isaaclab", "habitat_sim", "carb"])
    def test_all_sim_runtime_modules_skipped(self, tmp_path, sim_import):
        mod = tmp_path / "train.py"
        mod.write_text(f"import {sim_import}\n")
        assert validate_imports(str(mod), str(tmp_path))["status"] == "skipped_env_dependent"

    def test_non_sim_failure_still_fails(self, tmp_path):
        mod = tmp_path / "bad.py"
        mod.write_text("import nonexistent_module_xyz_123\n")
        result = validate_imports(str(mod), str(tmp_path))
        assert result["passed"] is False
        assert result["status"] == "failed"

    def test_pass_reports_status_passed(self, tmp_path):
        mod = tmp_path / "ok.py"
        mod.write_text("x = 1\n")
        result = validate_imports(str(mod), str(tmp_path),
                                  python_executable=sys.executable)
        assert result["passed"] is True
        assert result["status"] == "passed"

    def test_implement_skill_documents_env_python_and_allow_list(self):
        content = _read(SKILLS_DIR / "implement" / "SKILL.md")
        assert "python_executable" in content
        assert "skipped_env_dependent" in content


class TestHeadlessSimEnvForwarding:
    """D7 — baseline sim_env detection, experiment forwarding, env_vars CLI arg."""

    def test_baseline_skill_records_sim_env(self):
        content = _read(SKILLS_DIR / "baseline" / "SKILL.md")
        assert "profiling.sim_env" in content
        assert "MUJOCO_GL=egl" in content

    def test_experiment_skill_forwards_sim_env(self):
        content = _read(SKILLS_DIR / "experiment" / "SKILL.md")
        assert "profiling.sim_env" in content
        assert "env_vars_json" in content

    def test_cli_env_vars_arg(self, run_main, tmp_path):
        r = run_main(
            "experiment_setup.py", str(tmp_path), "echo hi", "0", "{}",
            "round-1-hp", '{"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}',
        )
        assert r.returncode == 0
        out = json.loads(r.stdout)
        content = Path(out["script_path"]).read_text()
        assert "export MUJOCO_GL=egl" in content
        assert "export PYOPENGL_PLATFORM=egl" in content

    def test_cli_invalid_env_vars_json_exits_1(self, run_main, tmp_path):
        r = run_main(
            "experiment_setup.py", str(tmp_path), "echo hi", "0", "{}",
            "round-1-hp", "not-json",
        )
        assert r.returncode == 1
