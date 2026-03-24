"""Tests for the ShinkaEvolve file-based handoff provider and evolve integration."""

import json
import threading
import time
from pathlib import Path

import pytest


# Import the handoff provider directly (stdlib only, no ShinkaEvolve deps)
# We import the module file directly to avoid ShinkaEvolve's __init__.py
# which requires dotenv and other dependencies
import importlib.util
_provider_path = str(Path(__file__).parent.parent / "skills" / "evolve" / "ShinkaEvolve" / "shinka" / "llm" / "file_handoff_provider.py")
_spec = importlib.util.spec_from_file_location("file_handoff_provider", _provider_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
set_handoff_dir = _mod.set_handoff_dir
query_file_handoff = _mod.query_file_handoff


# ======================================================================
# TestFileHandoffProvider
# ======================================================================


class TestFileHandoffProvider:
    """Tests for the file-based LLM handoff between ShinkaEvolve and Claude Code."""

    def test_set_handoff_dir(self, tmp_path):
        """set_handoff_dir creates pending/ and completed/ subdirectories."""
        result = set_handoff_dir(str(tmp_path))
        assert (tmp_path / "evolve" / "pending").is_dir()
        assert (tmp_path / "evolve" / "completed").is_dir()
        assert result == tmp_path / "evolve"

    def test_request_written_to_pending(self, tmp_path):
        """query_file_handoff writes a JSON request to the pending directory."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        # Simulate orchestrator responding after a short delay
        def respond():
            time.sleep(0.5)
            files = list(pending_dir.glob("*.json"))
            assert len(files) == 1
            request = json.loads(files[0].read_text())
            assert request["system_msg"] == "You are a code optimizer"
            assert request["user_msg"] == "Improve this function"
            assert request["model_name"] == "test-model"
            # Write response
            response_path = completed_dir / files[0].name
            response_path.write_text(json.dumps({"content": "Here is the improved code"}))

        t = threading.Thread(target=respond)
        t.start()

        result = query_file_handoff(
            "test-model", "Improve this function", "You are a code optimizer",
            timeout_seconds=5,
        )
        t.join()

        assert result["content"] == "Here is the improved code"
        assert result["model_name"] == "test-model"
        assert result["cost"] == 0.0

    def test_timeout_raises(self, tmp_path):
        """query_file_handoff raises TimeoutError when no response arrives."""
        set_handoff_dir(str(tmp_path))

        with pytest.raises(TimeoutError):
            query_file_handoff(
                "test-model", "msg", "sys",
                timeout_seconds=1,
            )

        # Pending file should be cleaned up after timeout
        pending_files = list((tmp_path / "evolve" / "pending").glob("*.json"))
        assert len(pending_files) == 0

    def test_cleanup_after_response(self, tmp_path):
        """Both pending and completed files are cleaned up after a successful handoff."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond():
            time.sleep(0.3)
            files = list(pending_dir.glob("*.json"))
            response_path = completed_dir / files[0].name
            response_path.write_text(json.dumps({"content": "response"}))

        t = threading.Thread(target=respond)
        t.start()
        query_file_handoff("m", "msg", "sys", timeout_seconds=5)
        t.join()

        assert len(list(pending_dir.glob("*.json"))) == 0
        assert len(list(completed_dir.glob("*.json"))) == 0

    def test_not_configured_raises(self):
        """query_file_handoff raises RuntimeError if handoff dir not set."""
        old = _mod.HANDOFF_DIR
        _mod.HANDOFF_DIR = None
        try:
            with pytest.raises(RuntimeError, match="not configured"):
                query_file_handoff("m", "msg", "sys", timeout_seconds=1)
        finally:
            _mod.HANDOFF_DIR = old

    def test_multiple_concurrent_requests(self, tmp_path):
        """Multiple handoff requests can be in-flight simultaneously."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond_all():
            time.sleep(0.5)
            for f in pending_dir.glob("*.json"):
                req = json.loads(f.read_text())
                (completed_dir / f.name).write_text(
                    json.dumps({"content": f"response to {req['id']}"})
                )

        results = [None, None]

        def run_query(idx, msg):
            results[idx] = query_file_handoff("m", msg, "sys", timeout_seconds=5)

        t_respond = threading.Thread(target=respond_all)
        t1 = threading.Thread(target=run_query, args=(0, "query-A"))
        t2 = threading.Thread(target=run_query, args=(1, "query-B"))

        t1.start()
        t2.start()
        t_respond.start()
        t1.join()
        t2.join()
        t_respond.join()

        assert results[0] is not None
        assert results[1] is not None
        assert results[0]["content"].startswith("response to")
        assert results[1]["content"].startswith("response to")


# ======================================================================
# TestEvolveSkillStructure
# ======================================================================


class TestEvolveSkillStructure:
    """Verify evolve skill and ShinkaEvolve submodule are correctly structured."""

    def test_evolve_skill_exists(self):
        plugin_root = Path(__file__).parent.parent
        assert (plugin_root / "skills" / "evolve" / "SKILL.md").exists()

    def test_shinka_submodule_exists(self):
        plugin_root = Path(__file__).parent.parent
        shinka_dir = plugin_root / "skills" / "evolve" / "ShinkaEvolve"
        assert shinka_dir.is_dir()
        assert (shinka_dir / "shinka" / "llm" / "query.py").exists()
        assert (shinka_dir / "shinka" / "llm" / "file_handoff_provider.py").exists()

    def test_shinka_skills_exist(self):
        plugin_root = Path(__file__).parent.parent
        skills_dir = plugin_root / "skills" / "evolve" / "ShinkaEvolve" / "skills"
        for skill in ["shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect"]:
            assert (skills_dir / skill / "SKILL.md").exists(), f"Missing: {skill}"

    def test_query_patched_for_claude_code(self):
        plugin_root = Path(__file__).parent.parent
        query_py = plugin_root / "skills" / "evolve" / "ShinkaEvolve" / "shinka" / "llm" / "query.py"
        content = query_py.read_text()
        assert "_CLAUDE_CODE_MODE" in content
        assert "file_handoff_provider" in content
        assert "SHINKA_PROVIDER" in content
