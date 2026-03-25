"""Tests for the ShinkaEvolve file-based handoff provider and evolve integration."""

import importlib.util
import json
import os
import re
import threading
import time
from pathlib import Path

import pytest


# Import the handoff provider directly (stdlib only, no ShinkaEvolve deps)
# We import the module file directly to avoid ShinkaEvolve's __init__.py
# which requires dotenv and other dependencies
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
        """query_file_handoff raises RuntimeError if handoff dir and env var not set."""
        old_dir = _mod.HANDOFF_DIR
        old_env = os.environ.get("SHINKA_HANDOFF_DIR")
        _mod.HANDOFF_DIR = None
        os.environ.pop("SHINKA_HANDOFF_DIR", None)
        try:
            with pytest.raises(RuntimeError, match="not configured"):
                query_file_handoff("m", "msg", "sys", timeout_seconds=1)
        finally:
            _mod.HANDOFF_DIR = old_dir
            if old_env is not None:
                os.environ["SHINKA_HANDOFF_DIR"] = old_env

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

    def test_query_patched_for_claude_code(self):
        plugin_root = Path(__file__).parent.parent
        query_py = plugin_root / "skills" / "evolve" / "ShinkaEvolve" / "shinka" / "llm" / "query.py"
        content = query_py.read_text()
        assert "_CLAUDE_CODE_MODE" in content
        assert "file_handoff_provider" in content
        assert "SHINKA_PROVIDER" in content

    def test_setup_script_exists_and_executable(self):
        plugin_root = Path(__file__).parent.parent
        setup = plugin_root / "scripts" / "setup_evolve.sh"
        assert setup.exists()
        assert os.access(str(setup), os.X_OK)

    def test_symlinks_point_to_submodule(self):
        plugin_root = Path(__file__).parent.parent
        for skill in ["shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect"]:
            p = plugin_root / "skills" / skill
            assert p.exists(), f"Missing: {skill}"
            assert (p / "SKILL.md").exists(), f"Missing SKILL.md in {skill}"



# ======================================================================
# TestQueryRouting
# ======================================================================


class TestQueryRouting:
    """Tests for the SHINKA_PROVIDER=claude_code query routing in patched query.py."""

    def test_query_function_has_claude_code_branch(self):
        """Both query() and query_async() have the _CLAUDE_CODE_MODE branch."""
        plugin_root = Path(__file__).parent.parent
        query_py = plugin_root / "skills" / "evolve" / "ShinkaEvolve" / "shinka" / "llm" / "query.py"
        content = query_py.read_text()
        # Should have at least 2 occurrences of the check (sync + async)
        assert content.count("if _CLAUDE_CODE_MODE:") >= 2

    def test_async_query_uses_run_in_executor(self):
        """query_async wraps the sync handoff in run_in_executor for async compat."""
        plugin_root = Path(__file__).parent.parent
        query_py = plugin_root / "skills" / "evolve" / "ShinkaEvolve" / "shinka" / "llm" / "query.py"
        content = query_py.read_text()
        assert "run_in_executor" in content


# ======================================================================
# TestMockShinkaEvolveSimulation
# ======================================================================


class TestMockShinkaEvolveSimulation:
    """Simulate what ShinkaEvolve does: write mutation requests, get responses.

    This tests the full file-based handoff pipeline without ShinkaEvolve's
    heavy dependencies (hydra, omegaconf, aiosqlite).
    """

    def test_mock_evolution_generation(self, tmp_path):
        """Simulate one generation of evolution: 3 mutation requests, 3 responses."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        # Simulate orchestrator that responds to each request
        def orchestrator():
            responded = set()
            for _ in range(30):  # poll for 3 seconds
                time.sleep(0.1)
                for f in pending_dir.glob("*.json"):
                    if f.name not in responded:
                        req = json.loads(f.read_text())
                        mutation = f"mutation for {req['id']}: add lr warmup"
                        (completed_dir / f.name).write_text(
                            json.dumps({"content": mutation})
                        )
                        responded.add(f.name)

        # Simulate ShinkaEvolve sending 3 mutation requests in parallel
        results = [None, None, None]

        def shinka_request(idx):
            results[idx] = query_file_handoff(
                "claude-opus", f"Mutate code variant {idx}",
                "You are an evolutionary code optimizer",
                timeout_seconds=5,
            )

        orch_thread = threading.Thread(target=orchestrator)
        threads = [threading.Thread(target=shinka_request, args=(i,)) for i in range(3)]

        orch_thread.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        orch_thread.join()

        # All 3 mutations should have responses
        for i, r in enumerate(results):
            assert r is not None, f"Mutation {i} got no response"
            assert "mutation for" in r["content"], f"Mutation {i} has wrong content"
            assert "lr warmup" in r["content"]

    def test_request_json_format(self, tmp_path):
        """Verify the request JSON has the fields ShinkaEvolve's prompt sampler expects."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond():
            time.sleep(0.3)
            files = list(pending_dir.glob("*.json"))
            req = json.loads(files[0].read_text())
            # Verify request structure
            assert "id" in req
            assert "system_msg" in req
            assert "user_msg" in req
            assert "model_name" in req
            (completed_dir / files[0].name).write_text(json.dumps({"content": "ok"}))

        t = threading.Thread(target=respond)
        t.start()
        query_file_handoff("test-model", "user prompt", "system prompt", timeout_seconds=5)
        t.join()

    def test_response_has_query_result_fields(self, tmp_path):
        """Verify the response dict has all fields expected by QueryResult."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond():
            time.sleep(0.3)
            for f in pending_dir.glob("*.json"):
                (completed_dir / f.name).write_text(json.dumps({"content": "code"}))

        t = threading.Thread(target=respond)
        t.start()
        result = query_file_handoff("model", "msg", "sys", timeout_seconds=5)
        t.join()

        # All QueryResult-compatible fields should be present
        assert "content" in result
        assert "cost" in result
        assert "model_name" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert result["cost"] == 0.0  # Claude Code handles billing


# ======================================================================
# TestFileHandoffEdgeCases
# ======================================================================


_PLUGIN_ROOT = Path(__file__).parent.parent


class TestFileHandoffEdgeCases:
    """Edge cases and error paths in the file-based handoff provider."""

    def test_malformed_json_response(self, tmp_path):
        """Provider retries when completed file contains invalid JSON."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond():
            time.sleep(0.3)
            files = list(pending_dir.glob("*.json"))
            resp_path = completed_dir / files[0].name
            # First write: malformed JSON (provider should retry via continue)
            resp_path.write_text("{invalid json!!!")
            time.sleep(1.5)
            # Second write: valid JSON (provider picks it up on next poll)
            resp_path.write_text(json.dumps({"content": "recovered"}))

        t = threading.Thread(target=respond)
        t.start()
        result = query_file_handoff("m", "msg", "sys", timeout_seconds=5)
        t.join()

        assert result["content"] == "recovered"

    def test_empty_response_file(self, tmp_path):
        """Provider retries when completed file is empty (0 bytes)."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond():
            time.sleep(0.3)
            files = list(pending_dir.glob("*.json"))
            resp_path = completed_dir / files[0].name
            # Write empty file first
            resp_path.write_text("")
            time.sleep(1.5)
            # Then write valid response
            resp_path.write_text(json.dumps({"content": "ok"}))

        t = threading.Thread(target=respond)
        t.start()
        result = query_file_handoff("m", "msg", "sys", timeout_seconds=5)
        t.join()

        assert result["content"] == "ok"

    def test_missing_content_field(self, tmp_path):
        """Response without 'content' key returns empty string (line 76 behavior)."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond():
            time.sleep(0.3)
            files = list(pending_dir.glob("*.json"))
            # Valid JSON but no "content" key
            (completed_dir / files[0].name).write_text(
                json.dumps({"result": "no content field here"})
            )

        t = threading.Thread(target=respond)
        t.start()
        result = query_file_handoff("m", "msg", "sys", timeout_seconds=5)
        t.join()

        assert result["content"] == ""
        assert result["cost"] == 0.0

    def test_timeout_zero(self, tmp_path):
        """timeout_seconds=0 raises TimeoutError immediately (no polling)."""
        set_handoff_dir(str(tmp_path))

        with pytest.raises(TimeoutError):
            query_file_handoff("m", "msg", "sys", timeout_seconds=0)

        # Pending file should still be cleaned up
        pending_files = list((tmp_path / "evolve" / "pending").glob("*.json"))
        assert len(pending_files) == 0

    def test_response_deleted_before_read(self, tmp_path):
        """Provider handles FileNotFoundError if completed file vanishes."""
        set_handoff_dir(str(tmp_path))
        pending_dir = tmp_path / "evolve" / "pending"
        completed_dir = tmp_path / "evolve" / "completed"

        def respond():
            time.sleep(0.3)
            files = list(pending_dir.glob("*.json"))
            resp_path = completed_dir / files[0].name
            # Create and immediately delete (simulates race condition)
            resp_path.write_text("will be deleted")
            resp_path.unlink()
            time.sleep(1.5)
            # Then write a stable response
            resp_path.write_text(json.dumps({"content": "stable"}))

        t = threading.Thread(target=respond)
        t.start()
        result = query_file_handoff("m", "msg", "sys", timeout_seconds=5)
        t.join()

        assert result["content"] == "stable"

    def test_idempotent_set_handoff_dir(self, tmp_path):
        """Calling set_handoff_dir() twice on the same path is safe."""
        result1 = set_handoff_dir(str(tmp_path))
        result2 = set_handoff_dir(str(tmp_path))

        assert result1 == result2
        assert (tmp_path / "evolve" / "pending").is_dir()
        assert (tmp_path / "evolve" / "completed").is_dir()

    def test_env_var_auto_detection(self, tmp_path):
        """SHINKA_HANDOFF_DIR env var auto-configures handoff when HANDOFF_DIR is None."""
        old_dir = _mod.HANDOFF_DIR
        old_env = os.environ.get("SHINKA_HANDOFF_DIR")
        _mod.HANDOFF_DIR = None
        os.environ["SHINKA_HANDOFF_DIR"] = str(tmp_path)

        try:
            pending_dir = tmp_path / "evolve" / "pending"
            completed_dir = tmp_path / "evolve" / "completed"

            def respond():
                time.sleep(0.3)
                for f in pending_dir.glob("*.json"):
                    (completed_dir / f.name).write_text(
                        json.dumps({"content": "auto-detected"})
                    )

            t = threading.Thread(target=respond)
            t.start()
            result = query_file_handoff("m", "msg", "sys", timeout_seconds=5)
            t.join()

            assert result["content"] == "auto-detected"
            # HANDOFF_DIR should now be set
            assert _mod.HANDOFF_DIR is not None
        finally:
            _mod.HANDOFF_DIR = old_dir
            if old_env is None:
                os.environ.pop("SHINKA_HANDOFF_DIR", None)
            else:
                os.environ["SHINKA_HANDOFF_DIR"] = old_env


# ======================================================================
# TestQueryRoutingRuntime
# ======================================================================


class TestQueryRoutingRuntime:
    """Runtime tests for SHINKA_PROVIDER env var routing in query.py.

    query.py imports pydantic (not in our deps), so we can't import it
    directly. Instead we extract and evaluate the routing expression.
    """

    def _get_mode_expression(self):
        """Extract the _CLAUDE_CODE_MODE expression from query.py source."""
        query_py = (
            _PLUGIN_ROOT / "skills" / "evolve" / "ShinkaEvolve"
            / "shinka" / "llm" / "query.py"
        )
        for line in query_py.read_text().splitlines():
            if "_CLAUDE_CODE_MODE" in line and "=" in line and "if" not in line:
                # Line like: _CLAUDE_CODE_MODE = os.environ.get("SHINKA_PROVIDER", "").lower() == "claude_code"
                _, expr = line.split("=", 1)
                return expr.strip()
        pytest.fail("_CLAUDE_CODE_MODE assignment not found in query.py")

    def test_env_var_activates_claude_code_mode(self):
        """With SHINKA_PROVIDER=claude_code, the routing expression is True."""
        expr = self._get_mode_expression()
        old = os.environ.get("SHINKA_PROVIDER")
        try:
            os.environ["SHINKA_PROVIDER"] = "claude_code"
            assert eval(expr) is True  # noqa: S307
        finally:
            if old is None:
                os.environ.pop("SHINKA_PROVIDER", None)
            else:
                os.environ["SHINKA_PROVIDER"] = old

    def test_env_var_unset_disables_mode(self):
        """Without SHINKA_PROVIDER, the routing expression is False."""
        expr = self._get_mode_expression()
        old = os.environ.get("SHINKA_PROVIDER")
        try:
            os.environ.pop("SHINKA_PROVIDER", None)
            assert eval(expr) is False  # noqa: S307
        finally:
            if old is not None:
                os.environ["SHINKA_PROVIDER"] = old

    def test_env_var_case_insensitive(self):
        """SHINKA_PROVIDER=CLAUDE_CODE (uppercase) also activates mode."""
        expr = self._get_mode_expression()
        old = os.environ.get("SHINKA_PROVIDER")
        try:
            os.environ["SHINKA_PROVIDER"] = "CLAUDE_CODE"
            assert eval(expr) is True  # noqa: S307
        finally:
            if old is None:
                os.environ.pop("SHINKA_PROVIDER", None)
            else:
                os.environ["SHINKA_PROVIDER"] = old


# ======================================================================
# TestShinkaSkillContent
# ======================================================================


class TestShinkaSkillContent:
    """Validate shinka skill SKILL.md content and contracts."""

    def _parse_frontmatter(self, skill_path):
        """Extract YAML frontmatter from a SKILL.md file."""
        content = skill_path.read_text()
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        assert match, f"No frontmatter in {skill_path}"
        fm = {}
        for line in match.group(1).strip().split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                fm[key.strip()] = val.strip()
        return fm, content

    @pytest.mark.parametrize("skill", [
        "shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect",
    ])
    def test_shinka_skills_have_required_frontmatter(self, skill):
        """All 4 shinka skills have name and description in frontmatter."""
        skill_path = _PLUGIN_ROOT / "skills" / skill / "SKILL.md"
        fm, _ = self._parse_frontmatter(skill_path)
        assert "name" in fm, f"{skill} missing 'name'"
        assert fm["name"] == skill, f"{skill} name mismatch: {fm['name']}"
        assert "description" in fm, f"{skill} missing 'description'"
        assert len(fm["description"]) > 20, f"{skill} description too short"

    def test_shinka_setup_has_evaluator_contract(self):
        """shinka-setup SKILL.md references evaluator scoring fields."""
        content = (_PLUGIN_ROOT / "skills" / "shinka-setup" / "SKILL.md").read_text()
        # The evaluator contract uses combined_score for ranking
        assert "EVOLVE-BLOCK" in content or "evolve" in content.lower()
        assert "evaluate.py" in content
        assert "initial" in content

    def test_shinka_run_has_required_cli_args(self):
        """shinka-run SKILL.md references --results_dir and generation count."""
        content = (_PLUGIN_ROOT / "skills" / "shinka-run" / "SKILL.md").read_text()
        assert "results_dir" in content
        assert "generation" in content.lower()

    def test_shinka_inspect_has_load_programs(self):
        """shinka-inspect SKILL.md references load_programs_to_df."""
        content = (_PLUGIN_ROOT / "skills" / "shinka-inspect" / "SKILL.md").read_text()
        assert "load_programs_to_df" in content
        assert "combined_score" in content

    def test_shinka_convert_has_evolve_blocks(self):
        """shinka-convert SKILL.md references EVOLVE-BLOCK markers."""
        content = (_PLUGIN_ROOT / "skills" / "shinka-convert" / "SKILL.md").read_text()
        assert "EVOLVE-BLOCK" in content or "evolve" in content.lower()
        assert "evaluate.py" in content


# ======================================================================
# TestSetupScriptValidation
# ======================================================================


class TestSetupScriptValidation:
    """Validate setup_evolve.sh content and safety guards."""

    def test_setup_script_references_all_symlinks(self):
        """setup_evolve.sh creates symlinks for all 4 shinka skills."""
        content = (_PLUGIN_ROOT / "scripts" / "setup_evolve.sh").read_text()
        for skill in ["shinka-setup", "shinka-convert", "shinka-run", "shinka-inspect"]:
            assert skill in content, f"Missing symlink for {skill}"
        # Verify relative symlink target path
        assert "evolve/ShinkaEvolve/skills/" in content

    def test_setup_script_has_safety_check(self):
        """setup_evolve.sh doesn't overwrite existing directories."""
        content = (_PLUGIN_ROOT / "scripts" / "setup_evolve.sh").read_text()
        # Script checks for existing directory and skips
        assert "-d" in content, "Missing directory existence check"
        assert "Skipping" in content or "skip" in content.lower(), \
            "No skip message for existing directories"
