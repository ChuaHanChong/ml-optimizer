"""Tests for gitnexus_utils.py — GitNexus code-graph indexing wrapper."""

import json
import os
import subprocess
import sys

from conftest import SCRIPTS_DIR

import gitnexus_utils
from gitnexus_utils import (
    available,
    index,
    is_indexed,
    mcp_registered,
    _exclude_gitnexus_artifact,
)


SCRIPT = str(SCRIPTS_DIR / "gitnexus_utils.py")


# ---------------------------------------------------------------------------
# TestAvailable
# ---------------------------------------------------------------------------

class TestAvailable:
    """The available() check reflects whether the gitnexus binary is on PATH."""

    def test_true_when_on_path(self, monkeypatch):
        """available() returns True when shutil.which resolves the binary."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")
        assert available() is True

    def test_false_when_absent(self, monkeypatch):
        """available() returns False when shutil.which finds nothing."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: None)
        assert available() is False

    def test_queries_gitnexus_binary(self, monkeypatch):
        """available() probes for the binary named 'gitnexus'."""
        seen = {}

        def fake_which(name):
            seen["name"] = name
            return None

        monkeypatch.setattr(gitnexus_utils.shutil, "which", fake_which)
        available()
        assert seen["name"] == "gitnexus"


# ---------------------------------------------------------------------------
# TestIsIndexed
# ---------------------------------------------------------------------------

class TestIsIndexed:
    """The is_indexed() check detects a valid .gitnexus index directory."""

    def test_false_when_missing(self, tmp_path):
        """is_indexed() is False when no .gitnexus directory exists."""
        assert is_indexed(str(tmp_path)) is False

    def test_true_when_present(self, tmp_path):
        """is_indexed() is True when a .gitnexus directory is present."""
        (tmp_path / ".gitnexus").mkdir()
        assert is_indexed(str(tmp_path)) is True

    def test_false_when_file_not_dir(self, tmp_path):
        """A regular file named .gitnexus is not treated as a valid index."""
        (tmp_path / ".gitnexus").write_text("not a dir")
        assert is_indexed(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# TestIndex — wrapper never raises; reports success:False on failure
# ---------------------------------------------------------------------------

class TestIndex:
    """The index() wrapper never raises and reports success:False on any failure."""

    def test_returns_failure_when_absent(self, tmp_path, monkeypatch):
        """gitnexus absent -> success:False, no exception raised (caller halts)."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: None)
        result = index(str(tmp_path))
        assert result["success"] is False
        assert result["error"]
        assert result["graph_path"] == str(tmp_path / ".gitnexus")

    def test_does_not_invoke_subprocess_when_absent(self, tmp_path, monkeypatch):
        """index() skips subprocess.run entirely when the CLI is absent."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: None)

        def boom(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("subprocess.run should not be called when CLI absent")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", boom)
        result = index(str(tmp_path))
        assert result["success"] is False

    def test_success_path(self, tmp_path, monkeypatch):
        """A successful analyze run uses --index-only and yields success:True with no error."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")
        # Isolate the analyze call from the git-exclude side effect.
        monkeypatch.setattr(gitnexus_utils, "_exclude_gitnexus_artifact", lambda path: None)

        def fake_run(cmd, **kwargs):
            assert cmd == ["gitnexus", "analyze", str(tmp_path), "--index-only"]
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        result = index(str(tmp_path))
        assert result["success"] is True
        assert result["error"] is None
        assert result["already_indexed"] is False
        assert result["graph_path"] == str(tmp_path / ".gitnexus")

    def test_handles_called_process_error(self, tmp_path, monkeypatch):
        """A non-zero exit surfaces the stderr in the error field."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")

        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="boom")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        result = index(str(tmp_path))
        assert result["success"] is False
        assert "boom" in result["error"]

    def test_handles_timeout(self, tmp_path, monkeypatch):
        """A subprocess timeout is reported as a timed-out error, not an exception."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 600))

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        result = index(str(tmp_path), timeout=5)
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_handles_file_not_found(self, tmp_path, monkeypatch):
        """A FileNotFoundError race (which() lies) is caught and reported."""
        # which() reports available but subprocess can't find the binary (race).
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("gitnexus")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        result = index(str(tmp_path))
        assert result["success"] is False
        assert result["error"]

    def test_passes_timeout_through(self, tmp_path, monkeypatch):
        """The timeout argument is forwarded to subprocess.run."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")
        monkeypatch.setattr(gitnexus_utils, "_exclude_gitnexus_artifact", lambda path: None)
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        index(str(tmp_path), timeout=42)
        assert seen["timeout"] == 42

    def test_skips_when_already_indexed(self, tmp_path, monkeypatch):
        """An already-indexed path returns already_indexed:True without re-running."""
        (tmp_path / ".gitnexus").mkdir()
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")

        def boom(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("subprocess.run should not be called when already indexed")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", boom)
        result = index(str(tmp_path))
        assert result["success"] is True
        assert result["already_indexed"] is True
        assert result["error"] is None

    def test_force_reindexes_when_already_indexed(self, tmp_path, monkeypatch):
        """force=True re-runs analyze with --force even when already indexed."""
        (tmp_path / ".gitnexus").mkdir()
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")
        monkeypatch.setattr(gitnexus_utils, "_exclude_gitnexus_artifact", lambda path: None)

        def fake_run(cmd, **kwargs):
            assert cmd == ["gitnexus", "analyze", str(tmp_path), "--index-only", "--force"]
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        result = index(str(tmp_path), force=True)
        assert result["success"] is True
        assert result["already_indexed"] is False

    def test_success_invokes_exclude(self, tmp_path, monkeypatch):
        """A successful index adds the .gitnexus artifact to the git exclude."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/gitnexus")
        seen = {}

        def record_exclude(path):
            seen["path"] = path

        monkeypatch.setattr(gitnexus_utils, "_exclude_gitnexus_artifact", record_exclude)
        monkeypatch.setattr(
            gitnexus_utils.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        )
        index(str(tmp_path))
        assert seen["path"] == str(tmp_path)


# ---------------------------------------------------------------------------
# TestMcpRegistered — best-effort MCP-server registration probe
# ---------------------------------------------------------------------------

class TestMcpRegistered:
    """mcp_registered() reports whether the gitnexus MCP server is registered."""

    def test_none_when_claude_absent(self, monkeypatch):
        """Returns None when the `claude` CLI is not on PATH (cannot determine)."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: None)
        assert mcp_registered() is None

    def test_true_when_registered(self, monkeypatch):
        """Returns True when `claude mcp get gitnexus` succeeds and names the server."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/claude")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="gitnexus:\n  Status: Connected", stderr="")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        assert mcp_registered() is True

    def test_false_when_not_registered(self, monkeypatch):
        """Returns False when `claude` runs but the server is not registered."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/claude")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No MCP server found")

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        assert mcp_registered() is False

    def test_none_on_exception(self, monkeypatch):
        """Returns None (never raises) when the probe subprocess errors out."""
        monkeypatch.setattr(gitnexus_utils.shutil, "which", lambda name: "/usr/bin/claude")

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 20))

        monkeypatch.setattr(gitnexus_utils.subprocess, "run", fake_run)
        assert mcp_registered() is None


# ---------------------------------------------------------------------------
# TestExcludeArtifact — git-exclude side effect (real git)
# ---------------------------------------------------------------------------

class TestExcludeArtifact:
    """_exclude_gitnexus_artifact() keeps .gitnexus/ out of git status; never raises."""

    def _git_init(self, path):
        """Initialize a git repo at *path* for exclude-file testing."""
        subprocess.run(["git", "init", "-q", str(path)], check=True)

    def test_appends_to_git_exclude(self, tmp_path):
        """After the call, the repo's info/exclude lists .gitnexus/."""
        self._git_init(tmp_path)
        _exclude_gitnexus_artifact(str(tmp_path))
        exclude = tmp_path / ".git" / "info" / "exclude"
        assert ".gitnexus/" in exclude.read_text()

    def test_idempotent(self, tmp_path):
        """Calling twice does not duplicate the .gitnexus/ entry."""
        self._git_init(tmp_path)
        _exclude_gitnexus_artifact(str(tmp_path))
        _exclude_gitnexus_artifact(str(tmp_path))
        exclude = tmp_path / ".git" / "info" / "exclude"
        assert exclude.read_text().count(".gitnexus/") == 1

    def test_noop_for_non_git_path(self, tmp_path):
        """A non-git path is a silent no-op (no exception, no .git created)."""
        _exclude_gitnexus_artifact(str(tmp_path))
        assert not (tmp_path / ".git").exists()


# ---------------------------------------------------------------------------
# TestCli — argument parsing + JSON output (subprocess, real environment)
# ---------------------------------------------------------------------------

class TestCli:
    """The command-line interface parses arguments and emits JSON in a real environment."""

    def _run(self, *args):
        """Invoke the gitnexus_utils CLI as a subprocess and capture its output."""
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            capture_output=True,
            text=True,
        )

    def test_no_args_usage(self):
        """Running with no arguments prints usage and exits non-zero."""
        proc = self._run()
        assert proc.returncode == 1
        assert "Usage" in proc.stdout

    def test_available_outputs_json(self):
        """The `available` mode emits a JSON object with a boolean available flag."""
        proc = self._run("available")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert "available" in data
        assert isinstance(data["available"], bool)

    def test_is_indexed_true(self, tmp_path):
        """The `is-indexed` mode reports indexed:True when .gitnexus exists."""
        (tmp_path / ".gitnexus").mkdir()
        proc = self._run("is-indexed", str(tmp_path))
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["indexed"] is True

    def test_is_indexed_false(self, tmp_path):
        """The `is-indexed` mode reports indexed:False for an unindexed path."""
        proc = self._run("is-indexed", str(tmp_path))
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["indexed"] is False

    def test_is_indexed_missing_path_arg(self):
        """The `is-indexed` mode requires a path argument and errors without one."""
        proc = self._run("is-indexed")
        assert proc.returncode == 1
        assert "Usage" in proc.stdout

    def test_index_missing_path_arg(self):
        """The `index` mode requires a path argument and errors without one."""
        proc = self._run("index")
        assert proc.returncode == 1
        assert "Usage" in proc.stdout

    def test_index_outputs_json(self, tmp_path):
        """The `index` mode emits JSON with success/graph_path/error and a matching exit code."""
        # Force the CLI absent (empty PATH) for a deterministic success:False,
        # exit 1 regardless of whether gitnexus is installed on the host.
        proc = self._run_without_gitnexus("index", str(tmp_path))
        data = json.loads(proc.stdout)
        assert "success" in data
        assert "graph_path" in data
        assert "error" in data
        assert data["success"] is False
        assert proc.returncode == 1

    def test_index_accepts_force_flag(self, tmp_path):
        """The `index` mode parses --force without treating it as the path."""
        # CLI absent -> deterministic failure, but the flag must parse cleanly and
        # the resolved path must still be the directory (not "--force").
        proc = self._run_without_gitnexus("index", str(tmp_path), "--force")
        data = json.loads(proc.stdout)
        assert data["graph_path"] == str(tmp_path / ".gitnexus")
        assert proc.returncode == 1

    def test_mcp_registered_outputs_json(self):
        """The `mcp-registered` mode emits {"registered": ...} and always exits 0."""
        # CLI absent (no `claude`) -> registered is null; key must be present.
        proc = self._run_without_gitnexus("mcp-registered")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert "registered" in data

    def test_unknown_mode(self):
        """An unrecognized mode prints an 'unknown mode' message and exits non-zero."""
        proc = self._run("bogus")
        assert proc.returncode == 1
        assert "unknown mode" in proc.stdout.lower()

    def _run_without_gitnexus(self, *args):
        """Invoke the CLI with an empty PATH so the gitnexus binary cannot resolve."""
        # Run with an empty PATH so `shutil.which("gitnexus")` cannot resolve,
        # guaranteeing the CLI is reported absent regardless of the host env.
        env = dict(os.environ)
        env["PATH"] = ""
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_require_exits_nonzero_when_absent(self):
        """The `require` mode fails loudly with install guidance when gitnexus is absent."""
        # GitNexus is a HARD PREREQUISITE: `require` must fail loudly when absent.
        proc = self._run_without_gitnexus("require")
        assert proc.returncode != 0
        data = json.loads(proc.stdout)
        assert data["available"] is False
        assert data["required"] is True
        assert data["error"]
        # Install guidance is surfaced so the caller can relay it.
        assert any("npm install -g gitnexus" in line for line in data["install"])
        assert any("gitnexus setup" in line for line in data["install"])

    def test_require_outputs_required_flag(self):
        """The `require` mode always reports required:True with a matching exit code."""
        # `require` always reports required:True regardless of availability.
        proc = self._run("require")
        data = json.loads(proc.stdout)
        assert data["required"] is True
        assert "available" in data
        # Exit code mirrors availability.
        assert proc.returncode == (0 if data["available"] else 1)
