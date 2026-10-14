"""Security and sandboxing tests: path traversal, command allowlisting, timeouts, env scrubbing."""

import os
from pathlib import Path
import pytest

from rootcause.core.exceptions import (
    CommandExecutionError,
    CommandTimeoutError,
    SecurityViolation,
)
from rootcause.security.path_guard import is_path_safe, validate_safe_path
from rootcause.security.sandbox import SubprocessRunner


def create_filesystem_link(link_path: Path, target_path: Path) -> bool:
    """Create a genuine filesystem link (symlink on POSIX, directory junction on Windows)."""
    try:
        target_is_dir = target_path.is_dir()
        link_path.symlink_to(target_path, target_is_directory=target_is_dir)
        return True
    except (OSError, NotImplementedError):
        if os.name == "nt" and target_path.is_dir():
            try:
                import _winapi
                _winapi.CreateJunction(str(target_path), str(link_path))
                return True
            except Exception:
                pass
        return False


class TestPathGuard:
    """Tests for path traversal and filesystem isolation."""

    def test_safe_relative_path(self, tmp_path: Path):
        file_path = tmp_path / "valid.txt"
        file_path.write_text("hello", encoding="utf-8")
        
        result = validate_safe_path(tmp_path, "valid.txt")
        assert result == file_path.resolve()
        assert is_path_safe(tmp_path, "valid.txt") is True

    def test_safe_nested_path(self, tmp_path: Path):
        subdir = tmp_path / "sub" / "dir"
        subdir.mkdir(parents=True)
        file_path = subdir / "nested.txt"
        file_path.write_text("nested", encoding="utf-8")
        
        result = validate_safe_path(tmp_path, "sub/dir/nested.txt")
        assert result == file_path.resolve()

    def test_reject_directory_traversal_up(self, tmp_path: Path):
        outside_file = tmp_path.parent / "escape.txt"
        outside_file.write_text("secret", encoding="utf-8")

        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "../escape.txt")

        assert is_path_safe(tmp_path, "../escape.txt") is False

    def test_reject_deep_directory_traversal(self, tmp_path: Path):
        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "../../../../../etc/passwd")

    def test_reject_absolute_path_outside_root(self, tmp_path: Path):
        outside_dir = tmp_path.parent / "other_dir"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "outside.txt"
        outside_file.write_text("secret", encoding="utf-8")

        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, str(outside_file))

    def test_reject_null_byte_in_path(self, tmp_path: Path):
        with pytest.raises(SecurityViolation, match="Null byte detected"):
            validate_safe_path(tmp_path, "file\x00.txt")

    def test_nonexistent_path_rejected_by_default(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            validate_safe_path(tmp_path, "nonexistent.txt", allow_nonexistent=False)

    def test_allow_nonexistent_path_within_root(self, tmp_path: Path):
        result = validate_safe_path(tmp_path, "new_file.txt", allow_nonexistent=True)
        assert result == (tmp_path / "new_file.txt").resolve()

    def test_reject_nonexistent_path_outside_root(self, tmp_path: Path):
        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "../new_file.txt", allow_nonexistent=True)

    def test_symlink_escape_rejection(self, tmp_path: Path):
        """Verify that symlinks / junctions pointing outside the repository root are rejected."""
        outside_dir = tmp_path.parent / "outside_target"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("classified data", encoding="utf-8")

        symlink_path = tmp_path / "link_to_outside"
        created = create_filesystem_link(symlink_path, outside_dir)
        assert created, "Failed to create filesystem link for test."

        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "link_to_outside")

        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "link_to_outside/secret.txt")

        assert is_path_safe(tmp_path, "link_to_outside") is False
        assert is_path_safe(tmp_path, "link_to_outside/secret.txt") is False

    def test_nested_symlink_escape_rejection(self, tmp_path: Path):
        """Verify that deeply nested symlinks pointing outside the repository root are rejected."""
        outside_dir = tmp_path.parent / "outside_nested"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.txt").write_text("classified data", encoding="utf-8")

        nested_dir = tmp_path / "deep" / "sub" / "directory"
        nested_dir.mkdir(parents=True)
        nested_link = nested_dir / "link_out"
        created = create_filesystem_link(nested_link, outside_dir)
        assert created, "Failed to create nested link for test."

        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "deep/sub/directory/link_out/secret.txt")

        assert is_path_safe(tmp_path, "deep/sub/directory/link_out/secret.txt") is False

    def test_chained_symlink_escape_rejection(self, tmp_path: Path):
        """Verify that chained multi-hop symlinks escaping the repository root are rejected."""
        outside_dir = tmp_path.parent / "outside_chained"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.txt").write_text("classified data", encoding="utf-8")

        internal_target = tmp_path / "internal_dir"
        internal_target.mkdir()

        # Link 1 inside repo points to internal_dir
        hop1 = tmp_path / "hop1"
        created1 = create_filesystem_link(hop1, internal_target)
        assert created1

        # Link 2 inside internal_dir points outside repo
        hop2 = internal_target / "hop2"
        created2 = create_filesystem_link(hop2, outside_dir)
        assert created2

        # Accessing via chained link must be rejected
        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "hop1/hop2/secret.txt")

        assert is_path_safe(tmp_path, "hop1/hop2/secret.txt") is False

    def test_symlink_nonexistent_target_escape(self, tmp_path: Path):
        """Verify that nonexistent paths under an escaping symlink are rejected as SecurityViolation."""
        outside_dir = tmp_path.parent / "outside_nonexistent_test"
        outside_dir.mkdir(exist_ok=True)

        link_path = tmp_path / "link_outside"
        created = create_filesystem_link(link_path, outside_dir)
        assert created

        # allow_nonexistent=False: must raise SecurityViolation, NOT FileNotFoundError
        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "link_outside/missing_file.txt", allow_nonexistent=False)

        # allow_nonexistent=True: must raise SecurityViolation
        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "link_outside/missing_file.txt", allow_nonexistent=True)

    def test_internal_symlink_allowed(self, tmp_path: Path):
        """Verify that legitimate symlinks pointing within the repository root are allowed."""
        internal_dir = tmp_path / "shared"
        internal_dir.mkdir()
        shared_file = internal_dir / "config.json"
        shared_file.write_text('{"key": "value"}', encoding="utf-8")

        link_path = tmp_path / "module" / "shared_link"
        link_path.parent.mkdir(parents=True)
        created = create_filesystem_link(link_path, internal_dir)
        assert created

        resolved = validate_safe_path(tmp_path, "module/shared_link/config.json")
        assert resolved == shared_file.resolve()
        assert is_path_safe(tmp_path, "module/shared_link/config.json") is True

    def test_simulated_symlink_escape_rejection(self, tmp_path: Path, monkeypatch):
        """Simulate a resolved file symlink escape to test detection on platforms lacking symlink privileges."""
        fake_file = tmp_path / "fake_symlink.txt"
        fake_file.write_text("dummy", encoding="utf-8")
        escaped_target = (tmp_path.parent / "escaped.txt").resolve()

        orig_resolve = Path.resolve

        def mock_resolve(path_obj, *args, **kwargs):
            if path_obj.name == "fake_symlink.txt":
                return escaped_target
            return orig_resolve(path_obj, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", mock_resolve)
        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            validate_safe_path(tmp_path, "fake_symlink.txt")


class TestSubprocessRunner:
    """Tests for controlled subprocess execution, allowlisting, and timeouts."""

    def test_disallowed_executables(self, tmp_path: Path):
        runner = SubprocessRunner(allowed_root=tmp_path)
        for bad_cmd in [["sh", "-c", "ls"], ["bash", "-c", "ls"], ["curl", "https://example.com"], ["powershell", "-Command", "dir"], ["cmd", "/c", "dir"]]:
            with pytest.raises(SecurityViolation, match="not in the allowlist"):
                runner.run(bad_cmd)

    def test_arbitrary_python_execution_disallowed(self, tmp_path: Path):
        runner = SubprocessRunner(allowed_root=tmp_path)
        with pytest.raises(SecurityViolation, match="Python can only be invoked with '-m pytest'"):
            runner.run(["python", "-c", "import os; print(os.environ)"])

        with pytest.raises(SecurityViolation, match="Python can only be invoked with '-m pytest'"):
            runner.run(["python", "malicious_script.py"])

        with pytest.raises(SecurityViolation, match="Python module 'http.server' is not allowed"):
            runner.run(["python", "-m", "http.server"])

    def test_disallowed_git_subcommands(self, tmp_path: Path):
        runner = SubprocessRunner(allowed_root=tmp_path)
        for bad_git in [
            ["git", "push", "origin", "main"],
            ["git", "clone", "https://github.com/evil/repo.git"],
            ["git", "remote", "add", "evil", "http://evil.com"],
            ["git", "config", "user.name", "hacker"],
        ]:
            with pytest.raises(SecurityViolation, match="forbidden by security policy"):
                runner.run(bad_git)

    def test_disallowed_git_inline_config(self, tmp_path: Path):
        runner = SubprocessRunner(allowed_root=tmp_path)
        with pytest.raises(SecurityViolation, match="Inline git '-c' configuration overrides are forbidden"):
            runner.run(["git", "-c", "core.editor=notepad", "status"])

    def test_working_directory_boundary_enforcement(self, tmp_path: Path):
        runner = SubprocessRunner(allowed_root=tmp_path)
        outside_dir = tmp_path.parent
        with pytest.raises(SecurityViolation, match="Path traversal detected"):
            runner.run(["git", "status"], cwd=outside_dir)

    def test_environment_variable_scrubbing(self, tmp_path: Path, monkeypatch):
        """Verify sensitive credentials are removed before subprocess launch."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret123456789")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-xyz")
        monkeypatch.setenv("MY_AUTH_TOKEN", "token-abc")
        monkeypatch.setenv("SAFE_SYSTEM_TEST_VAR", "visible")

        runner = SubprocessRunner(allowed_root=tmp_path)
        clean_env = runner._sanitize_env()

        assert "OPENAI_API_KEY" not in clean_env
        assert "AWS_SECRET_ACCESS_KEY" not in clean_env
        assert "MY_AUTH_TOKEN" not in clean_env

        # Injecting sensitive variables in extra_env must fail
        with pytest.raises(SecurityViolation, match="Attempted to inject sensitive environment variable"):
            runner._sanitize_env(extra_env={"OPENAI_API_KEY": "sk-123"})

    def test_timeout_enforcement(self, tmp_path: Path):
        """Verify hanging commands are terminated after timeout."""
        runner = SubprocessRunner(allowed_root=tmp_path)
        # Using pytest with a test that sleeps longer than timeout
        test_file = tmp_path / "test_hanging.py"
        test_file.write_text(
            "import time\n"
            "def test_sleep():\n"
            "    time.sleep(5)\n",
            encoding="utf-8",
        )

        with pytest.raises(CommandTimeoutError) as exc_info:
            runner.run(["python", "-m", "pytest", "test_hanging.py"], cwd=tmp_path, timeout=0.5)

        assert exc_info.value.timeout_seconds == 0.5
