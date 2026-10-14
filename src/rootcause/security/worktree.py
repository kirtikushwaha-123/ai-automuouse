"""Isolated Git worktree management for safe, isolated patch testing."""

import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Optional, Union

from rootcause.core.exceptions import (
    CommandExecutionError,
    InvalidRepositoryError,
    WorktreeError,
)
from rootcause.security.sandbox import SubprocessRunner


def _remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree on Windows read-only files."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class IsolatedWorktree:
    """Context manager representing a disposable Git worktree."""

    def __init__(
        self,
        original_repo_path: Union[Path, str],
        commit_or_ref: str = "HEAD",
        base_temp_dir: Optional[Union[Path, str]] = None,
    ):
        self.original_repo_path = Path(original_repo_path).resolve()
        if not self.original_repo_path.exists():
            raise InvalidRepositoryError(
                f"Repository path does not exist: {self.original_repo_path}"
            )
        self.commit_or_ref = commit_or_ref
        self.base_temp_dir = Path(base_temp_dir).resolve() if base_temp_dir else None
        self.worktree_path: Optional[Path] = None
        self._runner = SubprocessRunner(allowed_root=self.original_repo_path)
        self._cleaned_up = False

    def __enter__(self) -> "IsolatedWorktree":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def create(self) -> Path:
        """Create a detached git worktree in an isolated temporary directory."""
        if not self.original_repo_path.exists():
            raise InvalidRepositoryError(
                f"Repository path does not exist: {self.original_repo_path}"
            )

        # Verify target is a valid git repository
        try:
            res = self._runner.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.original_repo_path,
                check=True,
            )
            if "true" not in res.stdout.strip().lower():
                raise InvalidRepositoryError(
                    f"Path is not inside a git worktree: {self.original_repo_path}"
                )
        except (CommandExecutionError, Exception) as exc:
            raise InvalidRepositoryError(
                f"Invalid or unreadable git repository at {self.original_repo_path}: {exc}"
            ) from exc

        # Create temporary directory
        temp_dir_str = tempfile.mkdtemp(
            prefix="rootcause_wt_",
            dir=str(self.base_temp_dir) if self.base_temp_dir else None,
        )
        self.worktree_path = Path(temp_dir_str).resolve()

        # Add detached worktree
        try:
            self._runner.run(
                ["git", "worktree", "add", "--detach", str(self.worktree_path), self.commit_or_ref],
                cwd=self.original_repo_path,
                check=True,
            )
        except CommandExecutionError as exc:
            # Clean up the empty temp directory on failure
            if self.worktree_path and self.worktree_path.exists():
                shutil.rmtree(self.worktree_path, ignore_errors=True)
            raise WorktreeError(
                f"Failed to create git worktree at {self.worktree_path}: {exc.stderr or exc.stdout}"
            ) from exc

        return self.worktree_path

    def cleanup(self) -> None:
        """Remove the worktree and clean up temporary files."""
        if self._cleaned_up or not self.worktree_path:
            return

        self._cleaned_up = True
        
        # 1. Ask git to remove the worktree
        try:
            self._runner.run(
                ["git", "worktree", "remove", "--force", str(self.worktree_path)],
                cwd=self.original_repo_path,
            )
        except Exception:
            pass

        # 2. Prune git worktree records
        try:
            self._runner.run(
                ["git", "worktree", "prune"],
                cwd=self.original_repo_path,
            )
        except Exception:
            pass

        # 3. Clean up any remaining filesystem directory
        if self.worktree_path.exists():
            shutil.rmtree(self.worktree_path, onerror=_remove_readonly)


def create_isolated_worktree(
    repo_path: Union[Path, str],
    commit_or_ref: str = "HEAD",
    base_temp_dir: Optional[Union[Path, str]] = None,
) -> IsolatedWorktree:
    """Convenience helper to create an IsolatedWorktree context manager."""
    return IsolatedWorktree(
        original_repo_path=repo_path,
        commit_or_ref=commit_or_ref,
        base_temp_dir=base_temp_dir,
    )
