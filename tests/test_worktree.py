"""Tests for isolated Git worktree management and cleanup."""

from pathlib import Path
import subprocess
import pytest

from rootcause.core.exceptions import InvalidRepositoryError
from rootcause.security.worktree import IsolatedWorktree, create_isolated_worktree


class TestIsolatedWorktree:
    """Verify disposable worktrees, filesystem isolation, and reliable cleanup."""

    def test_worktree_creation_and_isolation(self, temp_git_repo: Path):
        calc_file = temp_git_repo / "src" / "app" / "calculator.py"
        original_content = calc_file.read_text(encoding="utf-8")

        with create_isolated_worktree(temp_git_repo) as wt:
            assert wt.worktree_path is not None
            assert wt.worktree_path.exists()
            assert wt.worktree_path != temp_git_repo

            # Worktree has copied files from HEAD
            wt_calc = wt.worktree_path / "src" / "app" / "calculator.py"
            assert wt_calc.exists()
            assert wt_calc.read_text(encoding="utf-8") == original_content

            # Mutate the file inside the worktree
            wt_calc.write_text("# MODIFIED IN WORKTREE\n", encoding="utf-8")
            assert wt_calc.read_text(encoding="utf-8") == "# MODIFIED IN WORKTREE\n"

            # The original repository MUST remain completely untouched!
            assert calc_file.read_text(encoding="utf-8") == original_content

            # Original git status is completely clean
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(temp_git_repo),
                capture_output=True,
                text=True,
            )
            assert status.stdout.strip() == ""

        # After exiting context manager, worktree directory is cleaned up
        assert not wt.worktree_path.exists()

    def test_cleanup_after_exception(self, temp_git_repo: Path):
        wt_path_holder = []
        with pytest.raises(RuntimeError, match="Simulated error inside worktree"):
            with create_isolated_worktree(temp_git_repo) as wt:
                wt_path_holder.append(wt.worktree_path)
                assert wt.worktree_path.exists()
                raise RuntimeError("Simulated error inside worktree")

        # Verify cleanup ran even when an exception was raised
        assert len(wt_path_holder) == 1
        assert not wt_path_holder[0].exists()

    def test_invalid_repository_raises_error(self, tmp_path: Path):
        non_git_dir = tmp_path / "not_a_git_repo"
        non_git_dir.mkdir()

        with pytest.raises(InvalidRepositoryError):
            with create_isolated_worktree(non_git_dir):
                pass

    def test_nonexistent_repository_raises_error(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(InvalidRepositoryError):
            with create_isolated_worktree(nonexistent):
                pass
