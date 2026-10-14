"""Git inspection tools using controlled subprocess runner. Never modifies repository."""

from pathlib import Path
import re
from typing import List, Optional, Union

from rootcause.core.exceptions import InvalidRepositoryError, SecurityViolation
from rootcause.core.models import GitCommit
from rootcause.security.path_guard import validate_safe_path
from rootcause.security.sandbox import SubprocessRunner

# Safe reference regex: commit hashes, HEAD references, and branch/tag names
# Rejects flag injections (e.g. leading dash) and shell metacharacters
SAFE_REF_RE = re.compile(r"^(?:[0-9a-fA-F]{4,40}|HEAD[~^0-9]*|[a-zA-Z0-9][a-zA-Z0-9_\-./]*)$")


def _get_runner(repo_path: Union[Path, str]) -> SubprocessRunner:
    """Create a runner for the repository, verifying it is a git repo."""
    resolved = Path(repo_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Directory does not exist: {resolved}")
    runner = SubprocessRunner(allowed_root=resolved)
    res = runner.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=resolved)
    if res.exit_code != 0 or "true" not in res.stdout.strip().lower():
        raise InvalidRepositoryError(f"Directory is not a valid git repository: {resolved}")
    return runner


def get_git_status(repo_path: Union[Path, str]) -> str:
    """Get the working tree status without modifying the repository."""
    runner = _get_runner(repo_path)
    res = runner.run(["git", "status", "--short"], cwd=repo_path)
    return res.stdout.strip()


def get_recent_commits(
    repo_path: Union[Path, str],
    max_count: int = 10,
) -> List[GitCommit]:
    """Retrieve recent commits with structured metadata.
    
    Returns:
        List of GitCommit models. Empty list if repository has no commits yet.
    """
    runner = _get_runner(repo_path)
    count = max(1, min(max_count, 100))
    res = runner.run(
        ["git", "log", f"-n{count}", "--pretty=format:%H|%an|%ad|%s", "--date=iso"],
        cwd=repo_path,
    )
    
    commits: List[GitCommit] = []
    # If repo has no commits yet, git log returns non-zero exit code
    if res.exit_code != 0 or not res.stdout.strip():
        return commits

    for line in res.stdout.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append(GitCommit(
                hash=parts[0],
                author=parts[1],
                date=parts[2],
                message=parts[3],
            ))
    return commits


def get_commit_diff(repo_path: Union[Path, str], commit_ref: str) -> str:
    """Get the full diff and patch for a specific commit. Read-only operation."""
    if not commit_ref or not SAFE_REF_RE.match(commit_ref) or commit_ref.startswith("-"):
        raise SecurityViolation(f"Invalid or unsafe commit reference: '{commit_ref}'")
    runner = _get_runner(repo_path)
    res = runner.run(["git", "show", "--patch", "--stat", commit_ref], cwd=repo_path)
    if res.exit_code != 0:
        raise ValueError(f"Failed to retrieve diff for ref '{commit_ref}': {res.stderr or res.stdout}")
    return res.stdout


def get_working_tree_diff(repo_path: Union[Path, str], relative_file: Optional[str] = None) -> str:
    """Get uncommitted diffs against HEAD. Read-only operation."""
    runner = _get_runner(repo_path)
    cmd = ["git", "diff", "HEAD"]
    if relative_file:
        safe_file = validate_safe_path(repo_path, relative_file)
        rel = str(safe_file.relative_to(Path(repo_path).resolve())).replace("\\", "/")
        cmd.extend(["--", rel])
    res = runner.run(cmd, cwd=repo_path)
    return res.stdout
