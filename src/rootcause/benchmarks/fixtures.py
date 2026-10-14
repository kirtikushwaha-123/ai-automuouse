"""Deterministic local Git fixture repositories for RootCause AI benchmarks."""

from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

from rootcause.core.models import TestResult
from rootcause.security.sandbox import SubprocessRunner
from rootcause.security.worktree import IsolatedWorktree
from rootcause.tools.test_runner import check_test_reproduces_bug, run_pytest
from rootcause.verification.patch_engine import apply_diff_to_worktree


def _init_git_repo(repo_dir: Path) -> SubprocessRunner:
    """Initialize a git repo with standard test user config."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "benchmark@rootcause.ai"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Benchmark Agent"], cwd=str(repo_dir), check=True, capture_output=True)
    (repo_dir / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n*.pyc\n", encoding="utf-8")
    return SubprocessRunner(allowed_root=repo_dir)


def setup_bm01_conditional_logic(target_dir: Path) -> Path:
    """Fixture BM01: Cart discount threshold comparison bug (> vs >=)."""
    runner = _init_git_repo(target_dir)

    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "discount.py").write_text(
        'def calculate_discount(total: float) -> float:\n'
        '    """Calculate discount amount.\n\n'
        '    Orders of $100.00 or more qualify for a 10% discount.\n'
        '    """\n'
        '    if total > 100.0:\n'
        '        return round(total * 0.10, 2)\n'
        '    return 0.0\n',
        encoding="utf-8",
    )

    test_dir = target_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test_discount.py").write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).parent.parent / "src"))\n'
        'from discount import calculate_discount\n\n'
        'def test_small_order_no_discount():\n'
        '    assert calculate_discount(50.0) == 0.0\n\n'
        'def test_large_order_gets_discount():\n'
        '    assert calculate_discount(200.0) == 20.0\n',
        encoding="utf-8",
    )

    runner.run(["git", "add", "."], cwd=target_dir, check=True)
    runner.run(["git", "commit", "-m", "Initial commit: discount calculator"], cwd=target_dir, check=True)
    return target_dir


def setup_bm02_boundary_off_by_one(target_dir: Path) -> Path:
    """Fixture BM02: Pagination slice off-by-one upper bound bug (end_idx - 1)."""
    runner = _init_git_repo(target_dir)

    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "pagination.py").write_text(
        'from typing import Any, List\n\n'
        'def paginate_items(items: List[Any], page: int, page_size: int) -> List[Any]:\n'
        '    """Return items for the requested 1-indexed page."""\n'
        '    if page < 1 or page_size < 1:\n'
        '        return []\n'
        '    start_idx = (page - 1) * page_size\n'
        '    end_idx = start_idx + page_size\n'
        '    # Off-by-one bug: subtracting 1 omits the last item on full page\n'
        '    return items[start_idx : end_idx - 1]\n',
        encoding="utf-8",
    )

    test_dir = target_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test_pagination.py").write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).parent.parent / "src"))\n'
        'from pagination import paginate_items\n\n'
        'def test_empty_list_returns_empty():\n'
        '    assert paginate_items([], 1, 5) == []\n\n'
        'def test_invalid_page_returns_empty():\n'
        '    assert paginate_items([1, 2, 3], 0, 5) == []\n',
        encoding="utf-8",
    )

    runner.run(["git", "add", "."], cwd=target_dir, check=True)
    runner.run(["git", "commit", "-m", "Initial commit: pagination utility"], cwd=target_dir, check=True)
    return target_dir


def setup_bm03_data_handling_none(target_dir: Path) -> Path:
    """Fixture BM03: User profile formatting unhandled None AttributeError."""
    runner = _init_git_repo(target_dir)

    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "profile.py").write_text(
        'from typing import Dict, Optional\n\n'
        'def format_user_bio(profile: Dict[str, Optional[str]]) -> str:\n'
        '    """Format user biography for display."""\n'
        '    bio = profile.get("bio")\n'
        '    # Unhandled None: calling .strip() on None raises AttributeError\n'
        '    return bio.strip().title()\n',
        encoding="utf-8",
    )

    test_dir = target_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test_profile.py").write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).parent.parent / "src"))\n'
        'from profile import format_user_bio\n\n'
        'def test_valid_user_bio():\n'
        '    profile = {"username": "alice", "bio": "  software engineer  "}\n'
        '    assert format_user_bio(profile) == "Software Engineer"\n',
        encoding="utf-8",
    )

    runner.run(["git", "add", "."], cwd=target_dir, check=True)
    runner.run(["git", "commit", "-m", "Initial commit: profile formatting"], cwd=target_dir, check=True)
    return target_dir


def setup_bm04_code_regression(target_dir: Path) -> Path:
    """Fixture BM04: Multi-commit sequence introducing regression in title formatter."""
    runner = _init_git_repo(target_dir)

    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    test_dir = target_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Commit 1: Initial working version
    (src_dir / "formatter.py").write_text(
        'def sanitize_input(text: str) -> str:\n'
        '    return text.strip().lower()\n\n'
        'def format_title(text: str) -> str:\n'
        '    cleaned = sanitize_input(text)\n'
        '    return f"Title: {cleaned.capitalize()}"\n',
        encoding="utf-8",
    )
    (test_dir / "test_formatter.py").write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).parent.parent / "src"))\n'
        'from formatter import sanitize_input\n\n'
        'def test_sanitize_basic():\n'
        '    assert sanitize_input("  HELLO  ") == "hello"\n',
        encoding="utf-8",
    )
    runner.run(["git", "add", "."], cwd=target_dir, check=True)
    runner.run(["git", "commit", "-m", "Initial commit: string sanitizer and title formatter"], cwd=target_dir, check=True)

    # Commit 2: Refactoring sanitize_input return type introduces regression in format_title
    (src_dir / "formatter.py").write_text(
        'from typing import Tuple\n\n'
        'def sanitize_input(text: str) -> Tuple[str, bool]:\n'
        '    cleaned = text.strip().lower()\n'
        '    modified = (cleaned != text)\n'
        '    return cleaned, modified\n\n'
        'def format_title(text: str) -> str:\n'
        '    # Regression: sanitize_input now returns tuple, but format_title expects str\n'
        '    cleaned = sanitize_input(text)\n'
        '    return f"Title: {cleaned.capitalize()}"\n',
        encoding="utf-8",
    )
    (test_dir / "test_formatter.py").write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).parent.parent / "src"))\n'
        'from formatter import sanitize_input\n\n'
        'def test_sanitize_tuple_return():\n'
        '    cleaned, modified = sanitize_input("  HELLO  ")\n'
        '    assert cleaned == "hello"\n'
        '    assert modified is True\n',
        encoding="utf-8",
    )
    runner.run(["git", "add", "."], cwd=target_dir, check=True)
    runner.run(["git", "commit", "-m", "Refactor sanitize_input to return modification status tuple"], cwd=target_dir, check=True)
    return target_dir


def setup_bm05_zero_division(target_dir: Path) -> Path:
    """Fixture BM05: Statistics calculation ZeroDivisionError on empty input list."""
    runner = _init_git_repo(target_dir)

    src_dir = target_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "stats.py").write_text(
        'from typing import List\n\n'
        'def calculate_average(scores: List[float]) -> float:\n'
        '    """Compute arithmetic mean of scores.\n\n'
        '    Returns 0.0 when score list is empty.\n'
        '    """\n'
        '    # Bug: direct division without empty-list check raises ZeroDivisionError\n'
        '    return round(sum(scores) / len(scores), 2)\n',
        encoding="utf-8",
    )

    test_dir = target_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test_stats.py").write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).parent.parent / "src"))\n'
        'from stats import calculate_average\n\n'
        'def test_average_non_empty():\n'
        '    assert calculate_average([10.0, 20.0, 30.0]) == 20.0\n',
        encoding="utf-8",
    )

    runner.run(["git", "add", "."], cwd=target_dir, check=True)
    runner.run(["git", "commit", "-m", "Initial commit: statistics calculator"], cwd=target_dir, check=True)
    return target_dir


FIXTURE_SETUP_FUNCTIONS = {
    "BM01_CONDITIONAL_LOGIC": setup_bm01_conditional_logic,
    "BM02_BOUNDARY_OFF_BY_ONE": setup_bm02_boundary_off_by_one,
    "BM03_DATA_HANDLING_NONE": setup_bm03_data_handling_none,
    "BM04_CODE_REGRESSION": setup_bm04_code_regression,
    "BM05_ZERO_DIVISION": setup_bm05_zero_division,
}


def create_fixture_repository(case_id: str, target_dir: Path) -> Path:
    """Instantiate a clean fixture git repository on disk.
    
    Args:
        case_id: Identifier of benchmark case.
        target_dir: Destination directory.
        
    Returns:
        Path to initialized fixture repository.
    """
    if case_id not in FIXTURE_SETUP_FUNCTIONS:
        raise ValueError(f"Unknown benchmark fixture case ID: '{case_id}'")
    
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    setup_fn = FIXTURE_SETUP_FUNCTIONS[case_id]
    return setup_fn(target_dir)


def validate_fixture(
    case_id: str,
    repo_path: Path,
    bug_description: str,
    stack_trace: Optional[str],
    regression_test_file: str,
    regression_test_code: str,
    known_good_patch: str,
) -> Dict[str, Any]:
    """Execute complete 5-point acceptance criteria check for a fixture.
    
    Validates:
      1. Pre-existing test suite passes on clean baseline.
      2. Regression test fails on clean baseline with expected symptom.
      3. Known-good patch applies cleanly to baseline worktree.
      4. Regression test passes on patched worktree.
      5. Full pre-existing test suite passes on patched worktree with 0 regressions.
      
    Returns:
      Validation summary dictionary.
    """
    runner = SubprocessRunner(allowed_root=repo_path)
    head_hash = runner.run(["git", "rev-parse", "HEAD"], cwd=repo_path, check=True).stdout.strip()

    # Step 1: Pre-existing tests must pass on baseline
    baseline_existing = run_pytest(repo_path, test_target=None)
    if not baseline_existing.passed or baseline_existing.failures > 0 or baseline_existing.errors > 0:
        return {
            "valid": False,
            "stage_failed": "baseline_existing_tests",
            "message": f"Pre-existing tests failed on baseline: {baseline_existing.output}",
            "commit_hash": head_hash,
        }

    # Step 2: Write regression test in isolated worktree and verify failure & symptom reproduction
    with IsolatedWorktree(repo_path, commit_or_ref="HEAD") as wt:
        wt_path = wt.worktree_path
        repro_path = wt_path / regression_test_file
        repro_path.parent.mkdir(parents=True, exist_ok=True)
        repro_path.write_text(regression_test_code, encoding="utf-8")

        rel_repro = str(repro_path.relative_to(wt_path)).replace("\\", "/")
        baseline_repro = run_pytest(wt_path, test_target=rel_repro)

        if baseline_repro.passed:
            return {
                "valid": False,
                "stage_failed": "baseline_regression_test_must_fail",
                "message": "Regression test unexpectedly passed on clean baseline.",
                "commit_hash": head_hash,
            }

        reproduces, reason = check_test_reproduces_bug(
            baseline_repro, bug_description, stack_trace
        )
        if not reproduces:
            return {
                "valid": False,
                "stage_failed": "symptom_reproduction",
                "message": f"Regression test failed, but symptom did not match bug report: {reason}",
                "commit_hash": head_hash,
            }

        # Step 3: Apply known-good patch
        try:
            apply_diff_to_worktree(wt_path, known_good_patch)
        except Exception as exc:
            return {
                "valid": False,
                "stage_failed": "apply_known_good_patch",
                "message": f"Known-good patch failed to apply: {exc}",
                "commit_hash": head_hash,
            }

        # Step 4: Regression test must pass on patched worktree
        patched_repro = run_pytest(wt_path, test_target=rel_repro)
        if not patched_repro.passed or patched_repro.failures > 0 or patched_repro.errors > 0:
            return {
                "valid": False,
                "stage_failed": "patched_regression_test",
                "message": f"Regression test failed after applying known-good patch: {patched_repro.output}",
                "commit_hash": head_hash,
            }

        # Step 5: Full suite must pass on patched worktree
        patched_existing = run_pytest(wt_path, test_target=None)
        if not patched_existing.passed or patched_existing.failures > 0 or patched_existing.errors > 0:
            return {
                "valid": False,
                "stage_failed": "patched_full_suite",
                "message": f"Regressions detected in full test suite after patch: {patched_existing.output}",
                "commit_hash": head_hash,
            }

    # Verify original repository was untouched
    status_res = runner.run(["git", "status", "--porcelain"], cwd=repo_path, check=True)
    if status_res.stdout.strip():
        return {
            "valid": False,
            "stage_failed": "immutability_check",
            "message": "Original fixture repository working tree was modified.",
            "commit_hash": head_hash,
        }

    return {
        "valid": True,
        "stage_failed": None,
        "message": "All 5 acceptance validation stages passed successfully.",
        "commit_hash": head_hash,
        "pre_existing_tests_count": baseline_existing.passed_count,
        "reproduction_reason": reason,
    }
