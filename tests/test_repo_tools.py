"""Tests for repository inspection and testing tools."""

from pathlib import Path
import subprocess
import pytest

from rootcause.core.exceptions import InvalidRepositoryError, SecurityViolation
from rootcause.tools.file_tools import list_repo_files, safe_read_file
from rootcause.tools.git_tools import (
    get_commit_diff,
    get_git_status,
    get_recent_commits,
    get_working_tree_diff,
)
from rootcause.core.models import TestResult as PydanticTestResult
from rootcause.tools.search_tools import search_in_files
from rootcause.tools.test_runner import check_test_reproduces_bug, discover_tests, run_pytest


class TestFileTools:
    """Verify safe file listing and reading."""

    def test_list_repo_files(self, temp_git_repo: Path):
        files = list_repo_files(temp_git_repo)
        paths = [f["path"] for f in files]

        assert "src/app/calculator.py" in paths
        assert "tests/test_calculator.py" in paths
        # .git folder must never be exposed in file list
        assert not any(".git" in p for p in paths)

    def test_list_repo_files_empty_repo(self, tmp_path: Path):
        files = list_repo_files(tmp_path)
        assert files == []

    def test_list_repo_files_truncation(self, tmp_path: Path):
        # Create 15 files
        for idx in range(15):
            (tmp_path / f"file_{idx:02d}.txt").write_text(f"content {idx}", encoding="utf-8")
        files = list_repo_files(tmp_path, max_files=5)
        assert len(files) == 5

    def test_safe_read_file_full(self, temp_git_repo: Path):
        res = safe_read_file(temp_git_repo, "src/app/calculator.py")
        assert res["file_path"] == "src/app/calculator.py"
        assert res["total_lines"] > 0
        assert "def add" in res["content"]
        assert "   1 | def add" in res["numbered_content"]
        assert res["truncated"] is False

    def test_safe_read_file_slice(self, temp_git_repo: Path):
        res = safe_read_file(temp_git_repo, "src/app/calculator.py", start_line=1, end_line=2)
        assert res["start_line"] == 1
        assert res["end_line"] == 2
        assert "def add" in res["content"]
        assert "def divide" not in res["content"]

    def test_safe_read_file_truncation(self, tmp_path: Path):
        # Create file with 100 lines
        big_file = tmp_path / "big.txt"
        big_file.write_text("\n".join(f"line {idx}" for idx in range(100)), encoding="utf-8")
        res = safe_read_file(tmp_path, "big.txt", max_lines=20)
        assert res.total_lines == 100
        assert res.truncated is True
        assert len(res.content.splitlines()) == 20

    def test_safe_read_file_binary_detection(self, tmp_path: Path):
        # Create binary file with null bytes
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\x04\xff")
        res = safe_read_file(tmp_path, "data.bin")
        assert res.is_binary is True
        assert "[Binary file not displayed]" in res.content

    def test_safe_read_file_empty_file(self, tmp_path: Path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        res = safe_read_file(tmp_path, "empty.txt")
        assert res.total_lines == 0
        assert res.content == ""
        assert res.truncated is False

    def test_safe_read_file_missing_file_raises_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            safe_read_file(tmp_path, "nonexistent.py")

    def test_safe_read_file_rejects_path_traversal(self, temp_git_repo: Path):
        with pytest.raises(SecurityViolation):
            safe_read_file(temp_git_repo, "../outside.py")


class TestSearchTools:
    """Verify search across repository files."""

    def test_literal_search(self, temp_git_repo: Path):
        matches = search_in_files(temp_git_repo, "ZeroDivisionError")
        assert len(matches) >= 1
        assert matches[0]["file_path"] == "src/app/calculator.py"
        assert "ZeroDivisionError" in matches[0]["line_content"]

    def test_regex_search(self, temp_git_repo: Path):
        matches = search_in_files(temp_git_repo, r"def\s+divide", is_regex=True)
        assert len(matches) == 1
        assert matches[0]["file_path"] == "src/app/calculator.py"

    def test_search_no_match(self, temp_git_repo: Path):
        matches = search_in_files(temp_git_repo, "nonexistent_token_xyz_123")
        assert matches == []

    def test_search_empty_repo(self, tmp_path: Path):
        matches = search_in_files(tmp_path, "any_token")
        assert matches == []

    def test_search_invalid_regex_raises_value_error(self, temp_git_repo: Path):
        with pytest.raises(ValueError, match="Invalid regular expression pattern"):
            search_in_files(temp_git_repo, "[unclosed_regex", is_regex=True)

    def test_search_truncation_max_results(self, tmp_path: Path):
        target_file = tmp_path / "hits.txt"
        target_file.write_text("\n".join(f"MATCH LINE {idx}" for idx in range(20)), encoding="utf-8")
        matches = search_in_files(tmp_path, "MATCH LINE", max_results=5)
        assert len(matches) == 5

    def test_search_skips_binary_files(self, tmp_path: Path):
        bin_file = tmp_path / "binary.bin"
        bin_file.write_bytes(b"\x00SEARCHABLE_SECRET\x00")
        matches = search_in_files(tmp_path, "SEARCHABLE_SECRET")
        assert matches == []

    def test_search_nonexistent_repo_raises_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            search_in_files(tmp_path / "does_not_exist", "pattern")


class TestGitTools:
    """Verify Git inspection without modifying repository."""

    def test_get_git_status(self, temp_git_repo: Path):
        status = get_git_status(temp_git_repo)
        assert status == ""

    def test_get_git_status_untracked_file(self, temp_git_repo: Path):
        new_file = temp_git_repo / "untracked.txt"
        new_file.write_text("untracked content", encoding="utf-8")
        status = get_git_status(temp_git_repo)
        assert "untracked.txt" in status
        # Original status inspection must remain read-only
        new_file.unlink()

    def test_get_recent_commits(self, temp_git_repo: Path):
        commits = get_recent_commits(temp_git_repo, max_count=5)
        assert len(commits) == 1
        assert "Initial commit" in commits[0]["message"]
        assert len(commits[0]["hash"]) == 40

    def test_get_recent_commits_empty_repo(self, tmp_path: Path):
        empty_git = tmp_path / "empty_git"
        empty_git.mkdir()
        subprocess.run(["git", "init"], cwd=str(empty_git), check=True, capture_output=True)
        commits = get_recent_commits(empty_git)
        assert commits == []

    def test_malformed_git_repo_raises_error(self, tmp_path: Path):
        not_git = tmp_path / "not_git"
        not_git.mkdir()
        with pytest.raises(InvalidRepositoryError):
            get_git_status(not_git)

        with pytest.raises(InvalidRepositoryError):
            get_recent_commits(not_git)

    def test_get_commit_diff(self, temp_git_repo: Path):
        commits = get_recent_commits(temp_git_repo, max_count=1)
        diff = get_commit_diff(temp_git_repo, commits[0]["hash"])
        assert "calculator.py" in diff
        assert "def add" in diff

    def test_unsafe_commit_ref_raises_security_violation(self, temp_git_repo: Path):
        with pytest.raises(SecurityViolation, match="Invalid or unsafe commit reference"):
            get_commit_diff(temp_git_repo, "--output=/tmp/pwn")

        with pytest.raises(SecurityViolation, match="Invalid or unsafe commit reference"):
            get_commit_diff(temp_git_repo, "; rm -rf /")

    def test_get_working_tree_diff(self, temp_git_repo: Path):
        diff = get_working_tree_diff(temp_git_repo)
        assert diff == ""


class TestTestRunner:
    """Verify controlled pytest execution and structured parsing."""

    def test_run_pytest_passing(self, temp_git_repo: Path):
        result = run_pytest(temp_git_repo)
        assert result.passed is True
        assert result.exit_code == 0
        assert result.tests_ran == 2
        assert result.failures == 0
        assert result.errors == 0
        assert result.timed_out is False

    def test_run_pytest_failing(self, failing_git_repo: Path):
        result = run_pytest(failing_git_repo)
        assert result.passed is False
        assert result.exit_code == 1
        assert result.failures == 1
        assert len(result.failure_details) >= 1
        assert result.timed_out is False

    def test_run_pytest_target_file(self, temp_git_repo: Path):
        result = run_pytest(temp_git_repo, test_target="tests/test_calculator.py")
        assert result.passed is True
        assert result.tests_ran == 2

    def test_discover_tests(self, temp_git_repo: Path):
        disco = discover_tests(temp_git_repo)
        assert disco.total_count == 2
        assert any("test_add" in node for node in disco.test_node_ids)
        assert any("test_divide" in node for node in disco.test_node_ids)

    def test_run_pytest_timeout(self, tmp_path: Path):
        # Create a test file that sleeps
        test_file = tmp_path / "test_hanging.py"
        test_file.write_text(
            "import time\n"
            "def test_hang():\n"
            "    time.sleep(5)\n",
            encoding="utf-8",
        )
        result = run_pytest(tmp_path, timeout=0.5)
        assert result.passed is False
        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.output.lower()

    def test_run_pytest_no_tests_collected(self, tmp_path: Path):
        # Running pytest in empty folder returns code 5
        result = run_pytest(tmp_path)
        assert result.passed is False
        assert result.exit_code == 5
        assert result.tests_ran == 0


class TestCheckTestReproducesBug:
    """Verify bug reproduction correlation and false positive rejection."""

    def test_reproduces_distinctive_exception_positive(self):
        result = PydanticTestResult(
            passed=False,
            exit_code=1,
            duration_seconds=0.5,
            output="FAILED tests/test_calc.py::test_zero - ZeroDivisionError: division by zero",
            tests_ran=1,
            failures=1,
            failure_details=["FAILED tests/test_calc.py::test_zero - ZeroDivisionError: division by zero"],
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="Calculator raises ZeroDivisionError when dividing by zero.",
            stack_trace="ZeroDivisionError: division by zero",
        )
        assert repro is True
        assert "zerodivisionerror" in reason.lower()

    def test_reproduces_broad_exception_with_corroborating_keyword_positive(self):
        result = PydanticTestResult(
            passed=False,
            exit_code=1,
            duration_seconds=0.5,
            output="FAILED tests/test_parser.py::test_date - ValueError: invalid date string format",
            tests_ran=1,
            failures=1,
            failure_details=["FAILED tests/test_parser.py::test_date - ValueError: invalid date string format"],
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="Date parser raises ValueError when parsing ISO timestamp.",
            stack_trace=None,
        )
        assert repro is True
        assert "valueerror" in reason.lower()
        assert "parser" in reason.lower() or "date" in reason.lower()

    def test_reproduces_multiple_keywords_non_exception_positive(self):
        result = PydanticTestResult(
            passed=False,
            exit_code=1,
            duration_seconds=0.5,
            output="FAILED tests/test_cart.py::test_checkout - AssertionError: total discount coupon calculation negative",
            tests_ran=1,
            failures=1,
            failure_details=["FAILED tests/test_cart.py::test_checkout - AssertionError: total discount coupon calculation negative"],
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="Discount coupon calculation results in negative total price.",
            stack_trace=None,
        )
        assert repro is True
        assert "keywords" in reason.lower()

    def test_rejects_single_keyword_overlap_negative(self):
        """A single generic word overlap (e.g. 'calculator' or 'user') must NOT claim bug reproduction."""
        result = PydanticTestResult(
            passed=False,
            exit_code=1,
            duration_seconds=0.5,
            output="FAILED tests/test_calculator.py::test_history - AttributeError: 'NoneType' object has no attribute 'history'",
            tests_ran=1,
            failures=1,
            failure_details=["FAILED tests/test_calculator.py::test_history - AttributeError: 'NoneType' object has no attribute 'history'"],
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="Calculator raises ZeroDivisionError when dividing by zero.",
            stack_trace=None,
        )
        assert repro is False
        assert "did not match reported bug symptoms" in reason

    def test_rejects_broad_exception_without_corroborating_keyword_negative(self):
        """A broad exception (e.g. ValueError) in an unrelated test must NOT reproduce the bug."""
        result = PydanticTestResult(
            passed=False,
            exit_code=1,
            duration_seconds=0.5,
            output="FAILED tests/test_math.py::test_sqrt - ValueError: math domain error",
            tests_ran=1,
            failures=1,
            failure_details=["FAILED tests/test_math.py::test_sqrt - ValueError: math domain error"],
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="User authentication raises ValueError when token signature is malformed.",
            stack_trace=None,
        )
        assert repro is False
        assert "lacks corroborating keywords" in reason

    def test_rejects_unrelated_failure_negative(self):
        result = PydanticTestResult(
            passed=False,
            exit_code=1,
            duration_seconds=0.5,
            output="FAILED tests/test_network.py::test_connect - ConnectionRefusedError: port 8080",
            tests_ran=1,
            failures=1,
            failure_details=["FAILED tests/test_network.py::test_connect - ConnectionRefusedError: port 8080"],
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="Database query times out during transaction commit.",
            stack_trace=None,
        )
        assert repro is False
        assert "did not match reported bug symptoms" in reason

    def test_rejects_passing_test_result_negative(self):
        result = PydanticTestResult(
            passed=True,
            exit_code=0,
            duration_seconds=0.1,
            output="2 passed",
            tests_ran=2,
            passed_count=2,
            failures=0,
            errors=0,
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="ZeroDivisionError in calculator",
        )
        assert repro is False
        assert "All tests passed" in reason

    def test_rejects_failure_without_details_negative(self):
        result = PydanticTestResult(
            passed=False,
            exit_code=2,
            duration_seconds=0.1,
            output="pytest: error: unrecognized arguments",
            tests_ran=0,
            failures=0,
            errors=0,
            failure_details=[],
        )
        repro, reason = check_test_reproduces_bug(
            test_result=result,
            bug_description="ZeroDivisionError in calculator",
        )
        assert repro is False
        assert "no specific failure details were captured" in reason

