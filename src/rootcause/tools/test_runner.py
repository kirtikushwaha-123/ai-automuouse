"""Controlled pytest execution and test discovery tools."""

from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

from rootcause.core.exceptions import CommandTimeoutError
from rootcause.core.models import TestDiscoveryResult, TestResult
from rootcause.security.path_guard import validate_safe_path
from rootcause.security.sandbox import SubprocessRunner

# Patterns to extract pytest summary counts
PASSED_RE = re.compile(r"(\d+)\s+passed")
FAILED_RE = re.compile(r"(\d+)\s+failed")
ERRORS_RE = re.compile(r"(\d+)\s+error")
SKIPPED_RE = re.compile(r"(\d+)\s+skipped")


def discover_tests(
    repo_or_worktree_path: Union[Path, str],
    test_target: Optional[str] = None,
    timeout: float = 30.0,
) -> TestDiscoveryResult:
    """Discover available test nodes safely using pytest --collect-only without running them.
    
    Args:
        repo_or_worktree_path: Path to target repository or worktree.
        test_target: Optional sub-folder or file to limit collection.
        timeout: Execution timeout in seconds.
        
    Returns:
        TestDiscoveryResult with list of collected test node IDs and count.
    """
    resolved_root = Path(repo_or_worktree_path).resolve()
    runner = SubprocessRunner(allowed_root=resolved_root, default_timeout=timeout)

    command: List[str] = ["python", "-m", "pytest", "--collect-only", "-q"]

    if test_target:
        target_path_part = test_target.split("::")[0]
        if target_path_part:
            safe_target = validate_safe_path(resolved_root, target_path_part)
            rel_target = str(safe_target.relative_to(resolved_root)).replace("\\", "/")
            if "::" in test_target:
                node_part = test_target[len(target_path_part):]
                command.append(f"{rel_target}{node_part}")
            else:
                command.append(rel_target)

    try:
        exec_res = runner.run(command, cwd=resolved_root, timeout=timeout)
    except CommandTimeoutError as exc:
        return TestDiscoveryResult(
            test_node_ids=[],
            total_count=0,
            output=f"Test discovery timed out after {exc.timeout_seconds}s",
        )

    output = f"{exec_res.stdout}\n{exec_res.stderr}".strip()
    node_ids: List[str] = []

    for line in output.splitlines():
        trimmed = line.strip()
        # Pytest -q --collect-only outputs node IDs like 'tests/test_foo.py::test_bar'
        if "::" in trimmed and not trimmed.startswith("ERROR") and not trimmed.startswith("!"):
            node_ids.append(trimmed)

    return TestDiscoveryResult(
        test_node_ids=node_ids,
        total_count=len(node_ids),
        output=output,
    )


def run_pytest(
    repo_or_worktree_path: Union[Path, str],
    test_target: Optional[str] = None,
    timeout: float = 30.0,
) -> TestResult:
    """Execute pytest under controlled sandbox constraints and parse structured results.
    
    Args:
        repo_or_worktree_path: Path to target repository or isolated worktree.
        test_target: Optional test file path or specific test node (e.g. tests/test_calc.py).
        timeout: Execution timeout in seconds.
        
    Returns:
        Structured TestResult model.
    """
    resolved_root = Path(repo_or_worktree_path).resolve()
    runner = SubprocessRunner(allowed_root=resolved_root, default_timeout=timeout)

    command: List[str] = ["python", "-m", "pytest", "-q", "--tb=short"]

    if test_target:
        # If test_target is a file path or node id, validate path portion
        target_path_part = test_target.split("::")[0]
        if target_path_part:
            safe_target = validate_safe_path(resolved_root, target_path_part)
            rel_target = str(safe_target.relative_to(resolved_root)).replace("\\", "/")
            if "::" in test_target:
                node_part = test_target[len(target_path_part):]
                command.append(f"{rel_target}{node_part}")
            else:
                command.append(rel_target)

    try:
        exec_res = runner.run(command, cwd=resolved_root, timeout=timeout)
    except CommandTimeoutError as exc:
        return TestResult(
            passed=False,
            exit_code=-1,
            duration_seconds=exc.timeout_seconds,
            output=f"Pytest execution timed out after {exc.timeout_seconds}s",
            timed_out=True,
            failures=1,
            failure_details=["TimeoutExpired: Test run took longer than the configured limit."],
        )

    output = f"{exec_res.stdout}\n{exec_res.stderr}".strip()

    # Parse counts
    passed_m = PASSED_RE.search(output)
    failed_m = FAILED_RE.search(output)
    errors_m = ERRORS_RE.search(output)
    skipped_m = SKIPPED_RE.search(output)

    passed_count = int(passed_m.group(1)) if passed_m else 0
    failed_count = int(failed_m.group(1)) if failed_m else 0
    errors_count = int(errors_m.group(1)) if errors_m else 0
    skipped_count = int(skipped_m.group(1)) if skipped_m else 0

    tests_ran = passed_count + failed_count + errors_count

    # Extract failure lines
    failure_details: List[str] = []
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR ") or "AssertionError" in line:
            failure_details.append(line.strip())

    return TestResult(
        passed=(exec_res.exit_code == 0),
        exit_code=exec_res.exit_code,
        duration_seconds=exec_res.duration_seconds,
        output=output,
        timed_out=False,
        tests_ran=tests_ran,
        passed_count=passed_count,
        failures=failed_count,
        errors=errors_count,
        skipped=skipped_count,
        failure_details=failure_details,
    )


# Distinctive exception types that unambiguously pinpoint a failure mode
DISTINCTIVE_EXCEPTIONS = {
    "zerodivisionerror",
    "keyerror",
    "indexerror",
    "filenotfounderror",
    "modulenotfounderror",
    "importerror",
    "recursionerror",
    "overflowerror",
    "syntaxerror",
    "permissionerror",
    "timeouterror",
    "connectionerror",
    "jsondecodeerror",
    "unicodedecodeerror",
    "notimplementederror",
}

# Broad exceptions commonly raised in unrelated unit tests
BROAD_EXCEPTIONS = {
    "valueerror",
    "typeerror",
    "attributeerror",
    "runtimeerror",
}

# Non-distinctive / generic words and pytest terminology to ignore
STOPWORDS = {
    "when", "raises", "with", "from", "that", "this", "then", "have",
    "test", "tests", "error", "errors", "file", "files", "line", "lines",
    "code", "fail", "fails", "failed", "failure", "failures", "pass", "passes", "passed",
    "true", "false", "none", "null", "expected", "actual", "assert", "assertion",
    "assertionerror", "function", "method", "class", "module", "object",
    "return", "returns", "value", "values", "call", "called", "calling",
    "input", "output", "should", "would", "could", "about", "after",
    "before", "during", "under", "where", "which", "while", "using", "used",
    "into", "some", "such", "only", "also", "more", "most", "been", "being",
    "were", "what", "does", "done", "case", "cases", "check", "checks", "testcase",
    "exception", "baseexception",
} | DISTINCTIVE_EXCEPTIONS | BROAD_EXCEPTIONS


def check_test_reproduces_bug(
    test_result: TestResult,
    bug_description: str,
    stack_trace: Optional[str] = None,
) -> Tuple[bool, str]:
    """Analyze if test failure symptoms align with the reported bug.
    
    Args:
        test_result: Result from pytest run.
        bug_description: Ingested bug description.
        stack_trace: Optional raw traceback from bug report.
        
    Returns:
        (reproduces_bug: bool, justification: str)
    """
    if test_result.passed:
        return False, "All tests passed; reported bug was not reproduced."

    if not test_result.failure_details or (test_result.failures == 0 and test_result.errors == 0):
        return False, f"Test suite failed (exit code {test_result.exit_code}), but no specific failure details were captured."

    ref_text = f"{bug_description} {stack_trace or ''}".lower()

    # Prefer specific failure details and traceback failure sections over full raw stdout
    failure_snippets = list(test_result.failure_details)
    if "=== FAILURES ===" in test_result.output:
        fail_section = test_result.output.split("=== FAILURES ===")[-1]
        if "=== short test summary info ===" in fail_section:
            fail_section = fail_section.split("=== short test summary info ===")[0]
        failure_snippets.append(fail_section)
    elif test_result.output:
        failure_snippets.append(test_result.output)

    combined_failures = " ".join(failure_snippets).lower()
    failure_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", combined_failures))

    # Meaningful identifier keywords (>= 4 chars) from bug description/trace, excluding generic exception names
    ref_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", ref_text))
    meaningful_words = {w for w in ref_tokens if w not in STOPWORDS}
    matched_keywords = sorted(meaningful_words & failure_tokens)

    # Exception matching
    matched_distinctive = [
        exc for exc in sorted(DISTINCTIVE_EXCEPTIONS)
        if exc in ref_text and (exc in failure_tokens or exc in combined_failures)
    ]
    matched_broad = [
        exc for exc in sorted(BROAD_EXCEPTIONS)
        if exc in ref_text and (exc in failure_tokens or exc in combined_failures)
    ]

    # Case 1: Distinctive exception match
    if matched_distinctive:
        exc_str = ", ".join(matched_distinctive)
        if matched_keywords:
            kw_str = ", ".join(matched_keywords[:3])
            return True, f"Failure matches reported symptom: exception '{exc_str}' (corroborating keywords: '{kw_str}')"
        return True, f"Failure matches reported symptom: exception '{exc_str}'"

    # Case 2: Broad exception match (requires at least 1 corroborating keyword)
    if matched_broad:
        exc_str = ", ".join(matched_broad)
        if matched_keywords:
            kw_str = ", ".join(matched_keywords[:3])
            return True, f"Failure matches reported symptom: exception '{exc_str}' (corroborating keywords: '{kw_str}')"
        return False, (
            f"Test failed with broad exception '{exc_str}', but lacks corroborating keywords "
            f"from the bug description to confirm bug reproduction."
        )

    # Case 3: Non-exception or generic failure (requires >= 2 distinctive keywords)
    if len(matched_keywords) >= 2:
        kw_str = ", ".join(matched_keywords[:4])
        return True, f"Failure matches reported symptom: keywords '{kw_str}'"
    elif len(matched_keywords) == 1:
        return False, (
            f"Test failed with exit code {test_result.exit_code}, but failure details did not match "
            f"reported bug symptoms (insufficient symptom match: only 1 keyword '{matched_keywords[0]}' "
            f"matched; minimum 2 required for non-exception failures)."
        )

    return False, f"Test failed with exit code {test_result.exit_code}, but failure details did not match reported bug symptoms."


