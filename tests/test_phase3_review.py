"""Comprehensive review tests verifying Phase 3 audit requirements.

Covers:
1. Pytest 1-pass / 1-fail accurate breakdown, failing test detection, and bug reproduction.
2. Evidence validation: every hypothesis references real evidence, and evidence contains real tool output.
3. COMPLETED status semantics: diagnostic workflow completed, not fix verified.
4. Honest representation of insufficient evidence and tool execution failures.
"""

from pathlib import Path
import subprocess
import pytest

from rootcause.core.exceptions import InvalidEvidenceReferenceError
from rootcause.core.models import (
    EvidenceType,
    InvestigationRequest,
    InvestigationStatus,
)
from rootcause.llm.base import LLMResponse, ToolCall
from rootcause.llm.mock_provider import MockLLMProvider
from rootcause.orchestrator.agent import InvestigationOrchestrator
from rootcause.tools.registry import ToolRegistry
from rootcause.tools.test_runner import check_test_reproduces_bug, run_pytest


@pytest.fixture
def repo_with_one_pass_one_fail(tmp_path: Path) -> Path:
    """Create a temporary git repository with 1 passing test and 1 failing test reproducing ZeroDivisionError."""
    repo = tmp_path / "mixed_repo"
    repo.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True, capture_output=True)

    src_dir = repo / "src" / "math_app"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "calc.py").write_text(
        "def add(x: int, y: int) -> int:\n"
        "    return x + y\n\n"
        "def divide(x: int, y: int) -> float:\n"
        "    if y == 0:\n"
        "        raise ZeroDivisionError('integer division or modulo by zero')\n"
        "    return x / y\n",
        encoding="utf-8",
    )

    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_calc.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
        "from math_app.calc import add, divide\n\n"
        "def test_add_works():\n"
        "    assert add(10, 20) == 30\n\n"
        "def test_divide_zero_reproduces_bug():\n"
        "    # Deliberately fails by asserting divide(10, 0) == 0 instead of catching exception\n"
        "    assert divide(10, 0) == 0\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit with 1 passing and 1 failing test"], cwd=str(repo), check=True, capture_output=True)

    return repo


class TestPhase3ReviewAudit:
    """Verify all 8 Phase 3 audit requirements."""

    def test_pytest_one_pass_one_fail_accurate_reporting(self, repo_with_one_pass_one_fail: Path):
        """Verify that a test run with 1 passing and 1 failing test is accurately parsed and correlated."""
        result = run_pytest(repo_with_one_pass_one_fail)

        # 1. Verification of test count breakdown
        assert result.passed is False
        assert result.exit_code == 1
        assert result.tests_ran == 2
        assert result.passed_count == 1
        assert result.failures == 1
        assert result.errors == 0
        assert result.timed_out is False

        # 2. Verification that the specific failing test is identified
        assert any("test_divide_zero_reproduces_bug" in line for line in result.failure_details)

        # 3. Correlation with bug report: matching symptoms reproduce the bug
        repro_true, reason_true = check_test_reproduces_bug(
            test_result=result,
            bug_description="Calculator raises ZeroDivisionError when dividing by zero.",
            stack_trace="ZeroDivisionError: integer division or modulo by zero",
        )
        assert repro_true is True
        assert "zerodivisionerror" in reason_true.lower()

        # 4. Correlation with unrelated bug report: should honestly report no reproduction
        repro_false, reason_false = check_test_reproduces_bug(
            test_result=result,
            bug_description="Authentication token expired unexpectedly during OAuth flow.",
            stack_trace=None,
        )
        assert repro_false is False
        assert "did not match reported bug symptoms" in reason_false

    def test_evidence_extraction_stores_real_outputs(self, repo_with_one_pass_one_fail: Path):
        """Verify each evidence item contains real, empirical tool outputs."""
        registry = ToolRegistry(
            repo_path=repo_with_one_pass_one_fail,
            bug_description="Calculator raises ZeroDivisionError",
        )

        # 1. File read produces real content
        read_res = registry.execute("read_repo_file", {"relative_path": "src/math_app/calc.py"})
        assert read_res.success is True
        assert len(read_res.evidence_items) == 1
        ev_file = read_res.evidence_items[0]
        assert ev_file.evidence_type == EvidenceType.SOURCE_CODE
        assert "def divide" in ev_file.content
        assert ev_file.file_path == "src/math_app/calc.py"

        # 2. Search produces real matching line and line numbers
        search_res = registry.execute("search_code", {"query": "ZeroDivisionError"})
        assert search_res.success is True
        assert len(search_res.evidence_items) >= 1
        ev_search = search_res.evidence_items[0]
        assert "ZeroDivisionError" in ev_search.content
        assert ev_search.line_start is not None

        # 3. Pytest execution produces accurate breakdown in evidence
        test_res = registry.execute("run_pytest", {})
        assert test_res.success is True
        assert len(test_res.evidence_items) == 1
        ev_test = test_res.evidence_items[0]
        assert ev_test.evidence_type == EvidenceType.TEST_OUTPUT
        assert ev_test.metadata["passed_count"] == 1
        assert ev_test.metadata["failures"] == 1
        assert ev_test.metadata["tests_ran"] == 2
        assert "test_divide_zero_reproduces_bug" in ev_test.metadata["failing_tests"][0]
        assert ev_test.metadata["reproduces_bug"] is True
        assert "1 passed, 1 failed" in ev_test.why_relevant

    def test_reject_hypotheses_without_supporting_evidence(self, repo_with_one_pass_one_fail: Path):
        """Hypotheses lacking supporting evidence must be rejected as ungrounded."""
        provider = MockLLMProvider()
        # Model attempts to submit hypothesis without supporting evidence IDs
        provider.queue_tool_call(
            "submit_hypotheses",
            {
                "hypotheses": [
                    {
                        "title": "Ungrounded Guess",
                        "description": "Guessing without linking any evidence",
                        "why_plausible": "Intuition only",
                        "supporting_evidence_ids": [],
                    }
                ]
            },
        )

        orchestrator = InvestigationOrchestrator(provider=provider)
        req = InvestigationRequest(
            repo_path=str(repo_with_one_pass_one_fail),
            bug_description="Investigate division bug",
        )
        state = orchestrator.run_investigation(req)

        assert state.status == InvestigationStatus.FAILED
        assert "must cite at least one valid supporting evidence ID" in state.error_message

    def test_completed_status_semantics_and_report_generation(self, repo_with_one_pass_one_fail: Path):
        """Verify COMPLETED means diagnostic workflow concluded, not that a fix was verified."""
        orch_holder = []

        def dynamic_llm(messages, tools, step):
            if step == 1:
                return LLMResponse(tool_calls=[ToolCall(name="read_repo_file", arguments={"relative_path": "src/math_app/calc.py"})])
            elif step == 2:
                return LLMResponse(tool_calls=[ToolCall(name="run_pytest", arguments={})])
            else:
                orch = orch_holder[0]
                real_evidence_id = orch.active_state.facts[0].id
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            name="submit_hypotheses",
                            arguments={
                                "hypotheses": [
                                    {
                                        "title": "Missing ZeroDivisionError guard",
                                        "description": "calc.py raises ZeroDivisionError when y == 0.",
                                        "why_plausible": "Observed code raise and failing test.",
                                        "confidence_score": 0.9,
                                        "uncertainty_level": "LOW",
                                        "supporting_evidence_ids": [real_evidence_id],
                                        "recommended_next_step": "Implement zero-check in calc.py",
                                    }
                                ]
                            },
                        )
                    ]
                )

        provider = MockLLMProvider(handler=dynamic_llm)
        orchestrator = InvestigationOrchestrator(provider=provider)
        orch_holder.append(orchestrator)

        req = InvestigationRequest(
            repo_path=str(repo_with_one_pass_one_fail),
            bug_description="Calculator raises ZeroDivisionError when dividing by zero.",
        )
        state = orchestrator.run_investigation(req)

        assert state.status == InvestigationStatus.COMPLETED

        # Generate report
        report = orchestrator.generate_report(state)

        # Confirm COMPLETED status is explicitly clarified as diagnostic
        assert "**Workflow Status**: `COMPLETED`" in report.markdown_report
        assert "Diagnostic only" in report.markdown_report

        # Confirm test breakdown is clearly shown in report
        assert "1 passed, 1 failed (2 total)" in report.markdown_report
        assert "Failing Tests:" in report.markdown_report
        assert "test_divide_zero_reproduces_bug" in report.markdown_report
        assert "**Reproduces Reported Bug:** YES" in report.markdown_report

        # Confirm target repository was NOT modified (read-only diagnosis)
        diff_proc = subprocess.run(["git", "diff"], cwd=str(repo_with_one_pass_one_fail), capture_output=True, text=True)
        assert diff_proc.stdout == ""

    def test_insufficient_evidence_represented_honestly(self, repo_with_one_pass_one_fail: Path):
        """Verify that declaring insufficient evidence is honestly reflected in report."""
        provider = MockLLMProvider()
        provider.queue_tool_call(
            "declare_insufficient_evidence",
            {
                "reason": "Missing stack trace and reproduction steps.",
                "missing_information": "Production error logs and input parameters.",
            },
        )

        orchestrator = InvestigationOrchestrator(provider=provider)
        req = InvestigationRequest(
            repo_path=str(repo_with_one_pass_one_fail),
            bug_description="Intermittent failure in batch job",
        )
        state = orchestrator.run_investigation(req)

        assert state.status == InvestigationStatus.COMPLETED
        report = orchestrator.generate_report(state)

        assert "**Insufficient Evidence Notice:** Missing stack trace and reproduction steps." in report.markdown_report
        assert "**Missing Information Needed:** Production error logs and input parameters." in report.markdown_report
        assert "No conclusive root-cause hypotheses could be verified" in report.markdown_report

    def test_failed_tool_execution_honestly_represented(self, repo_with_one_pass_one_fail: Path):
        """Verify that tool execution failures record ERROR status and do not inject fake facts."""
        provider = MockLLMProvider()
        # Attempt to read non-existent file
        provider.queue_tool_call("read_repo_file", {"relative_path": "non_existent_module.py"})
        provider.queue_tool_call("declare_insufficient_evidence", {"reason": "File not found"})

        orchestrator = InvestigationOrchestrator(provider=provider)
        req = InvestigationRequest(
            repo_path=str(repo_with_one_pass_one_fail),
            bug_description="Bug in missing file",
        )
        state = orchestrator.run_investigation(req)

        # Audit step must be ERROR
        read_step = [s for s in state.audit_trail if s.selected_tool == "read_repo_file"][0]
        assert read_step.status == "ERROR"
        assert read_step.evidence_collected_ids == []
        assert "Execution error: FileNotFoundError" in read_step.tool_result_summary

        # No facts collected for this failed step
        assert not any(f.source_tool == "read_repo_file" for f in state.facts)
