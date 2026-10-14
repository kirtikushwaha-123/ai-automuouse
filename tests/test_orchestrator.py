"""Tests for autonomous investigation orchestrator."""

from pathlib import Path
import pytest

from rootcause.core.exceptions import (
    InvalidEvidenceReferenceError,
    ToolArgumentValidationError,
    UnregisteredToolError,
)
from rootcause.core.models import (
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
)
from rootcause.db.repository import InvestigationRepository
from rootcause.llm.base import LLMResponse, ToolCall
from rootcause.llm.mock_provider import MockLLMProvider
from rootcause.orchestrator.agent import InvestigationOrchestrator


class TestInvestigationOrchestrator:
    """Verify autonomous investigation workflow, evidence anchoring, and limits."""

    def test_mock_investigation_identifies_known_bug(self, temp_git_repo: Path):
        db = InvestigationRepository()
        orch_holder = []

        def dynamic_llm(messages, tools, step):
            if step == 1:
                return LLMResponse(tool_calls=[ToolCall(name="read_repo_file", arguments={"relative_path": "src/app/calculator.py"})])
            elif step == 2:
                return LLMResponse(tool_calls=[ToolCall(name="run_pytest", arguments={})])
            else:
                active_orch = orch_holder[0]
                assert active_orch.active_state is not None
                assert len(active_orch.active_state.facts) > 0
                real_id = active_orch.active_state.facts[0].id
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            name="submit_hypotheses",
                            arguments={
                                "hypotheses": [
                                    {
                                        "title": "Unhandled ZeroDivisionError in divide()",
                                        "description": "divide(a, b) raises ZeroDivisionError when denominator is 0.",
                                        "why_plausible": "Observed code raise in calculator.py",
                                        "confidence_score": 0.9,
                                        "uncertainty_level": "LOW",
                                        "supporting_evidence_ids": [real_id],
                                        "contradictory_evidence_ids": [],
                                        "recommended_next_step": "Add input validation or zero check",
                                    }
                                ]
                            },
                        ),
                    ]
                )

        provider = MockLLMProvider(handler=dynamic_llm)
        orch = InvestigationOrchestrator(provider=provider, db=db, default_max_iterations=5)
        orch_holder.append(orch)

        req = InvestigationRequest(
            repo_path=str(temp_git_repo),
            bug_description="Calculator raises ZeroDivisionError when dividing by zero.",
            stack_trace="Traceback (most recent call last):\n  File \"src/app/calculator.py\", line 6, in divide\nZeroDivisionError: division by zero",
        )
        state = orch.run_investigation(req)

        assert state.status == InvestigationStatus.COMPLETED
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].title == "Unhandled ZeroDivisionError in divide()"
        assert len(state.hypotheses[0].supporting_evidence_ids) == 1
        assert len(state.facts) >= 2  # parsed traceback + read_repo_file + run_pytest
        assert len(state.audit_trail) >= 3

        # Test report generation
        report = orch.generate_report(state)
        assert report.investigation_id == state.investigation_id
        assert "ZeroDivisionError in divide()" in report.markdown_report
        assert "Supporting Evidence IDs" in report.markdown_report

    def test_reject_fabricated_evidence_references(self, temp_git_repo: Path):
        provider = MockLLMProvider()
        # Immediately try to submit a hypothesis citing a fake evidence ID
        provider.queue_tool_call(
            "submit_hypotheses",
            {
                "hypotheses": [
                    {
                        "title": "Fabricated Bug",
                        "description": "Hallucinated root cause",
                        "why_plausible": "No real facts",
                        "supporting_evidence_ids": ["totally-fabricated-id-99999"],
                    }
                ]
            },
        )

        orchestrator = InvestigationOrchestrator(provider=provider, default_max_iterations=3)
        req = InvestigationRequest(
            repo_path=str(temp_git_repo),
            bug_description="Bug with fake evidence",
        )

        state = orchestrator.run_investigation(req)
        # Investigation must transition to FAILED due to fabricated evidence rejection!
        assert state.status == InvestigationStatus.FAILED
        assert "non-existent supporting evidence ID" in state.error_message

    def test_iteration_limit_enforcement(self, temp_git_repo: Path):
        provider = MockLLMProvider()
        # Queue continuous read calls that never conclude
        for _ in range(10):
            provider.queue_tool_call("get_git_status", {})

        orchestrator = InvestigationOrchestrator(provider=provider, default_max_iterations=3)
        req = InvestigationRequest(
            repo_path=str(temp_git_repo),
            bug_description="Unresolved bug reaching limit",
            max_iterations=3,
        )

        state = orchestrator.run_investigation(req)
        assert state.status == InvestigationStatus.INSUFFICIENT_EVIDENCE
        # Because no hypotheses were formed, audit trail must record iteration limit
        assert any("iteration limit reached" in step.tool_result_summary.lower() for step in state.audit_trail)

    def test_declare_insufficient_evidence(self, tmp_path: Path):
        provider = MockLLMProvider()
        provider.queue_tool_call(
            "declare_insufficient_evidence",
            {
                "reason": "Repository is empty and no stack trace was provided.",
                "missing_information": "Need source files and reproduction steps.",
            },
        )

        orchestrator = InvestigationOrchestrator(provider=provider)
        req = InvestigationRequest(
            repo_path=str(tmp_path),
            bug_description="Bug in empty folder",
        )

        state = orchestrator.run_investigation(req)
        assert state.status == InvestigationStatus.COMPLETED
        assert any("insufficient evidence" in step.tool_result_summary.lower() for step in state.audit_trail)

    def test_unregistered_tool_call_handled_safely(self, temp_git_repo: Path):
        provider = MockLLMProvider()
        # Model tries to call dangerous unauthorized bash tool
        provider.queue_tool_call("execute_bash", {"command": "rm -rf /"})
        provider.queue_tool_call(
            "declare_insufficient_evidence",
            {"reason": "Unauthorized tool rejected"},
        )

        orchestrator = InvestigationOrchestrator(provider=provider)
        req = InvestigationRequest(
            repo_path=str(temp_git_repo),
            bug_description="Test tool guard",
        )

        state = orchestrator.run_investigation(req)
        assert state.status == InvestigationStatus.COMPLETED
        # The audit trail must record ERROR for execute_bash
        bash_step = [s for s in state.audit_trail if s.selected_tool == "execute_bash"]
        assert len(bash_step) == 1
        assert bash_step[0].status == "ERROR"
        assert "not in registered tools" in bash_step[0].tool_result_summary
