"""Autonomous investigation orchestrator driving evidence-based bug diagnosis."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from rootcause.core.exceptions import (
    InvalidEvidenceReferenceError,
    InvalidRepositoryError,
    InvalidStateTransitionError,
    RootCauseException,
    ToolArgumentValidationError,
    UnregisteredToolError,
)
from rootcause.core.models import (
    AuditStep,
    Evidence,
    EvidenceType,
    Hypothesis,
    InvestigationReport,
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
)
from rootcause.db.repository import InvestigationRepository
from rootcause.llm.base import BaseLLMProvider
from rootcause.tools.registry import ToolRegistry, redact_arguments


class InvestigationOrchestrator:
    """Manages the lifecycle and iterative tool execution loop of a bug investigation."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        db: Optional[InvestigationRepository] = None,
        default_max_iterations: int = 10,
    ):
        self.provider = provider
        self.db = db
        self.default_max_iterations = default_max_iterations

    def run_investigation(
        self,
        request: InvestigationRequest,
    ) -> InvestigationState:
        """Run an autonomous bug investigation from start to completion.
        
        Args:
            request: Ingestion payload containing repository path and bug description.
            
        Returns:
            Final InvestigationState snapshot.
        """
        resolved_repo = Path(request.repo_path).resolve()
        if not resolved_repo.exists() or not resolved_repo.is_dir():
            raise FileNotFoundError(f"Repository directory does not exist: {resolved_repo}")

        state = InvestigationState(
            investigation_id=str(uuid4()),
            status=InvestigationStatus.RECEIVED,
            request=request,
            facts=[],
            hypotheses=[],
            audit_trail=[],
        )
        self.active_state = state

        if self.db:
            self.db.save(state)

        try:
            # 1. Transition RECEIVED -> INVESTIGATING
            state.transition_to(InvestigationStatus.INVESTIGATING)
            if self.db:
                self.db.save(state)

            registry = ToolRegistry(
                repo_path=resolved_repo,
                bug_description=request.bug_description,
                stack_trace=request.stack_trace,
            )

            # 2. Ingest initial stack trace evidence if provided
            if request.stack_trace:
                trace_res = registry.execute(
                    "parse_traceback",
                    {"trace_text": request.stack_trace},
                    step_index=0,
                )
                if trace_res.success and trace_res.evidence_items:
                    state.facts.extend(trace_res.evidence_items)
                    state.audit_trail.append(AuditStep(
                        step_index=0,
                        selected_tool="parse_traceback",
                        arguments={"trace_text": "[RAW_STACK_TRACE]"},
                        status="SUCCESS",
                        tool_result_summary="Parsed initial input stack trace.",
                        evidence_collected_ids=[e.id for e in trace_res.evidence_items],
                        reasoning="Initial stack trace intake.",
                    ))

            # 3. Iterative Tool Loop
            max_iter = min(request.max_iterations, self.default_max_iterations)
            completed = False

            # Add submission tools to registry
            self._register_orchestrator_tools(registry, state)
            tools_schema = registry.get_all_tool_definitions()

            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are RootCause AI, an evidence-based autonomous bug investigation agent.\n"
                        "Your mission is to investigate software bugs using repository files, git history, and tests.\n\n"
                        "STRICT RULES:\n"
                        "1. Observe real facts: read files, search code, inspect git commits, or run tests.\n"
                        "2. Every hypothesis you propose MUST link to real Evidence IDs gathered in this investigation.\n"
                        "3. Never hallucinate line numbers, file names, or test outputs.\n"
                        "4. When evidence contradicts a hypothesis, mark the contradictory evidence ID or revise it.\n"
                        "5. When you have sufficient evidence, call 'submit_hypotheses'.\n"
                        "6. If you cannot find sufficient evidence, call 'declare_insufficient_evidence'.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt_context(state),
                },
            ]

            step = 1
            while step <= max_iter and not completed:
                response = self.provider.generate(messages, tools=tools_schema)

                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.name
                        tool_args = tool_call.arguments or {}

                        if tool_name == "submit_hypotheses":
                            completed = self._handle_submit_hypotheses(
                                state=state,
                                raw_args=tool_args,
                                step_index=step,
                            )
                            if completed:
                                break

                        elif tool_name == "declare_insufficient_evidence":
                            completed = self._handle_insufficient_evidence(
                                state=state,
                                raw_args=tool_args,
                                step_index=step,
                            )
                            if completed:
                                break

                        else:
                            # Standard registered tool execution
                            clean_args = redact_arguments(tool_args)
                            exec_result = registry.execute(
                                tool_name=tool_name,
                                raw_args=tool_args,
                                step_index=step,
                            )

                            if exec_result.success and exec_result.evidence_items:
                                state.facts.extend(exec_result.evidence_items)

                            evidence_ids = [e.id for e in exec_result.evidence_items]
                            audit_step = AuditStep(
                                step_index=step,
                                selected_tool=tool_name,
                                arguments=clean_args,
                                status="SUCCESS" if exec_result.success else "ERROR",
                                tool_result_summary=exec_result.output_summary,
                                evidence_collected_ids=evidence_ids,
                                reasoning=response.content,
                            )
                            state.audit_trail.append(audit_step)

                            # Append tool result to dialogue
                            tool_msg_content = exec_result.output_summary
                            if evidence_ids:
                                tool_msg_content = f"[Evidence IDs: {', '.join(evidence_ids)}]\n" + tool_msg_content

                            messages.append({
                                "role": "assistant",
                                "content": response.content,
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(tool_args),
                                        },
                                    }
                                ],
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_msg_content,
                            })

                else:
                    # Model produced text without tool calls
                    reasoning_text = response.content or ""
                    state.audit_trail.append(AuditStep(
                        step_index=step,
                        selected_tool="reasoning",
                        arguments={},
                        status="SUCCESS",
                        tool_result_summary=reasoning_text[:500],
                        evidence_collected_ids=[],
                        reasoning=reasoning_text,
                    ))
                    # Check if model text signals end of investigation
                    if "insufficient evidence" in reasoning_text.lower():
                        self._handle_insufficient_evidence(
                            state=state,
                            raw_args={"reason": reasoning_text},
                            step_index=step,
                        )
                        completed = True
                        break
                    else:
                        messages.append({"role": "assistant", "content": reasoning_text})
                        messages.append({
                            "role": "user",
                            "content": "Please continue investigating using tools, or call submit_hypotheses / declare_insufficient_evidence.",
                        })

                if self.db:
                    self.db.save(state)
                step += 1

            # 4. Handle iteration limit or unfinished investigations
            if not completed:
                if state.hypotheses:
                    if state.status == InvestigationStatus.INVESTIGATING:
                        state.transition_to(InvestigationStatus.HYPOTHESIS_FORMED)
                    state.transition_to(InvestigationStatus.COMPLETED)
                else:
                    state.insufficient_evidence_reason = (
                        f"Investigation reached maximum iteration limit ({max_iter}) "
                        "without finding sufficient evidence to form a root-cause hypothesis."
                    )
                    state.missing_information = "Additional logs, reproduction steps, or targeted test cases are required."
                    state.transition_to(InvestigationStatus.INSUFFICIENT_EVIDENCE)

                state.audit_trail.append(AuditStep(
                    step_index=step,
                    selected_tool="orchestrator",
                    arguments={},
                    status="SUCCESS",
                    tool_result_summary="Investigation stopped: iteration limit reached.",
                    reasoning=f"Reached maximum limit of {max_iter} iterations without explicit conclusion.",
                ))

        except Exception as exc:
            state.error_message = f"{type(exc).__name__}: {str(exc)}"
            try:
                state.transition_to(InvestigationStatus.FAILED)
            except Exception:
                state.status = InvestigationStatus.FAILED
            state.audit_trail.append(AuditStep(
                step_index=len(state.audit_trail) + 1,
                selected_tool="orchestrator",
                arguments={},
                status="ERROR",
                tool_result_summary=f"Investigation aborted due to error: {exc}",
                reasoning=str(exc),
            ))

        if self.db:
            self.db.save(state)

        return state

    def _build_prompt_context(self, state: InvestigationState) -> str:
        """Construct user message prompt with current known facts and problem."""
        prompt = [
            f"Bug Description: {state.request.bug_description}",
        ]
        if state.request.error_log:
            prompt.append(f"Error Log:\n{state.request.error_log[:1500]}")

        if state.facts:
            prompt.append("\nObserved Facts So Far:")
            for f in state.facts:
                loc = f" in {f.file_path}:{f.line_start}" if f.file_path else ""
                prompt.append(f"- [Evidence ID: {f.id}] ({f.evidence_type.value}{loc}): {f.content[:200]}")

        return "\n".join(prompt)

    def _handle_submit_hypotheses(
        self,
        state: InvestigationState,
        raw_args: Dict[str, Any],
        step_index: int,
    ) -> bool:
        """Validate evidence references and record final hypotheses."""
        raw_hypotheses = raw_args.get("hypotheses", [])
        if not raw_hypotheses and "title" in raw_args:
            raw_hypotheses = [raw_args]

        known_evidence_ids = {f.id for f in state.facts}
        validated_hypotheses: List[Hypothesis] = []

        for h_data in raw_hypotheses:
            supporting_ids = h_data.get("supporting_evidence_ids", [])
            contradictory_ids = h_data.get("contradictory_evidence_ids", [])

            # Reject hypotheses without ANY supporting evidence!
            if not supporting_ids:
                raise InvalidEvidenceReferenceError(
                    f"Hypothesis '{h_data.get('title', 'Untitled')}' must cite at least one "
                    f"valid supporting evidence ID from observed facts. Ungrounded hypotheses are rejected."
                )

            # Reject fabricated evidence references!
            for eid in supporting_ids:
                if eid not in known_evidence_ids:
                    raise InvalidEvidenceReferenceError(
                        f"Hypothesis references non-existent supporting evidence ID: '{eid}'. "
                        f"Must reference an observed fact from {list(known_evidence_ids)}."
                    )
            for eid in contradictory_ids:
                if eid not in known_evidence_ids:
                    raise InvalidEvidenceReferenceError(
                        f"Hypothesis references non-existent contradictory evidence ID: '{eid}'. "
                        f"Must reference an observed fact from {list(known_evidence_ids)}."
                    )

            hypo = Hypothesis(
                rank=len(validated_hypotheses) + 1,
                title=h_data.get("title", "Proposed Root Cause"),
                description=h_data.get("description", ""),
                why_plausible=h_data.get("why_plausible", ""),
                confidence_score=float(h_data.get("confidence_score", 0.5)),
                uncertainty_level=h_data.get("uncertainty_level", "MODERATE"),
                supporting_evidence_ids=supporting_ids,
                contradictory_evidence_ids=contradictory_ids,
                recommended_next_step=h_data.get("recommended_next_step"),
            )
            validated_hypotheses.append(hypo)

        state.hypotheses = validated_hypotheses

        # Transition: INVESTIGATING -> HYPOTHESIS_FORMED -> COMPLETED
        state.transition_to(InvestigationStatus.HYPOTHESIS_FORMED)
        state.transition_to(InvestigationStatus.COMPLETED)

        state.audit_trail.append(AuditStep(
            step_index=step_index,
            selected_tool="submit_hypotheses",
            arguments=redact_arguments(raw_args),
            status="SUCCESS",
            tool_result_summary=f"Formed {len(validated_hypotheses)} evidence-backed hypothesis/es.",
            evidence_collected_ids=[],
            reasoning="Sufficient evidence gathered to form root-cause hypothesis.",
        ))
        return True

    def _handle_insufficient_evidence(
        self,
        state: InvestigationState,
        raw_args: Dict[str, Any],
        step_index: int,
    ) -> bool:
        """Handle agent declaring insufficient evidence."""
        reason = raw_args.get("reason", "Lacks sufficient repository or log evidence.")
        missing_info = raw_args.get("missing_information", "Reproduction steps or logs")

        state.insufficient_evidence_reason = reason
        state.missing_information = missing_info

        state.transition_to(InvestigationStatus.INSUFFICIENT_EVIDENCE)
        state.transition_to(InvestigationStatus.COMPLETED)

        state.audit_trail.append(AuditStep(
            step_index=step_index,
            selected_tool="declare_insufficient_evidence",
            arguments=redact_arguments(raw_args),
            status="SUCCESS",
            tool_result_summary=f"Investigation concluded: Insufficient evidence. Reason: {reason}",
            reasoning=f"Missing: {missing_info}",
        ))
        return True

    def _register_orchestrator_tools(self, registry: ToolRegistry, state: InvestigationState) -> None:
        """Register conclusion pseudo-tools for the LLM."""
        from pydantic import BaseModel, Field

        class SubmitHypothesisItem(BaseModel):
            title: str
            description: str
            why_plausible: str
            confidence_score: float = Field(0.5, ge=0.0, le=1.0)
            uncertainty_level: str = "MODERATE"
            supporting_evidence_ids: List[str] = Field(..., min_length=1, description="List of observed fact IDs supporting this hypothesis")
            contradictory_evidence_ids: List[str] = Field(default_factory=list)
            recommended_next_step: Optional[str] = None

        class SubmitHypothesesArgs(BaseModel):
            hypotheses: List[SubmitHypothesisItem]

        class InsufficientEvidenceArgs(BaseModel):
            reason: str
            missing_information: Optional[str] = None

        registry.register(
            name="submit_hypotheses",
            description="Submit final evidence-backed root-cause hypotheses and conclude investigation.",
            schema=SubmitHypothesesArgs,
            handler=lambda repo, args: args,
        )

        registry.register(
            name="declare_insufficient_evidence",
            description="Conclude investigation when available repository evidence and logs are insufficient.",
            schema=InsufficientEvidenceArgs,
            handler=lambda repo, args: args,
        )

    def generate_report(self, state: InvestigationState) -> InvestigationReport:
        """Produce a structured investigation report in Markdown and JSON format."""
        status_clarification = (
            "Investigation workflow completed (Diagnostic only — no code modifications or fix verifications performed)"
            if state.status == InvestigationStatus.COMPLETED
            else f"Investigation {state.status.value}"
        )

        lines = [
            f"# RootCause AI Investigation Report: {state.investigation_id}",
            f"\n**Repository**: `{state.request.repo_path}`",
            f"**Workflow Status**: `{state.status.value}` — {status_clarification}",
            f"**Bug Description**: {state.request.bug_description}",
            f"**Investigated At**: {state.updated_at.isoformat()}\n",
            "## 1. Root-Cause Hypotheses\n",
        ]

        if not state.hypotheses:
            lines.append("*No conclusive root-cause hypotheses could be verified with the available evidence.*")
            if state.insufficient_evidence_reason:
                lines.append(f"\n> **Insufficient Evidence Notice:** {state.insufficient_evidence_reason}")
            if state.missing_information:
                lines.append(f"> **Missing Information Needed:** {state.missing_information}\n")
        else:
            for idx, h in enumerate(state.hypotheses, start=1):
                lines.append(f"### Hypothesis {idx}: {h.title} (Rank {h.rank})")
                lines.append(f"- **Description**: {h.description}")
                lines.append(f"- **Why Plausible**: {h.why_plausible}")
                lines.append(f"- **Uncertainty Level**: `{h.uncertainty_level}` (Heuristic score: {h.confidence_score:.2f})")
                lines.append(f"- **Supporting Evidence IDs**: {', '.join(f'`{eid}`' for eid in h.supporting_evidence_ids) or 'None'}")
                if h.contradictory_evidence_ids:
                    lines.append(f"- **Contradictory Evidence IDs**: {', '.join(f'`{eid}`' for eid in h.contradictory_evidence_ids)}")
                if h.recommended_next_step:
                    lines.append(f"- **Recommended Next Action**: {h.recommended_next_step}")
                lines.append("")

        lines.append("## 2. Collected Empirical Evidence\n")
        if not state.facts:
            lines.append("*No empirical evidence collected.*")
        else:
            for f in state.facts:
                loc = f"`{f.file_path}:{f.line_start}`" if f.file_path else f"`{f.source}`"
                lines.append(f"- **[{f.id[:8]}]** ({f.evidence_type.value}) {loc}: {f.why_relevant or ''}")
                if f.evidence_type == EvidenceType.TEST_OUTPUT and f.metadata:
                    m = f.metadata
                    passed_c = m.get("passed_count", 0)
                    failed_c = m.get("failures", 0) + m.get("errors", 0)
                    total_c = m.get("tests_ran", passed_c + failed_c)
                    failing_t = m.get("failing_tests", [])
                    repro = m.get("reproduces_bug", False)
                    lines.append(f"  - **Test Breakdown:** {passed_c} passed, {failed_c} failed ({total_c} total)")
                    if failing_t:
                        lines.append(f"  - **Failing Tests:** {', '.join(failing_t)}")
                    lines.append(f"  - **Reproduces Reported Bug:** {'YES' if repro else 'NO'} ({m.get('reproduction_reason', '')})")
                lines.append(f"  ```\n  {f.content[:300].strip()}\n  ```")

        lines.append("\n## 3. Investigation Audit Trail\n")
        for step in state.audit_trail:
            lines.append(f"- **Step {step.step_index}**: `{step.selected_tool}` [{step.status}] — {step.tool_result_summary[:200]}")

        markdown_report = "\n".join(lines)

        return InvestigationReport(
            investigation_id=state.investigation_id,
            repo_path=state.request.repo_path,
            bug_description=state.request.bug_description,
            final_status=state.status,
            hypotheses=state.hypotheses,
            evidence=state.facts,
            audit_trail=state.audit_trail,
            markdown_report=markdown_report,
        )
