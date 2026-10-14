"""Deterministic simulated LLM provider for benchmark plumbing evaluation."""

import json
import re
from typing import Any, Dict, List, Optional

from rootcause.benchmarks.models import BenchmarkCase
from rootcause.llm.base import BaseLLMProvider, LLMResponse, ToolCall
from rootcause.llm.mock_provider import MockLLMProvider

UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def create_simulated_benchmark_provider(case: BenchmarkCase) -> MockLLMProvider:
    """Create a deterministic MockLLMProvider that simulates tool-based diagnosis.
    
    The simulation exercises:
      1. Inspecting the relevant source file with `read_repo_file`.
      2. Extracting the real captured evidence ID from tool output.
      3. Submitting an explicit evidence-grounded hypothesis labeled '[SIMULATED]'.
      4. Synthesizing a unified diff patch and regression test from ground truth.
      
    Args:
        case: BenchmarkCase specification.
        
    Returns:
        Configured MockLLMProvider instance.
    """
    call_index = [0]
    captured_evidence_ids: List[str] = []

    def handler(
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        step: int,
    ) -> LLMResponse:
        call_index[0] += 1
        current_step = call_index[0]

        # Check if the prompt is asking for candidate patch generation
        last_msg = messages[-1] if messages else {}
        last_content = str(last_msg.get("content", ""))

        if "Generate a minimal unified diff to fix the bug" in last_content or "unified diff" in last_content:
            patch_data = {
                "diff": case.known_good_patch,
                "test_file_path": case.regression_test_file,
                "test_name": "test_reproduction",
                "test_content": case.regression_test_code,
                "explanation": f"[SIMULATED] Ground-truth patch for {case.case_id}",
            }
            return LLMResponse(
                content=json.dumps(patch_data),
                tool_calls=[],
                finish_reason="stop",
            )

        # Scan incoming messages for newly collected Evidence IDs
        for msg in messages:
            content = str(msg.get("content", ""))
            # Check for [Evidence IDs: ...] prefix
            evidence_header_match = re.search(r"\[Evidence IDs:\s*([^\]]+)\]", content)
            if evidence_header_match:
                for eid in evidence_header_match.group(1).split(","):
                    clean_eid = eid.strip()
                    if clean_eid and clean_eid not in captured_evidence_ids:
                        captured_evidence_ids.append(clean_eid)
            # Catch any UUID strings
            for uid in UUID_PATTERN.findall(content):
                if uid not in captured_evidence_ids:
                    captured_evidence_ids.append(uid)

        # Investigation Step 1: Read the primary relevant repository file
        if current_step == 1:
            target_file = case.relevant_files[0] if case.relevant_files else "src/app.py"
            return LLMResponse(
                content=f"[SIMULATED] Inspecting source file {target_file} for root cause analysis.",
                tool_calls=[
                    ToolCall(
                        name="read_repo_file",
                        arguments={"relative_path": target_file},
                    )
                ],
                finish_reason="tool_calls",
            )

        # Investigation Step 2: Form and submit evidence-backed hypothesis
        evidence_id = captured_evidence_ids[0] if captured_evidence_ids else "ev_missing"
        target_file = case.relevant_files[0] if case.relevant_files else "unknown"
        keywords_str = ", ".join(case.expected_root_cause_keywords)

        simulated_hyp = {
            "title": f"[SIMULATED] {case.name} Root Cause in {target_file}",
            "description": (
                f"[SIMULATED] In {target_file}: {case.known_root_cause}. "
                f"Root cause involves {keywords_str}."
            ),
            "why_plausible": (
                f"[SIMULATED] Empirical evidence in {target_file} confirms {keywords_str} discrepancy "
                f"causing expected symptom: '{case.expected_symptom}'."
            ),
            "confidence_score": 0.95,
            "uncertainty_level": "LOW",
            "supporting_evidence_ids": [evidence_id],
            "contradictory_evidence_ids": [],
            "recommended_next_step": f"[SIMULATED] Apply verified patch to {target_file}.",
        }

        return LLMResponse(
            content="[SIMULATED] Sufficient evidence gathered. Submitting root-cause hypothesis.",
            tool_calls=[
                ToolCall(
                    name="submit_hypotheses",
                    arguments={"hypotheses": [simulated_hyp]},
                )
            ],
            finish_reason="tool_calls",
        )

    return MockLLMProvider(handler=handler)
