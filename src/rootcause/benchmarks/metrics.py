"""Auditable scoring and metric calculation for RootCause AI benchmarks."""

import statistics
from typing import Dict, List, Optional, Tuple

from rootcause.benchmarks.models import (
    ArtifactSource,
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkMetrics,
    GateMetric,
    RootCauseScore,
)
from rootcause.core.models import InvestigationState, InvestigationStatus

# Pricing per 1M tokens in USD: (prompt_price, completion_price)
KNOWN_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
}


def estimate_model_cost(
    model_name: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    """Calculate estimated cost in USD if model pricing is known, else None."""
    if not model_name or prompt_tokens is None or completion_tokens is None:
        return None
    model_lower = model_name.lower()
    for prefix, (p_price, c_price) in KNOWN_MODEL_PRICING.items():
        if prefix in model_lower:
            cost = (prompt_tokens / 1_000_000.0) * p_price + (completion_tokens / 1_000_000.0) * c_price
            return round(cost, 6)
    return None


def evaluate_root_cause_match(
    case: BenchmarkCase,
    state: InvestigationState,
) -> RootCauseScore:
    """Evaluate whether an investigation's diagnosis accurately identifies the root cause.
    
    Uses deterministic, rule-based matching against ground truth:
      1. Primary hypothesis must exist, be non-empty, and reference valid observed evidence.
      2. Primary hypothesis must identify at least one relevant source file (via evidence or text).
      3. Primary hypothesis text must contain at least 2 (or all if <2) expected
         distinctive keywords.
          
    This metric represents heuristic matching accuracy rather than proof of
    semantic correctness.
    
    Args:
        case: BenchmarkCase specification.
        state: Final InvestigationState.
        
    Returns:
        RootCauseScore with detailed match breakdown.
    """
    if not state.hypotheses:
        return RootCauseScore(
            matched=False,
            metric_label="heuristic matching accuracy",
            matched_keywords=[],
            missing_keywords=list(case.expected_root_cause_keywords),
            matched_files=[],
            reason="No hypotheses were formed during the investigation.",
        )

    # Primary hypothesis is rank 1 (or first in list)
    primary_hyp = state.hypotheses[0]
    
    # Check for empty or whitespace-only content across all structured fields
    hyp_parts = [
        primary_hyp.title or "",
        primary_hyp.description or "",
        primary_hyp.why_plausible or "",
        primary_hyp.recommended_next_step or "",
    ]
    hyp_text = " ".join(hyp_parts).strip().lower()
    if not hyp_text:
        return RootCauseScore(
            matched=False,
            metric_label="heuristic matching accuracy",
            matched_keywords=[],
            missing_keywords=list(case.expected_root_cause_keywords),
            matched_files=[],
            reason="Primary hypothesis is empty or contains no descriptive content.",
        )

    # Verify that supporting evidence IDs actually exist in state.facts
    fact_id_map = {f.id: f for f in state.facts}
    valid_supporting_facts = [
        fact_id_map[eid] for eid in primary_hyp.supporting_evidence_ids if eid in fact_id_map
    ]
    if not primary_hyp.supporting_evidence_ids:
        return RootCauseScore(
            matched=False,
            metric_label="heuristic matching accuracy",
            matched_keywords=[],
            missing_keywords=list(case.expected_root_cause_keywords),
            matched_files=[],
            reason="Primary hypothesis lacks supporting evidence references.",
        )
    if not valid_supporting_facts:
        return RootCauseScore(
            matched=False,
            metric_label="heuristic matching accuracy",
            matched_keywords=[],
            missing_keywords=list(case.expected_root_cause_keywords),
            matched_files=[],
            reason=f"Primary hypothesis references nonexistent evidence IDs: {primary_hyp.supporting_evidence_ids}.",
        )

    # Collect source files referenced in supporting evidence or hypothesis text
    fact_files = {
        f.file_path.replace("\\", "/").lower()
        for f in valid_supporting_facts
        if f.file_path
    }

    matched_files = [
        rf for rf in case.relevant_files
        if any(rf.lower() in ff for ff in fact_files) or rf.lower() in hyp_text
    ]

    matched_keywords = [
        kw for kw in case.expected_root_cause_keywords
        if kw.lower() in hyp_text
    ]
    missing_keywords = [
        kw for kw in case.expected_root_cause_keywords
        if kw.lower() not in hyp_text
    ]

    min_required_keywords = min(2, len(case.expected_root_cause_keywords))
    files_ok = len(matched_files) > 0
    keywords_ok = len(matched_keywords) >= min_required_keywords

    matched = files_ok and keywords_ok

    if matched:
        reason = (
            f"Diagnosis identified relevant file(s) {matched_files} with verified evidence "
            f"[{', '.join(f.id for f in valid_supporting_facts)}] and matched "
            f"{len(matched_keywords)}/{len(case.expected_root_cause_keywords)} distinctive keywords: {matched_keywords}."
        )
    elif not files_ok and not keywords_ok:
        reason = (
            f"Diagnosis failed both criteria: no relevant source files identified, "
            f"and matched {len(matched_keywords)}/{len(case.expected_root_cause_keywords)} keywords (missing: {missing_keywords})."
        )
    elif not files_ok:
        reason = (
            f"Diagnosis matched keywords {matched_keywords}, but failed to identify relevant source files {case.relevant_files}."
        )
    else:
        reason = (
            f"Diagnosis identified file(s) {matched_files}, but matched insufficient distinctive keywords "
            f"({len(matched_keywords)}/{min_required_keywords} required; missing: {missing_keywords})."
        )

    return RootCauseScore(
        matched=matched,
        metric_label="heuristic matching accuracy",
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        matched_files=matched_files,
        reason=reason,
    )


def _fmt_pct(rate: Optional[float]) -> Optional[str]:
    """Format float rate as percentage string, or None if rate is None."""
    return f"{rate * 100:.1f}%" if rate is not None else None


def calculate_benchmark_metrics(case_results: List[BenchmarkCaseResult]) -> BenchmarkMetrics:
    """Aggregate benchmark results into structured counts and population rates.
    
    Computes per-stage metrics with exact numerators and denominators.
    Undefined rates (zero denominator) are explicitly set to None with an explanation.
    
    Args:
        case_results: List of evaluated case results.
        
    Returns:
        BenchmarkMetrics model.
    """
    total = len(case_results)

    # Completed investigations: requires legitimate investigation completion with evidence or formed hypotheses
    completed_statuses = {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.PATCH_PROPOSED,
        InvestigationStatus.REGRESSION_TEST_CREATED,
        InvestigationStatus.VERIFYING,
        InvestigationStatus.PATCH_VERIFIED,
        InvestigationStatus.VERIFICATION_FAILED,
    }
    completed_count = sum(
        1 for r in case_results
        if r.investigation_status in completed_statuses and (r.evidence_count > 0 or r.hypothesis_source != ArtifactSource.NONE)
    )
    completion_rate = round(completed_count / total, 4) if total > 0 else None

    # Root cause match
    matched_count = sum(1 for r in case_results if r.root_cause_score.matched)
    match_rate = round(matched_count / total, 4) if total > 0 else None

    # Patch generation
    patches_count = sum(1 for r in case_results if r.patch_generated)
    patch_rate = round(patches_count / total, 4) if total > 0 else None

    # Regression test generation
    tests_count = sum(1 for r in case_results if r.regression_test_generated)
    test_rate = round(tests_count / total, 4) if total > 0 else None

    # Gate A: Baseline reproduction
    gate_a_attempts = sum(1 for r in case_results if r.gate_a_attempted)
    gate_a_successes = sum(1 for r in case_results if r.gate_a_passed is True)
    gate_a_failures = sum(1 for r in case_results if r.gate_a_attempted and r.gate_a_passed is False and not r.timed_out)
    gate_a_timeouts = sum(1 for r in case_results if r.timed_out and r.failing_gate == "GATE_A")
    gate_a_inconclusive = sum(1 for r in case_results if r.investigation_status == InvestigationStatus.INVESTIGATION_INCONCLUSIVE)
    gate_a_rate = round(gate_a_successes / gate_a_attempts, 4) if gate_a_attempts > 0 else None

    gate_a_metric = GateMetric(
        gate_name="GATE_A",
        eligible_attempts=gate_a_attempts,
        successes=gate_a_successes,
        failures=gate_a_failures,
        timeouts=gate_a_timeouts,
        inconclusive=gate_a_inconclusive,
        pass_rate=gate_a_rate,
        pass_rate_percentage=_fmt_pct(gate_a_rate),
        denominator_description="Cases that generated candidate patch & test and attempted Gate A",
        zero_denominator_reason="No benchmark cases attempted Gate A reproduction" if gate_a_attempts == 0 else None,
    )

    # Gate B: Patched regression test
    gate_b_attempts = sum(1 for r in case_results if r.gate_b_attempted)
    gate_b_successes = sum(1 for r in case_results if r.gate_b_passed is True)
    gate_b_failures = sum(1 for r in case_results if r.gate_b_attempted and r.gate_b_passed is False and not r.timed_out)
    gate_b_timeouts = sum(1 for r in case_results if r.timed_out and r.failing_gate == "GATE_B")
    gate_b_inconclusive = 0
    gate_b_rate = round(gate_b_successes / gate_b_attempts, 4) if gate_b_attempts > 0 else None

    gate_b_metric = GateMetric(
        gate_name="GATE_B",
        eligible_attempts=gate_b_attempts,
        successes=gate_b_successes,
        failures=gate_b_failures,
        timeouts=gate_b_timeouts,
        inconclusive=gate_b_inconclusive,
        pass_rate=gate_b_rate,
        pass_rate_percentage=_fmt_pct(gate_b_rate),
        denominator_description="Cases that passed Gate A and attempted Gate B",
        zero_denominator_reason="No benchmark cases passed Gate A to attempt Gate B" if gate_b_attempts == 0 else None,
    )

    # Gate C: Full test suite regressions
    gate_c_attempts = sum(1 for r in case_results if r.gate_c_attempted)
    gate_c_successes = sum(1 for r in case_results if r.gate_c_passed is True)
    gate_c_failures = sum(1 for r in case_results if r.gate_c_attempted and r.gate_c_passed is False and not r.timed_out)
    gate_c_timeouts = sum(1 for r in case_results if r.timed_out and r.failing_gate == "GATE_C")
    gate_c_inconclusive = 0
    gate_c_rate = round(gate_c_successes / gate_c_attempts, 4) if gate_c_attempts > 0 else None

    gate_c_metric = GateMetric(
        gate_name="GATE_C",
        eligible_attempts=gate_c_attempts,
        successes=gate_c_successes,
        failures=gate_c_failures,
        timeouts=gate_c_timeouts,
        inconclusive=gate_c_inconclusive,
        pass_rate=gate_c_rate,
        pass_rate_percentage=_fmt_pct(gate_c_rate),
        denominator_description="Cases that passed Gate B and attempted Gate C",
        zero_denominator_reason="No benchmark cases passed Gate B to attempt Gate C" if gate_c_attempts == 0 else None,
    )

    # Overall verified patches
    verified_count = sum(1 for r in case_results if r.verified and r.investigation_status == InvestigationStatus.PATCH_VERIFIED)
    verification_rate = round(verified_count / total, 4) if total > 0 else None

    failed_count = sum(1 for r in case_results if r.investigation_status in {InvestigationStatus.FAILED, InvestigationStatus.VERIFICATION_FAILED})
    inconclusive_count = sum(
        1 for r in case_results
        if r.investigation_status in {InvestigationStatus.INVESTIGATION_INCONCLUSIVE, InvestigationStatus.INSUFFICIENT_EVIDENCE}
    )
    timeouts_count = sum(1 for r in case_results if r.timed_out)
    errors_count = sum(1 for r in case_results if r.error_message is not None)
    rate_limit_count = sum(1 for r in case_results if r.rate_limited or r.error_type == "RateLimitError")
    auth_error_count = sum(1 for r in case_results if r.error_type == "AuthenticationError")
    config_error_count = sum(1 for r in case_results if r.error_type in ("ConfigurationError", "MissingCredentialsError"))

    # Token and cost accounting
    has_prompt = any(r.prompt_tokens is not None for r in case_results)
    total_prompt = sum(r.prompt_tokens for r in case_results if r.prompt_tokens is not None) if has_prompt else None

    has_completion = any(r.completion_tokens is not None for r in case_results)
    total_completion = sum(r.completion_tokens for r in case_results if r.completion_tokens is not None) if has_completion else None

    has_total = any(r.total_tokens is not None for r in case_results)
    total_tokens = sum(r.total_tokens for r in case_results if r.total_tokens is not None) if has_total else None

    has_cost = any(r.estimated_cost_usd is not None for r in case_results)
    total_cost = round(sum(r.estimated_cost_usd for r in case_results if r.estimated_cost_usd is not None), 6) if has_cost else None

    durations = [r.duration_seconds for r in case_results if r.duration_seconds > 0]
    mean_runtime = round(statistics.mean(durations), 4) if durations else None
    median_runtime = round(statistics.median(durations), 4) if durations else None

    return BenchmarkMetrics(
        total_cases=total,
        completed_investigations=completed_count,
        investigation_completion_rate=completion_rate,
        investigation_completion_rate_percentage=_fmt_pct(completion_rate),
        root_cause_matches=matched_count,
        root_cause_match_rate=match_rate,
        root_cause_match_rate_percentage=_fmt_pct(match_rate),
        patches_generated=patches_count,
        patch_generation_rate=patch_rate,
        patch_generation_rate_percentage=_fmt_pct(patch_rate),
        regression_tests_generated=tests_count,
        regression_test_generation_rate=test_rate,
        regression_test_generation_rate_percentage=_fmt_pct(test_rate),
        gate_a=gate_a_metric,
        gate_b=gate_b_metric,
        gate_c=gate_c_metric,
        verified_patches=verified_count,
        overall_verification_rate=verification_rate,
        overall_verification_rate_percentage=_fmt_pct(verification_rate),
        failed_cases=failed_count,
        inconclusive_cases=inconclusive_count,
        timeouts=timeouts_count,
        execution_errors=errors_count,
        rate_limit_errors=rate_limit_count,
        authentication_errors=auth_error_count,
        configuration_errors=config_error_count,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_tokens=total_tokens,
        total_estimated_cost_usd=total_cost,
        mean_runtime_seconds=mean_runtime,
        median_runtime_seconds=median_runtime,
    )
