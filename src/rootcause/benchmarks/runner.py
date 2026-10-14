"""Benchmark execution runner and reporting engine for RootCause AI."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from rootcause.benchmarks.dataset import (
    get_benchmark_case,
    get_benchmark_cases,
)
from rootcause.benchmarks.fixtures import create_fixture_repository
from rootcause.benchmarks.metrics import (
    calculate_benchmark_metrics,
    estimate_model_cost,
    evaluate_root_cause_match,
)
from rootcause.benchmarks.models import (
    ArtifactSource,
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    EvaluationMode,
    RootCauseScore,
)
from rootcause.benchmarks.simulated_provider import create_simulated_benchmark_provider
from rootcause.benchmarks.tracking import UsageTrackingProviderWrapper
from rootcause.core.exceptions import (
    AuthenticationError,
    CommandTimeoutError,
    ConfigurationError,
    MissingCredentialsError,
    RateLimitError,
    RootCauseException,
)
from rootcause.core.models import (
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
)
from rootcause.llm import get_llm_provider
from rootcause.llm.base import BaseLLMProvider
from rootcause.llm.mock_provider import MockLLMProvider
from rootcause.orchestrator.agent import InvestigationOrchestrator
from rootcause.security.sandbox import SubprocessRunner
from rootcause.verification.patch_engine import generate_candidate_patch
from rootcause.verification.verifier import VerificationService

SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+[a-zA-Z0-9_\-\.]{8,})"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{8,}['\"]?)"),
    re.compile(r"(?i)(secret\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{8,}['\"]?)"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
]

SAFE_BENCHMARK_KEYS = {
    "matched_keywords",
    "missing_keywords",
    "expected_root_cause_keywords",
    "token_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_prompt_tokens",
    "total_completion_tokens",
}


def redact_secrets(data: Any) -> Any:
    """Recursively scrub sensitive keys and token patterns from benchmark result data."""
    if isinstance(data, str):
        scrubbed = data
        for pattern in SECRET_PATTERNS:
            scrubbed = pattern.sub("[REDACTED_SECRET]", scrubbed)
        return scrubbed
    elif isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            k_lower = k.lower()
            if k_lower in SAFE_BENCHMARK_KEYS:
                cleaned[k] = redact_secrets(v)
            elif (
                any(term == k_lower for term in ("key", "secret", "token", "password", "auth", "api_key"))
                or any(k_lower.endswith(suf) for suf in ("_key", "_secret", "_token", "_password", "_auth"))
            ):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_secrets(v)
        return cleaned
    elif isinstance(data, list):
        return [redact_secrets(item) for item in data]
    return data


def _remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree on Windows read-only git pack files."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class BenchmarkRunner:
    """Orchestrates reproducible evaluation across benchmark cases in disposable environments."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        base_temp_dir: Optional[Path] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.SIMULATED_OFFLINE,
        requested_model: Optional[str] = None,
        max_retries: int = 1,
    ):
        self.base_temp_dir = Path(base_temp_dir).resolve() if base_temp_dir else None
        self.evaluation_mode = evaluation_mode
        self.requested_model = requested_model
        self.max_retries = max_retries

        if provider is not None:
            self.provider = provider
            self.evaluation_mode = evaluation_mode or EvaluationMode.LIVE_PROVIDER
        elif self.evaluation_mode == EvaluationMode.LIVE_PROVIDER:
            # Fail closed immediately if credentials are missing
            self.provider = get_llm_provider("openai", model=requested_model)
        else:
            self.provider = None

    def run_case(
        self,
        case_id: str,
        provider: Optional[BaseLLMProvider] = None,
        seed: Optional[int] = None,
    ) -> BenchmarkCaseResult:
        """Run an end-to-end benchmark evaluation against a single fixture case.
        
        Args:
            case_id: Unique benchmark case ID.
            provider: Optional LLM provider override.
            seed: Optional random seed for reproducible runs.
            
        Returns:
            BenchmarkCaseResult documenting the complete diagnosis and verification outcome.
        """
        case = get_benchmark_case(case_id)

        # Determine active provider and evaluation mode
        if provider is not None:
            raw_provider = provider
            is_simulated = False
            eval_mode = EvaluationMode.LIVE_PROVIDER
        elif self.evaluation_mode == EvaluationMode.LIVE_PROVIDER:
            if self.provider is not None:
                raw_provider = self.provider
            else:
                raw_provider = get_llm_provider("openai", model=self.requested_model)
            is_simulated = False
            eval_mode = EvaluationMode.LIVE_PROVIDER
        elif self.provider is not None:
            raw_provider = self.provider
            is_simulated = False
            eval_mode = self.evaluation_mode
        else:
            # Deterministic simulated provider for offline benchmark
            raw_provider = create_simulated_benchmark_provider(case)
            is_simulated = True
            eval_mode = EvaluationMode.SIMULATED_OFFLINE

        # Wrap in usage tracker for token and error telemetry
        if isinstance(raw_provider, UsageTrackingProviderWrapper):
            tracker = raw_provider
            tracker.reset_case_accounting()
        else:
            tracker = UsageTrackingProviderWrapper(
                inner_provider=raw_provider,
                max_retries=self.max_retries,
            )
        active_provider = tracker

        t0 = time.perf_counter()
        temp_dir_str = tempfile.mkdtemp(
            prefix=f"rootcause_bm_{case_id}_",
            dir=str(self.base_temp_dir) if self.base_temp_dir else None,
        )
        temp_repo_path = Path(temp_dir_str).resolve()

        investigation_status = InvestigationStatus.FAILED
        root_cause_score = RootCauseScore(
            matched=False,
            metric_label="heuristic matching accuracy",
            matched_keywords=[],
            missing_keywords=list(case.expected_root_cause_keywords),
            matched_files=[],
            reason="Execution did not start.",
        )
        evidence_count = 0
        evidence_ids: List[str] = []
        hypothesis_source = ArtifactSource.NONE
        hypothesis_text: Optional[str] = None
        evidence_references: List[str] = []
        patch_generated = False
        patch_source = ArtifactSource.NONE
        reg_test_generated = False
        regression_test_source = ArtifactSource.NONE
        gate_a_attempted = False
        gate_a_passed: Optional[bool] = None
        gate_b_attempted = False
        gate_b_passed: Optional[bool] = None
        gate_c_attempted = False
        gate_c_passed: Optional[bool] = None
        verified = False
        failing_gate: Optional[str] = None
        error_message: Optional[str] = None
        error_type: Optional[str] = None
        timed_out = False
        commit_hash: Optional[str] = None

        try:
            # 1. Initialize clean fixture repository in disposable directory
            create_fixture_repository(case_id, temp_repo_path)
            runner = SubprocessRunner(allowed_root=temp_repo_path)
            commit_run = runner.run(["git", "rev-parse", "HEAD"], cwd=temp_repo_path, check=True)
            commit_hash = commit_run.stdout.strip()

            # 2. Execute Investigation Workflow
            orchestrator = InvestigationOrchestrator(
                provider=active_provider,
                db=None,  # Benchmarks run in-memory / stateless
            )
            request = InvestigationRequest(
                repo_path=str(temp_repo_path),
                bug_description=case.bug_description,
                stack_trace=case.stack_trace,
            )
            state = orchestrator.run_investigation(request)
            investigation_status = state.status
            evidence_count = len(state.facts)
            evidence_ids = [f.id for f in state.facts]

            if state.error_message:
                error_message = redact_secrets(state.error_message)
                if ":" in state.error_message:
                    error_type = state.error_message.split(":")[0].strip()
                else:
                    error_type = "InvestigationError"
            elif tracker.rate_limited:
                error_type = "RateLimitError"
                error_message = "LLM rate limit reached during investigation."

            if state.hypotheses:
                hypothesis_source = ArtifactSource.SIMULATED if is_simulated else ArtifactSource.LIVE_LLM
                primary_hyp = state.hypotheses[0]
                hypothesis_text = f"{primary_hyp.title}: {primary_hyp.description}"
                evidence_references = list(primary_hyp.supporting_evidence_ids)

            # 3. Evaluate Root Cause Accuracy via rule-based matching
            root_cause_score = evaluate_root_cause_match(case, state)

            # 4. Generate Candidate Patch and Regression Test
            if state.hypotheses and state.status in {
                InvestigationStatus.COMPLETED,
                InvestigationStatus.HYPOTHESIS_FORMED,
                InvestigationStatus.BUG_REPRODUCED,
                InvestigationStatus.ROOT_CAUSE_HYPOTHESIS,
            }:
                try:
                    cand_patch, reg_test = generate_candidate_patch(
                        provider=active_provider,
                        state=state,
                        hypothesis=state.hypotheses[0],
                    )
                    state.candidate_patch = cand_patch
                    state.regression_test = reg_test
                    patch_generated = True
                    patch_source = ArtifactSource.SIMULATED if is_simulated else ArtifactSource.LIVE_LLM
                    reg_test_generated = True
                    regression_test_source = ArtifactSource.SIMULATED if is_simulated else ArtifactSource.LIVE_LLM
                except Exception as patch_exc:
                    error_message = f"Patch generation failed: {patch_exc}"
                    error_type = type(patch_exc).__name__

            # NO SILENT FALLBACK: If candidate patch was not generated, do not substitute ground-truth patch!

            # 5. Execute 3-Point Verification Pipeline (only if candidate patch and test exist)
            if patch_generated and reg_test_generated:
                verifier = VerificationService(default_timeout=30.0)
                verif_res = verifier.verify(state)

                investigation_status = state.status
                verified = verif_res.verified
                failing_gate = verif_res.failing_gate

                # Gate A
                gate_a_attempted = True
                gate_a_passed = verif_res.gate_a.passed

                # Gate B
                if verif_res.gate_b is not None:
                    gate_b_attempted = True
                    gate_b_passed = verif_res.gate_b.passed

                # Gate C
                if verif_res.gate_c is not None:
                    gate_c_attempted = True
                    gate_c_passed = verif_res.gate_c.passed

        except RateLimitError as exc:
            error_type = "RateLimitError"
            error_message = f"Rate limit exceeded: {redact_secrets(str(exc))}"
            investigation_status = InvestigationStatus.FAILED
        except AuthenticationError as exc:
            error_type = "AuthenticationError"
            error_message = f"Authentication failure: {redact_secrets(str(exc))}"
            investigation_status = InvestigationStatus.FAILED
        except MissingCredentialsError as exc:
            error_type = "MissingCredentialsError"
            error_message = f"Missing credentials: {redact_secrets(str(exc))}"
            investigation_status = InvestigationStatus.FAILED
        except ConfigurationError as exc:
            error_type = "ConfigurationError"
            error_message = f"Configuration error: {redact_secrets(str(exc))}"
            investigation_status = InvestigationStatus.FAILED
        except CommandTimeoutError as exc:
            timed_out = True
            error_type = "CommandTimeoutError"
            error_message = f"Command timed out: {exc}"
            investigation_status = InvestigationStatus.FAILED
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = f"Benchmark case failed with exception: {redact_secrets(str(exc))}"
            investigation_status = InvestigationStatus.FAILED
        finally:
            # Reliably clean up temporary fixture repository
            if temp_repo_path.exists():
                shutil.rmtree(str(temp_repo_path), onerror=_remove_readonly)

        duration = round(time.perf_counter() - t0, 4)

        prompt_tokens = tracker.prompt_tokens
        completion_tokens = tracker.completion_tokens
        total_tokens = tracker.total_tokens
        actual_model = tracker.actual_model
        requested_model = tracker.requested_model
        retry_count = tracker.retry_count
        rate_limited = tracker.rate_limited
        effective_model = actual_model or requested_model
        cost_usd = estimate_model_cost(effective_model, prompt_tokens, completion_tokens)
        token_usage_str = (
            f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
            if total_tokens is not None
            else "unavailable"
        )

        return BenchmarkCaseResult(
            case_id=case.case_id,
            category=case.category,
            evaluation_mode=eval_mode,
            duration_seconds=duration,
            investigation_status=investigation_status,
            root_cause_score=root_cause_score,
            evidence_count=evidence_count,
            evidence_ids=evidence_ids,
            hypothesis_source=hypothesis_source,
            hypothesis_text=hypothesis_text,
            evidence_references=evidence_references,
            patch_generated=patch_generated,
            patch_source=patch_source,
            regression_test_generated=reg_test_generated,
            regression_test_source=regression_test_source,
            gate_a_attempted=gate_a_attempted,
            gate_a_passed=gate_a_passed,
            gate_b_attempted=gate_b_attempted,
            gate_b_passed=gate_b_passed,
            gate_c_attempted=gate_c_attempted,
            gate_c_passed=gate_c_passed,
            verification_executed_by="REAL_WORKFLOW",
            verified=verified,
            failing_gate=failing_gate,
            error_message=error_message,
            error_type=error_type,
            timed_out=timed_out,
            token_usage=token_usage_str,
            requested_model=requested_model,
            actual_model=actual_model,
            retry_count=retry_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
            rate_limited=rate_limited,
            fixture_commit_hash=commit_hash,
        )

    def run_suite(
        self,
        case_ids: Optional[List[str]] = None,
        provider: Optional[BaseLLMProvider] = None,
        seed: Optional[int] = None,
    ) -> BenchmarkSuiteResult:
        """Run all or specified benchmark cases and return aggregated results."""
        all_cases = get_benchmark_cases()
        selected_ids = case_ids if case_ids else list(all_cases.keys())

        results: List[BenchmarkCaseResult] = []

        for cid in selected_ids:
            case_res = self.run_case(cid, provider=provider, seed=seed)
            results.append(case_res)

        metrics = calculate_benchmark_metrics(results)

        # Determine suite evaluation mode and description
        suite_mode = (
            EvaluationMode.SIMULATED_OFFLINE
            if any(r.evaluation_mode == EvaluationMode.SIMULATED_OFFLINE for r in results)
            else EvaluationMode.LIVE_PROVIDER
        )
        if suite_mode == EvaluationMode.SIMULATED_OFFLINE:
            provider_name = "MockLLMProvider (simulated offline)"
            mode_clarification = (
                "SIMULATED_OFFLINE: Results exercise orchestration, evidence gathering, and verification plumbing "
                "using deterministic simulation. Not presented as autonomous LLM diagnostic performance."
            )
        else:
            active_p = provider or self.provider
            provider_name = active_p.__class__.__name__ if active_p else "LiveLLMProvider"
            mode_clarification = (
                "LIVE_PROVIDER: Results evaluate autonomous root-cause diagnosis and verification using an external "
                "live LLM provider. No ground-truth patches or synthetic hypotheses were injected."
            )

        # Suite level token and model aggregates
        has_prompt = any(r.prompt_tokens is not None for r in results)
        total_prompt = sum(r.prompt_tokens for r in results if r.prompt_tokens is not None) if has_prompt else None

        has_completion = any(r.completion_tokens is not None for r in results)
        total_completion = sum(r.completion_tokens for r in results if r.completion_tokens is not None) if has_completion else None

        has_total = any(r.total_tokens is not None for r in results)
        total_tokens = sum(r.total_tokens for r in results if r.total_tokens is not None) if has_total else None

        has_cost = any(r.estimated_cost_usd is not None for r in results)
        total_cost = round(sum(r.estimated_cost_usd for r in results if r.estimated_cost_usd is not None), 6) if has_cost else None

        actual_models = {r.actual_model for r in results if r.actual_model}
        actual_model_str = ", ".join(sorted(actual_models)) if actual_models else None

        suite_result = BenchmarkSuiteResult(
            run_id=str(uuid4()),
            benchmark_version="0.1.0",
            timestamp=datetime.now(timezone.utc),
            evaluation_mode=suite_mode,
            mode_clarification=mode_clarification,
            provider_info=provider_name,
            requested_model=self.requested_model or getattr(provider, "model", None),
            actual_model=actual_model_str,
            seed=seed,
            total_cases=len(results),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            total_estimated_cost_usd=total_cost,
            case_results=results,
            metrics=metrics,
        )

        return suite_result


def main() -> None:
    """CLI entry point for running the RootCause AI benchmark suite."""
    parser = argparse.ArgumentParser(description="RootCause AI Reproducible Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["simulated", "live", "offline"],
        default="simulated",
        help="Evaluation mode: 'simulated' (deterministic offline) or 'live' (live LLM provider)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider name for live mode (e.g. 'openai', 'groq', 'mock')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. 'gpt-4o-mini', 'gpt-4o')",
    )
    parser.add_argument("--case", type=str, default=None, help="Specific benchmark case ID to run")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to write JSON results")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")

    args = parser.parse_args()

    mode_arg = args.mode.lower()
    if mode_arg in ("simulated", "offline"):
        eval_mode = EvaluationMode.SIMULATED_OFFLINE
        runner = BenchmarkRunner(evaluation_mode=eval_mode)
        live_provider = None
    else:
        eval_mode = EvaluationMode.LIVE_PROVIDER
        # Fail closed if live provider credentials cannot be loaded
        try:
            live_provider = get_llm_provider(
                provider_type=args.provider,
                model=args.model,
            )
            runner = BenchmarkRunner(
                provider=live_provider,
                evaluation_mode=eval_mode,
                requested_model=args.model,
            )
        except MissingCredentialsError as exc:
            print("\n" + "!" * 70)
            print("LIVE BENCHMARK NOT RUN: Missing credentials")
            print("To evaluate live LLM diagnostic accuracy, set one of:")
            print("  - ROOTCAUSE_LLM_API_KEY=<your-api-key>")
            print("  - OPENAI_API_KEY=<your-api-key>")
            print("Optional:")
            print("  - ROOTCAUSE_LLM_BASE_URL=<api-base-url>")
            print("  - ROOTCAUSE_LLM_MODEL=<model-name> (default: gpt-4o-mini)")
            print("Per Phase 5.2 integrity rules, live mode will not fall back to simulation.")
            print("!" * 70 + "\n")
            return
        except ConfigurationError as exc:
            print("\n" + "!" * 70)
            print(f"LIVE BENCHMARK NOT RUN: Configuration error: {exc}")
            print("Per Phase 5.2 integrity rules, live mode will not fall back to simulation.")
            print("!" * 70 + "\n")
            return

    case_ids = [args.case] if args.case else None
    suite_result = runner.run_suite(case_ids=case_ids, provider=live_provider, seed=args.seed)

    raw_dict = suite_result.model_dump(mode="json")
    clean_dict = redact_secrets(raw_dict)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean_dict, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"RootCause AI Benchmark Suite — Run ID: {suite_result.run_id}")
    print(f"Evaluation Mode: {suite_result.evaluation_mode.value}")
    print(f"Provider: {suite_result.provider_info}")
    if suite_result.requested_model:
        print(f"Requested Model: {suite_result.requested_model}")
    if suite_result.actual_model:
        print(f"Actual Model(s): {suite_result.actual_model}")
    print(f"Total Cases Evaluated: {suite_result.total_cases}")
    print(f"Completed Investigations: {suite_result.metrics.completed_investigations}/{suite_result.total_cases} ({suite_result.metrics.investigation_completion_rate_percentage or 'N/A'})")
    print(f"Root Cause Matches: {suite_result.metrics.root_cause_matches}/{suite_result.total_cases} ({suite_result.metrics.root_cause_match_rate_percentage or 'N/A'})")
    print(f"Gate A Reproduction: {suite_result.metrics.gate_a.successes}/{suite_result.metrics.gate_a.eligible_attempts} ({suite_result.metrics.gate_a.pass_rate_percentage or 'N/A'})")
    print(f"Gate B Patched Test: {suite_result.metrics.gate_b.successes}/{suite_result.metrics.gate_b.eligible_attempts} ({suite_result.metrics.gate_b.pass_rate_percentage or 'N/A'})")
    print(f"Gate C Full Suite: {suite_result.metrics.gate_c.successes}/{suite_result.metrics.gate_c.eligible_attempts} ({suite_result.metrics.gate_c.pass_rate_percentage or 'N/A'})")
    print(f"Verified Patches: {suite_result.metrics.verified_patches}/{suite_result.total_cases} ({suite_result.metrics.overall_verification_rate_percentage or 'N/A'})")
    if suite_result.total_tokens is not None:
        print(f"Total Tokens: {suite_result.total_tokens} (prompt: {suite_result.total_prompt_tokens}, completion: {suite_result.total_completion_tokens})")
    if suite_result.total_estimated_cost_usd is not None:
        print(f"Estimated Cost: ${suite_result.total_estimated_cost_usd:.6f}")
    if suite_result.metrics.rate_limit_errors > 0:
        print(f"Rate Limit Errors: {suite_result.metrics.rate_limit_errors}")
    print(f"Integrity Note: {suite_result.mode_clarification}")
    print(f"Results written to: {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
