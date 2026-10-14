"""Comprehensive test suite for RootCause AI Phase 5.1 & 5.1.1: Benchmark Infrastructure & Evaluation Integrity."""

import json
from pathlib import Path
import shutil
import tempfile
from typing import Generator
import pytest
from pydantic import ValidationError

from rootcause.benchmarks.dataset import (
    get_benchmark_case,
    get_benchmark_cases,
    validate_all_fixtures,
)
from rootcause.benchmarks.fixtures import (
    create_fixture_repository,
    validate_fixture,
)
from rootcause.benchmarks.metrics import (
    calculate_benchmark_metrics,
    evaluate_root_cause_match,
)
from rootcause.benchmarks.models import (
    ArtifactSource,
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkCategory,
    BenchmarkDifficulty,
    EvaluationMode,
    RootCauseScore,
)
from rootcause.benchmarks.runner import BenchmarkRunner, redact_secrets
from rootcause.core.models import (
    Evidence,
    EvidenceType,
    Hypothesis,
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
)
from rootcause.llm.base import LLMResponse, ToolCall
from rootcause.llm.mock_provider import MockLLMProvider
from rootcause.security.sandbox import SubprocessRunner


@pytest.fixture
def temp_fixtures_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for building disposable fixture repos."""
    temp_dir = tempfile.mkdtemp(prefix="rootcause_test_fixtures_")
    path = Path(temp_dir).resolve()
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(str(path), ignore_errors=True)


class TestBenchmarkDatasetAndSchemas:
    """Acceptance Criterion A & E: Verify curated cases and Pydantic schemas."""

    def test_benchmark_cases_registry_integrity(self):
        cases = get_benchmark_cases()
        assert len(cases) == 5, f"Expected exactly 5 curated cases, found {len(cases)}"
        expected_ids = {
            "BM01_CONDITIONAL_LOGIC",
            "BM02_BOUNDARY_OFF_BY_ONE",
            "BM03_DATA_HANDLING_NONE",
            "BM04_CODE_REGRESSION",
            "BM05_ZERO_DIVISION",
        }
        assert set(cases.keys()) == expected_ids

        for cid, case in cases.items():
            assert case.case_id == cid
            assert case.name
            assert case.known_root_cause
            assert len(case.expected_root_cause_keywords) >= 2, f"{cid} has fewer than 2 keywords"
            assert len(case.relevant_files) >= 1, f"{cid} has no relevant files"
            assert "--- a/" in case.known_good_patch
            assert "+++ b/" in case.known_good_patch
            assert "def test_" in case.regression_test_code

    def test_benchmark_case_schema_validation(self):
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        assert case.category == BenchmarkCategory.CONDITIONAL_LOGIC
        assert case.difficulty == BenchmarkDifficulty.EASY
        assert case.expected_verification_outcome == InvestigationStatus.PATCH_VERIFIED

        # Test invalid case raises ValidationError
        with pytest.raises(ValidationError):
            BenchmarkCase(
                case_id="INVALID",
                name="Missing fields",
                category="NOT_A_CATEGORY",  # type: ignore
                difficulty=BenchmarkDifficulty.EASY,
                bug_description="missing fields",
                known_root_cause="none",
                expected_symptom="err",
                fixture_relative_dir="dir",
                known_good_patch="diff",
                regression_test_code="test",
            )


class TestBenchmarkFixtures:
    """Acceptance Criterion A: All 5 fixture repositories must pass 5-point validation."""

    @pytest.mark.parametrize("case_id", [
        "BM01_CONDITIONAL_LOGIC",
        "BM02_BOUNDARY_OFF_BY_ONE",
        "BM03_DATA_HANDLING_NONE",
        "BM04_CODE_REGRESSION",
        "BM05_ZERO_DIVISION",
    ])
    def test_all_fixture_repositories_validate(self, case_id: str, temp_fixtures_dir: Path):
        case = get_benchmark_case(case_id)
        fixture_dir = temp_fixtures_dir / case.fixture_relative_dir

        # Setup fixture repository
        create_fixture_repository(case_id, fixture_dir)

        # Run acceptance criteria verification
        result = validate_fixture(
            case_id=case_id,
            repo_path=fixture_dir,
            bug_description=case.bug_description,
            stack_trace=case.stack_trace,
            regression_test_file=case.regression_test_file,
            regression_test_code=case.regression_test_code,
            known_good_patch=case.known_good_patch,
        )

        assert result["valid"] is True, f"Fixture {case_id} failed stage: {result.get('stage_failed')} — {result.get('message')}"
        assert result["stage_failed"] is None
        assert result["pre_existing_tests_count"] > 0
        assert result["commit_hash"] is not None


class TestCodeRegressionCommitSequence:
    """Verify BM04 code regression fixture reproduces through an explicit multi-commit sequence."""

    def test_bm04_multi_commit_sequence(self, temp_fixtures_dir: Path):
        fixture_dir = temp_fixtures_dir / "BM04_COMMIT_TEST"
        create_fixture_repository("BM04_CODE_REGRESSION", fixture_dir)

        runner = SubprocessRunner(allowed_root=fixture_dir)
        log_res = runner.run(["git", "log", "--oneline"], cwd=fixture_dir, check=True)
        commits = log_res.stdout.strip().splitlines()

        # Must have at least 2 distinct commits documenting the refactoring regression
        assert len(commits) >= 2
        assert any("refactor sanitize_input" in c.lower() for c in commits)
        assert any("initial commit" in c.lower() for c in commits)


class TestRuleBasedScorer:
    """Acceptance Criterion C & Phase 5.1.1 Step 3: Comprehensive root-cause scorer tests."""

    def test_scorer_positive_match(self):
        """Correct root cause with matching keywords and relevant files."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(
            repo_path="/test",
            bug_description=case.bug_description,
        )
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/discount.py:4-6",
            file_path="src/discount.py",
            content="if total > 100.0:\n    return round(total * 0.10, 2)",
        )
        hyp = Hypothesis(
            title="Incorrect threshold comparison",
            description="The discount calculation in src/discount.py uses strictly greater instead of greater than or equal.",
            why_plausible="The threshold check fails on orders of exactly 100.",
            confidence_score=0.9,
            supporting_evidence_ids=[fact.id],
        )
        state = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[hyp],
        )

        score = evaluate_root_cause_match(case, state)
        assert score.matched is True
        assert score.metric_label == "heuristic matching accuracy"
        assert "src/discount.py" in score.matched_files
        assert len(score.matched_keywords) >= 2
        assert "discount" in score.matched_keywords
        assert "verified evidence" in score.reason.lower()

    def test_scorer_generic_overlapping_keywords_rejected(self):
        """Incorrect root cause containing generic overlapping keywords without distinctive domain terms."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(
            repo_path="/test",
            bug_description=case.bug_description,
        )
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/discount.py",
            file_path="src/discount.py",
            content="some code",
        )
        # Uses generic words like 'error', 'bug', 'fails', 'broken', but none of 'discount', 'threshold', 'greater'
        hyp = Hypothesis(
            title="Generic error bug in file",
            description="There is an unexpected failure and broken logic causing an error in the calculation routine.",
            why_plausible="Observed bug issue during execution.",
            confidence_score=0.5,
            supporting_evidence_ids=[fact.id],
        )
        state = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[hyp],
        )

        score = evaluate_root_cause_match(case, state)
        assert score.matched is False
        assert "insufficient distinctive keywords" in score.reason.lower()
        assert len(score.matched_keywords) == 0

    def test_scorer_correct_keywords_wrong_file_rejected(self):
        """Correct keywords but incorrect source files."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(
            repo_path="/test",
            bug_description=case.bug_description,
        )
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/other_service.py",
            file_path="src/other_service.py",
            content="some content",
        )
        hyp = Hypothesis(
            title="Discount threshold error",
            description="The discount threshold check uses greater than in src/other_service.py.",
            why_plausible="Checked other service logic.",
            confidence_score=0.5,
            supporting_evidence_ids=[fact.id],
        )
        state = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[hyp],
        )

        score = evaluate_root_cause_match(case, state)
        assert score.matched is False
        assert "failed to identify relevant source files" in score.reason.lower()

    def test_scorer_correct_file_unrelated_cause_rejected(self):
        """Correct source files but unrelated root cause."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(
            repo_path="/test",
            bug_description=case.bug_description,
        )
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/discount.py",
            file_path="src/discount.py",
            content="if total > 100.0:",
        )
        hyp = Hypothesis(
            title="Database Connection Timeout",
            description="src/discount.py crashes because PostgreSQL connection pool timed out on read.",
            why_plausible="Network latency spike in container.",
            confidence_score=0.8,
            supporting_evidence_ids=[fact.id],
        )
        state = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[hyp],
        )

        score = evaluate_root_cause_match(case, state)
        assert score.matched is False
        assert "insufficient distinctive keywords" in score.reason.lower()

    def test_scorer_missing_hypothesis_rejected(self):
        """Missing hypothesis (empty list)."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(repo_path="/test", bug_description="bug")
        state = InvestigationState(request=request, status=InvestigationStatus.FAILED, facts=[], hypotheses=[])
        score = evaluate_root_cause_match(case, state)
        assert score.matched is False
        assert "no hypotheses were formed" in score.reason.lower()

    def test_scorer_empty_or_whitespace_hypothesis_rejected(self):
        """Empty or whitespace-only hypothesis content rejected."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(repo_path="/test", bug_description="bug")
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/discount.py",
            file_path="src/discount.py",
            content="code",
        )
        hyp = Hypothesis(
            title="   ",
            description="",
            why_plausible="\t\n",
            confidence_score=0.0,
            supporting_evidence_ids=[fact.id],
        )
        state = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[hyp],
        )

        score = evaluate_root_cause_match(case, state)
        assert score.matched is False
        assert "empty or contains no descriptive content" in score.reason.lower()

    def test_scorer_nonexistent_evidence_id_rejected(self):
        """Hypothesis citing fabricated or nonexistent evidence IDs is rejected."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(repo_path="/test", bug_description="bug")
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/discount.py",
            file_path="src/discount.py",
            content="code",
        )
        hyp = Hypothesis(
            title="Discount threshold error",
            description="The discount threshold check uses greater than in src/discount.py.",
            why_plausible="Plausible",
            confidence_score=0.9,
            supporting_evidence_ids=["ev_completely_fabricated_id_99999"],
        )
        state = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[hyp],
        )

        score = evaluate_root_cause_match(case, state)
        assert score.matched is False
        assert "nonexistent evidence ids" in score.reason.lower()

    def test_scorer_evidence_citation_alone_not_enough(self):
        """Citing real evidence without matching distinctive keywords does not pass."""
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")
        request = InvestigationRequest(repo_path="/test", bug_description="bug")
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/discount.py",
            file_path="src/discount.py",
            content="if total > 100.0:",
        )
        # References the real fact ID, but text contains none of the keywords
        hyp = Hypothesis(
            title="File anomaly",
            description="Inspected the file and noted arbitrary code style.",
            why_plausible="Nothing specific.",
            confidence_score=0.2,
            supporting_evidence_ids=[fact.id],
        )
        state = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[hyp],
        )

        score = evaluate_root_cause_match(case, state)
        assert score.matched is False
        assert "insufficient distinctive keywords" in score.reason.lower()

    def test_scorer_structured_vs_natural_language_representations(self):
        """Verify that both structured multi-field and single narrative representations succeed."""
        case = get_benchmark_case("BM02_BOUNDARY_OFF_BY_ONE")
        request = InvestigationRequest(repo_path="/test", bug_description=case.bug_description)
        fact = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="src/pagination.py",
            file_path="src/pagination.py",
            content="return items[start_idx : end_idx - 1]",
        )

        # 1. Structured representation: key terms spread across fields
        structured_hyp = Hypothesis(
            title="Pagination slice boundary calculation",
            description="The slice calculation subtracts 1 from end index.",
            why_plausible="Boundary off-by-one drops the last page item.",
            recommended_next_step="Fix slice in src/pagination.py",
            confidence_score=0.9,
            supporting_evidence_ids=[fact.id],
        )
        state_structured = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[structured_hyp],
        )
        score_struct = evaluate_root_cause_match(case, state_structured)
        assert score_struct.matched is True

        # 2. Natural language narrative representation: everything in description
        narrative_hyp = Hypothesis(
            title="Root Cause Diagnosis",
            description=(
                "In src/pagination.py, the pagination function calculates the slice boundary "
                "with an off-by-one subtraction (end_idx - 1), omitting the final item."
            ),
            why_plausible="Empirical fact",
            confidence_score=0.9,
            supporting_evidence_ids=[fact.id],
        )
        state_narrative = InvestigationState(
            request=request,
            status=InvestigationStatus.COMPLETED,
            facts=[fact],
            hypotheses=[narrative_hyp],
        )
        score_narr = evaluate_root_cause_match(case, state_narrative)
        assert score_narr.matched is True


class TestMetricCalculationsAndDenominators:
    """Acceptance Criterion B: Correct metric populations, denominators, and zero handling."""

    def test_zero_denominator_yields_null_with_reason(self):
        metrics = calculate_benchmark_metrics([])
        assert metrics.total_cases == 0
        assert metrics.completed_investigations == 0
        assert metrics.investigation_completion_rate is None
        assert metrics.gate_a.pass_rate is None
        assert metrics.gate_a.zero_denominator_reason is not None
        assert metrics.gate_b.pass_rate is None
        assert metrics.gate_b.zero_denominator_reason is not None
        assert metrics.gate_c.pass_rate is None
        assert metrics.gate_c.zero_denominator_reason is not None
        assert metrics.overall_verification_rate is None

    def test_per_stage_denominators_accurate(self):
        r1 = BenchmarkCaseResult(
            case_id="C1",
            category=BenchmarkCategory.CONDITIONAL_LOGIC,
            duration_seconds=1.0,
            investigation_status=InvestigationStatus.PATCH_VERIFIED,
            root_cause_score=RootCauseScore(matched=True, reason="ok"),
            evidence_count=1,
            hypothesis_source=ArtifactSource.SIMULATED,
            patch_generated=True,
            patch_source=ArtifactSource.SIMULATED,
            regression_test_generated=True,
            gate_a_attempted=True,
            gate_a_passed=True,
            gate_b_attempted=True,
            gate_b_passed=True,
            gate_c_attempted=True,
            gate_c_passed=True,
            verified=True,
        )
        r2 = BenchmarkCaseResult(
            case_id="C2",
            category=BenchmarkCategory.BOUNDARY_CONDITION,
            duration_seconds=2.0,
            investigation_status=InvestigationStatus.VERIFICATION_FAILED,
            root_cause_score=RootCauseScore(matched=True, reason="ok"),
            evidence_count=1,
            hypothesis_source=ArtifactSource.SIMULATED,
            patch_generated=True,
            patch_source=ArtifactSource.SIMULATED,
            regression_test_generated=True,
            gate_a_attempted=True,
            gate_a_passed=True,
            gate_b_attempted=True,
            gate_b_passed=False,
            gate_c_attempted=False,
            gate_c_passed=None,
            verified=False,
            failing_gate="GATE_B",
        )
        r3 = BenchmarkCaseResult(
            case_id="C3",
            category=BenchmarkCategory.DATA_HANDLING,
            duration_seconds=3.0,
            investigation_status=InvestigationStatus.INVESTIGATION_INCONCLUSIVE,
            root_cause_score=RootCauseScore(matched=False, reason="no"),
            evidence_count=1,
            hypothesis_source=ArtifactSource.SIMULATED,
            patch_generated=True,
            patch_source=ArtifactSource.SIMULATED,
            regression_test_generated=True,
            gate_a_attempted=True,
            gate_a_passed=False,
            gate_b_attempted=False,
            gate_b_passed=None,
            gate_c_attempted=False,
            gate_c_passed=None,
            verified=False,
            failing_gate="GATE_A",
        )
        r4 = BenchmarkCaseResult(
            case_id="C4",
            category=BenchmarkCategory.NUMERIC_ERROR,
            duration_seconds=4.0,
            investigation_status=InvestigationStatus.INSUFFICIENT_EVIDENCE,
            root_cause_score=RootCauseScore(matched=False, reason="no"),
            evidence_count=0,
            hypothesis_source=ArtifactSource.NONE,
            patch_generated=False,
            patch_source=ArtifactSource.NONE,
            regression_test_generated=False,
            gate_a_attempted=False,
            gate_a_passed=None,
            gate_b_attempted=False,
            gate_b_passed=None,
            gate_c_attempted=False,
            gate_c_passed=None,
            verified=False,
        )

        metrics = calculate_benchmark_metrics([r1, r2, r3, r4])

        assert metrics.total_cases == 4
        # C1 and C2 completed with hypotheses
        assert metrics.completed_investigations == 2
        assert metrics.investigation_completion_rate == 0.5000

        # Gate A: 3 attempts, 2 successes -> 2/3 = 0.6667
        assert metrics.gate_a.eligible_attempts == 3
        assert metrics.gate_a.successes == 2
        assert metrics.gate_a.pass_rate == 0.6667

        # Gate B: 2 attempts, 1 success -> 1/2 = 0.5000
        assert metrics.gate_b.eligible_attempts == 2
        assert metrics.gate_b.successes == 1
        assert metrics.gate_b.pass_rate == 0.5000

        # Gate C: 1 attempt, 1 success -> 1/1 = 1.0000
        assert metrics.gate_c.eligible_attempts == 1
        assert metrics.gate_c.successes == 1
        assert metrics.gate_c.pass_rate == 1.0000

        # Overall verified: 1 / 4 = 0.2500
        assert metrics.verified_patches == 1
        assert metrics.overall_verification_rate == 0.2500

        # Inconclusive cases count (C3 + C4)
        assert metrics.inconclusive_cases == 2

        # Mean and median runtime: (1 + 2 + 3 + 4) / 4 = 2.5
        assert metrics.mean_runtime_seconds == 2.5
        assert metrics.median_runtime_seconds == 2.5


class TestSecretRedaction:
    """Acceptance Criterion E: Redact secrets before serialization."""

    def test_redact_secrets_in_nested_dict(self):
        raw = {
            "api_key": "sk-secret12345678901234567890",
            "normal_field": "hello world",
            "nested": {
                "auth_token": "Bearer abcdef123456",
                "safe": "data",
            },
            "error_msg": "Failed with token: sk-mysecretkey99999999999999999",
        }
        redacted = redact_secrets(raw)

        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["normal_field"] == "hello world"
        assert redacted["nested"]["auth_token"] == "[REDACTED]"
        assert redacted["nested"]["safe"] == "data"
        assert "[REDACTED_SECRET]" in redacted["error_msg"]
        assert "sk-mysecretkey" not in redacted["error_msg"]


class TestBenchmarkRunnerExecution:
    """Acceptance Criteria D & E & Integrity Rules: Isolated execution, simulated provenance, and honest failure."""

    def test_simulated_benchmark_execution(self, temp_fixtures_dir: Path):
        """Simulated benchmark execution exercises real tools and verification pipeline."""
        runner = BenchmarkRunner(base_temp_dir=temp_fixtures_dir)

        result = runner.run_case("BM01_CONDITIONAL_LOGIC")

        assert result.case_id == "BM01_CONDITIONAL_LOGIC"
        assert result.category == BenchmarkCategory.CONDITIONAL_LOGIC
        assert result.evaluation_mode == EvaluationMode.SIMULATED_OFFLINE
        assert result.hypothesis_source == ArtifactSource.SIMULATED
        assert result.patch_source == ArtifactSource.SIMULATED
        assert result.regression_test_source == ArtifactSource.SIMULATED
        assert result.verification_executed_by == "REAL_WORKFLOW"

        assert result.patch_generated is True
        assert result.regression_test_generated is True
        assert result.gate_a_attempted is True
        assert result.gate_a_passed is True
        assert result.gate_b_passed is True
        assert result.gate_c_passed is True
        assert result.verified is True
        assert result.investigation_status == InvestigationStatus.PATCH_VERIFIED
        assert result.root_cause_score.matched is True
        assert result.token_usage == "unavailable"

    def test_unconfigured_mock_provider_fails_honestly(self, temp_fixtures_dir: Path):
        """An unconfigured mock provider must NOT cause a successful investigation or silent patch generation."""
        runner = BenchmarkRunner(base_temp_dir=temp_fixtures_dir)
        unconfigured_provider = MockLLMProvider()

        result = runner.run_case("BM01_CONDITIONAL_LOGIC", provider=unconfigured_provider)

        assert result.case_id == "BM01_CONDITIONAL_LOGIC"
        assert result.evaluation_mode == EvaluationMode.LIVE_PROVIDER
        assert result.investigation_status == InvestigationStatus.INSUFFICIENT_EVIDENCE
        assert result.hypothesis_source == ArtifactSource.NONE
        assert result.patch_generated is False
        assert result.patch_source == ArtifactSource.NONE
        assert result.regression_test_generated is False
        assert result.regression_test_source == ArtifactSource.NONE
        assert result.gate_a_attempted is False
        assert result.verified is False
        assert result.root_cause_score.matched is False
        assert "no hypotheses were formed" in result.root_cause_score.reason.lower()

    def test_missing_patch_does_not_silently_fallback(self, temp_fixtures_dir: Path):
        """When provider fails to generate a patch, ground truth is NOT silently injected."""
        # Provider forms a hypothesis but fails to generate a valid diff
        case = get_benchmark_case("BM01_CONDITIONAL_LOGIC")

        def provider_without_patch(messages, tools, step):
            if step == 1:
                return LLMResponse(tool_calls=[ToolCall(name="read_repo_file", arguments={"relative_path": "src/discount.py"})])
            elif step == 2:
                # Find evidence ID in messages
                import re
                eids = re.findall(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", str(messages))
                eid = eids[0] if eids else "ev_fallback"
                return LLMResponse(tool_calls=[ToolCall(
                    name="submit_hypotheses",
                    arguments={"hypotheses": [{
                        "title": "Discount threshold",
                        "description": "Threshold comparison greater than in src/discount.py",
                        "why_plausible": "Code inspection",
                        "confidence_score": 0.9,
                        "supporting_evidence_ids": [eid],
                    }]},
                )])
            else:
                # Patch generation prompt: returns invalid text without diff
                return LLMResponse(content="I cannot create a diff for this.", tool_calls=[])

        provider = MockLLMProvider(handler=provider_without_patch)
        runner = BenchmarkRunner(base_temp_dir=temp_fixtures_dir)

        result = runner.run_case("BM01_CONDITIONAL_LOGIC", provider=provider)

        assert result.root_cause_score.matched is True  # Diagnosis was grounded
        assert result.patch_generated is False          # Patch generation failed
        assert result.regression_test_generated is False
        assert result.verified is False                 # Never verified!
        assert result.gate_a_attempted is False
        assert result.patch_source == ArtifactSource.NONE

    def test_runner_suite_execution_resilient_to_individual_failure(self, temp_fixtures_dir: Path):
        """Verify that suite execution cleanly records per-case results with mode."""
        runner = BenchmarkRunner(base_temp_dir=temp_fixtures_dir)

        suite_res = runner.run_suite(case_ids=["BM01_CONDITIONAL_LOGIC", "BM02_BOUNDARY_OFF_BY_ONE"])

        assert suite_res.total_cases == 2
        assert len(suite_res.case_results) == 2
        assert suite_res.evaluation_mode == EvaluationMode.SIMULATED_OFFLINE
        assert "SIMULATED_OFFLINE" in suite_res.mode_clarification
        assert suite_res.case_results[0].case_id == "BM01_CONDITIONAL_LOGIC"
        assert suite_res.case_results[1].case_id == "BM02_BOUNDARY_OFF_BY_ONE"
        assert suite_res.metrics.total_cases == 2
        assert suite_res.metrics.root_cause_matches == 2

    def test_runner_cleans_up_workspace_after_execution(self, temp_fixtures_dir: Path):
        runner = BenchmarkRunner(base_temp_dir=temp_fixtures_dir)

        runner.run_case("BM05_ZERO_DIVISION")

        # After execution, no temporary bm_* directory should linger in base_temp_dir
        lingering = list(temp_fixtures_dir.glob("rootcause_bm_*"))
        assert len(lingering) == 0, f"Lingering temporary directories detected: {lingering}"
