"""Data models for RootCause AI benchmark framework using Pydantic v2."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from rootcause.core.models import InvestigationStatus


class EvaluationMode(str, Enum):
    """Execution mode distinguishing deterministic simulation from live model evaluation."""
    SIMULATED_OFFLINE = "SIMULATED_OFFLINE"
    LIVE_PROVIDER = "LIVE_PROVIDER"


class ArtifactSource(str, Enum):
    """Provenance tracking for individual artifacts in benchmark runs."""
    SIMULATED = "SIMULATED"
    LIVE_LLM = "LIVE_LLM"
    NONE = "NONE"


class BenchmarkCategory(str, Enum):
    """Classification of bug archetype for benchmark cases."""
    CONDITIONAL_LOGIC = "CONDITIONAL_LOGIC"
    BOUNDARY_CONDITION = "BOUNDARY_CONDITION"
    DATA_HANDLING = "DATA_HANDLING"
    REGRESSION = "REGRESSION"
    NUMERIC_ERROR = "NUMERIC_ERROR"


class BenchmarkDifficulty(str, Enum):
    """Estimated difficulty rating of benchmark bug scenario."""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class BenchmarkCase(BaseModel):
    """Specification of a reproducible bug scenario in the benchmark suite."""
    case_id: str = Field(..., description="Unique case identifier, e.g. BM01_CONDITIONAL_LOGIC")
    name: str = Field(..., description="Human-readable title of the benchmark case")
    category: BenchmarkCategory
    difficulty: BenchmarkDifficulty
    bug_description: str = Field(..., description="Human or issue-tracker bug report")
    stack_trace: Optional[str] = Field(None, description="Optional raw traceback from bug symptom")
    known_root_cause: str = Field(..., description="Documented ground-truth root cause description")
    expected_root_cause_keywords: List[str] = Field(
        default_factory=list,
        description="Distinctive non-generic keywords required to verify diagnosis",
    )
    expected_symptom: str = Field(..., description="Specific failure message or assertion observed")
    relevant_files: List[str] = Field(
        default_factory=list,
        description="Repository-relative paths to files containing the bug or relevant to fix",
    )
    expected_verification_outcome: InvestigationStatus = Field(
        InvestigationStatus.PATCH_VERIFIED,
        description="Target verification state after successful fix",
    )
    fixture_relative_dir: str = Field(
        ...,
        description="Relative subdirectory where fixture files or template reside",
    )
    known_good_patch: str = Field(
        ...,
        description="Verified unified diff that resolves the bug without regressions",
    )
    regression_test_code: str = Field(
        ...,
        description="Python regression test source that fails on baseline and passes on patched",
    )
    regression_test_file: str = Field(
        "tests/test_reproduction.py",
        description="Relative file path where the regression test should be written",
    )


class RootCauseScore(BaseModel):
    """Auditable, rule-based comparison of diagnosed hypothesis against ground truth."""
    matched: bool = Field(..., description="Whether diagnosis matched expected ground truth")
    metric_label: str = Field("heuristic matching accuracy", description="Auditable metric label")
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    matched_files: List[str] = Field(default_factory=list)
    reason: str = Field(..., description="Detailed explanation of matching determination")


class GateMetric(BaseModel):
    """Per-stage metrics for an individual verification gate (A, B, or C)."""
    gate_name: str
    eligible_attempts: int = Field(0, description="Number of cases that reached and attempted this gate")
    successes: int = Field(0, description="Number of cases that passed this gate")
    failures: int = Field(0, description="Number of cases that failed this gate")
    timeouts: int = Field(0, description="Number of cases that timed out during this gate")
    inconclusive: int = Field(0, description="Number of cases that terminated inconclusive at this gate")
    pass_rate: Optional[float] = Field(
        None,
        description="Pass rate (successes / eligible_attempts), or null if denominator is zero",
    )
    pass_rate_percentage: Optional[str] = Field(
        None,
        description="Formatted percentage string or null",
    )
    denominator_description: str = Field(
        ...,
        description="Exact population description used as denominator",
    )
    zero_denominator_reason: Optional[str] = Field(
        None,
        description="Explanation when eligible_attempts is zero",
    )


class BenchmarkCaseResult(BaseModel):
    """Evaluation result for an individual benchmark case execution."""
    case_id: str
    category: BenchmarkCategory
    evaluation_mode: EvaluationMode = Field(
        EvaluationMode.SIMULATED_OFFLINE,
        description="Explicit mode: SIMULATED_OFFLINE vs LIVE_PROVIDER",
    )
    duration_seconds: float
    investigation_status: InvestigationStatus
    root_cause_score: RootCauseScore
    evidence_count: int = 0
    evidence_ids: List[str] = Field(default_factory=list)
    hypothesis_source: ArtifactSource = Field(
        ArtifactSource.NONE,
        description="Source of diagnosed hypothesis: SIMULATED, LIVE_LLM, or NONE",
    )
    patch_generated: bool = False
    patch_source: ArtifactSource = Field(
        ArtifactSource.NONE,
        description="Source of candidate patch: SIMULATED, LIVE_LLM, or NONE",
    )
    regression_test_generated: bool = False
    regression_test_source: ArtifactSource = Field(
        ArtifactSource.NONE,
        description="Source of regression test: SIMULATED, LIVE_LLM, or NONE",
    )
    gate_a_attempted: bool = False
    gate_a_passed: Optional[bool] = None
    gate_b_attempted: bool = False
    gate_b_passed: Optional[bool] = None
    gate_c_attempted: bool = False
    gate_c_passed: Optional[bool] = None
    verification_executed_by: str = Field(
        "REAL_WORKFLOW",
        description="Subsystem executing the verification pipeline",
    )
    verified: bool = False
    failing_gate: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    timed_out: bool = False
    token_usage: Optional[str] = Field("unavailable", description="Token count or 'unavailable'")
    requested_model: Optional[str] = None
    actual_model: Optional[str] = None
    hypothesis_text: Optional[str] = None
    evidence_references: List[str] = Field(default_factory=list)
    retry_count: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    rate_limited: bool = False
    fixture_commit_hash: Optional[str] = None


class BenchmarkMetrics(BaseModel):
    """Aggregated population statistics and rates across benchmark suite."""
    total_cases: int
    completed_investigations: int
    investigation_completion_rate: Optional[float]
    investigation_completion_rate_percentage: Optional[str]
    root_cause_matches: int
    root_cause_match_rate: Optional[float]
    root_cause_match_rate_percentage: Optional[str]
    patches_generated: int
    patch_generation_rate: Optional[float]
    patch_generation_rate_percentage: Optional[str]
    regression_tests_generated: int
    regression_test_generation_rate: Optional[float]
    regression_test_generation_rate_percentage: Optional[str]
    gate_a: GateMetric
    gate_b: GateMetric
    gate_c: GateMetric
    verified_patches: int
    overall_verification_rate: Optional[float]
    overall_verification_rate_percentage: Optional[str]
    failed_cases: int
    inconclusive_cases: int
    timeouts: int
    execution_errors: int
    rate_limit_errors: int = 0
    authentication_errors: int = 0
    configuration_errors: int = 0
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_estimated_cost_usd: Optional[float] = None
    mean_runtime_seconds: Optional[float] = None
    median_runtime_seconds: Optional[float] = None


class BenchmarkSuiteResult(BaseModel):
    """Comprehensive benchmark execution report serialized to JSON."""
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    benchmark_version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluation_mode: EvaluationMode = Field(
        EvaluationMode.SIMULATED_OFFLINE,
        description="Overall evaluation mode: SIMULATED_OFFLINE or LIVE_PROVIDER",
    )
    mode_clarification: str = Field(
        (
            "SIMULATED_OFFLINE: Results exercise orchestration, evidence gathering, and verification plumbing "
            "using deterministic simulation. Not presented as autonomous LLM diagnostic performance."
        ),
        description="Statement of integrity regarding evaluation mode and results interpretation",
    )
    provider_info: str = Field("MockLLMProvider", description="LLM provider and model description")
    requested_model: Optional[str] = None
    actual_model: Optional[str] = None
    seed: Optional[int] = None
    total_cases: int
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_estimated_cost_usd: Optional[float] = None
    case_results: List[BenchmarkCaseResult] = Field(default_factory=list)
    metrics: BenchmarkMetrics
