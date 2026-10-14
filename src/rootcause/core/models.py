"""Core domain data models for RootCause AI using Pydantic v2."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from rootcause.core.exceptions import InvalidStateTransitionError


class InvestigationStatus(str, Enum):
    """Investigation lifecycle and outcome statuses."""
    RECEIVED = "RECEIVED"
    PENDING = "PENDING"
    INVESTIGATING = "INVESTIGATING"
    HYPOTHESIS_FORMED = "HYPOTHESIS_FORMED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    COMPLETED = "COMPLETED"  # Diagnostic investigation workflow concluded (does NOT imply fix verified)
    FAILED = "FAILED"
    # Extended verification gate statuses
    BUG_REPRODUCED = "BUG_REPRODUCED"
    ROOT_CAUSE_HYPOTHESIS = "ROOT_CAUSE_HYPOTHESIS"
    PATCH_PROPOSED = "PATCH_PROPOSED"
    REGRESSION_TEST_CREATED = "REGRESSION_TEST_CREATED"
    VERIFYING = "VERIFYING"
    PATCH_VERIFIED = "PATCH_VERIFIED"
    PATCH_APPLIED = "PATCH_APPLIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    INVESTIGATION_INCONCLUSIVE = "INVESTIGATION_INCONCLUSIVE"


VALID_TRANSITIONS: Dict[InvestigationStatus, Set[InvestigationStatus]] = {
    InvestigationStatus.RECEIVED: {
        InvestigationStatus.INVESTIGATING,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.PENDING: {
        InvestigationStatus.RECEIVED,
        InvestigationStatus.INVESTIGATING,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.INVESTIGATING: {
        InvestigationStatus.HYPOTHESIS_FORMED,
        InvestigationStatus.INSUFFICIENT_EVIDENCE,
        InvestigationStatus.BUG_REPRODUCED,
        InvestigationStatus.ROOT_CAUSE_HYPOTHESIS,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.HYPOTHESIS_FORMED: {
        InvestigationStatus.INVESTIGATING,
        InvestigationStatus.INSUFFICIENT_EVIDENCE,
        InvestigationStatus.COMPLETED,
        InvestigationStatus.PATCH_PROPOSED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.INSUFFICIENT_EVIDENCE: {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.BUG_REPRODUCED: {
        InvestigationStatus.INVESTIGATING,
        InvestigationStatus.HYPOTHESIS_FORMED,
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.ROOT_CAUSE_HYPOTHESIS: {
        InvestigationStatus.INVESTIGATING,
        InvestigationStatus.COMPLETED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.COMPLETED: {
        InvestigationStatus.PATCH_PROPOSED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.PATCH_PROPOSED: {
        InvestigationStatus.REGRESSION_TEST_CREATED,
        InvestigationStatus.VERIFYING,
        InvestigationStatus.VERIFICATION_FAILED,
        InvestigationStatus.INVESTIGATION_INCONCLUSIVE,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.REGRESSION_TEST_CREATED: {
        InvestigationStatus.VERIFYING,
        InvestigationStatus.PATCH_VERIFIED,
        InvestigationStatus.VERIFICATION_FAILED,
        InvestigationStatus.INVESTIGATION_INCONCLUSIVE,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.VERIFYING: {
        InvestigationStatus.PATCH_VERIFIED,
        InvestigationStatus.VERIFICATION_FAILED,
        InvestigationStatus.INVESTIGATION_INCONCLUSIVE,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.PATCH_VERIFIED: {
        InvestigationStatus.PATCH_APPLIED,
    },
    InvestigationStatus.PATCH_APPLIED: set(),
    InvestigationStatus.VERIFICATION_FAILED: {
        InvestigationStatus.PATCH_PROPOSED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.INVESTIGATION_INCONCLUSIVE: {
        InvestigationStatus.PATCH_PROPOSED,
        InvestigationStatus.INVESTIGATING,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.FAILED: set(),
}


class EvidenceType(str, Enum):
    """Type of empirical evidence collected during investigation."""
    SOURCE_CODE = "SOURCE_CODE"
    STACK_TRACE = "STACK_TRACE"
    LOG_EXCERPT = "LOG_EXCERPT"
    GIT_COMMIT = "GIT_COMMIT"
    GIT_DIFF = "GIT_DIFF"
    TEST_OUTPUT = "TEST_OUTPUT"
    REPRODUCTION_RESULT = "REPRODUCTION_RESULT"
    COMMAND_EXECUTION = "COMMAND_EXECUTION"


class Evidence(BaseModel):
    """Structured empirical fact collected from repository, logs, or test execution."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_type: EvidenceType
    source: str = Field(..., description="Path, commit hash, test name, or command that generated this evidence")
    file_path: Optional[str] = Field(None, description="Repository-relative file path, where applicable")
    line_start: Optional[int] = Field(None, description="Starting line number in source file, if applicable")
    line_end: Optional[int] = Field(None, description="Ending line number in source file, if applicable")
    content: str = Field(..., description="Observed raw text or data snippet")
    why_relevant: Optional[str] = Field(None, description="Explanation of why this evidence matters to the investigation")
    source_tool: Optional[str] = Field(None, description="Name of the tool that collected this evidence")
    step_index: int = Field(0, description="Investigation step number when this fact was gathered")
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary structured metadata")

    @field_validator("line_end")
    @classmethod
    def validate_line_bounds(cls, v: Optional[int], info) -> Optional[int]:
        start = info.data.get("line_start")
        if v is not None and start is not None and v < start:
            raise ValueError(f"line_end ({v}) cannot be less than line_start ({start})")
        return v


class Hypothesis(BaseModel):
    """Evidence-based explanation of the root cause."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    rank: int = Field(1, ge=1, description="Relative priority rank (1 is highest)")
    title: str = Field(..., description="Short summary of the hypothesized root cause")
    description: str = Field(..., description="Detailed explanation of what failed and why")
    why_plausible: str = Field(..., description="Explanation of why the available evidence supports this hypothesis")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence rating between 0.0 and 1.0 (uncalibrated heuristic)")
    uncertainty_level: str = Field("MODERATE", description="Qualitative uncertainty assessment: LOW, MODERATE, HIGH, UNCONFIRMED")
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="IDs of supporting Evidence items")
    contradictory_evidence_ids: List[str] = Field(default_factory=list, description="IDs of contradictory Evidence items")
    recommended_next_step: Optional[str] = Field(None, description="Suggested next action or verification step")
    insufficient_evidence_reason: Optional[str] = Field(
        None, description="Explanation when evidence is insufficient or inconclusive"
    )


class AuditStep(BaseModel):
    """Immutable record of an individual investigation step."""
    step_index: int
    selected_tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments executed with sensitive data redacted")
    status: str = Field("SUCCESS", description="Execution status: SUCCESS, ERROR, TIMEOUT")
    tool_result_summary: str = Field("", description="Bounded summary or excerpt of tool execution output")
    evidence_collected_ids: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = Field(None, description="Agent's rationale for calling this tool or continuing")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CommandExecutionResult(BaseModel):
    """Record of a controlled subprocess execution."""
    command: List[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    cwd: str


class ParsedFrame(BaseModel):
    """Single frame extracted from a Python stack trace."""
    file_path: str
    line_number: int
    function_name: str
    code_line: Optional[str] = None


class ParsedTraceback(BaseModel):
    """Structured decomposition of a Python traceback."""
    error_type: str
    error_message: str
    frames: List[ParsedFrame] = Field(default_factory=list)
    raw_traceback: str
    innermost_frame: Optional[ParsedFrame] = None


class FileInfo(BaseModel):
    """Metadata for a repository file."""
    path: str
    size_bytes: int
    is_file: bool = True
    is_binary: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class FileReadResult(BaseModel):
    """Structured result of reading a repository file."""
    file_path: str
    total_lines: int
    start_line: int
    end_line: int
    content: str
    numbered_content: str
    truncated: bool = False
    file_size_bytes: int = 0
    is_binary: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class SearchMatch(BaseModel):
    """Single match occurrence from code search."""
    file_path: str
    line_number: int
    line_content: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class GitCommit(BaseModel):
    """Structured representation of a Git commit."""
    hash: str
    author: str
    date: str
    message: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class TestDiscoveryResult(BaseModel):
    """Result of discovering test targets in a repository."""
    test_node_ids: List[str] = Field(default_factory=list)
    total_count: int = 0
    output: str = ""


class TestResult(BaseModel):
    """Structured result of executing a test suite or single test."""
    passed: bool
    exit_code: int
    duration_seconds: float
    output: str
    timed_out: bool = False
    tests_ran: int = 0
    passed_count: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    failure_details: List[str] = Field(default_factory=list)


class CandidatePatch(BaseModel):
    """Unified diff proposed to resolve the identified root cause."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    investigation_id: str
    hypothesis_id: str
    diff: str = Field(..., description="Minimal unified diff")
    target_files: List[str] = Field(default_factory=list, description="List of target relative file paths modified by this diff")
    explanation: str = Field("", description="Explanation of why this patch resolves the issue")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegressionTest(BaseModel):
    """Automated reproduction and regression test."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    investigation_id: str
    test_file_path: str = Field(..., description="Repository-relative path for the regression test file")
    test_name: str = Field(..., description="Name of the test function or node ID")
    test_content: str = Field(..., description="Full Python source code of the regression test")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VerificationGateResult(BaseModel):
    """Result of evaluating an individual verification gate (A, B, or C)."""
    gate_name: str = Field(..., description="Name of the gate: GATE_A, GATE_B, or GATE_C")
    passed: bool = Field(..., description="Whether this gate passed")
    message: str = Field("", description="Detailed explanation of the outcome or reason for failure")
    test_result: Optional[TestResult] = Field(None, description="Test execution result if executed")
    duration_seconds: float = Field(0.0, description="Duration of this gate's execution in seconds")


class VerificationResult(BaseModel):
    """Comprehensive result of the 3-point independent verification workflow."""
    verification_attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    investigation_id: str
    patch_id: str
    regression_test_id: str
    commit_tested: str = Field(..., description="Git commit hash tested against")
    status: InvestigationStatus
    gate_a: VerificationGateResult
    gate_b: Optional[VerificationGateResult] = None
    gate_c: Optional[VerificationGateResult] = None
    verified: bool = Field(False, description="True only if Gates A, B, and C all passed in this single attempt")
    failing_gate: Optional[str] = Field(None, description="Name of the first failing gate, if any")
    summary: str = Field("", description="Human-readable summary of the verification outcome")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InvestigationRequest(BaseModel):
    """Input payload initiating an autonomous bug investigation."""
    repo_path: str = Field(..., description="Absolute path to the target local Git repository")
    bug_description: str = Field(..., description="Human or reporter description of the bug")
    error_log: Optional[str] = Field(None, description="Optional raw error log")
    stack_trace: Optional[str] = Field(None, description="Optional Python stack trace")
    test_target: Optional[str] = Field(None, description="Optional test file or expression hint")
    max_iterations: int = Field(10, ge=1, le=50, description="Maximum iterative tool steps")


class InvestigationState(BaseModel):
    """Complete state snapshot of an active or concluded bug investigation."""
    investigation_id: str = Field(default_factory=lambda: str(uuid4()))
    status: InvestigationStatus = InvestigationStatus.RECEIVED
    request: InvestigationRequest
    facts: List[Evidence] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    audit_trail: List[AuditStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    insufficient_evidence_reason: Optional[str] = None
    missing_information: Optional[str] = None
    candidate_patch: Optional[CandidatePatch] = None
    regression_test: Optional[RegressionTest] = None
    verification_result: Optional[VerificationResult] = None

    def transition_to(self, new_status: InvestigationStatus) -> None:
        """Validate and transition to a new investigation status.
        
        Raises:
            InvalidStateTransitionError: If the transition is not allowed.
        """
        allowed = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidStateTransitionError(self.status.value, new_status.value)
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)


class InvestigationReport(BaseModel):
    """Structured summary report of a completed investigation."""
    investigation_id: str
    repo_path: str
    bug_description: str
    final_status: InvestigationStatus
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    audit_trail: List[AuditStep] = Field(default_factory=list)
    markdown_report: str
    candidate_patch: Optional[CandidatePatch] = None
    verification_result: Optional[VerificationResult] = None
