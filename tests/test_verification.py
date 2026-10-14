"""Comprehensive test suite for RootCause AI Phase 4: Patch Generation and Verification."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from rootcause.api.app import create_app
from rootcause.core.exceptions import (
    ConfirmationRequiredError,
    InvalidPatchError,
    PatchApplicationError,
    SecurityViolation,
    StaleVerificationError,
)
from rootcause.core.models import (
    CandidatePatch,
    Evidence,
    EvidenceType,
    Hypothesis,
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
    RegressionTest,
    TestResult as PydanticTestResult,
)
from rootcause.db.repository import InvestigationRepository
from rootcause.llm.base import LLMResponse
from rootcause.llm.mock_provider import MockLLMProvider
from rootcause.security.sandbox import SubprocessRunner
from rootcause.verification.patch_engine import (
    apply_diff_to_worktree,
    generate_candidate_patch,
    parse_diff_target_files,
    validate_patch_safety,
)
from rootcause.verification.verifier import (
    VerificationService,
    apply_verified_patch_to_original,
)


@pytest.fixture
def temp_git_repo() -> Generator[Path, None, None]:
    """Create a temporary git repository with a buggy module and existing tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="rootcause_test_repo_")).resolve()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=str(temp_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@rootcause.ai"], cwd=str(temp_dir), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(temp_dir), check=True)

    # Create module with a bug: divide_numbers raises ZeroDivisionError on denominator 0
    src_dir = temp_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    calc_file = src_dir / "calc.py"
    calc_file.write_text(
        "def divide_numbers(a: float, b: float) -> float:\n"
        "    return a / b\n"
        "\n"
        "def add_numbers(a: float, b: float) -> float:\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    # Create existing test suite
    test_dir = temp_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_existing = test_dir / "test_existing.py"
    test_existing.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
        "import calc\n"
        "\n"
        "def test_addition():\n"
        "    assert calc.add_numbers(2, 3) == 5\n"
        "\n"
        "def test_valid_division():\n"
        "    assert calc.divide_numbers(10, 2) == 5\n",
        encoding="utf-8",
    )

    # Initial commit
    subprocess.run(["git", "add", "."], cwd=str(temp_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit with buggy calc"], cwd=str(temp_dir), check=True, capture_output=True)

    yield temp_dir

    # Cleanup
    shutil.rmtree(str(temp_dir), ignore_errors=True)


@pytest.fixture
def sample_investigation_state(temp_git_repo: Path) -> InvestigationState:
    """Create a sample investigation state that reached COMPLETED with supporting evidence."""
    request = InvestigationRequest(
        repo_path=str(temp_git_repo),
        bug_description="divide_numbers crashes with ZeroDivisionError when denominator is zero",
        stack_trace="ZeroDivisionError: division by zero in calc.py line 2",
    )
    fact = Evidence(
        evidence_type=EvidenceType.SOURCE_CODE,
        source="src/calc.py",
        file_path="src/calc.py",
        line_start=1,
        line_end=2,
        content="return a / b",
        why_relevant="Direct zero division without guard check",
    )
    hypothesis = Hypothesis(
        title="Missing zero-denominator guard",
        description="divide_numbers executes a / b directly without checking if b == 0, causing ZeroDivisionError.",
        why_plausible="Source code directly executes division without guard.",
        confidence_score=0.95,
        supporting_evidence_ids=[fact.id],
    )
    state = InvestigationState(
        request=request,
        status=InvestigationStatus.COMPLETED,
        facts=[fact],
        hypotheses=[hypothesis],
    )
    return state


SAMPLE_VALID_DIFF = """--- a/src/calc.py
+++ b/src/calc.py
@@ -1,2 +1,4 @@
 def divide_numbers(a: float, b: float) -> float:
+    if b == 0:
+        return 0.0
     return a / b
"""

SAMPLE_REGRESSION_TEST_CONTENT = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import calc
import pytest

def test_divide_zero_reproduction():
    # Calling with b=0 should raise ZeroDivisionError before patch, and return 0.0 after patch
    val = calc.divide_numbers(10, 0)
    assert val == 0.0
"""


class TestPatchEngine:
    """Unit tests for unified diff parsing, validation, and generation."""

    def test_parse_diff_target_files(self):
        targets = parse_diff_target_files(SAMPLE_VALID_DIFF)
        assert targets == ["src/calc.py"]

    def test_parse_diff_with_multiple_targets(self):
        diff = (
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-1\n+2\n"
            "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-3\n+4\n"
        )
        targets = parse_diff_target_files(diff)
        assert targets == ["src/a.py", "src/b.py"]

    def test_empty_diff_rejected(self, temp_git_repo: Path):
        with pytest.raises(InvalidPatchError, match="cannot be empty"):
            validate_patch_safety(temp_git_repo, "")
        with pytest.raises(InvalidPatchError, match="cannot be empty"):
            validate_patch_safety(temp_git_repo, "   \n  \t ")

    def test_malformed_diff_rejected(self, temp_git_repo: Path):
        with pytest.raises(InvalidPatchError, match="missing hunk header"):
            validate_patch_safety(temp_git_repo, "--- a/src/calc.py\n+++ b/src/calc.py\nSome invalid text")

    def test_path_traversal_diff_rejected(self, temp_git_repo: Path):
        diff = (
            "--- a/../../escape.py\n"
            "+++ b/../../escape.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-bad\n"
            "+good\n"
        )
        with pytest.raises(SecurityViolation, match="Path traversal"):
            validate_patch_safety(temp_git_repo, diff)

    def test_forbidden_path_diff_rejected(self, temp_git_repo: Path):
        diff = (
            "--- a/.git/config\n"
            "+++ b/.git/config\n"
            "@@ -1,1 +1,1 @@\n"
            "-bad\n"
            "+good\n"
        )
        with pytest.raises(SecurityViolation, match="forbidden path"):
            validate_patch_safety(temp_git_repo, diff)

    def test_apply_diff_to_worktree_success(self, temp_git_repo: Path):
        apply_diff_to_worktree(temp_git_repo, SAMPLE_VALID_DIFF)
        calc_content = (temp_git_repo / "src" / "calc.py").read_text(encoding="utf-8")
        assert "if b == 0:" in calc_content
        assert "return 0.0" in calc_content

    def test_apply_diff_conflicts_raise_patch_application_error(self, temp_git_repo: Path):
        # A diff that does not match calc.py content
        conflicting_diff = (
            "--- a/src/calc.py\n"
            "+++ b/src/calc.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-def non_existent_function(x, y):\n"
            "+def replacement_function(x, y):\n"
            "     return x * y\n"
        )
        with pytest.raises(PatchApplicationError, match="dry-run check"):
            apply_diff_to_worktree(temp_git_repo, conflicting_diff)

    def test_generate_candidate_patch_requires_evidence(self, sample_investigation_state: InvestigationState):
        sample_investigation_state.hypotheses[0].supporting_evidence_ids = []
        provider = MockLLMProvider()
        with pytest.raises(InvalidPatchError, match="no supporting evidence"):
            generate_candidate_patch(provider, sample_investigation_state)

    def test_generate_candidate_patch_with_mock_provider(self, sample_investigation_state: InvestigationState):
        provider = MockLLMProvider()
        mock_payload = {
            "diff": SAMPLE_VALID_DIFF,
            "test_file_path": "tests/test_calc_repro.py",
            "test_name": "test_divide_zero_reproduction",
            "test_content": SAMPLE_REGRESSION_TEST_CONTENT,
            "explanation": "Add zero-denominator guard to calc.divide_numbers",
        }
        import json
        provider.queue_text(json.dumps(mock_payload))

        patch, reg_test = generate_candidate_patch(provider, sample_investigation_state)
        assert patch.diff == SAMPLE_VALID_DIFF
        assert patch.target_files == ["src/calc.py"]
        assert reg_test.test_file_path == "tests/test_calc_repro.py"
        assert "test_divide_zero_reproduction" in reg_test.test_content


class TestVerificationGates:
    """Tests evaluating individual verification gates within disposable worktrees."""

    def test_gate_a_baseline_reproduction_passes_when_test_fails_expectedly(
        self, sample_investigation_state: InvestigationState
    ):
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)

        assert result.gate_a.passed is True
        assert "baseline failure reproduced bug symptom" in result.gate_a.message.lower()

    def test_gate_a_fails_when_baseline_test_passes(
        self, sample_investigation_state: InvestigationState
    ):
        # A test that already passes on the baseline repository
        passing_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_already_passing",
            test_content="def test_already_passing():\n    assert True\n",
        )
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=passing_test)

        assert result.verified is False
        assert result.gate_a.passed is False
        assert result.status == InvestigationStatus.INVESTIGATION_INCONCLUSIVE
        assert result.failing_gate == "GATE_A"
        assert "passed on clean baseline" in result.gate_a.message

    def test_gate_a_fails_when_test_fails_for_unrelated_reason(
        self, sample_investigation_state: InvestigationState
    ):
        # A test that fails with an unrelated error (AssertionError on string match)
        unrelated_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_unrelated",
            test_content="def test_unrelated():\n    assert 'hello' == 'world'\n",
        )
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=unrelated_test)

        assert result.verified is False
        assert result.gate_a.passed is False
        assert result.status == InvestigationStatus.INVESTIGATION_INCONCLUSIVE
        assert result.failing_gate == "GATE_A"
        assert "did not reproduce" in result.gate_a.message.lower()

    def test_gate_a_fails_on_syntax_or_collection_error(
        self, sample_investigation_state: InvestigationState
    ):
        broken_syntax_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_syntax_error",
            test_content="def test_syntax_error(\n    # unclosed parenthesis\n",
        )
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=broken_syntax_test)

        assert result.verified is False
        assert result.gate_a.passed is False
        assert result.status == InvestigationStatus.INVESTIGATION_INCONCLUSIVE
        assert "collection or import error" in result.gate_a.message.lower()

    def test_gate_b_fails_when_regression_test_still_fails_after_patch(
        self, sample_investigation_state: InvestigationState
    ):
        # A patch that does not fix the bug (just adds a comment)
        ineffective_diff = """--- a/src/calc.py
+++ b/src/calc.py
@@ -1,2 +1,3 @@
 def divide_numbers(a: float, b: float) -> float:
+    # Ineffective comment
     return a / b
"""
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=ineffective_diff,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)

        assert result.gate_a.passed is True
        assert result.gate_b.passed is False
        assert result.verified is False
        assert result.status == InvestigationStatus.VERIFICATION_FAILED
        assert result.failing_gate == "GATE_B"
        assert "did not pass on patched worktree" in result.gate_b.message

    def test_gate_c_fails_when_patch_breaks_existing_tests(
        self, sample_investigation_state: InvestigationState
    ):
        # A patch that fixes division but breaks addition
        breaking_diff = """--- a/src/calc.py
+++ b/src/calc.py
@@ -1,5 +1,7 @@
 def divide_numbers(a: float, b: float) -> float:
+    if b == 0:
+        return 0.0
     return a / b
 
 def add_numbers(a: float, b: float) -> float:
-    return a + b
+    return a - b
"""
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=breaking_diff,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)

        assert result.gate_a.passed is True
        assert result.gate_b.passed is True
        assert result.gate_c.passed is False
        assert result.verified is False
        assert result.status == InvestigationStatus.VERIFICATION_FAILED
        assert result.failing_gate == "GATE_C"
        assert "regressions detected" in result.gate_c.message.lower()

    def test_all_3_gates_pass_successfully(
        self, sample_investigation_state: InvestigationState
    ):
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)

        assert result.verified is True
        assert result.gate_a.passed is True
        assert result.gate_b.passed is True
        assert result.gate_c.passed is True
        assert result.status == InvestigationStatus.PATCH_VERIFIED
        assert sample_investigation_state.status == InvestigationStatus.PATCH_VERIFIED
        assert result.failing_gate is None


class TestWorkingTreeImmutability:
    """Verify that generation and verification never touch the user's working tree."""

    def test_original_working_tree_remains_unmodified_after_verification(
        self, temp_git_repo: Path, sample_investigation_state: InvestigationState
    ):
        # Record pre-verification git status and HEAD
        runner = SubprocessRunner(allowed_root=temp_git_repo)
        status_before = runner.run(["git", "status", "--porcelain"], cwd=temp_git_repo).stdout
        head_before = runner.run(["git", "rev-parse", "HEAD"], cwd=temp_git_repo).stdout

        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )

        verifier = VerificationService()
        result = verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)
        assert result.verified is True

        # Check post-verification git status and HEAD
        status_after = runner.run(["git", "status", "--porcelain"], cwd=temp_git_repo).stdout
        head_after = runner.run(["git", "rev-parse", "HEAD"], cwd=temp_git_repo).stdout

        assert status_before == status_after == ""
        assert head_before == head_after

        # Original source file must still be unchanged
        calc_content = (temp_git_repo / "src" / "calc.py").read_text(encoding="utf-8")
        assert "if b == 0:" not in calc_content


class TestPatchApplicationGuard:
    """Test safe patch application to original repository with strict human confirmation."""

    def test_apply_patch_requires_explicit_confirmation(
        self, temp_git_repo: Path, sample_investigation_state: InvestigationState
    ):
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )
        sample_investigation_state.candidate_patch = patch
        sample_investigation_state.status = InvestigationStatus.PATCH_VERIFIED

        with pytest.raises(ConfirmationRequiredError, match="Explicit human confirmation"):
            apply_verified_patch_to_original(temp_git_repo, sample_investigation_state, confirm=False)

    def test_apply_patch_rejects_unverified_investigations(
        self, temp_git_repo: Path, sample_investigation_state: InvestigationState
    ):
        sample_investigation_state.status = InvestigationStatus.COMPLETED

        with pytest.raises(InvalidPatchError, match="PATCH_VERIFIED"):
            apply_verified_patch_to_original(temp_git_repo, sample_investigation_state, confirm=True)

    def test_apply_patch_rejects_dirty_original_repo(
        self, temp_git_repo: Path, sample_investigation_state: InvestigationState
    ):
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )
        verifier = VerificationService()
        verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)

        # Make original repo dirty
        dirty_file = temp_git_repo / "dirty.txt"
        dirty_file.write_text("uncommitted changes", encoding="utf-8")

        with pytest.raises(PatchApplicationError, match="uncommitted changes"):
            apply_verified_patch_to_original(temp_git_repo, sample_investigation_state, confirm=True)

    def test_apply_patch_rejects_stale_commit_head(
        self, temp_git_repo: Path, sample_investigation_state: InvestigationState
    ):
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )
        verifier = VerificationService()
        verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)

        # Commit a new change to move HEAD
        new_file = temp_git_repo / "new.txt"
        new_file.write_text("new file", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(temp_git_repo), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "advance head"], cwd=str(temp_git_repo), check=True, capture_output=True)

        with pytest.raises(StaleVerificationError, match="Verification is stale"):
            apply_verified_patch_to_original(temp_git_repo, sample_investigation_state, confirm=True)

    def test_apply_patch_applies_cleanly_when_confirmed(
        self, temp_git_repo: Path, sample_investigation_state: InvestigationState
    ):
        patch = CandidatePatch(
            investigation_id=sample_investigation_state.investigation_id,
            hypothesis_id=sample_investigation_state.hypotheses[0].id,
            diff=SAMPLE_VALID_DIFF,
            target_files=["src/calc.py"],
        )
        reg_test = RegressionTest(
            investigation_id=sample_investigation_state.investigation_id,
            test_file_path="tests/test_repro.py",
            test_name="test_divide_zero_reproduction",
            test_content=SAMPLE_REGRESSION_TEST_CONTENT,
        )
        verifier = VerificationService()
        verifier.verify(sample_investigation_state, patch=patch, regression_test=reg_test)

        # Apply confirmed patch
        apply_verified_patch_to_original(temp_git_repo, sample_investigation_state, confirm=True)

        assert sample_investigation_state.status == InvestigationStatus.PATCH_APPLIED
        calc_content = (temp_git_repo / "src" / "calc.py").read_text(encoding="utf-8")
        assert "if b == 0:" in calc_content


class TestVerificationApi:
    """Test REST API endpoints for patch proposing, verification, and application."""

    def test_api_patch_and_verify_lifecycle(
        self, temp_git_repo: Path, sample_investigation_state: InvestigationState
    ):
        repo = InvestigationRepository()
        repo.save(sample_investigation_state)

        app = create_app(db=repo)
        client = TestClient(app)

        # 1. Propose candidate patch
        propose_resp = client.post(
            f"/api/investigations/{sample_investigation_state.investigation_id}/propose-patch",
            json={
                "diff": SAMPLE_VALID_DIFF,
                "test_file_path": "tests/test_repro.py",
                "test_name": "test_divide_zero_reproduction",
                "test_content": SAMPLE_REGRESSION_TEST_CONTENT,
                "explanation": "Add zero-denominator guard",
            },
        )
        assert propose_resp.status_code == 200
        data = propose_resp.json()
        assert data["status"] == InvestigationStatus.REGRESSION_TEST_CREATED.value
        assert data["candidate_patch"]["diff"] == SAMPLE_VALID_DIFF

        # 2. Get patch
        get_patch_resp = client.get(f"/api/investigations/{sample_investigation_state.investigation_id}/patch")
        assert get_patch_resp.status_code == 200
        assert get_patch_resp.json()["candidate_patch"]["target_files"] == ["src/calc.py"]

        # 3. Verify patch
        verify_resp = client.post(f"/api/investigations/{sample_investigation_state.investigation_id}/verify")
        assert verify_resp.status_code == 200
        verify_data = verify_resp.json()
        assert verify_data["verified"] is True
        assert verify_data["status"] == InvestigationStatus.PATCH_VERIFIED.value

        # 4. Get verification details
        verif_details_resp = client.get(f"/api/investigations/{sample_investigation_state.investigation_id}/verification")
        assert verif_details_resp.status_code == 200
        assert verif_details_resp.json()["gate_a"]["passed"] is True
        assert verif_details_resp.json()["gate_b"]["passed"] is True
        assert verif_details_resp.json()["gate_c"]["passed"] is True

        # 5. Apply patch without confirmation -> 400
        apply_unconf = client.post(f"/api/investigations/{sample_investigation_state.investigation_id}/apply-patch?confirm=false")
        assert apply_unconf.status_code == 400

        # 6. Apply patch with confirmation -> 200
        apply_conf = client.post(f"/api/investigations/{sample_investigation_state.investigation_id}/apply-patch?confirm=true")
        assert apply_conf.status_code == 200
        assert apply_conf.json()["status"] == InvestigationStatus.PATCH_APPLIED.value
