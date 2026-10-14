"""Independent 3-point verification service and patch application guard."""

from pathlib import Path
import time
from typing import Optional, Union
from uuid import uuid4

from rootcause.core.exceptions import (
    ConfirmationRequiredError,
    InvalidPatchError,
    InvalidStateTransitionError,
    PatchApplicationError,
    StaleVerificationError,
)
from rootcause.core.models import (
    CandidatePatch,
    InvestigationState,
    InvestigationStatus,
    RegressionTest,
    VerificationGateResult,
    VerificationResult,
)
from rootcause.security.path_guard import validate_safe_path
from rootcause.security.sandbox import SubprocessRunner
from rootcause.security.worktree import IsolatedWorktree
from rootcause.tools.test_runner import check_test_reproduces_bug, run_pytest
from rootcause.verification.patch_engine import apply_diff_to_worktree


class VerificationService:
    """Orchestrates independent, deterministic 3-point verification within disposable worktrees."""

    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout

    def verify(
        self,
        state: InvestigationState,
        patch: Optional[CandidatePatch] = None,
        regression_test: Optional[RegressionTest] = None,
        timeout: Optional[float] = None,
    ) -> VerificationResult:
        """Execute the 3-point verification gate against candidate patch and regression test.
        
        Gates evaluated:
          - Gate A: Regression test fails on clean baseline and reproduces reported bug symptom.
          - Gate B: Regression test passes cleanly after candidate patch is applied to worktree.
          - Gate C: Full existing test suite passes on patched worktree with 0 regressions.
          
        Args:
            state: Active investigation state.
            patch: Optional CandidatePatch (defaults to state.candidate_patch).
            regression_test: Optional RegressionTest (defaults to state.regression_test).
            timeout: Subprocess execution timeout in seconds.
            
        Returns:
            VerificationResult with outcome details.
            
        Raises:
            InvalidPatchError: If patch or test is missing.
            InvalidStateTransitionError: If state cannot transition to verification states.
        """
        gate_timeout = timeout if timeout is not None else self.default_timeout

        # Resolve candidate patch and regression test
        cand_patch = patch or state.candidate_patch
        if not cand_patch:
            raise InvalidPatchError("No candidate patch provided or attached to investigation state.")

        reg_test = regression_test or state.regression_test
        if not reg_test:
            raise InvalidPatchError("No regression test provided or attached to investigation state.")

        # Ensure state holds references
        state.candidate_patch = cand_patch
        state.regression_test = reg_test

        # Advance state machine cleanly to VERIFYING
        if state.status in {InvestigationStatus.COMPLETED, InvestigationStatus.HYPOTHESIS_FORMED}:
            state.transition_to(InvestigationStatus.PATCH_PROPOSED)

        if state.status == InvestigationStatus.PATCH_PROPOSED:
            state.transition_to(InvestigationStatus.REGRESSION_TEST_CREATED)

        if state.status == InvestigationStatus.REGRESSION_TEST_CREATED:
            state.transition_to(InvestigationStatus.VERIFYING)
        elif state.status != InvestigationStatus.VERIFYING:
            # If state is in VERIFICATION_FAILED or INVESTIGATION_INCONCLUSIVE, transition back to PATCH_PROPOSED -> REGRESSION_TEST_CREATED -> VERIFYING
            if state.status in {InvestigationStatus.VERIFICATION_FAILED, InvestigationStatus.INVESTIGATION_INCONCLUSIVE}:
                state.transition_to(InvestigationStatus.PATCH_PROPOSED)
                state.transition_to(InvestigationStatus.REGRESSION_TEST_CREATED)
                state.transition_to(InvestigationStatus.VERIFYING)
            else:
                raise InvalidStateTransitionError(state.status.value, InvestigationStatus.VERIFYING.value)

        repo_root = Path(state.request.repo_path).resolve()
        runner = SubprocessRunner(allowed_root=repo_root)

        # Record commit being tested
        head_run = runner.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True)
        commit_tested = head_run.stdout.strip()

        attempt_id = str(uuid4())

        # All verification operations execute inside disposable isolated worktree
        with IsolatedWorktree(repo_root, commit_or_ref="HEAD") as worktree:
            wt_path = worktree.worktree_path

            # Write regression test inside worktree
            safe_test_path = validate_safe_path(wt_path, reg_test.test_file_path, allow_nonexistent=True)
            safe_test_path.parent.mkdir(parents=True, exist_ok=True)
            safe_test_path.write_text(reg_test.test_content, encoding="utf-8")

            rel_test_target = str(safe_test_path.relative_to(wt_path)).replace("\\", "/")

            # =========================================================================
            # GATE A: Baseline Reproduction
            # =========================================================================
            t0 = time.perf_counter()
            test_res_a = run_pytest(wt_path, test_target=rel_test_target, timeout=gate_timeout)
            dur_a = round(time.perf_counter() - t0, 4)

            gate_a_passed = False
            gate_a_message = ""

            if test_res_a.timed_out:
                gate_a_message = f"Gate A failed: regression test timed out after {gate_timeout}s on clean baseline."
            elif (
                test_res_a.exit_code in {2, 4}
                or "error collecting" in test_res_a.output.lower()
                or "collection failure" in test_res_a.output.lower()
                or (test_res_a.errors > 0 and test_res_a.failures == 0)
            ):
                gate_a_message = (
                    f"Gate A failed: regression test encountered collection or import error on clean baseline: "
                    f"{test_res_a.output}"
                )
            elif test_res_a.passed:
                gate_a_message = (
                    "Gate A failed: regression test passed on clean baseline; "
                    "expected failure reproducing the reported bug."
                )
            else:
                reproduces, reason = check_test_reproduces_bug(
                    test_res_a,
                    state.request.bug_description,
                    state.request.stack_trace,
                )
                if not reproduces:
                    gate_a_message = (
                        f"Gate A failed: test failed on clean baseline, but did not reproduce the reported bug symptom: {reason}"
                    )
                else:
                    gate_a_passed = True
                    gate_a_message = f"Gate A passed: baseline failure reproduced bug symptom ({reason})."

            gate_a_result = VerificationGateResult(
                gate_name="GATE_A",
                passed=gate_a_passed,
                message=gate_a_message,
                test_result=test_res_a,
                duration_seconds=dur_a,
            )

            if not gate_a_passed:
                verification_result = VerificationResult(
                    verification_attempt_id=attempt_id,
                    investigation_id=state.investigation_id,
                    patch_id=cand_patch.id,
                    regression_test_id=reg_test.id,
                    commit_tested=commit_tested,
                    status=InvestigationStatus.INVESTIGATION_INCONCLUSIVE,
                    gate_a=gate_a_result,
                    verified=False,
                    failing_gate="GATE_A",
                    summary=f"Verification stopped at Gate A: {gate_a_message}",
                )
                state.transition_to(InvestigationStatus.INVESTIGATION_INCONCLUSIVE)
                state.verification_result = verification_result
                return verification_result

            # =========================================================================
            # GATE B: Patched Regression Test
            # =========================================================================
            t0 = time.perf_counter()
            try:
                apply_diff_to_worktree(wt_path, cand_patch.diff)
            except Exception as exc:
                dur_b = round(time.perf_counter() - t0, 4)
                gate_b_result = VerificationGateResult(
                    gate_name="GATE_B",
                    passed=False,
                    message=f"Gate B failed during patch application: {exc}",
                    duration_seconds=dur_b,
                )
                verification_result = VerificationResult(
                    verification_attempt_id=attempt_id,
                    investigation_id=state.investigation_id,
                    patch_id=cand_patch.id,
                    regression_test_id=reg_test.id,
                    commit_tested=commit_tested,
                    status=InvestigationStatus.VERIFICATION_FAILED,
                    gate_a=gate_a_result,
                    gate_b=gate_b_result,
                    verified=False,
                    failing_gate="GATE_B",
                    summary=f"Verification stopped at Gate B: failed to apply patch ({exc})",
                )
                state.transition_to(InvestigationStatus.VERIFICATION_FAILED)
                state.verification_result = verification_result
                return verification_result

            test_res_b = run_pytest(wt_path, test_target=rel_test_target, timeout=gate_timeout)
            dur_b = round(time.perf_counter() - t0, 4)

            gate_b_passed = False
            gate_b_message = ""

            if test_res_b.timed_out:
                gate_b_message = f"Gate B failed: regression test timed out after {gate_timeout}s on patched worktree."
            elif not test_res_b.passed or test_res_b.failures > 0 or test_res_b.errors > 0:
                gate_b_message = (
                    f"Gate B failed: regression test did not pass on patched worktree "
                    f"({test_res_b.failures} failed, {test_res_b.errors} errors)."
                )
            else:
                gate_b_passed = True
                gate_b_message = f"Gate B passed: regression test passed cleanly on patched worktree ({test_res_b.passed_count} passed)."

            gate_b_result = VerificationGateResult(
                gate_name="GATE_B",
                passed=gate_b_passed,
                message=gate_b_message,
                test_result=test_res_b,
                duration_seconds=dur_b,
            )

            if not gate_b_passed:
                verification_result = VerificationResult(
                    verification_attempt_id=attempt_id,
                    investigation_id=state.investigation_id,
                    patch_id=cand_patch.id,
                    regression_test_id=reg_test.id,
                    commit_tested=commit_tested,
                    status=InvestigationStatus.VERIFICATION_FAILED,
                    gate_a=gate_a_result,
                    gate_b=gate_b_result,
                    verified=False,
                    failing_gate="GATE_B",
                    summary=f"Verification stopped at Gate B: {gate_b_message}",
                )
                state.transition_to(InvestigationStatus.VERIFICATION_FAILED)
                state.verification_result = verification_result
                return verification_result

            # =========================================================================
            # GATE C: Full Existing Test Suite (Zero Regressions)
            # =========================================================================
            t0 = time.perf_counter()
            test_res_c = run_pytest(wt_path, test_target=None, timeout=gate_timeout * 2)
            dur_c = round(time.perf_counter() - t0, 4)

            gate_c_passed = False
            gate_c_message = ""

            if test_res_c.timed_out:
                gate_c_message = "Gate C failed: full test suite timed out on patched worktree."
            elif not test_res_c.passed or test_res_c.failures > 0 or test_res_c.errors > 0:
                gate_c_message = (
                    f"Gate C failed: regressions detected in full test suite after applying patch "
                    f"({test_res_c.failures} failed, {test_res_c.errors} errors)."
                )
            else:
                gate_c_passed = True
                gate_c_message = (
                    f"Gate C passed: full test suite passed with 0 regressions "
                    f"({test_res_c.passed_count} passed)."
                )

            gate_c_result = VerificationGateResult(
                gate_name="GATE_C",
                passed=gate_c_passed,
                message=gate_c_message,
                test_result=test_res_c,
                duration_seconds=dur_c,
            )

            if not gate_c_passed:
                verification_result = VerificationResult(
                    verification_attempt_id=attempt_id,
                    investigation_id=state.investigation_id,
                    patch_id=cand_patch.id,
                    regression_test_id=reg_test.id,
                    commit_tested=commit_tested,
                    status=InvestigationStatus.VERIFICATION_FAILED,
                    gate_a=gate_a_result,
                    gate_b=gate_b_result,
                    gate_c=gate_c_result,
                    verified=False,
                    failing_gate="GATE_C",
                    summary=f"Verification stopped at Gate C: {gate_c_message}",
                )
                state.transition_to(InvestigationStatus.VERIFICATION_FAILED)
                state.verification_result = verification_result
                return verification_result

            # =========================================================================
            # SUCCESS: All 3 Gates Passed in the Same Attempt
            # =========================================================================
            state.transition_to(InvestigationStatus.PATCH_VERIFIED)

            verification_result = VerificationResult(
                verification_attempt_id=attempt_id,
                investigation_id=state.investigation_id,
                patch_id=cand_patch.id,
                regression_test_id=reg_test.id,
                commit_tested=commit_tested,
                status=InvestigationStatus.PATCH_VERIFIED,
                gate_a=gate_a_result,
                gate_b=gate_b_result,
                gate_c=gate_c_result,
                verified=True,
                failing_gate=None,
                summary=(
                    "All 3 verification gates passed successfully: baseline failure reproduced symptom, "
                    "patch fixed regression test, and full test suite passed with 0 regressions."
                ),
            )
            state.verification_result = verification_result
            return verification_result


def apply_verified_patch_to_original(
    original_repo: Union[Path, str],
    state: InvestigationState,
    confirm: bool = False,
) -> None:
    """Apply a verified patch to the original repository with strict guardrails.
    
    Requires:
      1. Explicit human confirmation (confirm=True).
      2. Investigation status is PATCH_VERIFIED with valid 3-gate pass.
      3. Working tree of original repository is clean (no unstaged or uncommitted changes).
      4. Current HEAD of original repository matches the exact commit tested.
      
    Args:
        original_repo: Path to target user repository.
        state: InvestigationState that has reached PATCH_VERIFIED.
        confirm: Explicit human confirmation flag.
        
    Raises:
        ConfirmationRequiredError: If confirm is False.
        InvalidPatchError: If state is not PATCH_VERIFIED or patch is missing.
        StaleVerificationError: If repository HEAD has moved since verification.
        PatchApplicationError: If working tree is dirty or diff fails to apply.
    """
    if not confirm:
        raise ConfirmationRequiredError(
            "Explicit human confirmation (confirm=True) is required to apply patch to original repository."
        )

    if state.status != InvestigationStatus.PATCH_VERIFIED:
        raise InvalidPatchError(
            f"Cannot apply patch in state '{state.status.value}'. Investigation status must be 'PATCH_VERIFIED'."
        )

    if not state.verification_result or not state.verification_result.verified:
        raise InvalidPatchError(
            "Cannot apply patch: no verified 3-gate result recorded for this investigation."
        )

    if not state.candidate_patch:
        raise InvalidPatchError("No candidate patch found on investigation state.")

    repo_path = Path(original_repo).resolve()
    runner = SubprocessRunner(allowed_root=repo_path)

    # Verify commit matches
    head_res = runner.run(["git", "rev-parse", "HEAD"], cwd=repo_path, check=True)
    current_head = head_res.stdout.strip()
    if current_head != state.verification_result.commit_tested:
        raise StaleVerificationError(
            f"Repository HEAD ({current_head}) differs from verified commit "
            f"({state.verification_result.commit_tested}). Verification is stale."
        )

    # Verify working tree is completely clean
    status_res = runner.run(["git", "status", "--porcelain"], cwd=repo_path, check=True)
    if status_res.stdout.strip():
        raise PatchApplicationError(
            "Original repository has uncommitted changes. Working tree must be clean to apply verified patch."
        )

    # Apply diff safely to original repository
    apply_diff_to_worktree(repo_path, state.candidate_patch.diff)

    # Transition state to PATCH_APPLIED
    state.transition_to(InvestigationStatus.PATCH_APPLIED)
