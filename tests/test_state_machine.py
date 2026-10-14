"""Tests for the investigation state machine and allowed transitions."""

from pathlib import Path
import pytest

from rootcause.core.exceptions import InvalidStateTransitionError
from rootcause.core.models import (
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
)


def _make_state() -> InvestigationState:
    return InvestigationState(
        request=InvestigationRequest(
            repo_path=str(Path.cwd()),
            bug_description="Sample bug for state test",
        ),
        status=InvestigationStatus.RECEIVED,
    )


class TestInvestigationStateMachine:
    """Verify permitted and rejected state transitions."""

    def test_valid_lifecycle_with_hypotheses(self):
        state = _make_state()
        assert state.status == InvestigationStatus.RECEIVED

        state.transition_to(InvestigationStatus.INVESTIGATING)
        assert state.status == InvestigationStatus.INVESTIGATING

        state.transition_to(InvestigationStatus.HYPOTHESIS_FORMED)
        assert state.status == InvestigationStatus.HYPOTHESIS_FORMED

        state.transition_to(InvestigationStatus.COMPLETED)
        assert state.status == InvestigationStatus.COMPLETED

    def test_valid_lifecycle_with_insufficient_evidence(self):
        state = _make_state()
        state.transition_to(InvestigationStatus.INVESTIGATING)
        state.transition_to(InvestigationStatus.INSUFFICIENT_EVIDENCE)
        state.transition_to(InvestigationStatus.COMPLETED)
        assert state.status == InvestigationStatus.COMPLETED

    def test_valid_transition_to_failed(self):
        state = _make_state()
        state.transition_to(InvestigationStatus.INVESTIGATING)
        state.transition_to(InvestigationStatus.FAILED)
        assert state.status == InvestigationStatus.FAILED

    def test_reject_direct_jump_from_received_to_completed(self):
        state = _make_state()
        with pytest.raises(InvalidStateTransitionError, match="from 'RECEIVED' to 'COMPLETED'"):
            state.transition_to(InvestigationStatus.COMPLETED)

    def test_reject_direct_jump_from_investigating_to_completed(self):
        state = _make_state()
        state.transition_to(InvestigationStatus.INVESTIGATING)
        with pytest.raises(InvalidStateTransitionError, match="from 'INVESTIGATING' to 'COMPLETED'"):
            state.transition_to(InvestigationStatus.COMPLETED)

    def test_reject_transitions_out_of_completed(self):
        state = _make_state()
        state.transition_to(InvestigationStatus.INVESTIGATING)
        state.transition_to(InvestigationStatus.HYPOTHESIS_FORMED)
        state.transition_to(InvestigationStatus.COMPLETED)

        with pytest.raises(InvalidStateTransitionError, match="from 'COMPLETED' to 'INVESTIGATING'"):
            state.transition_to(InvestigationStatus.INVESTIGATING)

    def test_reject_transitions_out_of_failed(self):
        state = _make_state()
        state.transition_to(InvestigationStatus.FAILED)

        with pytest.raises(InvalidStateTransitionError, match="from 'FAILED' to 'RECEIVED'"):
            state.transition_to(InvestigationStatus.RECEIVED)
