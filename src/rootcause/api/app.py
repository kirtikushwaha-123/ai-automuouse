"""FastAPI application for RootCause AI investigation API."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from rootcause.core.exceptions import ConfigurationError, RootCauseException
from rootcause.core.models import (
    InvestigationReport,
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
)
from rootcause.db.repository import InvestigationRepository
from rootcause.llm import get_llm_provider
from rootcause.llm.base import BaseLLMProvider
from rootcause.orchestrator.agent import InvestigationOrchestrator


class InvestigationSubmitResponse(BaseModel):
    """Initial response upon submitting an investigation."""
    investigation_id: str
    status: InvestigationStatus
    repo_path: str
    bug_description: str
    message: str


def create_app(
    db: Optional[InvestigationRepository] = None,
    provider: Optional[BaseLLMProvider] = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="RootCause AI API",
        description="Evidence-Based Autonomous Bug Investigation Agent API",
        version="0.1.0",
    )

    # State containers
    app.state.db = db if db is not None else InvestigationRepository()
    app.state.provider = provider

    def _get_orchestrator() -> InvestigationOrchestrator:
        if app.state.provider:
            active_provider = app.state.provider
        else:
            try:
                active_provider = get_llm_provider()
            except ConfigurationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"LLM Provider not configured: {exc}",
                )
        return InvestigationOrchestrator(provider=active_provider, db=app.state.db)

    @app.get("/health", tags=["Health"])
    def health_check() -> Dict[str, str]:
        return {"status": "ok", "service": "rootcause-ai", "version": "0.1.0"}

    @app.post(
        "/api/investigations",
        status_code=status.HTTP_201_CREATED,
        response_model=InvestigationSubmitResponse,
        tags=["Investigations"],
    )
    def submit_investigation(
        request: InvestigationRequest,
        background_tasks: BackgroundTasks,
        sync: bool = Query(False, description="Run synchronously before returning (useful for automated testing)"),
    ) -> InvestigationSubmitResponse:
        # Validate repo path exists
        repo = Path(request.repo_path).resolve()
        if not repo.exists() or not repo.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Repository path does not exist or is not a directory: {request.repo_path}",
            )

        orchestrator = _get_orchestrator()

        if sync:
            state = orchestrator.run_investigation(request)
            return InvestigationSubmitResponse(
                investigation_id=state.investigation_id,
                status=state.status,
                repo_path=str(repo),
                bug_description=request.bug_description,
                message="Investigation completed synchronously.",
            )
        else:
            # Create initial RECEIVED record in database
            initial_state = InvestigationState(
                request=request,
                status=InvestigationStatus.RECEIVED,
            )
            app.state.db.save(initial_state)

            background_tasks.add_task(orchestrator.run_investigation, request)

            return InvestigationSubmitResponse(
                investigation_id=initial_state.investigation_id,
                status=InvestigationStatus.RECEIVED,
                repo_path=str(repo),
                bug_description=request.bug_description,
                message="Investigation queued and running in background.",
            )

    @app.get(
        "/api/investigations/{investigation_id}",
        response_model=InvestigationState,
        tags=["Investigations"],
    )
    def get_investigation_status(investigation_id: str) -> InvestigationState:
        state = app.state.db.get(investigation_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation with ID '{investigation_id}' was not found.",
            )
        return state

    @app.get(
        "/api/investigations/{investigation_id}/report",
        response_model=InvestigationReport,
        tags=["Investigations"],
    )
    def get_investigation_report(investigation_id: str) -> InvestigationReport:
        state = app.state.db.get(investigation_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation with ID '{investigation_id}' was not found.",
            )
        orchestrator = _get_orchestrator()
        return orchestrator.generate_report(state)

    @app.get(
        "/api/investigations",
        tags=["Investigations"],
    )
    def list_investigations(limit: int = Query(50, ge=1, le=100)) -> List[Dict[str, Any]]:
        return app.state.db.list_all(limit=limit)

    # -------------------------------------------------------------------------
    # Phase 4: Patch Generation and Verification Endpoints
    # -------------------------------------------------------------------------

    @app.post(
        "/api/investigations/{investigation_id}/propose-patch",
        tags=["Patch"],
    )
    def propose_patch(
        investigation_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Propose a candidate patch and regression test for a diagnosed bug."""
        from rootcause.core.exceptions import InvalidPatchError, SecurityViolation
        from rootcause.core.models import CandidatePatch, RegressionTest
        from rootcause.verification.patch_engine import (
            generate_candidate_patch,
            parse_diff_target_files,
            validate_patch_safety,
        )

        state = app.state.db.get(investigation_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation with ID '{investigation_id}' was not found.",
            )

        payload = payload or {}
        diff_str = payload.get("diff")

        try:
            if diff_str:
                repo_root = Path(state.request.repo_path)
                targets = validate_patch_safety(repo_root, diff_str)
                hypothesis_id = payload.get("hypothesis_id") or (
                    state.hypotheses[0].id if state.hypotheses else "hyp_custom"
                )
                candidate_patch = CandidatePatch(
                    investigation_id=state.investigation_id,
                    hypothesis_id=hypothesis_id,
                    diff=diff_str,
                    target_files=targets,
                    explanation=payload.get("explanation", "User-provided candidate patch"),
                )
                regression_test = RegressionTest(
                    investigation_id=state.investigation_id,
                    test_file_path=payload.get("test_file_path", "tests/test_rootcause_repro.py"),
                    test_name=payload.get("test_name", "test_reproduce_bug"),
                    test_content=payload.get("test_content", "import pytest\ndef test_reproduce_bug():\n    assert True\n"),
                )
            else:
                provider = app.state.provider
                if not provider:
                    try:
                        provider = get_llm_provider()
                    except ConfigurationError as exc:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"LLM Provider not configured: {exc}",
                        )
                candidate_patch, regression_test = generate_candidate_patch(provider, state)

            state.candidate_patch = candidate_patch
            state.regression_test = regression_test

            if state.status in {InvestigationStatus.COMPLETED, InvestigationStatus.HYPOTHESIS_FORMED}:
                state.transition_to(InvestigationStatus.PATCH_PROPOSED)
            if state.status == InvestigationStatus.PATCH_PROPOSED:
                state.transition_to(InvestigationStatus.REGRESSION_TEST_CREATED)

            app.state.db.save(state)

            return {
                "investigation_id": state.investigation_id,
                "status": state.status,
                "candidate_patch": candidate_patch.model_dump(mode="json"),
                "regression_test": regression_test.model_dump(mode="json"),
            }
        except (InvalidPatchError, SecurityViolation, RootCauseException) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

    @app.get(
        "/api/investigations/{investigation_id}/patch",
        tags=["Patch"],
    )
    def get_patch(investigation_id: str) -> Dict[str, Any]:
        """Inspect the current candidate patch and regression test."""
        state = app.state.db.get(investigation_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation with ID '{investigation_id}' was not found.",
            )
        if not state.candidate_patch:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No candidate patch proposed for investigation '{investigation_id}'.",
            )
        return {
            "investigation_id": state.investigation_id,
            "candidate_patch": state.candidate_patch.model_dump(mode="json"),
            "regression_test": state.regression_test.model_dump(mode="json") if state.regression_test else None,
        }

    @app.post(
        "/api/investigations/{investigation_id}/verify",
        tags=["Verification"],
    )
    def verify_investigation(
        investigation_id: str,
        timeout: Optional[float] = Query(None, description="Timeout per gate in seconds"),
    ) -> Dict[str, Any]:
        """Run independent 3-point verification for candidate patch in disposable worktree."""
        from rootcause.core.exceptions import InvalidPatchError, RootCauseException
        from rootcause.verification.verifier import VerificationService

        state = app.state.db.get(investigation_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation with ID '{investigation_id}' was not found.",
            )

        if not state.candidate_patch or not state.regression_test:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Investigation must have both a candidate patch and a regression test before verification.",
            )

        try:
            verifier = VerificationService(default_timeout=timeout or 30.0)
            result = verifier.verify(state, timeout=timeout)
            app.state.db.save(state)
            return result.model_dump(mode="json")
        except RootCauseException as exc:
            app.state.db.save(state)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

    @app.get(
        "/api/investigations/{investigation_id}/verification",
        tags=["Verification"],
    )
    def get_verification_result(investigation_id: str) -> Dict[str, Any]:
        """Retrieve the 3-point verification gate results."""
        state = app.state.db.get(investigation_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation with ID '{investigation_id}' was not found.",
            )
        if not state.verification_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No verification result recorded for investigation '{investigation_id}'.",
            )
        return state.verification_result.model_dump(mode="json")

    @app.post(
        "/api/investigations/{investigation_id}/apply-patch",
        tags=["Patch"],
    )
    def apply_patch(
        investigation_id: str,
        confirm: bool = Query(False, description="Explicit human confirmation to apply to original repository"),
    ) -> Dict[str, Any]:
        """Apply a verified patch to the original repository with human confirmation."""
        from rootcause.core.exceptions import (
            ConfirmationRequiredError,
            InvalidPatchError,
            PatchApplicationError,
            StaleVerificationError,
        )
        from rootcause.verification.verifier import apply_verified_patch_to_original

        if not confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Explicit confirmation (confirm=true) is required to apply patch to original repository.",
            )

        state = app.state.db.get(investigation_id)
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation with ID '{investigation_id}' was not found.",
            )

        try:
            apply_verified_patch_to_original(state.request.repo_path, state, confirm=True)
            app.state.db.save(state)
            return {
                "message": "Verified patch applied successfully to original repository.",
                "investigation_id": state.investigation_id,
                "status": state.status,
            }
        except ConfirmationRequiredError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except StaleVerificationError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
        except (InvalidPatchError, PatchApplicationError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return app


# Default ASGI application instance for uvicorn (e.g. uvicorn rootcause.api.app:app)
app = create_app()
