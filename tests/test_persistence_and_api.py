"""Tests for SQLite persistence and FastAPI endpoints."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from rootcause.api import create_app
from rootcause.core.models import (
    Evidence,
    EvidenceType,
    Hypothesis,
    InvestigationRequest,
    InvestigationState,
    InvestigationStatus,
)
from rootcause.db.repository import InvestigationRepository
from rootcause.llm.base import LLMResponse, ToolCall
from rootcause.llm.mock_provider import MockLLMProvider


class TestSQLitePersistence:
    """Verify SQLite storage, retrieval, and full model serialization."""

    def test_save_and_retrieve_investigation_state(self, tmp_path: Path):
        db_file = tmp_path / "test_store.db"
        repo = InvestigationRepository(db_path=db_file)

        req = InvestigationRequest(
            repo_path=str(tmp_path),
            bug_description="Persisted investigation bug",
        )
        state = InvestigationState(
            request=req,
            status=InvestigationStatus.HYPOTHESIS_FORMED,
        )

        ev = Evidence(
            evidence_type=EvidenceType.SOURCE_CODE,
            source="calc.py:10",
            content="assert 1 == 2",
            why_relevant="Broken assertion",
        )
        state.facts.append(ev)

        hypo = Hypothesis(
            title="Assertion mismatch",
            description="Expected 1 to equal 2",
            why_plausible="Code observed",
            confidence_score=0.85,
            supporting_evidence_ids=[ev.id],
        )
        state.hypotheses.append(hypo)

        repo.save(state)

        # Retrieve and verify exact state reconstruction
        retrieved = repo.get(state.investigation_id)
        assert retrieved is not None
        assert retrieved.investigation_id == state.investigation_id
        assert retrieved.status == InvestigationStatus.HYPOTHESIS_FORMED
        assert len(retrieved.facts) == 1
        assert retrieved.facts[0].id == ev.id
        assert retrieved.facts[0].content == "assert 1 == 2"
        assert len(retrieved.hypotheses) == 1
        assert retrieved.hypotheses[0].supporting_evidence_ids == [ev.id]

    def test_list_and_delete_investigations(self, tmp_path: Path):
        repo = InvestigationRepository(db_path=tmp_path / "list_del.db")
        s1 = InvestigationState(
            request=InvestigationRequest(repo_path=str(tmp_path), bug_description="Bug 1"),
        )
        s2 = InvestigationState(
            request=InvestigationRequest(repo_path=str(tmp_path), bug_description="Bug 2"),
        )
        repo.save(s1)
        repo.save(s2)

        records = repo.list_all()
        assert len(records) == 2
        ids = [r["id"] for r in records]
        assert s1.investigation_id in ids
        assert s2.investigation_id in ids

        deleted = repo.delete(s1.investigation_id)
        assert deleted is True
        assert repo.get(s1.investigation_id) is None
        assert len(repo.list_all()) == 1


class TestFastAPIEndpoints:
    """Verify REST API routes, request validation, error handling, and reporting."""

    def test_health_check(self):
        app = create_app()
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_submit_investigation_sync(self, temp_git_repo: Path):
        db = InvestigationRepository()
        provider = MockLLMProvider()
        # Script a quick declaration of insufficient evidence
        provider.queue_tool_call(
            "declare_insufficient_evidence",
            {"reason": "Fast test pass"},
        )

        app = create_app(db=db, provider=provider)
        client = TestClient(app)

        payload = {
            "repo_path": str(temp_git_repo),
            "bug_description": "API submission test",
            "max_iterations": 2,
        }

        # Run synchronously via sync=true query param
        res = client.post("/api/investigations?sync=true", json=payload)
        assert res.status_code == 201
        data = res.json()
        inv_id = data["investigation_id"]
        assert data["status"] == "COMPLETED"

        # Fetch status
        get_res = client.get(f"/api/investigations/{inv_id}")
        assert get_res.status_code == 200
        state_data = get_res.json()
        assert state_data["status"] == "COMPLETED"
        assert len(state_data["audit_trail"]) >= 1

        # Fetch report
        rep_res = client.get(f"/api/investigations/{inv_id}/report")
        assert rep_res.status_code == 200
        rep_data = rep_res.json()
        assert rep_data["investigation_id"] == inv_id
        assert "# RootCause AI Investigation Report" in rep_data["markdown_report"]

    def test_get_nonexistent_investigation_returns_404(self):
        app = create_app()
        client = TestClient(app)
        res = client.get("/api/investigations/nonexistent-id-000")
        assert res.status_code == 404
        assert "was not found" in res.json()["detail"]

    def test_submit_invalid_repo_path_returns_400(self):
        app = create_app()
        client = TestClient(app)
        payload = {
            "repo_path": "C:/nonexistent/invalid/repo/path",
            "bug_description": "Bad path test",
        }
        res = client.post("/api/investigations", json=payload)
        assert res.status_code == 400
        assert "does not exist" in res.json()["detail"]

    def test_missing_provider_configuration_returns_503(self, temp_git_repo: Path, monkeypatch):
        monkeypatch.delenv("ROOTCAUSE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ROOTCAUSE_LLM_PROVIDER", "openai")

        # App created with no pre-injected provider
        app = create_app(provider=None)
        client = TestClient(app)

        payload = {
            "repo_path": str(temp_git_repo),
            "bug_description": "Missing key test",
        }
        res = client.post("/api/investigations?sync=true", json=payload)
        assert res.status_code == 503
        assert "LLM Provider not configured" in res.json()["detail"]
