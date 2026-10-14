"""Comprehensive test suite for RootCause AI Phase 5.2: Live-Provider Evaluation.

Verifies:
1. Fail-closed behavior when credentials or configuration are missing (no simulation fallback).
2. Accurate token accounting, distinguishing missing from zero usage.
3. Model tracking (requested vs actual).
4. Error taxonomy and failure semantics (rate limits, auth errors, unattempted verification gates).
5. No ground-truth patch injection in live evaluation mode.
6. Cost estimation rules.
7. Secret scrubbing and repository immutability.
"""

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest

from rootcause.benchmarks.metrics import (
    calculate_benchmark_metrics,
    estimate_model_cost,
)
from rootcause.benchmarks.models import (
    ArtifactSource,
    BenchmarkCaseResult,
    BenchmarkCategory,
    EvaluationMode,
    RootCauseScore,
)
from rootcause.benchmarks.runner import (
    BenchmarkRunner,
    main,
    redact_secrets,
)
from rootcause.benchmarks.tracking import UsageTrackingProviderWrapper
from rootcause.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    MissingCredentialsError,
    RateLimitError,
)
from rootcause.core.models import (
    Evidence,
    EvidenceType,
    Hypothesis,
    InvestigationState,
    InvestigationStatus,
)
from rootcause.llm.base import BaseLLMProvider, LLMResponse, ToolCall
from rootcause.llm.openai_provider import OpenAICompatibleProvider


class DummyLiveLLMProvider(BaseLLMProvider):
    """Controllable LLM provider for simulating live provider interactions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        actual_model: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        responses: Optional[List[LLMResponse]] = None,
        side_effect: Optional[Exception] = None,
    ):
        self.model = model
        self.actual_model = actual_model or model
        self.usage = usage
        self.responses = list(responses or [])
        self.side_effect = side_effect
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        self.call_history.append({"messages": messages, "tools": tools})
        if self.side_effect:
            raise self.side_effect
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content="I have analyzed the code and found the root cause.",
            tool_calls=[],
            finish_reason="stop",
            usage=self.usage,
            model=self.actual_model,
        )


class TestLiveProviderConfigurationFailClosed:
    """Requirement: Live mode must fail closed without credentials or fallback."""

    def test_live_mode_fails_closed_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ROOTCAUSE_LLM_API_KEY", raising=False)

        with pytest.raises(MissingCredentialsError) as exc_info:
            BenchmarkRunner(evaluation_mode=EvaluationMode.LIVE_PROVIDER)

        assert "No LLM API key configured" in str(exc_info.value)

    def test_live_mode_never_calls_simulated_provider(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ROOTCAUSE_LLM_API_KEY", raising=False)

        with patch("rootcause.benchmarks.runner.create_simulated_benchmark_provider") as mock_sim:
            with pytest.raises(MissingCredentialsError):
                BenchmarkRunner(evaluation_mode=EvaluationMode.LIVE_PROVIDER)
            mock_sim.assert_not_called()

    def test_cli_live_mode_without_credentials_reports_not_run(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ROOTCAUSE_LLM_API_KEY", raising=False)

        with patch("sys.argv", ["runner.py", "--mode", "live"]):
            main()

        captured = capsys.readouterr()
        assert "LIVE BENCHMARK NOT RUN: Missing credentials" in captured.out
        assert "ROOTCAUSE_LLM_API_KEY" in captured.out
        assert "Per Phase 5.2 integrity rules, live mode will not fall back to simulation." in captured.out

    def test_openai_provider_raises_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ROOTCAUSE_LLM_API_KEY", raising=False)

        with pytest.raises(MissingCredentialsError):
            OpenAICompatibleProvider()

    def test_openai_provider_raises_authentication_error_on_401(self):
        mock_response = MagicMock()
        mock_response.status_code = 401

        provider = OpenAICompatibleProvider(api_key="test-key-12345")
        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(AuthenticationError) as exc_info:
                provider.generate([{"role": "user", "content": "hi"}])
            assert "Unauthorized" in str(exc_info.value)

    def test_openai_provider_raises_rate_limit_error_on_429(self):
        mock_response = MagicMock()
        mock_response.status_code = 429

        provider = OpenAICompatibleProvider(api_key="test-key-12345")
        with patch("httpx.Client.post", return_value=mock_response):
            with pytest.raises(RateLimitError) as exc_info:
                provider.generate([{"role": "user", "content": "hi"}])
            assert "429" in str(exc_info.value)

    def test_openai_provider_parses_usage_and_model(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4o-mini-2024-07-18",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "total_tokens": 165,
            },
        }

        provider = OpenAICompatibleProvider(api_key="test-key-12345", model="gpt-4o-mini")
        with patch("httpx.Client.post", return_value=mock_response):
            res = provider.generate([{"role": "user", "content": "hi"}])

        assert res.model == "gpt-4o-mini-2024-07-18"
        assert res.usage == {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
        }


class TestUsageTrackingAndAccounting:
    """Requirement: Accurate usage accounting, distinguishing missing from zero usage."""

    def test_tracker_aggregates_tokens_across_calls(self):
        resp1 = LLMResponse(
            content="one",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            model="gpt-4o-mini",
        )
        resp2 = LLMResponse(
            content="two",
            usage={"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110},
            model="gpt-4o-mini",
        )
        inner = DummyLiveLLMProvider(responses=[resp1, resp2])
        tracker = UsageTrackingProviderWrapper(inner)

        tracker.generate([{"role": "user", "content": "1"}])
        tracker.generate([{"role": "user", "content": "2"}])

        assert tracker.call_count == 2
        assert tracker.prompt_tokens == 130
        assert tracker.completion_tokens == 50
        assert tracker.total_tokens == 180

    def test_tracker_distinguishes_missing_from_zero_usage(self):
        # Case A: usage is None -> tracker tokens remain None
        resp_missing = LLMResponse(content="none", usage=None, model="gpt-4o-mini")
        tracker_missing = UsageTrackingProviderWrapper(DummyLiveLLMProvider(responses=[resp_missing]))
        tracker_missing.generate([{"role": "user", "content": "1"}])
        assert tracker_missing.prompt_tokens is None
        assert tracker_missing.completion_tokens is None
        assert tracker_missing.total_tokens is None

        # Case B: usage is 0 -> tracker tokens record 0
        resp_zero = LLMResponse(
            content="zero",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="gpt-4o-mini",
        )
        tracker_zero = UsageTrackingProviderWrapper(DummyLiveLLMProvider(responses=[resp_zero]))
        tracker_zero.generate([{"role": "user", "content": "1"}])
        assert tracker_zero.prompt_tokens == 0
        assert tracker_zero.completion_tokens == 0
        assert tracker_zero.total_tokens == 0

    def test_tracker_captures_actual_model_separately(self):
        resp = LLMResponse(
            content="ok",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="gpt-4o-mini-2024-07-18",
        )
        inner = DummyLiveLLMProvider(model="gpt-4o-mini", responses=[resp])
        tracker = UsageTrackingProviderWrapper(inner)

        assert tracker.requested_model == "gpt-4o-mini"
        tracker.generate([{"role": "user", "content": "test"}])
        assert tracker.actual_model == "gpt-4o-mini-2024-07-18"

    def test_tracker_retries_on_rate_limit_and_tracks_count(self):
        call_counter = [0]

        class FlakyRateLimitProvider(BaseLLMProvider):
            def generate(self, messages, tools=None):
                call_counter[0] += 1
                if call_counter[0] == 1:
                    raise RateLimitError("Rate limit HTTP 429")
                return LLMResponse(
                    content="success",
                    usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                    model="gpt-4o-mini",
                )

        flaky = FlakyRateLimitProvider()
        tracker = UsageTrackingProviderWrapper(flaky, max_retries=2, retry_delay=0.01)

        res = tracker.generate([{"role": "user", "content": "hello"}])
        assert res.content == "success"
        assert tracker.retry_count == 1
        assert tracker.rate_limited is True
        assert tracker.prompt_tokens == 20

    def test_tracker_exhausts_retries_and_raises_rate_limit(self):
        class PermanentRateLimitProvider(BaseLLMProvider):
            def generate(self, messages, tools=None):
                raise RateLimitError("Permanent 429")

        permanent = PermanentRateLimitProvider()
        tracker = UsageTrackingProviderWrapper(permanent, max_retries=2, retry_delay=0.01)

        with pytest.raises(RateLimitError):
            tracker.generate([{"role": "user", "content": "hello"}])

        assert tracker.retry_count == 2
        assert tracker.rate_limited is True


class TestLiveBenchmarkRunnerExecution:
    """Requirement: Live runner execution, telemetry collection, failure semantics."""

    def test_live_runner_captures_usage_and_model_metadata(self):
        provider = DummyLiveLLMProvider(
            model="gpt-4o-mini",
            actual_model="gpt-4o-mini-2024-07-18",
            usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )
        runner = BenchmarkRunner(
            provider=provider,
            evaluation_mode=EvaluationMode.LIVE_PROVIDER,
            requested_model="gpt-4o-mini",
        )

        res = runner.run_case("BM01_CONDITIONAL_LOGIC")
        assert res.evaluation_mode == EvaluationMode.LIVE_PROVIDER
        assert res.requested_model == "gpt-4o-mini"
        assert res.actual_model == "gpt-4o-mini-2024-07-18"
        assert res.prompt_tokens is not None and res.prompt_tokens >= 200
        assert res.completion_tokens is not None and res.completion_tokens >= 80
        assert res.total_tokens is not None and res.total_tokens >= 280
        assert res.estimated_cost_usd is not None
        assert res.estimated_cost_usd > 0.0

    def test_live_runner_rate_limit_failure_semantics(self):
        provider = DummyLiveLLMProvider(
            model="gpt-4o-mini",
            side_effect=RateLimitError("429 Too Many Requests"),
        )
        runner = BenchmarkRunner(
            provider=provider,
            evaluation_mode=EvaluationMode.LIVE_PROVIDER,
            max_retries=0,
        )

        res = runner.run_case("BM01_CONDITIONAL_LOGIC")
        assert res.investigation_status == InvestigationStatus.FAILED
        assert res.error_type == "RateLimitError"
        assert res.rate_limited is True
        # Verification gates must be unattempted
        assert res.gate_a_attempted is False
        assert res.gate_a_passed is None
        assert res.gate_b_attempted is False
        assert res.gate_b_passed is None
        assert res.gate_c_attempted is False
        assert res.gate_c_passed is None
        assert res.verified is False

    def test_live_runner_auth_failure_semantics(self):
        provider = DummyLiveLLMProvider(
            model="gpt-4o-mini",
            side_effect=AuthenticationError("401 Unauthorized API key"),
        )
        runner = BenchmarkRunner(
            provider=provider,
            evaluation_mode=EvaluationMode.LIVE_PROVIDER,
        )

        res = runner.run_case("BM01_CONDITIONAL_LOGIC")
        assert res.investigation_status == InvestigationStatus.FAILED
        assert res.error_type == "AuthenticationError"
        assert res.gate_a_attempted is False
        assert res.gate_a_passed is None
        assert res.verified is False

    def test_no_silent_ground_truth_injection_in_live_mode(self):
        """If the model produces no candidate patch, ground truth must NOT be injected."""
        # A provider that generates an answer with no patch tool calls
        provider = DummyLiveLLMProvider(
            model="gpt-4o-mini",
            responses=[
                LLMResponse(
                    content="I investigated but could not find a patch.",
                    finish_reason="stop",
                    usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                    model="gpt-4o-mini",
                )
            ],
        )
        runner = BenchmarkRunner(
            provider=provider,
            evaluation_mode=EvaluationMode.LIVE_PROVIDER,
        )

        res = runner.run_case("BM01_CONDITIONAL_LOGIC")
        # Ensure ground truth patch was NOT injected
        assert res.patch_generated is False
        assert res.patch_source == ArtifactSource.NONE
        assert res.regression_test_generated is False
        assert res.regression_test_source == ArtifactSource.NONE
        assert res.gate_a_attempted is False
        assert res.verified is False


class TestCostEstimation:
    """Requirement: Estimate monetary cost only when model and pricing are known."""

    def test_gpt_4o_mini_pricing(self):
        # 1M input = $0.150, 1M output = $0.600
        # 100,000 input = $0.015, 50,000 output = $0.030 -> total $0.045
        cost = estimate_model_cost("gpt-4o-mini", 100_000, 50_000)
        assert cost == 0.045

    def test_gpt_4o_pricing(self):
        # 1M input = $2.50, 1M output = $10.00
        # 1,000,000 input = $2.50, 1,000,000 output = $10.00 -> total $12.50
        cost = estimate_model_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == 12.50

    def test_unknown_model_reports_none(self):
        assert estimate_model_cost("my-custom-model-v1", 1000, 500) is None
        assert estimate_model_cost(None, 1000, 500) is None
        assert estimate_model_cost("gpt-4o-mini", None, 500) is None
        assert estimate_model_cost("gpt-4o-mini", 500, None) is None


class TestSecretScrubbing:
    """Requirement: Scrub secrets from logs, exception messages, and JSON artifacts."""

    def test_redact_secrets_cleans_sensitive_patterns(self):
        data = {
            "error": "Error with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 and sk-1234567890123456789012345",
            "api_key": "supersecretkey12345",
            "safe_field": "regular text",
            "prompt_tokens": 120,
            "total_tokens": 200,
        }
        cleaned = redact_secrets(data)
        assert cleaned["api_key"] == "[REDACTED]"
        assert "[REDACTED_SECRET]" in cleaned["error"]
        assert "eyJhbGciOi" not in cleaned["error"]
        assert "sk-123456" not in cleaned["error"]
        assert cleaned["safe_field"] == "regular text"
        assert cleaned["prompt_tokens"] == 120
        assert cleaned["total_tokens"] == 200
