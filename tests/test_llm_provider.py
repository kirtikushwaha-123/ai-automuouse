"""Tests for LLM provider adapters, configuration validation, and error handling."""

import os
from unittest.mock import MagicMock
import httpx
import pytest

from rootcause.core.exceptions import ConfigurationError, LLMProviderError
from rootcause.llm import MockLLMProvider, OpenAICompatibleProvider, get_llm_provider
from rootcause.llm.base import LLMResponse, ToolCall


class TestMockLLMProvider:
    """Verify deterministic mock provider behavior."""

    def test_mock_provider_queue_and_generate(self):
        provider = MockLLMProvider()
        provider.queue_tool_call("read_repo_file", {"relative_path": "app.py"})
        provider.queue_text("Investigation finished.")

        resp1 = provider.generate([{"role": "user", "content": "Start"}])
        assert len(resp1.tool_calls) == 1
        assert resp1.tool_calls[0].name == "read_repo_file"

        resp2 = provider.generate([{"role": "user", "content": "Next"}])
        assert resp2.content == "Investigation finished."
        assert len(resp2.tool_calls) == 0

    def test_mock_provider_custom_handler(self):
        def dynamic_handler(messages, tools, step):
            return LLMResponse(content=f"Handled step {step}")

        provider = MockLLMProvider(handler=dynamic_handler)
        resp = provider.generate([{"role": "user", "content": "Query"}])
        assert resp.content == "Handled step 1"


class TestOpenAICompatibleProvider:
    """Verify OpenAI provider error handling, configuration guardrails, and parsing."""

    def test_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("ROOTCAUSE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ConfigurationError, match="No LLM API key configured"):
            OpenAICompatibleProvider(api_key=None)

    def test_factory_returns_configured_provider(self, monkeypatch):
        monkeypatch.setenv("ROOTCAUSE_LLM_PROVIDER", "mock")
        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)

    def test_factory_invalid_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider type 'unknown'"):
            get_llm_provider("unknown")

    def test_rate_limit_429_raises_provider_error(self, monkeypatch):
        provider = OpenAICompatibleProvider(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"

        def mock_post(*args, **kwargs):
            return mock_resp

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        with pytest.raises(LLMProviderError, match="rate limit reached"):
            provider.generate([{"role": "user", "content": "Hi"}])

    def test_timeout_raises_provider_error(self, monkeypatch):
        provider = OpenAICompatibleProvider(api_key="test-key", timeout=1.0)

        def mock_post_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr(httpx.Client, "post", mock_post_timeout)

        with pytest.raises(LLMProviderError, match="timed out"):
            provider.generate([{"role": "user", "content": "Hi"}])

    def test_successful_tool_call_parsing(self, monkeypatch):
        provider = OpenAICompatibleProvider(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "I will inspect the file.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "read_repo_file",
                                    "arguments": '{"relative_path": "main.py"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }

        monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: mock_resp)

        res = provider.generate([{"role": "user", "content": "Inspect"}], tools=[])
        assert res.content == "I will inspect the file."
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].name == "read_repo_file"
        assert res.tool_calls[0].arguments == {"relative_path": "main.py"}
