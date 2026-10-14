"""OpenAI-compatible LLM provider implementation."""

import json
import os
from typing import Any, Dict, List, Optional

import httpx

from rootcause.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMProviderError,
    MissingCredentialsError,
    RateLimitError,
)
from rootcause.llm.base import BaseLLMProvider, LLMResponse, ToolCall


class OpenAICompatibleProvider(BaseLLMProvider):
    """Replaceable LLM adapter supporting OpenAI, Groq, Ollama, DeepSeek, and LiteLLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("ROOTCAUSE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingCredentialsError(
                "No LLM API key configured. Please set the 'ROOTCAUSE_LLM_API_KEY' "
                "or 'OPENAI_API_KEY' environment variable."
            )
        self.base_url = (
            base_url
            or os.environ.get("ROOTCAUSE_LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("ROOTCAUSE_LLM_MODEL")
            or "gpt-4o-mini"
        )
        self.timeout = timeout

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # Low temperature for factual investigation
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"LLM request timed out after {self.timeout}s.") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Network error connecting to LLM provider at {url}: {exc}") from exc

        if response.status_code == 401:
            raise AuthenticationError("Unauthorized: Provided LLM API key was rejected by the provider.")
        if response.status_code == 429:
            raise RateLimitError("LLM rate limit reached (HTTP 429). Please retry later.")
        if response.status_code >= 500:
            raise LLMProviderError(f"LLM provider internal error (HTTP {response.status_code}): {response.text}")
        if response.status_code != 200:
            raise LLMProviderError(f"LLM provider error (HTTP {response.status_code}): {response.text}")

        try:
            data = response.json()
        except Exception as exc:
            raise LLMProviderError(f"Malformed JSON response from LLM provider: {response.text}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise LLMProviderError("LLM response contained no completion choices.")

        msg = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason", "stop")
        content = msg.get("content")

        tool_calls: List[ToolCall] = []
        raw_tool_calls = msg.get("tool_calls", [])
        for tc in raw_tool_calls:
            call_id = tc.get("id", "")
            func = tc.get("function", {})
            name = func.get("name", "")
            raw_args_str = func.get("arguments", "{}")
            try:
                args = json.loads(raw_args_str) if isinstance(raw_args_str, str) else raw_args_str
            except json.JSONDecodeError:
                args = {}

            tool_calls.append(ToolCall(
                id=call_id,
                name=name,
                arguments=args,
            ))

        usage_data = data.get("usage")
        usage: Optional[Dict[str, int]] = None
        if isinstance(usage_data, dict):
            usage = {
                "prompt_tokens": int(usage_data.get("prompt_tokens", 0)),
                "completion_tokens": int(usage_data.get("completion_tokens", 0)),
                "total_tokens": int(usage_data.get("total_tokens", 0)),
            }

        actual_model = data.get("model") or self.model

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=actual_model,
        )
