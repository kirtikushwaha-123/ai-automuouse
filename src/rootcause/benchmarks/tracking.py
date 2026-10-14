"""Usage and request tracking wrapper for LLM evaluation."""

import time
from typing import Any, Dict, List, Optional

from rootcause.core.exceptions import RateLimitError
from rootcause.llm.base import BaseLLMProvider, LLMResponse


class UsageTrackingProviderWrapper(BaseLLMProvider):
    """Wraps an LLM provider to capture usage, model names, retries, and rate limits."""

    def __init__(
        self,
        inner_provider: BaseLLMProvider,
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ):
        self.inner = inner_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.requested_model: Optional[str] = getattr(inner_provider, "model", None)
        self.actual_model: Optional[str] = None
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self.total_tokens: Optional[int] = None
        self.retry_count: int = 0
        self.rate_limited: bool = False
        self.call_count: int = 0

    def reset_case_accounting(self) -> None:
        """Reset per-case metrics before executing a new benchmark case."""
        self.actual_model = None
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.retry_count = 0
        self.rate_limited = False
        self.call_count = 0

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        attempts = 0
        while True:
            try:
                response = self.inner.generate(messages, tools=tools)
                self.call_count += 1

                # Capture actual model
                if response.model:
                    self.actual_model = response.model
                elif not self.actual_model and self.requested_model:
                    self.actual_model = self.requested_model

                # Accurate usage accounting: distinguish missing from zero
                if response.usage is not None:
                    p = response.usage.get("prompt_tokens")
                    c = response.usage.get("completion_tokens")
                    t = response.usage.get("total_tokens")

                    if p is not None:
                        self.prompt_tokens = (self.prompt_tokens or 0) + p
                    if c is not None:
                        self.completion_tokens = (self.completion_tokens or 0) + c
                    if t is not None:
                        self.total_tokens = (self.total_tokens or 0) + t
                    elif p is not None and c is not None:
                        self.total_tokens = (self.total_tokens or 0) + (p + c)

                return response

            except RateLimitError:
                self.rate_limited = True
                if attempts < self.max_retries:
                    attempts += 1
                    self.retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                raise
