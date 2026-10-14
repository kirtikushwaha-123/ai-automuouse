"""Deterministic mock LLM provider for zero-cost, offline testing."""

from typing import Any, Callable, Dict, List, Optional

from rootcause.core.exceptions import LLMProviderError
from rootcause.llm.base import BaseLLMProvider, LLMResponse, ToolCall


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, replaceable mock provider for testing agent orchestration."""

    def __init__(
        self,
        responses: Optional[List[LLMResponse]] = None,
        handler: Optional[Callable[[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], int], LLMResponse]] = None,
    ):
        self.responses = list(responses) if responses else []
        self.handler = handler
        self.call_count = 0
        self.history: List[List[Dict[str, Any]]] = []

    def queue_response(self, response: LLMResponse) -> None:
        """Add a response to the queue."""
        self.responses.append(response)

    def queue_tool_call(self, name: str, arguments: Dict[str, Any]) -> None:
        """Helper to queue a tool call."""
        self.responses.append(LLMResponse(
            content=None,
            tool_calls=[ToolCall(name=name, arguments=arguments)],
            finish_reason="tool_calls",
        ))

    def queue_text(self, text: str) -> None:
        """Helper to queue a textual response."""
        self.responses.append(LLMResponse(
            content=text,
            tool_calls=[],
            finish_reason="stop",
        ))

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        self.history.append(messages)
        self.call_count += 1

        if self.handler:
            return self.handler(messages, tools, self.call_count)

        if self.responses:
            return self.responses.pop(0)

        # Default fallback response if queue is exhausted
        return LLMResponse(
            content="[MockProvider] No further actions to take.",
            tool_calls=[],
            finish_reason="stop",
        )
