"""Base LLM provider interface and response abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Structured tool invocation emitted by an LLM."""
    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:8]}")
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Normalized response from an LLM provider."""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"  # "stop", "tool_calls", "length", "error"
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for replaceable LLM providers with tool-calling support."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate a completion given a conversation history and optional tool schemas.
        
        Args:
            messages: OpenAI-format chat messages.
            tools: Registered tool schemas in OpenAI function calling format.
            
        Returns:
            LLMResponse containing text content and/or tool calls.
            
        Raises:
            LLMProviderError: On network, rate-limit, timeout, or API failure.
        """
        pass
