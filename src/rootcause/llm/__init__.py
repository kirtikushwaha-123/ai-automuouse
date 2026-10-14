"""LLM provider adapters and abstractions."""

import os
from typing import Optional

from rootcause.llm.base import BaseLLMProvider, LLMResponse, ToolCall
from rootcause.llm.mock_provider import MockLLMProvider
from rootcause.llm.openai_provider import OpenAICompatibleProvider


def get_llm_provider(provider_type: Optional[str] = None, **kwargs) -> BaseLLMProvider:
    """Factory creating an LLM provider based on environment configuration.
    
    Args:
        provider_type: Explicit provider choice ('mock' or 'openai'). Defaults to ROOTCAUSE_LLM_PROVIDER.
        **kwargs: Provider-specific arguments passed to constructors.
        
    Returns:
        Configured BaseLLMProvider.
    """
    selected = provider_type or os.environ.get("ROOTCAUSE_LLM_PROVIDER", "openai").lower()

    if selected == "mock":
        return MockLLMProvider(**kwargs)
    elif selected in {"openai", "openai-compatible", "groq", "ollama", "deepseek"}:
        return OpenAICompatibleProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider type '{selected}'. Supported: 'mock', 'openai'.")


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "ToolCall",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "get_llm_provider",
]
