"""LLM abstraction layer for model providers."""

from .base import LLMProvider, LLMResponse
from .local_provider import LocalLLMProvider

__all__ = ["LLMProvider", "LLMResponse", "LocalLLMProvider"]
