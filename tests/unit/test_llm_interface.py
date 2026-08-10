"""Unit tests for LLM interface (without actual model)."""

import pytest
from typing import Dict, Any, List

from agent.llm.base import LLMProvider, LLMConfig, LLMResponse, LLMProviderType


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._mock_response = "Mock response"
    
    def initialize(self) -> bool:
        self._is_initialized = True
        return True
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(
            text=self._mock_response,
            model="mock-model",
            tokens_used=10,
            tokens_prompt=5,
            tokens_completion=5,
        )
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        return LLMResponse(
            text=self._mock_response,
            model="mock-model",
            tokens_used=10,
            tokens_prompt=5,
            tokens_completion=5,
        )
    
    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "initialized": self._is_initialized}
    
    def model_info(self) -> Dict[str, Any]:
        return {"model_name": "mock-model", "status": "mock"}
    
    def unload(self) -> bool:
        self._is_initialized = False
        return True


class TestLLMInterface:
    """Test LLM provider interface."""
    
    def test_provider_initialization(self):
        """Test provider can be initialized."""
        config = LLMConfig()
        provider = MockLLMProvider(config)
        
        assert not provider.is_initialized
        assert provider.initialize()
        assert provider.is_initialized
    
    def test_provider_generate(self):
        """Test provider can generate responses."""
        config = LLMConfig()
        provider = MockLLMProvider(config)
        provider.initialize()
        
        response = provider.generate("Test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.text == "Mock response"
        assert response.model == "mock-model"
        assert response.tokens_used == 10
    
    def test_provider_chat(self):
        """Test provider can handle chat messages."""
        config = LLMConfig()
        provider = MockLLMProvider(config)
        provider.initialize()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        response = provider.chat(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.text == "Mock response"
    
    def test_provider_health_check(self):
        """Test health check returns valid status."""
        config = LLMConfig()
        provider = MockLLMProvider(config)
        
        health = provider.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
    
    def test_provider_model_info(self):
        """Test model info returns valid data."""
        config = LLMConfig()
        provider = MockLLMProvider(config)
        provider.initialize()
        
        info = provider.model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
    
    def test_provider_unload(self):
        """Test provider can be unloaded."""
        config = LLMConfig()
        provider = MockLLMProvider(config)
        provider.initialize()
        
        assert provider.is_initialized
        assert provider.unload()
        assert not provider.is_initialized


class TestLLMResponse:
    """Test LLM response data class."""
    
    def test_response_creation(self):
        """Test response can be created."""
        response = LLMResponse(
            text="Hello",
            model="test",
            tokens_used=100,
            tokens_prompt=50,
            tokens_completion=50,
        )
        
        assert response.text == "Hello"
        assert response.model == "test"
        assert response.total_tokens == 100
    
    def test_response_total_tokens(self):
        """Test total tokens calculation."""
        response = LLMResponse(
            text="Hello",
            model="test",
            tokens_prompt=30,
            tokens_completion=70,
        )
        
        assert response.total_tokens == 100
    
    def test_response_defaults(self):
        """Test response default values."""
        response = LLMResponse(text="Hello", model="test")
        
        assert response.tokens_used == 0
        assert response.finish_reason == "stop"
        assert response.metadata == {}
