"""Integration tests for actual model inference."""

import pytest
import os
from pathlib import Path

from agent.core.config import Config
from agent.core.agent import Agent
from agent.llm.base import LLMConfig, LLMResponse
from agent.llm.local_provider import LocalLLMProvider


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config():
    """Load test configuration."""
    return Config()


@pytest.fixture(scope="module")
def agent(config):
    """Create and initialize agent for testing."""
    agent = Agent(config)
    if not agent.initialize():
        pytest.skip("Could not initialize agent (model not available)")
    yield agent
    agent.shutdown()


class TestLocalProvider:
    """Test local LLM provider with actual model."""
    
    def test_provider_initialization(self, config):
        """Test local provider can initialize."""
        llm_config = LLMConfig(
            model_path=str(config.get_model_full_path()),
            model_name=config.model_name,
            context_length=config.model_context_length,
            n_gpu_layers=config.llm_gpu_layers,
        )
        provider = LocalLLMProvider(llm_config)
        
        result = provider.initialize()
        assert result is True
        assert provider.is_initialized
        
        provider.unload()
    
    def test_model_health_check(self, config):
        """Test model health check."""
        llm_config = LLMConfig(
            model_path=str(config.get_model_full_path()),
            model_name=config.model_name,
            context_length=config.model_context_length,
            n_gpu_layers=config.llm_gpu_layers,
        )
        provider = LocalLLMProvider(llm_config)
        provider.initialize()
        
        health = provider.health_check()
        
        assert health["initialized"] is True
        assert health["model_exists"] is True
        assert health["llama_cpp_installed"] is True
        
        provider.unload()


class TestAgent:
    """Test agent with actual model."""
    
    def test_agent_generate(self, agent):
        """Test agent can generate text."""
        response = agent.generate("What is 2 + 2?")
        
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.tokens_used > 0
    
    def test_agent_chat(self, agent):
        """Test agent can chat."""
        response = agent.chat("Hello, what can you do?")
        
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0
        assert response.tokens_used > 0
    
    def test_agent_conversation_history(self, agent):
        """Test conversation history is maintained."""
        agent.clear_history()
        
        agent.chat("My name is Test")
        response = agent.chat("What is my name?")
        
        assert "Test" in response.text or "test" in response.text.lower()
    
    def test_agent_health_check(self, agent):
        """Test agent health check."""
        health = agent.health_check()
        
        assert health["agent"]["initialized"] is True
        assert health["agent"]["project"] == "Rose"
        assert "llm" in health
        assert health["llm"]["initialized"] is True
