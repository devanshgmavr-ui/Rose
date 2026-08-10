"""Integration tests for memory system with actual agent."""

import pytest
import tempfile
from pathlib import Path

from agent.core.config import Config
from agent.core.agent import Agent
from agent.llm.base import LLMResponse

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


class TestMemorySystem:
    """Test memory system integration with agent."""

    def test_memory_initialization(self, agent):
        """Test memory system is initialized with agent."""
        health = agent.health_check()
        assert "memory" in health
        assert health["memory"]["initialized"] is True

    def test_store_and_search_memory(self, agent):
        """Test storing and searching memories."""
        stored = agent.store_memory(
            content="User prefers Python for scripting",
            memory_type="user_preference",
            importance=0.9,
        )
        assert stored is True

        results = agent.search_memories("Python")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]

    def test_session_management(self, agent):
        """Test session creation and listing."""
        session_id = agent.start_new_session("Integration Test Session")
        assert session_id is not None

        sessions = agent.list_sessions()
        assert len(sessions) >= 1
        assert any(s["session_id"] == session_id for s in sessions)

    def test_memory_stats(self, agent):
        """Test memory statistics."""
        stats = agent.get_memory_stats()
        assert stats is not None
        assert "session_manager" in stats
        assert "conversation" in stats
        assert "long_term_memory" in stats
        assert "context" in stats
        assert "summarizer" in stats

    def test_chat_with_memory(self, agent):
        """Test chat uses memory system."""
        agent.clear_history()
        agent.start_new_session("Chat Memory Test")

        agent.store_memory(
            content="User's favorite color is blue",
            memory_type="user_preference",
            importance=0.9,
        )

        response = agent.chat("What is my favorite color?")
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0

    def test_conversation_in_memory(self, agent):
        """Test conversation is tracked in memory system."""
        agent.clear_history()
        agent.start_new_session("Conversation Tracking Test")

        agent.chat("Hello, remember that I like TypeScript")
        stats = agent.get_memory_stats()

        assert stats["conversation"]["message_count"] >= 2
        assert stats["conversation"]["user_messages"] >= 1

    def test_health_check_includes_memory(self, agent):
        """Test health check includes memory status."""
        health = agent.health_check()

        assert "memory" in health
        assert "long_term" in health["memory"]
        assert health["memory"]["long_term"]["status"] == "healthy"

    def test_multiple_sessions(self, agent):
        """Test multiple session creation."""
        id1 = agent.start_new_session("Session A")
        id2 = agent.start_new_session("Session B")

        sessions = agent.list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        assert id1 in session_ids
        assert id2 in session_ids
