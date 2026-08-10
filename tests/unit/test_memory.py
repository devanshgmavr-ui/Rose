"""Unit tests for memory system (Stage 1.2)."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

from agent.memory.base import (
    MemoryType,
    MessageRole,
    Message,
    MemoryRecord,
    Session,
)
from agent.memory.session import SessionManager
from agent.memory.conversation import ConversationManager
from agent.memory.long_term import LongTermMemory
from agent.memory.context import ContextManager
from agent.memory.summarizer import ConversationSummarizer


class TestBaseDataClasses:
    """Test base data classes."""

    def test_message_creation(self):
        msg = Message(role=MessageRole.USER, content="Hello world")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello world"
        assert msg.message_id is not None
        assert msg.timestamp > 0

    def test_message_to_dict(self):
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
        assert "message_id" in d
        assert "timestamp" in d

    def test_message_from_dict(self):
        original = Message(role=MessageRole.USER, content="Test")
        d = original.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content

    def test_memory_record_creation(self):
        record = MemoryRecord(
            content="User prefers dark mode",
            memory_type=MemoryType.USER_PREFERENCE,
            importance=0.8,
            confidence=0.9,
        )
        assert record.content == "User prefers dark mode"
        assert record.memory_type == MemoryType.USER_PREFERENCE
        assert record.importance == 0.8
        assert record.active is True

    def test_memory_record_to_dict(self):
        record = MemoryRecord(content="Test fact", memory_type=MemoryType.FACT)
        d = record.to_dict()
        assert d["content"] == "Test fact"
        assert d["memory_type"] == "fact"
        assert "memory_id" in d

    def test_memory_record_from_dict(self):
        original = MemoryRecord(content="Important decision", memory_type=MemoryType.DECISION)
        d = original.to_dict()
        restored = MemoryRecord.from_dict(d)
        assert restored.content == original.content
        assert restored.memory_type == original.memory_type

    def test_session_creation(self):
        session = Session(title="Test Session")
        assert session.title == "Test Session"
        assert session.messages == []
        assert session.session_id is not None

    def test_session_to_dict(self):
        session = Session(title="My Session")
        session.messages.append(Message(role=MessageRole.USER, content="Hi"))
        d = session.to_dict()
        assert d["title"] == "My Session"
        assert len(d["messages"]) == 1

    def test_session_from_dict(self):
        original = Session(title="Restore Test")
        original.messages.append(Message(role=MessageRole.ASSISTANT, content="Hello"))
        d = original.to_dict()
        restored = Session.from_dict(d)
        assert restored.title == original.title
        assert len(restored.messages) == 1


class TestSessionManager:
    """Test session persistence."""

    def test_create_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(data_dir=tmpdir)
            session = mgr.create_session("Test")
            assert session.title == "Test"
            assert mgr.active_session is not None

    def test_save_and_load_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(data_dir=tmpdir)
            session = mgr.create_session("Save Test")
            session.messages.append(Message(role=MessageRole.USER, content="Hello"))
            mgr.save_session(session)

            loaded = mgr.load_session(session.session_id)
            assert loaded is not None
            assert loaded.title == "Save Test"
            assert len(loaded.messages) == 1

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(data_dir=tmpdir)
            mgr.create_session("Session 1")
            mgr.create_session("Session 2")
            sessions = mgr.list_sessions()
            assert len(sessions) == 2

    def test_delete_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(data_dir=tmpdir)
            session = mgr.create_session("To Delete")
            deleted = mgr.delete_session(session.session_id)
            assert deleted is True
            loaded = mgr.load_session(session.session_id)
            assert loaded is None

    def test_session_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(data_dir=tmpdir)
            assert mgr.get_session_count() == 0
            mgr.create_session("S1")
            mgr.create_session("S2")
            assert mgr.get_session_count() == 2


class TestConversationManager:
    """Test conversation management."""

    def test_start_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(data_dir=tmpdir)
            cm = ConversationManager(sm)
            session = cm.start_conversation("Test Conv")
            assert session.title == "Test Conv"
            assert cm.current_session is not None

    def test_add_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(data_dir=tmpdir)
            cm = ConversationManager(sm)
            cm.start_conversation()

            msg1 = cm.add_user_message("Hello")
            msg2 = cm.add_assistant_message("Hi there")
            msg3 = cm.add_system_message("System note")

            assert msg1.role == MessageRole.USER
            assert msg2.role == MessageRole.ASSISTANT
            assert msg3.role == MessageRole.SYSTEM
            assert cm.get_message_count() == 3

    def test_get_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(data_dir=tmpdir)
            cm = ConversationManager(sm)
            cm.start_conversation()

            cm.add_user_message("Msg 1")
            cm.add_user_message("Msg 2")
            cm.add_user_message("Msg 3")

            msgs = cm.get_messages(limit=2)
            assert len(msgs) == 2
            assert msgs[0].content == "Msg 2"

    def test_get_last_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(data_dir=tmpdir)
            cm = ConversationManager(sm)
            cm.start_conversation()

            cm.add_user_message("User 1")
            cm.add_assistant_message("Bot 1")
            cm.add_user_message("User 2")

            last_user = cm.get_last_user_message()
            last_bot = cm.get_last_assistant_message()

            assert last_user.content == "User 2"
            assert last_bot.content == "Bot 1"

    def test_conversation_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(data_dir=tmpdir)
            cm = ConversationManager(sm)
            cm.start_conversation()

            cm.add_user_message("Hello")
            cm.add_assistant_message("Hi")

            stats = cm.get_conversation_stats()
            assert stats["message_count"] == 2
            assert stats["user_messages"] == 1
            assert stats["assistant_messages"] == 1

    def test_clear_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(data_dir=tmpdir)
            cm = ConversationManager(sm)
            cm.start_conversation()

            cm.add_user_message("Hello")
            cm.clear_conversation()
            assert cm.get_message_count() == 0


class TestLongTermMemory:
    """Test SQLite long-term memory."""

    def test_store_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            ltm = LongTermMemory(db_path=db_path)

            record = MemoryRecord(
                content="Python is a programming language",
                memory_type=MemoryType.FACT,
                importance=0.8,
            )
            stored = ltm.store(record)
            assert stored is True

            results = ltm.retrieve(query="Python")
            assert len(results) >= 1
            assert "Python" in results[0].content

    def test_retrieve_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            ltm = LongTermMemory(db_path=db_path)

            ltm.store(MemoryRecord(content="Fact 1", memory_type=MemoryType.FACT))
            ltm.store(MemoryRecord(content="Pref 1", memory_type=MemoryType.USER_PREFERENCE))
            ltm.store(MemoryRecord(content="Fact 2", memory_type=MemoryType.FACT))

            facts = ltm.retrieve(memory_type=MemoryType.FACT)
            assert len(facts) == 2

            prefs = ltm.retrieve(memory_type=MemoryType.USER_PREFERENCE)
            assert len(prefs) == 1

    def test_delete_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            ltm = LongTermMemory(db_path=db_path)

            record = MemoryRecord(content="To delete", memory_type=MemoryType.FACT)
            ltm.store(record)

            ltm.delete(record.memory_id)
            results = ltm.get_all()
            assert len(results) == 0

    def test_memory_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            ltm = LongTermMemory(db_path=db_path)

            assert ltm.get_count() == 0
            ltm.store(MemoryRecord(content="M1", memory_type=MemoryType.FACT))
            ltm.store(MemoryRecord(content="M2", memory_type=MemoryType.FACT))
            assert ltm.get_count() == 2

    def test_health_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            ltm = LongTermMemory(db_path=db_path)

            health = ltm.health_check()
            assert health["status"] == "healthy"
            assert health["total_memories"] == 0

    def test_type_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            ltm = LongTermMemory(db_path=db_path)

            ltm.store(MemoryRecord(content="F1", memory_type=MemoryType.FACT))
            ltm.store(MemoryRecord(content="F2", memory_type=MemoryType.FACT))
            ltm.store(MemoryRecord(content="P1", memory_type=MemoryType.USER_PREFERENCE))

            counts = ltm.get_type_counts()
            assert counts.get("fact", 0) == 2
            assert counts.get("user_preference", 0) == 1

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            ltm = LongTermMemory(db_path=db_path)

            ltm.store(MemoryRecord(content="M1", memory_type=MemoryType.FACT))
            ltm.store(MemoryRecord(content="M2", memory_type=MemoryType.FACT))
            ltm.clear()
            assert ltm.get_count() == 0


class TestContextManager:
    """Test context window management."""

    def _make_setup(self, tmpdir):
        sm = SessionManager(data_dir=tmpdir)
        cm = ConversationManager(sm)
        cm.start_conversation()
        ltm = LongTermMemory(db_path=str(Path(tmpdir) / "mem.db"))
        ctx = ContextManager(
            conversation_manager=cm,
            long_term_memory=ltm,
            system_prompt="You are helpful.",
            max_context_tokens=2000,
            reserved_output_tokens=200,
            recent_message_count=4,
            memory_retrieval_limit=3,
        )
        return cm, ltm, ctx

    def test_build_context_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, ctx = self._make_setup(tmpdir)
            cm.add_user_message("Hello")
            cm.add_assistant_message("Hi there!")

            context = ctx.build_context(user_query="How are you?")
            assert len(context) >= 2
            assert context[0]["role"] == "system"
            assert any(m["role"] == "user" for m in context)

    def test_build_context_with_memories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, ctx = self._make_setup(tmpdir)

            ltm.store(MemoryRecord(
                content="User prefers Python for scripting and automation",
                memory_type=MemoryType.USER_PREFERENCE,
                importance=0.9,
            ))

            ltm.store(MemoryRecord(
                content="What language should I use for this project?",
                memory_type=MemoryType.INTERACTION,
                importance=0.5,
            ))

            context = ctx.build_context(user_query="What language should I use?")
            memory_found = any("python" in m.get("content", "").lower() for m in context)
            assert memory_found

    def test_context_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, ctx = self._make_setup(tmpdir)
            cm.add_user_message("Test message")

            stats = ctx.get_context_stats()
            assert "system_prompt_tokens" in stats
            assert "estimated_usage" in stats
            assert stats["available_tokens"] == 1800

    def test_needs_summarization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, ctx = self._make_setup(tmpdir)

            for i in range(30):
                cm.add_user_message(f"Message number {i} with extra words to increase token count significantly " * 15)

            assert ctx.needs_summarization() is True

    def test_update_limits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, ctx = self._make_setup(tmpdir)
            ctx.update_limits(max_tokens=4000, reserved_output=600)
            stats = ctx.get_context_stats()
            assert stats["available_tokens"] == 3400


class TestConversationSummarizer:
    """Test conversation summarization."""

    def _make_setup(self, tmpdir):
        sm = SessionManager(data_dir=tmpdir)
        cm = ConversationManager(sm)
        cm.start_conversation()
        ltm = LongTermMemory(db_path=str(Path(tmpdir) / "mem.db"))
        summarizer = ConversationSummarizer(
            conversation_manager=cm,
            long_term_memory=ltm,
            max_context_tokens=2000,
            summary_threshold=500,
            preserve_recent=4,
        )
        return cm, ltm, summarizer

    def test_should_summarize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, summarizer = self._make_setup(tmpdir)

            assert summarizer.should_summarize() is False

            for i in range(10):
                cm.add_user_message(f"Message {i} " * 20)

            assert summarizer.should_summarize() is True

    def test_summarize_and_compress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, summarizer = self._make_setup(tmpdir)

            cm.add_user_message("I prefer Python for coding")
            cm.add_assistant_message("Python is great!")
            cm.add_user_message("Let's use FastAPI for the web framework")
            cm.add_assistant_message("Good choice!")
            cm.add_user_message("Remember to use type hints")
            cm.add_assistant_message("Will do!")

            summary = summarizer.summarize_and_compress()
            assert summary is not None

            stats = cm.get_conversation_stats()
            assert stats["message_count"] <= 6

    def test_extract_memories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, summarizer = self._make_setup(tmpdir)

            for i in range(10):
                cm.add_user_message(f"Message {i}: I prefer using Python for all my projects")

            summarizer.summarize_and_compress()

            memories = ltm.get_all()
            assert len(memories) >= 1

    def test_summarization_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, summarizer = self._make_setup(tmpdir)
            cm.add_user_message("Test")

            stats = summarizer.get_summarization_stats()
            assert "message_count" in stats
            assert "should_summarize" in stats
            assert "summary_threshold" in stats

    def test_update_limits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm, ltm, summarizer = self._make_setup(tmpdir)
            summarizer.update_limits(max_tokens=4000, threshold=1000, preserve=8)
            assert summarizer.max_context_tokens == 4000
            assert summarizer.summary_threshold == 1000
            assert summarizer.preserve_recent == 8
