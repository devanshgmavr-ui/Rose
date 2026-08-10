"""Tests for Stage 7.1 - Session Persistence."""

import pytest
import os
import json
import time
from agent.core.session_persistence import SessionPersistence, SessionState


class TestSessionState:
    def test_creation(self):
        s = SessionState(session_id="test_001", user_request="Do something")
        assert s.session_id == "test_001"
        assert s.status == "active"

    def test_to_dict(self):
        s = SessionState(session_id="test_001", tool_calls=5)
        d = s.to_dict()
        assert d["session_id"] == "test_001"
        assert d["tool_calls"] == 5

    def test_from_dict(self):
        d = {"session_id": "test_001", "user_request": "hello", "tool_calls": 3}
        s = SessionState.from_dict(d)
        assert s.session_id == "test_001"
        assert s.tool_calls == 3


class TestSessionPersistence:
    def test_init(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        assert os.path.exists(tmp_path / "sessions")

    def test_save_load_session(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        s = SessionState(session_id="s1", user_request="test")
        assert p.save_session(s) is True

        loaded = p.load_session("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.user_request == "test"

    def test_load_nonexistent(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        loaded = p.load_session("nonexistent")
        assert loaded is None

    def test_delete_session(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        s = SessionState(session_id="s1")
        p.save_session(s)
        assert p.delete_session("s1") is True
        assert p.load_session("s1") is None

    def test_list_sessions(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        p.save_session(SessionState(session_id="s1"))
        p.save_session(SessionState(session_id="s2"))
        sessions = p.list_sessions()
        assert len(sessions) == 2

    def test_get_session_from_memory(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        s = SessionState(session_id="s1")
        p.save_session(s)
        got = p.get_session("s1")
        assert got is not None
        assert got.session_id == "s1"

    def test_auto_save_check(self, tmp_path):
        p = SessionPersistence(
            sessions_dir=str(tmp_path / "sessions"),
            auto_save=True,
            auto_save_interval=0.1,
        )
        s = SessionState(session_id="s1")
        p.save_session(s)
        time.sleep(0.2)
        result = p.auto_save_check(s)
        assert result is True

    def test_auto_save_disabled(self, tmp_path):
        p = SessionPersistence(
            sessions_dir=str(tmp_path / "sessions"),
            auto_save=False,
        )
        s = SessionState(session_id="s1")
        p.save_session(s)
        result = p.auto_save_check(s)
        assert result is False

    def test_cleanup_old_sessions(self, tmp_path):
        p = SessionPersistence(
            sessions_dir=str(tmp_path / "sessions"),
            max_sessions=2,
        )
        for i in range(5):
            p.save_session(SessionState(session_id=f"s{i}"))
        removed = p.cleanup_old_sessions()
        assert removed >= 1

    def test_save_load_conversation(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        msgs = [{"role": "user", "content": "hello"}]
        assert p.save_conversation("s1", msgs) is True
        loaded = p.load_conversation("s1")
        assert len(loaded) == 1
        assert loaded[0]["content"] == "hello"

    def test_load_conversation_nonexistent(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        msgs = p.load_conversation("nonexistent")
        assert msgs == []

    def test_session_with_plan(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        plan = {"objective": "test", "steps": [{"description": "step 1"}]}
        s = SessionState(session_id="s1", current_plan=plan)
        p.save_session(s)
        loaded = p.load_session("s1")
        assert loaded.current_plan["objective"] == "test"

    def test_session_with_metadata(self, tmp_path):
        p = SessionPersistence(sessions_dir=str(tmp_path / "sessions"))
        s = SessionState(session_id="s1", metadata={"key": "value"})
        p.save_session(s)
        loaded = p.load_session("s1")
        assert loaded.metadata["key"] == "value"
