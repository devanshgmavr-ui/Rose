"""Session persistence for saving and restoring agent state.

Stage 7.1 - Session Persistence.

Provides:
- Session save/load
- Task state persistence
- Conversation history persistence
- Configuration persistence
- Auto-save
- Session recovery
"""

import json
import os
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    created_at: float = 0.0
    updated_at: float = 0.0
    status: str = "active"
    user_request: str = ""
    current_plan: Optional[Dict[str, Any]] = None
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    failed_steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionPersistence:
    """Manages session save/load operations."""

    def __init__(
        self,
        sessions_dir: str = "sessions",
        auto_save: bool = True,
        auto_save_interval: float = 30.0,
        max_sessions: int = 100,
    ):
        self._dir = sessions_dir
        self._auto_save = auto_save
        self._auto_save_interval = auto_save_interval
        self._max_sessions = max_sessions
        self._sessions: Dict[str, SessionState] = {}
        self._last_save: Dict[str, float] = {}

        os.makedirs(self._dir, exist_ok=True)

    def save_session(self, session: SessionState) -> bool:
        """Save a session to disk."""
        try:
            session.updated_at = time.time()
            path = self._get_session_path(session.session_id)
            data = session.to_dict()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._sessions[session.session_id] = session
            self._last_save[session.session_id] = time.time()
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session from disk."""
        try:
            path = self._get_session_path(session_id)
            if not os.path.exists(path):
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            session = SessionState.from_dict(data)
            self._sessions[session_id] = session
            return session

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from disk."""
        try:
            path = self._get_session_path(session_id)
            if os.path.exists(path):
                os.remove(path)
            self._sessions.pop(session_id, None)
            self._last_save.pop(session_id, None)
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions."""
        sessions = []
        try:
            for filename in os.listdir(self._dir):
                if filename.endswith(".json"):
                    path = os.path.join(self._dir, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id"),
                        "status": data.get("status"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    })
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")

        return sorted(sessions, key=lambda x: x.get("updated_at", 0), reverse=True)

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session from memory or disk."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.load_session(session_id)

    def auto_save_check(self, session: SessionState) -> bool:
        """Check if auto-save is needed."""
        if not self._auto_save:
            return False

        last = self._last_save.get(session.session_id, 0)
        if time.time() - last >= self._auto_save_interval:
            return self.save_session(session)
        return False

    def cleanup_old_sessions(self) -> int:
        """Remove old sessions beyond max limit."""
        sessions = self.list_sessions()
        if len(sessions) <= self._max_sessions:
            return 0

        removed = 0
        for session in sessions[self._max_sessions:]:
            if self.delete_session(session["session_id"]):
                removed += 1

        return removed

    def save_conversation(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> bool:
        """Save conversation history."""
        try:
            path = os.path.join(self._dir, f"{session_id}_conv.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False

    def load_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation history."""
        try:
            path = os.path.join(self._dir, f"{session_id}_conv.json")
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return []

    def _get_session_path(self, session_id: str) -> str:
        """Get file path for a session."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return os.path.join(self._dir, f"{safe_id}.json")
