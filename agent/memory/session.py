"""Session management for memory system."""

import json
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import Session, Message, MessageRole


class SessionManager:
    """Manages session persistence and retrieval."""

    def __init__(self, data_dir: str = "sessions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.active_session: Optional[Session] = None

    def create_session(self, title: str = "New Session") -> Session:
        session = Session(title=title)
        self.active_session = session
        self._save_session(session)
        return session

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        if session_id:
            loaded = self.load_session(session_id)
            if loaded:
                self.active_session = loaded
                return loaded
        if self.active_session:
            return self.active_session
        return self.create_session()

    def load_session(self, session_id: str) -> Optional[Session]:
        path = self.data_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Session.from_dict(data)
        except Exception:
            return None

    def save_session(self, session: Session) -> bool:
        session.updated_at = time.time()
        return self._save_session(session)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        sessions = []
        for path in sorted(self.data_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "title": data.get("title"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                    "summary": data.get("summary"),
                })
                if len(sessions) >= limit:
                    break
            except Exception:
                continue
        return sessions

    def delete_session(self, session_id: str) -> bool:
        path = self.data_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            if self.active_session and self.active_session.session_id == session_id:
                self.active_session = None
            return True
        return False

    def get_session_count(self) -> int:
        return len(list(self.data_dir.glob("*.json")))

    def _save_session(self, session: Session) -> bool:
        try:
            path = self.data_dir / f"{session.session_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
