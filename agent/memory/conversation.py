"""Conversation management for memory system."""

import time
from typing import List, Optional, Dict, Any

from .base import Message, MessageRole, Session
from .session import SessionManager


class ConversationManager:
    """Manages conversation flow and message handling."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.current_session: Optional[Session] = None

    def start_conversation(self, title: Optional[str] = None) -> Session:
        if title is None:
            title = f"Conversation {time.strftime('%Y-%m-%d %H:%M')}"
        self.current_session = self.session_manager.create_session(title)
        return self.current_session

    def load_conversation(self, session_id: str) -> Optional[Session]:
        session = self.session_manager.load_session(session_id)
        if session:
            self.current_session = session
        return session

    def add_message(self, role: MessageRole, content: str, **metadata) -> Message:
        if not self.current_session:
            self.start_conversation()

        msg = Message(role=role, content=content, metadata=metadata)
        self.current_session.messages.append(msg)
        self.current_session.updated_at = time.time()
        self.session_manager.save_session(self.current_session)
        return msg

    def add_user_message(self, content: str) -> Message:
        return self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str) -> Message:
        return self.add_message(MessageRole.ASSISTANT, content)

    def add_system_message(self, content: str) -> Message:
        return self.add_message(MessageRole.SYSTEM, content)

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        if not self.current_session:
            return []
        msgs = self.current_session.messages
        if limit:
            return msgs[-limit:]
        return msgs

    def get_last_user_message(self) -> Optional[Message]:
        for msg in reversed(self.current_session.messages if self.current_session else []):
            if msg.role == MessageRole.USER:
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[Message]:
        for msg in reversed(self.current_session.messages if self.current_session else []):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None

    def get_message_count(self) -> int:
        if not self.current_session:
            return 0
        return len(self.current_session.messages)

    def get_token_estimate(self) -> int:
        if not self.current_session:
            return 0
        return sum(len(m.content.split()) * 1.3 for m in self.current_session.messages)

    def clear_conversation(self) -> bool:
        if self.current_session:
            self.current_session.messages.clear()
            self.current_session.updated_at = time.time()
            return self.session_manager.save_session(self.current_session)
        return False

    def export_conversation(self) -> List[Dict[str, Any]]:
        if not self.current_session:
            return []
        return [m.to_dict() for m in self.current_session.messages]

    def get_conversation_stats(self) -> Dict[str, Any]:
        if not self.current_session:
            return {"message_count": 0, "user_messages": 0, "assistant_messages": 0}

        msgs = self.current_session.messages
        return {
            "message_count": len(msgs),
            "user_messages": sum(1 for m in msgs if m.role == MessageRole.USER),
            "assistant_messages": sum(1 for m in msgs if m.role == MessageRole.ASSISTANT),
            "system_messages": sum(1 for m in msgs if m.role == MessageRole.SYSTEM),
            "token_estimate": self.get_token_estimate(),
            "session_id": self.current_session.session_id,
            "title": self.current_session.title,
        }
