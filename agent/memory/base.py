"""Base memory interfaces and data classes."""

import uuid
import time
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class MemoryType(Enum):
    USER_PREFERENCE = "user_preference"
    FACT = "fact"
    DECISION = "decision"
    PROJECT_INFO = "project_info"
    INTERACTION = "interaction"
    SUMMARY = "summary"
    INSTRUCTION = "instruction"


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id", str(uuid.uuid4())[:8]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryRecord:
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    source: str = "conversation"
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    confidence: float = 0.8
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            content=data.get("content", ""),
            memory_type=MemoryType(data.get("memory_type", "fact")),
            source=data.get("source", "conversation"),
            timestamp=data.get("timestamp", time.time()),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 0.8),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
            active=data.get("active", True),
        )


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            title=data.get("title", "New Session"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            metadata=data.get("metadata", {}),
            summary=data.get("summary"),
        )


class MemoryProvider(ABC):
    @abstractmethod
    def store(self, entry: MemoryRecord) -> bool:
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 10) -> List[MemoryRecord]:
        pass

    @abstractmethod
    def clear(self) -> bool:
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        pass
