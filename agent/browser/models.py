"""Browser data models.

Stage 2.4.1 - Browser foundation data structures.

Provides structured representations of browser sessions and pages.
No sensitive browser data (cookies, passwords, tokens) is stored.
"""

import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


class SessionStatus(Enum):
    """Browser session status."""
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class BrowserPage:
    """Represents a browser page/tab within a session."""
    page_id: str
    created_at: float
    title: str = ""
    url: str = ""
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserPage":
        return cls(
            page_id=data.get("page_id", ""),
            created_at=data.get("created_at", 0.0),
            title=data.get("title", ""),
            url=data.get("url", ""),
            is_active=data.get("is_active", False),
        )


@dataclass
class BrowserSession:
    """Represents an isolated browser session.

    Stores only safe metadata. No passwords, cookies,
    or authentication tokens are stored.
    """
    session_id: str
    created_at: float
    headless: bool
    status: SessionStatus = SessionStatus.ACTIVE
    page_count: int = 1
    current_url: str = ""
    current_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserSession":
        return cls(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", 0.0),
            headless=data.get("headless", True),
            status=SessionStatus(data.get("status", "active")),
            page_count=data.get("page_count", 1),
            current_url=data.get("current_url", ""),
            current_title=data.get("current_title", ""),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a unique session ID."""
        return f"browser_{uuid.uuid4().hex[:12]}"
