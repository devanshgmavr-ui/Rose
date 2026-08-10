"""Base tool abstractions and data classes."""

import time
import uuid
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class Permission(Enum):
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    CODE_EXECUTE = "code.execute"
    COMMAND_EXECUTE = "command.execute"
    APP_LAUNCH = "app.launch"


class ConfirmationLevel(Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass
class ToolRequest:
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolRequest":
        return cls(
            tool=data.get("tool", ""),
            arguments=data.get("arguments", {}),
            request_id=data.get("request_id", str(uuid.uuid4())[:8]),
            session_id=data.get("session_id"),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ToolResult:
    success: bool
    tool_name: str
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "truncated": self.truncated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        return cls(
            success=data.get("success", False),
            tool_name=data.get("tool_name", ""),
            output=data.get("output", ""),
            error=data.get("error", ""),
            execution_time=data.get("execution_time", 0.0),
            truncated=data.get("truncated", False),
            metadata=data.get("metadata", {}),
        )


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def required_permissions(self) -> list:
        pass

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.ALLOW

    @property
    def timeout(self) -> float:
        return 30.0

    @abstractmethod
    def validate(self, arguments: Dict[str, Any]) -> tuple:
        pass

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "required_permissions": [p.value if isinstance(p, Permission) else p for p in self.required_permissions],
            "confirmation_level": self.confirmation_level.value,
            "timeout": self.timeout,
        }
