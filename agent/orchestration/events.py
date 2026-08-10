"""Task events for orchestration."""

import time
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class EventType(Enum):
    TASK_CREATED = "task_created"
    PLAN_CREATED = "plan_created"
    PLAN_VALIDATED = "plan_validated"
    STEP_STARTED = "step_started"
    TOOL_REQUESTED = "tool_requested"
    TOOL_COMPLETED = "tool_completed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    RETRY_STARTED = "retry_started"
    PLAN_REVISED = "plan_revISED"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_TIMEOUT = "task_timeout"
    DECISION_MADE = "decision_made"


@dataclass
class TaskEvent:
    event_type: EventType
    task_id: str
    session_id: Optional[str] = None
    step_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class EventLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / "task_events.jsonl"
        self._events: list = []

    def log(self, event: TaskEvent) -> bool:
        self._events.append(event)
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False

    def get_events(self, task_id: Optional[str] = None, limit: int = 100) -> list:
        events = self._events
        if task_id:
            events = [e for e in events if e.task_id == task_id]
        return [e.to_dict() for e in events[-limit:]]

    def clear(self):
        self._events.clear()
        if self._log_file.exists():
            self._log_file.unlink()
