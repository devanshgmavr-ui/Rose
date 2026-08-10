"""Task and Plan data models for orchestration."""

import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Decision(Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAIL = "fail"
    CANCEL = "cancel"


@dataclass
class PlanStep:
    step_id: str = field(default_factory=lambda: f"step_{str(uuid.uuid4())[:6]}")
    description: str = ""
    action: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    expected_result: str = ""
    actual_result: str = ""
    retry_count: int = 0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action": self.action,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "expected_result": self.expected_result,
            "actual_result": self.actual_result,
            "retry_count": self.retry_count,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        return cls(
            step_id=data.get("step_id", f"step_{str(uuid.uuid4())[:6]}"),
            description=data.get("description", ""),
            action=data.get("action", ""),
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            status=StepStatus(data.get("status", "pending")),
            dependencies=data.get("dependencies", []),
            expected_result=data.get("expected_result", ""),
            actual_result=data.get("actual_result", ""),
            retry_count=data.get("retry_count", 0),
            error=data.get("error", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Plan:
    task_id: str = ""
    objective: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "completion_criteria": self.completion_criteria,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            task_id=data.get("task_id", ""),
            objective=data.get("objective", ""),
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            completion_criteria=data.get("completion_criteria", []),
            created_at=data.get("created_at", time.time()),
            version=data.get("version", 1),
        )


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_request: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    plan: Optional[Plan] = None
    current_step_index: int = 0
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    tool_calls: int = 0
    replans: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "user_request": self.user_request,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "plan": self.plan.to_dict() if self.plan else None,
            "current_step_index": self.current_step_index,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "result": self.result,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "replans": self.replans,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        plan_data = data.get("plan")
        plan = Plan.from_dict(plan_data) if plan_data else None
        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            session_id=data.get("session_id"),
            user_request=data.get("user_request", ""),
            status=TaskStatus(data.get("status", "pending")),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            plan=plan,
            current_step_index=data.get("current_step_index", 0),
            completed_steps=data.get("completed_steps", []),
            failed_steps=data.get("failed_steps", []),
            result=data.get("result", ""),
            error=data.get("error", ""),
            tool_calls=data.get("tool_calls", 0),
            replans=data.get("replans", 0),
            metadata=data.get("metadata", {}),
        )
