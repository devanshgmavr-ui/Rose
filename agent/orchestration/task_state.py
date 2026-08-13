"""Extended task state model for autonomous execution.

Phase 9 - Execution State Management.

Tracks the full lifecycle of an autonomous task through
RECEIVED → UNDERSTANDING → PLANNING → EXECUTING → OBSERVING → VERIFYING → COMPLETED/FAILED.
"""

import time
import uuid
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TaskPhase(Enum):
    """Phases of autonomous task execution."""
    RECEIVED = "received"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    WAITING_CONFIRMATION = "waiting_confirmation"
    EXECUTING = "executing"
    OBSERVING = "observing"
    VERIFYING = "verifying"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepExecution:
    """Record of a single step execution."""
    step_id: str
    tool_name: str
    action: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = False
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    retry_count: int = 0
    observation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "action": self.action,
            "arguments": self.arguments,
            "result": self.result[:200] if self.result else "",
            "success": self.success,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else 0,
            "retry_count": self.retry_count,
            "observation": self.observation[:200] if self.observation else "",
        }


@dataclass
class TaskConstraints:
    """Constraints on task execution derived from user prompt."""
    prohibited_tools: List[str] = field(default_factory=list)
    preferred_tools: List[str] = field(default_factory=list)
    allowed_files: List[str] = field(default_factory=list)
    minimize_confirmations: bool = False
    use_screen_image: bool = False
    max_steps: Optional[int] = None
    timeout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prohibited_tools": self.prohibited_tools,
            "preferred_tools": self.preferred_tools,
            "allowed_files": self.allowed_files,
            "minimize_confirmations": self.minimize_confirmations,
            "use_screen_image": self.use_screen_image,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConstraints":
        return cls(
            prohibited_tools=data.get("prohibited_tools", []),
            preferred_tools=data.get("preferred_tools", []),
            allowed_files=data.get("allowed_files", []),
            minimize_confirmations=data.get("minimize_confirmations", False),
            use_screen_image=data.get("use_screen_image", False),
            max_steps=data.get("max_steps"),
            timeout=data.get("timeout"),
        )


@dataclass
class AutonomousTaskState:
    """Full state of an autonomous task."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    objective: str = ""
    user_prompt: str = ""
    phase: TaskPhase = TaskPhase.RECEIVED
    constraints: TaskConstraints = field(default_factory=TaskConstraints)

    # Execution tracking
    steps_executed: List[StepExecution] = field(default_factory=list)
    completed_step_ids: Set[str] = field(default_factory=set)
    failed_step_ids: Set[str] = field(default_factory=set)
    tools_used: Set[str] = field(default_factory=set)
    tools_unavailable: Set[str] = field(default_factory=set)

    # State tracking
    total_tool_calls: int = 0
    total_replans: int = 0
    current_step_index: int = 0
    error: str = ""
    result: str = ""

    # Timing
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    # Limits
    max_tool_calls: int = 20
    max_replans: int = 3
    max_retries_per_step: int = 2
    max_task_duration: float = 300.0

    def is_terminal(self) -> bool:
        return self.phase in (
            TaskPhase.COMPLETED,
            TaskPhase.FAILED,
            TaskPhase.CANCELLED,
        )

    def can_continue(self) -> bool:
        if self.is_terminal():
            return False
        if self.total_tool_calls >= self.max_tool_calls:
            return False
        if self.total_replans >= self.max_replans:
            return False
        if time.time() - self.created_at > self.max_task_duration:
            return False
        return True

    def advance_to(self, phase: TaskPhase):
        self.phase = phase
        self.updated_at = time.time()

    def record_step(self, execution: StepExecution):
        self.steps_executed.append(execution)
        self.tools_used.add(execution.tool_name)
        self.total_tool_calls += 1
        if execution.success:
            self.completed_step_ids.add(execution.step_id)
        else:
            self.failed_step_ids.add(execution.step_id)
        self.updated_at = time.time()

    def mark_tool_unavailable(self, tool_name: str):
        self.tools_unavailable.add(tool_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "user_prompt": self.user_prompt,
            "phase": self.phase.value,
            "constraints": self.constraints.to_dict(),
            "steps_executed": [s.to_dict() for s in self.steps_executed],
            "completed_steps": list(self.completed_step_ids),
            "failed_steps": list(self.failed_step_ids),
            "tools_used": list(self.tools_used),
            "tools_unavailable": list(self.tools_unavailable),
            "total_tool_calls": self.total_tool_calls,
            "total_replans": self.total_replans,
            "current_step_index": self.current_step_index,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed_time": time.time() - self.created_at,
        }

    def get_summary(self) -> str:
        """Get concise execution summary."""
        elapsed = time.time() - self.created_at
        completed = len(self.completed_step_ids)
        failed = len(self.failed_step_ids)
        tools = ", ".join(sorted(self.tools_used)) if self.tools_used else "none"

        lines = [
            f"[{self.phase.value.upper()}] {self.objective[:80]}",
            f"Steps: {completed} done, {failed} failed | Calls: {self.total_tool_calls} | Replans: {self.total_replans}",
            f"Tools: {tools}",
            f"Time: {elapsed:.1f}s",
        ]
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


@dataclass
class ExecutionTrace:
    """Concise execution trace for UI display (no private chain-of-thought)."""
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add_tool_selected(self, step_id: str, tool_name: str, reason: str):
        self.entries.append({
            "type": "tool_selected",
            "step_id": step_id,
            "tool": tool_name,
            "reason": reason,
            "timestamp": time.time(),
        })

    def add_step_started(self, step_id: str, description: str):
        self.entries.append({
            "type": "step_started",
            "step_id": step_id,
            "description": description,
            "timestamp": time.time(),
        })

    def add_step_completed(self, step_id: str, success: bool, summary: str = ""):
        self.entries.append({
            "type": "step_completed",
            "step_id": step_id,
            "success": success,
            "summary": summary[:100],
            "timestamp": time.time(),
        })

    def add_verification(self, passed: bool, summary: str = ""):
        self.entries.append({
            "type": "verification",
            "passed": passed,
            "summary": summary,
            "timestamp": time.time(),
        })

    def add_replan(self, reason: str, attempt: int):
        self.entries.append({
            "type": "replan",
            "reason": reason,
            "attempt": attempt,
            "timestamp": time.time(),
        })

    def add_error(self, error: str):
        self.entries.append({
            "type": "error",
            "error": error[:200],
            "timestamp": time.time(),
        })

    def add_completion(self, success: bool, result: str):
        self.entries.append({
            "type": "completion",
            "success": success,
            "result": result[:200],
            "timestamp": time.time(),
        })

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return list(self.entries)

    def to_concise_text(self) -> str:
        """Generate concise human-readable execution trace."""
        lines = []
        for entry in self.entries:
            if entry["type"] == "tool_selected":
                lines.append(f"→ Selected: {entry['tool']}")
                if entry.get("reason"):
                    lines.append(f"  {entry['reason']}")
            elif entry["type"] == "step_completed":
                icon = "✓" if entry["success"] else "✗"
                lines.append(f"{icon} {entry.get('summary', 'Step done')}")
            elif entry["type"] == "verification":
                icon = "✓" if entry["passed"] else "✗"
                lines.append(f"{icon} Verification {'passed' if entry['passed'] else 'failed'}")
            elif entry["type"] == "replan":
                lines.append(f"↻ Replanning (attempt {entry['attempt']}): {entry['reason']}")
            elif entry["type"] == "error":
                lines.append(f"✗ Error: {entry['error']}")
            elif entry["type"] == "completion":
                icon = "✓" if entry["success"] else "✗"
                lines.append(f"{icon} Task {'completed' if entry['success'] else 'failed'}")
        return "\n".join(lines)
