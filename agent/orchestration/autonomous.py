"""Autonomous task manager for multi-step task execution.

Stage 4.3 - Multi-Step Autonomous Tasks.

Integrates LLM, Memory, Planner, Executor, Tools, Vision,
and Verifier into a reliable autonomous task system with:
- Task state tracking
- Progress reporting
- Retries and failure recovery
- Cancellation
- Timeout
- Step and resource limits
- Verification
- Final summary
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from .models import Task, Plan, PlanStep, TaskStatus, StepStatus
from .planner import Planner
from .enhanced_planner import EnhancedPlanner
from .executor import TaskExecutor
from .verifier import Verifier
from .limits import OrchestrationLimits
from .events import EventLogger, EventType, TaskEvent
from ..tools.router import ToolRouter

logger = logging.getLogger(__name__)


@dataclass
class TaskProgress:
    """Progress information for a task."""
    task_id: str
    status: str
    objective: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    current_step: str
    elapsed_time: float
    tool_calls: int
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "objective": self.objective,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "current_step": self.current_step,
            "elapsed_time": self.elapsed_time,
            "tool_calls": self.tool_calls,
            "error": self.error,
        }

    def to_text(self) -> str:
        pct = (
            (self.completed_steps / max(self.total_steps, 1)) * 100
            if self.total_steps > 0
            else 0
        )
        lines = [
            f"[Task {self.task_id}] {self.status}",
            f"Objective: {self.objective}",
            f"Progress: {self.completed_steps}/{self.total_steps} ({pct:.0f}%)",
            f"Failed: {self.failed_steps}",
            f"Tool calls: {self.tool_calls}",
            f"Elapsed: {self.elapsed_time:.1f}s",
        ]
        if self.current_step:
            lines.append(f"Current: {self.current_step}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


@dataclass
class TaskResult:
    """Final result of an autonomous task."""
    success: bool
    task_id: str
    objective: str
    result: str
    error: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    total_tool_calls: int = 0
    total_replans: int = 0
    total_time: float = 0.0
    progress_history: List[TaskProgress] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "objective": self.objective,
            "result": self.result,
            "error": self.error,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "total_tool_calls": self.total_tool_calls,
            "total_replans": self.total_replans,
            "total_time": self.total_time,
        }

    def to_text(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"[TASK {status}]",
            f"Objective: {self.objective}",
            f"Result: {self.result}",
            f"Steps: {self.completed_steps}/{self.total_steps}",
            f"Tool calls: {self.total_tool_calls}",
            f"Time: {self.total_time:.1f}s",
        ]
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class AutonomousTaskManager:
    """Manages autonomous task execution with full lifecycle."""

    def __init__(
        self,
        tool_router: ToolRouter,
        llm_provider=None,
        limits: Optional[OrchestrationLimits] = None,
        event_logger: Optional[EventLogger] = None,
    ):
        self._tool_router = tool_router
        self._llm_provider = llm_provider
        self._limits = limits or OrchestrationLimits()
        self._event_logger = event_logger or EventLogger()
        self._planner = EnhancedPlanner(
            llm_provider=llm_provider,
            max_plan_steps=self._limits.max_plan_steps,
        )
        self._executor = TaskExecutor(
            tool_router=tool_router,
            limits=self._limits,
            event_logger=self._event_logger,
        )
        self._verifier = Verifier(tool_router)
        self._active_tasks: Dict[str, Task] = {}
        self._cancelled = False
        self._progress_callback: Optional[Callable[[TaskProgress], None]] = None

    def set_progress_callback(self, callback: Callable[[TaskProgress], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def cancel(self):
        """Cancel all active tasks."""
        self._cancelled = True

    def execute(
        self,
        user_request: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> TaskResult:
        """Execute a user request as an autonomous task.

        Args:
            user_request: The user's natural language request.
            task_id: Optional task ID.
            session_id: Optional session ID.

        Returns:
            TaskResult with the outcome.
        """
        start = time.time()
        self._cancelled = False

        if not task_id:
            task_id = f"task_{int(time.time())}"

        task = Task(
            task_id=task_id,
            session_id=session_id,
            user_request=user_request,
        )

        self._active_tasks[task_id] = task

        try:
            self._report_progress(task, start, "Planning...")

            plan = self._planner.create_plan(user_request, task_id)
            task.plan = plan

            self._event_logger.log(TaskEvent(
                event_type=EventType.PLAN_CREATED,
                task_id=task_id,
                data={"steps": len(plan.steps), "objective": plan.objective},
            ))

            self._report_progress(task, start, "Executing...")

            task = self._executor.execute_task(task)

            verified = self._verifier.verify(task)

            if task.status == TaskStatus.COMPLETED and not verified:
                task.status = TaskStatus.FAILED
                task.error = "Verification failed"

            elapsed = time.time() - start
            result = TaskResult(
                success=task.status == TaskStatus.COMPLETED,
                task_id=task_id,
                objective=user_request,
                result=task.result or task.error,
                error=task.error,
                total_steps=len(plan.steps) if plan else 0,
                completed_steps=len(task.completed_steps),
                failed_steps=len(task.failed_steps),
                total_tool_calls=task.tool_calls,
                total_replans=task.replans,
                total_time=elapsed,
            )

            self._report_progress(task, start, "Complete")

            return result

        except Exception as e:
            elapsed = time.time() - start
            return TaskResult(
                success=False,
                task_id=task_id,
                objective=user_request,
                result="",
                error=str(e),
                total_time=elapsed,
            )
        finally:
            self._active_tasks.pop(task_id, None)

    def _report_progress(self, task: Task, start: float, phase: str):
        """Report task progress."""
        if not self._progress_callback:
            return

        plan = task.plan
        total = len(plan.steps) if plan else 0
        current_step = ""
        if plan and task.current_step_index < len(plan.steps):
            current_step = plan.steps[task.current_step_index].description

        progress = TaskProgress(
            task_id=task.task_id,
            status=task.status.value,
            objective=task.user_request,
            total_steps=total,
            completed_steps=len(task.completed_steps),
            failed_steps=len(task.failed_steps),
            current_step=current_step,
            elapsed_time=time.time() - start,
            tool_calls=task.tool_calls,
            error=task.error,
        )

        try:
            self._progress_callback(progress)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._active_tasks.get(task_id)

    def get_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get current progress for a task."""
        task = self._active_tasks.get(task_id)
        if not task:
            return None

        plan = task.plan
        total = len(plan.steps) if plan else 0
        current_step = ""
        if plan and task.current_step_index < len(plan.steps):
            current_step = plan.steps[task.current_step_index].description

        return TaskProgress(
            task_id=task.task_id,
            status=task.status.value,
            objective=task.user_request,
            total_steps=total,
            completed_steps=len(task.completed_steps),
            failed_steps=len(task.failed_steps),
            current_step=current_step,
            elapsed_time=time.time() - task.created_at,
            tool_calls=task.tool_calls,
            error=task.error,
        )

    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks."""
        return [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "objective": t.user_request,
            }
            for t in self._active_tasks.values()
        ]
