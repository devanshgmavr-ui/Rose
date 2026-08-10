"""Task executor for orchestration."""

import logging
import time
from typing import Optional, Dict, Any, List

from .models import Task, Plan, PlanStep, TaskStatus, StepStatus, Decision
from .state import StateMachine
from .limits import OrchestrationLimits
from .events import EventLogger, EventType, TaskEvent
from .verifier import Verifier
from ..tools.base import ToolRequest, ToolResult
from ..tools.router import ToolRouter

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes tasks following the plan with tool integration."""

    def __init__(
        self,
        tool_router: ToolRouter,
        limits: Optional[OrchestrationLimits] = None,
        event_logger: Optional[EventLogger] = None,
    ):
        self._tool_router = tool_router
        self._limits = limits or OrchestrationLimits()
        self._event_logger = event_logger or EventLogger()
        self._verifier = Verifier(tool_router)
        self._action_history: List[str] = []

    def execute_task(self, task: Task) -> Task:
        state = StateMachine()
        state._current_status = task.status

        task.updated_at = time.time()
        start_time = time.time()

        self._event_logger.log(TaskEvent(
            event_type=EventType.TASK_CREATED,
            task_id=task.task_id,
            session_id=task.session_id,
            data={"user_request": task.user_request},
        ))

        if task.plan is None:
            return self._fail_task(task, state, "No plan provided")

        ok, errs = state.transition(TaskStatus.RUNNING)
        if not ok:
            return self._fail_task(task, state, f"Cannot start: {errs}")

        task.status = TaskStatus.RUNNING

        while not state.is_terminal():
            if time.time() - start_time > self._limits.max_task_duration:
                task.status = TaskStatus.TIMEOUT
                task.error = f"Task timed out after {self._limits.max_task_duration}s"
                state.transition(TaskStatus.TIMEOUT)
                self._event_logger.log(TaskEvent(
                    event_type=EventType.TASK_TIMEOUT,
                    task_id=task.task_id,
                    data={"duration": time.time() - start_time},
                ))
                break

            if task.tool_calls >= self._limits.max_tool_calls:
                return self._fail_task(task, state, "Max tool calls exceeded")

            if task.replans >= self._limits.max_replans:
                return self._fail_task(task, state, "Max replans exceeded")

            current_step = self._get_current_step(task)
            if current_step is None:
                task.status = TaskStatus.COMPLETED
                task.result = "All steps completed"
                state.transition(TaskStatus.COMPLETED)
                break

            if not self._dependencies_met(task, current_step):
                next_step = self._find_next_ready_step(task)
                if next_step is None:
                    return self._fail_task(task, state, "No ready steps - possible deadlock")
                task.current_step_index = self._get_step_index(task, next_step)
                current_step = next_step

            result = self._execute_step(task, current_step, state)
            decision = self._make_decision(task, current_step, result)

            self._event_logger.log(TaskEvent(
                event_type=EventType.DECISION_MADE,
                task_id=task.task_id,
                step_id=current_step.step_id,
                data={"decision": decision.value},
            ))

            if decision == Decision.COMPLETE:
                task.status = TaskStatus.COMPLETED
                task.result = f"Task completed: {task.plan.objective}"
                state.transition(TaskStatus.COMPLETED)
                break
            elif decision == Decision.FAIL:
                return self._fail_task(task, state, current_step.error or "Step failed")
            elif decision == Decision.RETRY:
                if current_step.retry_count >= self._limits.max_step_retries:
                    current_step.status = StepStatus.FAILED
                    task.failed_steps.append(current_step.step_id)
                    self._advance_step(task)
                else:
                    current_step.retry_count += 1
                    self._event_logger.log(TaskEvent(
                        event_type=EventType.RETRY_STARTED,
                        task_id=task.task_id,
                        step_id=current_step.step_id,
                        data={"retry_count": current_step.retry_count},
                    ))
            elif decision == Decision.REPLAN:
                task.replans += 1
                self._event_logger.log(TaskEvent(
                    event_type=EventType.PLAN_REVISED,
                    task_id=task.task_id,
                    data={"replans": task.replans},
                ))
                self._advance_step(task)
            elif decision == Decision.VERIFY:
                task.status = TaskStatus.VERIFYING
                state.transition(TaskStatus.VERIFYING)
                verified = self._verifier.verify(task)
                if verified:
                    task.status = TaskStatus.COMPLETED
                    task.result = "Verification passed"
                    state.transition(TaskStatus.COMPLETED)
                else:
                    task.status = TaskStatus.RUNNING
                    state.transition(TaskStatus.RUNNING)
                    self._advance_step(task)
            else:
                self._advance_step(task)

        task.updated_at = time.time()
        self._action_history.clear()
        return task

    def _execute_step(self, task: Task, step: PlanStep, state: StateMachine) -> ToolResult:
        step.status = StepStatus.RUNNING
        self._event_logger.log(TaskEvent(
            event_type=EventType.STEP_STARTED,
            task_id=task.task_id,
            step_id=step.step_id,
            data={"tool_name": step.tool_name, "description": step.description},
        ))

        if step.tool_name == "generate":
            result = ToolResult(
                success=True,
                tool_name="generate",
                output=f"Generated: {step.description}",
            )
        else:
            request = ToolRequest(
                tool=step.tool_name,
                arguments=step.arguments,
                session_id=task.session_id,
            )
            self._event_logger.log(TaskEvent(
                event_type=EventType.TOOL_REQUESTED,
                task_id=task.task_id,
                step_id=step.step_id,
                data={"tool": step.tool_name, "arguments": step.arguments},
            ))
            task.tool_calls += 1
            result = self._tool_router.route(request)

        step.actual_result = result.output if result.success else result.error
        self._event_logger.log(TaskEvent(
            event_type=EventType.TOOL_COMPLETED,
            task_id=task.task_id,
            step_id=step.step_id,
            data={"success": result.success},
        ))

        if result.success:
            step.status = StepStatus.COMPLETED
            task.completed_steps.append(step.step_id)
            self._event_logger.log(TaskEvent(
                event_type=EventType.STEP_COMPLETED,
                task_id=task.task_id,
                step_id=step.step_id,
            ))
        else:
            step.status = StepStatus.FAILED
            step.error = result.error
            self._event_logger.log(TaskEvent(
                event_type=EventType.STEP_FAILED,
                task_id=task.task_id,
                step_id=step.step_id,
                data={"error": result.error},
            ))

        return result

    def _make_decision(self, task: Task, step: PlanStep, result: ToolResult) -> Decision:
        if result.success:
            if step.tool_name != "generate" and result.output:
                action_key = f"{step.tool_name}:{step.arguments}"
                self._action_history.append(action_key)
                if self._action_history.count(action_key) > self._limits.max_repeated_actions:
                    return Decision.FAIL
            return Decision.CONTINUE
        else:
            if step.retry_count < self._limits.max_step_retries:
                return Decision.RETRY
            return Decision.FAIL

    def _get_current_step(self, task: Task) -> Optional[PlanStep]:
        if not task.plan or task.current_step_index >= len(task.plan.steps):
            return None
        return task.plan.steps[task.current_step_index]

    def _get_step_index(self, task: Task, step: PlanStep) -> int:
        if not task.plan:
            return 0
        for i, s in enumerate(task.plan.steps):
            if s.step_id == step.step_id:
                return i
        return 0

    def _dependencies_met(self, task: Task, step: PlanStep) -> bool:
        for dep_id in step.dependencies:
            if dep_id not in task.completed_steps:
                return False
        return True

    def _find_next_ready_step(self, task: Task) -> Optional[PlanStep]:
        if not task.plan:
            return None
        for step in task.plan.steps:
            if step.status == StepStatus.PENDING and self._dependencies_met(task, step):
                return step
        return None

    def _advance_step(self, task: Task):
        if task.plan:
            task.current_step_index += 1

    def _fail_task(self, task: Task, state: StateMachine, error: str) -> Task:
        task.status = TaskStatus.FAILED
        task.error = error
        task.updated_at = time.time()
        state.transition(TaskStatus.FAILED)
        self._event_logger.log(TaskEvent(
            event_type=EventType.TASK_FAILED,
            task_id=task.task_id,
            data={"error": error},
        ))
        return task

    def cancel_task(self, task: Task) -> Task:
        task.status = TaskStatus.CANCELLED
        task.updated_at = time.time()
        self._event_logger.log(TaskEvent(
            event_type=EventType.TASK_CANCELLED,
            task_id=task.task_id,
        ))
        return task
