"""Autonomous execution loop for Rose.

Phase 9 - Autonomous Prompt → Execution Pipeline.

Implements the core loop:
  PROMPT → UNDERSTAND → PLAN → EXECUTE → OBSERVE → VERIFY → REPEAT/COMPLETE

Selects tools dynamically at each step. Replanning on failure.
Respects permissions, constraints, and safety limits.
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable

from .capability_analyzer import CapabilityAnalyzer, CapabilityAnalysis
from .tool_scorer import ToolScorer, SelectionResult
from .task_state import (
    AutonomousTaskState, TaskPhase, TaskConstraints,
    StepExecution, ExecutionTrace,
)
from .tool_catalog import ToolMetadata, build_tool_catalog
from .limits import OrchestrationLimits
from .events import EventLogger, EventType, TaskEvent
from ..tools.base import ToolRequest, ToolResult
from ..tools.router import ToolRouter

logger = logging.getLogger(__name__)


class AutonomousLoop:
    """Core autonomous execution loop.

    Processes a user prompt through:
    1. Understanding (capability analysis)
    2. Planning (step generation)
    3. Execution (dynamic tool selection per step)
    4. Observation (result inspection)
    5. Verification (completion check)
    6. Replanning (on failure)
    """

    def __init__(
        self,
        tool_router: ToolRouter,
        permission_manager=None,
        limits: Optional[OrchestrationLimits] = None,
        event_logger: Optional[EventLogger] = None,
    ):
        self._tool_router = tool_router
        self._permission_manager = permission_manager
        self._limits = limits or OrchestrationLimits()
        self._event_logger = event_logger or EventLogger()
        self._analyzer = CapabilityAnalyzer()
        self._scorer = ToolScorer(
            tool_registry=None,
            permission_manager=permission_manager,
        )
        self._catalog = build_tool_catalog()
        self._cancelled = False
        self._progress_callback: Optional[Callable] = None
        self._confirmation_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        self._progress_callback = callback

    def set_confirmation_callback(self, callback: Callable):
        self._confirmation_callback = callback

    def cancel(self):
        self._cancelled = True

    def execute(
        self,
        user_prompt: str,
        task_id: Optional[str] = None,
    ) -> AutonomousTaskState:
        """Execute a user prompt autonomously.

        Args:
            user_prompt: Natural language objective from user.
            task_id: Optional task ID.

        Returns:
            AutonomousTaskState with full execution results.
        """
        state = AutonomousTaskState(
            task_id=task_id or f"auto_{int(time.time())}",
            user_prompt=user_prompt,
            max_tool_calls=self._limits.max_tool_calls,
            max_replans=self._limits.max_replans,
            max_task_duration=self._limits.max_task_duration,
        )
        trace = ExecutionTrace()

        try:
            # Phase 1: UNDERSTANDING
            state.advance_to(TaskPhase.UNDERSTANDING)
            analysis = self._analyzer.analyze(user_prompt)
            state.objective = analysis.capabilities[0].description if analysis.capabilities else user_prompt
            state.constraints = self._apply_constraints(analysis.constraints)

            # Phase 2: PLANNING
            state.advance_to(TaskPhase.PLANNING)
            steps = self._generate_steps(user_prompt, analysis, state)

            self._event_logger.log(TaskEvent(
                event_type=EventType.PLAN_CREATED,
                task_id=state.task_id,
                data={"steps": len(steps), "objective": state.objective},
            ))

            # Phase 3-6: EXECUTION LOOP
            step_index = 0
            while step_index < len(steps) and state.can_continue() and not self._cancelled:
                step = steps[step_index]
                state.current_step_index = step_index

                # Dynamic tool selection for this step
                state.advance_to(TaskPhase.EXECUTING)
                selection = self._select_tool_for_step(step, state, analysis)

                if selection.selected_tool is None:
                    trace.add_error(f"No tool available for step: {step.get('description', '')}")
                    state.error = f"No tool available for: {step.get('description', '')}"
                    break

                tool_score = selection.selected_tool
                tool_name = tool_score.tool_name
                trace.add_tool_selected(
                    step_id=step.get("step_id", f"step_{step_index}"),
                    tool_name=tool_name,
                    reason=selection.reason_summary,
                )

                # Permission check
                perm_result = self._check_permission(tool_name, step)
                if perm_result == "denied":
                    state.mark_tool_unavailable(tool_name)
                    self._scorer.record_failure(tool_name)
                    trace.add_step_completed(
                        step_id=step.get("step_id", f"step_{step_index}"),
                        success=False,
                        summary=f"{tool_name} permission denied",
                    )
                    step_index += 1
                    continue
                elif perm_result == "confirm":
                    state.advance_to(TaskPhase.WAITING_CONFIRMATION)
                    confirmed = self._request_confirmation(tool_name, step)
                    if not confirmed:
                        trace.add_step_completed(
                            step_id=step.get("step_id", f"step_{step_index}"),
                            success=False,
                            summary="User denied confirmation",
                        )
                        step_index += 1
                        continue
                    state.advance_to(TaskPhase.EXECUTING)

                # Execute the step
                trace.add_step_started(
                    step_id=step.get("step_id", f"step_{step_index}"),
                    description=step.get("description", ""),
                )
                execution = self._execute_step(step, tool_name, state)
                state.record_step(execution)

                if execution.success:
                    self._scorer.record_success(tool_name)
                else:
                    self._scorer.record_failure(tool_name)

                trace.add_step_completed(
                    step_id=execution.step_id,
                    success=execution.success,
                    summary=step.get("description", "")[:100],
                )

                # Phase 4: OBSERVING
                state.advance_to(TaskPhase.OBSERVING)
                observation = self._observe(execution)
                execution.observation = observation

                # Phase 5: VERIFYING
                state.advance_to(TaskPhase.VERIFYING)
                verified = self._verify_step(execution, step)
                trace.add_verification(
                    passed=verified,
                    summary=f"Step {execution.step_id}: {'passed' if verified else 'failed'}",
                )

                if verified:
                    step_index += 1
                else:
                    # Phase 6: REPLANNING
                    retries = execution.retry_count
                    if retries < state.max_retries_per_step:
                        execution.retry_count += 1
                        trace.add_replan(
                            reason=f"Step failed, retry {execution.retry_count}",
                            attempt=execution.retry_count,
                        )
                        # Don't advance - retry same step
                    elif state.total_replans < state.max_replans:
                        state.total_replans += 1
                        trace.add_replan(
                            reason="Max retries reached, replanning",
                            attempt=state.total_replans,
                        )
                        step_index += 1
                    else:
                        state.error = f"Step {execution.step_id} failed after {retries} retries"
                        break

            # Determine final result
            if self._cancelled:
                state.advance_to(TaskPhase.CANCELLED)
                trace.add_completion(False, "Task cancelled by user")
            elif state.error:
                state.advance_to(TaskPhase.FAILED)
                trace.add_completion(False, state.error)
            elif step_index >= len(steps):
                state.advance_to(TaskPhase.COMPLETED)
                state.result = f"Task completed: {state.objective}"
                trace.add_completion(True, state.result)
            else:
                state.advance_to(TaskPhase.FAILED)
                if not state.error:
                    state.error = "Task could not continue"
                trace.add_completion(False, state.error)

            state.completed_at = time.time()

            self._event_logger.log(TaskEvent(
                event_type=EventType.TASK_COMPLETED if state.phase == TaskPhase.COMPLETED else EventType.TASK_FAILED,
                task_id=state.task_id,
                data={"result": state.result or state.error},
            ))

        except Exception as e:
            logger.error(f"Autonomous loop error: {e}")
            state.advance_to(TaskPhase.FAILED)
            state.error = str(e)
            trace.add_error(str(e))

        return state

    def _apply_constraints(self, raw_constraints: Dict[str, Any]) -> TaskConstraints:
        """Convert raw constraints to TaskConstraints."""
        return TaskConstraints(
            prohibited_tools=raw_constraints.get("prohibited_tools", []),
            preferred_tools=raw_constraints.get("preferred_tools", []),
            allowed_files=raw_constraints.get("allowed_files", []),
            minimize_confirmations=raw_constraints.get("minimize_confirmations", False),
            use_screen_image=raw_constraints.get("use_screen_image", False),
        )

    def _generate_steps(
        self,
        user_prompt: str,
        analysis: CapabilityAnalysis,
        state: AutonomousTaskState,
    ) -> List[Dict[str, Any]]:
        """Generate execution steps from analysis."""
        steps = []
        step_num = 0

        for cap in analysis.capabilities:
            step_num += 1
            tools = []
            if cap.name in CAPABILITY_TOOLS_MAP:
                tools = CAPABILITY_TOOLS_MAP[cap.name]

            step = {
                "step_id": f"step_{step_num}",
                "description": cap.description,
                "capability": cap.name,
                "suggested_tools": tools,
                "confidence": cap.confidence,
                "arguments": {},
            }
            steps.append(step)

        if not steps:
            steps.append({
                "step_id": "step_1",
                "description": user_prompt,
                "capability": "general",
                "suggested_tools": [],
                "confidence": 0.5,
                "arguments": {"request": user_prompt},
            })

        return steps

    def _select_tool_for_step(
        self,
        step: Dict[str, Any],
        state: AutonomousTaskState,
        analysis: CapabilityAnalysis,
    ) -> SelectionResult:
        """Select the best tool for a specific step."""
        cap_name = step.get("capability", "general")
        capabilities = [Capability(
            name=cap_name,
            description=step.get("description", ""),
            confidence=step.get("confidence", 0.5),
        )]

        context = {
            "active_tools": list(state.tools_used),
            "recent_successes": [
                e.tool_name for e in state.steps_executed[-3:]
                if e.success
            ],
            "recent_failures": [
                e.tool_name for e in state.steps_executed[-3:]
                if not e.success
            ],
            "unavailable": list(state.tools_unavailable),
        }

        constraints_dict = state.constraints.to_dict()

        return self._scorer.select_tool(
            capabilities=capabilities,
            context=context,
            explicit_tools=analysis.explicit_tools,
            constraints=constraints_dict,
        )

    def _check_permission(self, tool_name: str, step: Dict[str, Any]) -> str:
        """Check permission for tool. Returns 'allow', 'confirm', or 'denied'."""
        if not self._permission_manager:
            return "allow"

        meta = self._catalog.get(tool_name)
        if not meta:
            return "allow"

        if not meta.permissions:
            return "allow"

        for perm in meta.permissions:
            status = self._permission_manager.check_permission(perm)
            if status.value == "deny":
                return "denied"
            elif status.value == "require_confirmation":
                return "confirm"

        return "allow"

    def _request_confirmation(self, tool_name: str, step: Dict[str, Any]) -> bool:
        """Request user confirmation for a dangerous action."""
        if not self._confirmation_callback:
            return False

        try:
            return self._confirmation_callback(
                tool_name=tool_name,
                description=step.get("description", ""),
                arguments=step.get("arguments", {}),
            )
        except Exception as e:
            logger.warning(f"Confirmation callback failed: {e}")
            return False

    def _execute_step(
        self,
        step: Dict[str, Any],
        tool_name: str,
        state: AutonomousTaskState,
    ) -> StepExecution:
        """Execute a single step using the selected tool."""
        execution = StepExecution(
            step_id=step.get("step_id", f"step_{state.current_step_index}"),
            tool_name=tool_name,
            action=step.get("action", ""),
            arguments=step.get("arguments", {}),
            start_time=time.time(),
        )

        try:
            request = ToolRequest(
                tool=tool_name,
                arguments=step.get("arguments", {}),
                session_id=state.task_id,
            )
            result = self._tool_router.route(request)
            execution.success = result.success
            execution.result = result.output if result.success else result.error
            execution.error = result.error if not result.success else ""
        except Exception as e:
            execution.success = False
            execution.error = str(e)
            execution.result = str(e)

        execution.end_time = time.time()
        return execution

    def _observe(self, execution: StepExecution) -> str:
        """Observe the result of a step execution."""
        if execution.success:
            return f"Step {execution.step_id} completed successfully"
        return f"Step {execution.step_id} failed: {execution.error}"

    def _verify_step(self, execution: StepExecution, step: Dict[str, Any]) -> bool:
        """Verify if a step achieved its goal."""
        if not execution.success:
            return False

        expected = step.get("expected_result", "")
        if not expected:
            return True

        return expected.lower() in execution.result.lower() if execution.result else True


# Mapping from capabilities to tool names (module-level for imports)
CAPABILITY_TOOLS_MAP: Dict[str, List[str]] = {
    "file_operations": ["filesystem"],
    "code_execution": ["python_sandbox"],
    "screen_capture": ["screen_capture"],
    "system_information": ["system_info"],
    "mouse_control": ["mouse"],
    "keyboard_input": ["keyboard"],
    "window_management": ["window"],
    "browser_automation": ["browser"],
    "browser_reading": ["browser"],
    "browser_interaction": ["browser"],
    "vision_analysis": ["vision_analyze", "image_analyze"],
    "visual_grounding": ["visual_ground"],
    "app_launch": ["launch_app"],
    "text_transcription": ["vision_analyze", "keyboard"],
    "verification": ["screen_capture", "vision_analyze"],
    "general": [],
}
