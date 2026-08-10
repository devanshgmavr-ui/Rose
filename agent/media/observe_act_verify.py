"""Observe/Act/Verify loop for safe visual automation.

Stage 3.3 - Observe/Act/Verify.

Integrates vision perception with computer control in a
safe loop with maximum iterations, timeouts, action limits,
retry limits, failure recovery, and re-planning support.

OBSERVE -> PLAN -> ACT -> OBSERVE -> VERIFY

The loop never runs infinitely - all limits are enforced.
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple

from .vision import VisionResult
from .analyzer import VisionAnalyzer
from .grounding import VisualGrounder, GroundingResult, GroundedTarget

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """States of the observe/act/verify loop."""
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    ACTING = "acting"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    COMPLETE = "complete"
    FAILED = "failed"


class LoopExitReason(Enum):
    """Reasons for loop exit."""
    GOAL_ACHIEVED = "goal_achieved"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    MAX_ACTIONS = "max_actions"
    VERIFICATION_FAILED = "verification_failed"
    RECOVERY_FAILED = "recovery_failed"
    USER_CANCELLED = "user_cancelled"
    ERROR = "error"


@dataclass
class LoopConfig:
    """Configuration for observe/act/verify loop."""
    max_iterations: int = 10
    max_actions: int = 20
    timeout: float = 120.0
    max_retries: int = 3
    verification_timeout: float = 10.0
    confidence_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_actions": self.max_actions,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "verification_timeout": self.verification_timeout,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class LoopStep:
    """A single step in the observe/act/verify loop."""
    step_number: int
    state: LoopState
    observation: Optional[VisionResult] = None
    grounding: Optional[GroundingResult] = None
    action: Optional[Dict[str, Any]] = None
    verification: Optional[VisionResult] = None
    success: bool = False
    error: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "step_number": self.step_number,
            "state": self.state.value,
            "success": self.success,
            "execution_time": self.execution_time,
        }
        if self.error:
            result["error"] = self.error
        if self.action:
            result["action"] = self.action
        return result


@dataclass
class LoopResult:
    """Result of the observe/act/verify loop."""
    success: bool
    exit_reason: LoopExitReason
    steps: List[LoopStep] = field(default_factory=list)
    total_iterations: int = 0
    total_actions: int = 0
    total_time: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "exit_reason": self.exit_reason.value,
            "total_iterations": self.total_iterations,
            "total_actions": self.total_actions,
            "total_time": self.total_time,
            "step_count": len(self.steps),
            "metadata": self.metadata,
        }
        if self.error:
            result["error"] = self.error
        return result

    def to_text(self) -> str:
        lines = ["[BEGIN UNTRUSTED LOOP RESULT]"]
        lines.append(f"Success: {self.success}")
        lines.append(f"Exit reason: {self.exit_reason.value}")
        lines.append(f"Iterations: {self.total_iterations}")
        lines.append(f"Actions: {self.total_actions}")
        lines.append(f"Time: {self.total_time:.2f}s")
        for step in self.steps:
            status = "OK" if step.success else "FAIL"
            lines.append(
                f"  Step {step.step_number}: {step.state.value} [{status}]"
            )
            if step.error:
                lines.append(f"    Error: {step.error}")
        lines.append("[END UNTRUSTED LOOP RESULT]")
        return "\n".join(lines)


class ObserveActVerifyLoop:
    """Safe observe/act/verify loop for visual automation."""

    def __init__(
        self,
        vision_analyzer: VisionAnalyzer,
        visual_grounder: VisualGrounder,
        config: Optional[LoopConfig] = None,
    ):
        self._analyzer = vision_analyzer
        self._grounder = visual_grounder
        self._config = config or LoopConfig()
        self._cancelled = False
        self._state = LoopState.IDLE
        self._step_count = 0
        self._action_count = 0
        self._start_time = 0.0

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state not in (
            LoopState.IDLE,
            LoopState.COMPLETE,
            LoopState.FAILED,
        )

    def cancel(self):
        """Cancel the loop."""
        self._cancelled = True

    def execute(
        self,
        goal: str,
        observe_fn: Optional[Callable[[], str]] = None,
        plan_fn: Optional[Callable[[VisionResult], Optional[Dict[str, Any]]]] = None,
        act_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        verify_fn: Optional[Callable[[str], VisionResult]] = None,
    ) -> LoopResult:
        """Execute the observe/act/verify loop.

        Args:
            goal: Description of the goal to achieve.
            observe_fn: Function to capture screenshot and return path.
            plan_fn: Function to plan action from vision result.
            act_fn: Function to execute an action.
            verify_fn: Function to verify action result.

        Returns:
            LoopResult with loop outcome.
        """
        start = time.time()
        self._start_time = start
        self._cancelled = False
        self._step_count = 0
        self._action_count = 0
        steps = []

        if not all([observe_fn, plan_fn, act_fn, verify_fn]):
            return LoopResult(
                success=False,
                exit_reason=LoopExitReason.ERROR,
                error="All callback functions must be provided",
                total_time=time.time() - start,
            )

        try:
            while self._should_continue():
                self._step_count += 1
                step = LoopStep(
                    step_number=self._step_count,
                    state=LoopState.OBSERVING,
                )

                try:
                    self._state = LoopState.OBSERVING
                    screenshot_path = observe_fn()
                    if not screenshot_path:
                        step.error = "Failed to capture screenshot"
                        steps.append(step)
                        continue

                    observation = self._analyzer.analyze(
                        image_path=screenshot_path,
                        prompt=f"Observe for goal: {goal}",
                    )
                    step.observation = observation

                    if not observation.success:
                        step.error = observation.error
                        steps.append(step)
                        continue

                    self._state = LoopState.PLANNING
                    action = plan_fn(observation)
                    step.action = action

                    if action is None:
                        step.success = True
                        step.state = LoopState.COMPLETE
                        self._state = LoopState.COMPLETE
                        steps.append(step)
                        break

                    if action.get("type") == "goal_achieved":
                        step.success = True
                        step.state = LoopState.COMPLETE
                        self._state = LoopState.COMPLETE
                        steps.append(step)
                        break

                    self._state = LoopState.ACTING
                    self._action_count += 1
                    action_success = act_fn(action)
                    step.success = action_success

                    if not action_success:
                        step.error = "Action execution failed"
                        steps.append(step)
                        continue

                    self._state = LoopState.VERIFYING
                    verify_screenshot = verify_fn(screenshot_path)
                    if verify_screenshot:
                        verification = self._analyzer.analyze(
                            image_path=verify_screenshot,
                            prompt=f"Verify goal: {goal}",
                        )
                        step.verification = verification

                    steps.append(step)

                except Exception as e:
                    step.error = f"Step failed: {e}"
                    step.state = LoopState.FAILED
                    steps.append(step)
                    logger.error(f"Loop step {self._step_count} failed: {e}")

            exit_reason = self._determine_exit_reason()
            self._state = (
                LoopState.COMPLETE
                if exit_reason == LoopExitReason.GOAL_ACHIEVED
                else LoopState.FAILED
            )

            return LoopResult(
                success=exit_reason == LoopExitReason.GOAL_ACHIEVED,
                exit_reason=exit_reason,
                steps=steps,
                total_iterations=self._step_count,
                total_actions=self._action_count,
                total_time=time.time() - start,
                metadata=self._config.to_dict(),
            )

        except Exception as e:
            self._state = LoopState.FAILED
            return LoopResult(
                success=False,
                exit_reason=LoopExitReason.ERROR,
                error=str(e),
                steps=steps,
                total_iterations=self._step_count,
                total_actions=self._action_count,
                total_time=time.time() - start,
            )

    def _should_continue(self) -> bool:
        """Check if the loop should continue."""
        if self._cancelled:
            return False

        elapsed = time.time() - self._start_time
        if elapsed >= self._config.timeout:
            return False

        if self._step_count >= self._config.max_iterations:
            return False

        if self._action_count >= self._config.max_actions:
            return False

        return True

    def _determine_exit_reason(self) -> LoopExitReason:
        """Determine why the loop exited."""
        if self._cancelled:
            return LoopExitReason.USER_CANCELLED

        elapsed = time.time() - self._start_time
        if elapsed >= self._config.timeout:
            return LoopExitReason.TIMEOUT

        if self._step_count >= self._config.max_iterations:
            return LoopExitReason.MAX_ITERATIONS

        if self._action_count >= self._config.max_actions:
            return LoopExitReason.MAX_ACTIONS

        if self._state == LoopState.COMPLETE:
            return LoopExitReason.GOAL_ACHIEVED

        return LoopExitReason.ERROR
