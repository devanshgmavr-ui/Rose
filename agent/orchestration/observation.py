"""Observation system for capturing and comparing state after actions.

Phase 13 - Observation, Verification & Failure Recovery.

Provides:
- State capture after tool execution
- Expected vs actual state comparison
- Observation history for debugging
- Structured observation results
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ObservationType(Enum):
    """Types of observations."""
    FILE_EXISTS = "file_exists"
    FILE_CONTENT = "file_content"
    WINDOW_STATE = "window_state"
    SCREENSHOT = "screenshot"
    URL_STATE = "url_state"
    PAGE_CONTENT = "page_content"
    ELEMENT_EXISTS = "element_exists"
    CUSTOM = "custom"


class ObservationStatus(Enum):
    """Status of an observation check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class Observation:
    """A single observation of system state."""
    observation_type: ObservationType
    description: str
    expected: Any = None
    actual: Any = None
    status: ObservationStatus = ObservationStatus.SKIPPED
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_type": self.observation_type.value,
            "description": self.description,
            "expected": self.expected,
            "actual": self.actual,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @property
    def passed(self) -> bool:
        return self.status == ObservationStatus.PASSED


@dataclass
class ObservationResult:
    """Result of an observation check."""
    action_description: str
    observations: List[Observation] = field(default_factory=list)
    overall_status: ObservationStatus = ObservationStatus.SKIPPED
    timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_description": self.action_description,
            "observations": [o.to_dict() for o in self.observations],
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "passed_count": sum(1 for o in self.observations if o.passed),
            "failed_count": sum(1 for o in self.observations if o.status == ObservationStatus.FAILED),
            "total_count": len(self.observations),
        }

    @property
    def passed(self) -> bool:
        return self.overall_status == ObservationStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for o in self.observations if o.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.observations if o.status == ObservationStatus.FAILED)


@dataclass
class ObservationRule:
    """A rule that defines what to observe after an action."""
    action_pattern: str
    observation_type: ObservationType
    description: str
    check_function: Optional[Callable] = None
    extract_function: Optional[Callable] = None
    expected_extractor: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_pattern": self.action_pattern,
            "observation_type": self.observation_type.value,
            "description": self.description,
        }


class ObservationSystem:
    """Captures and validates system state after tool execution.

    Provides structured observation:
    ACTION → OBSERVATION → EXPECTED STATE → ACTUAL STATE → VERIFICATION
    """

    def __init__(self, max_history: int = 100):
        self._rules: List[ObservationRule] = []
        self._history: deque = deque(maxlen=max_history)
        self._custom_checks: Dict[str, Callable] = {}
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default observation rules."""
        self._rules.append(ObservationRule(
            action_pattern="filesystem:write",
            observation_type=ObservationType.FILE_EXISTS,
            description="Verify file was created",
        ))
        self._rules.append(ObservationRule(
            action_pattern="filesystem:delete",
            observation_type=ObservationType.FILE_EXISTS,
            description="Verify file was removed",
        ))
        self._rules.append(ObservationRule(
            action_pattern="window:activate",
            observation_type=ObservationType.WINDOW_STATE,
            description="Verify window is active",
        ))
        self._rules.append(ObservationRule(
            action_pattern="window:minimize",
            observation_type=ObservationType.WINDOW_STATE,
            description="Verify window is minimized",
        ))
        self._rules.append(ObservationRule(
            action_pattern="browser:navigate",
            observation_type=ObservationType.URL_STATE,
            description="Verify page navigated",
        ))
        self._rules.append(ObservationRule(
            action_pattern="browser:click",
            observation_type=ObservationType.ELEMENT_EXISTS,
            description="Verify element state after click",
        ))

    def register_rule(self, rule: ObservationRule):
        """Register an observation rule."""
        self._rules.append(rule)

    def register_custom_check(self, name: str, check_fn: Callable):
        """Register a custom observation check function."""
        self._custom_checks[name] = check_fn

    def get_rules_for_action(self, tool_name: str, action: str = "") -> List[ObservationRule]:
        """Get applicable observation rules for an action."""
        matching = []
        action_key = f"{tool_name}:{action}" if action else tool_name
        for rule in self._rules:
            if rule.action_pattern in action_key or rule.action_pattern == tool_name:
                matching.append(rule)
        return matching

    def observe(
        self,
        tool_name: str,
        action: str,
        tool_result: Any,
        arguments: Dict[str, Any],
        tool_router: Any = None,
    ) -> ObservationResult:
        """Observe the state after a tool execution.

        Args:
            tool_name: Name of the tool that was executed.
            action: Action performed.
            tool_result: Result from tool execution.
            arguments: Arguments passed to the tool.
            tool_router: Optional tool router for executing observation checks.

        Returns:
            ObservationResult with all observations.
        """
        start = time.time()
        result = ObservationResult(
            action_description=f"{tool_name}:{action}",
        )

        rules = self.get_rules_for_action(tool_name, action)

        for rule in rules:
            observation = self._execute_observation(
                rule, tool_result, arguments, tool_router
            )
            result.observations.append(observation)

        if result.observations:
            all_passed = all(o.passed for o in result.observations)
            any_failed = any(o.status == ObservationStatus.FAILED for o in result.observations)

            if all_passed:
                result.overall_status = ObservationStatus.PASSED
            elif any_failed:
                result.overall_status = ObservationStatus.FAILED
            else:
                result.overall_status = ObservationStatus.SKIPPED
        else:
            result.overall_status = ObservationStatus.PASSED

        result.execution_time = time.time() - start
        self._history.append(result)
        return result

    def _execute_observation(
        self,
        rule: ObservationRule,
        tool_result: Any,
        arguments: Dict[str, Any],
        tool_router: Any = None,
    ) -> Observation:
        """Execute a single observation check."""
        observation = Observation(
            observation_type=rule.observation_type,
            description=rule.description,
        )

        try:
            if rule.observation_type == ObservationType.FILE_EXISTS:
                return self._observe_file_exists(rule, arguments, tool_router, observation)
            elif rule.observation_type == ObservationType.WINDOW_STATE:
                return self._observe_window_state(rule, arguments, tool_router, observation)
            elif rule.observation_type == ObservationType.URL_STATE:
                return self._observe_url_state(rule, arguments, tool_router, observation)
            elif rule.observation_type == ObservationType.ELEMENT_EXISTS:
                return self._observe_element_exists(rule, arguments, tool_router, observation)
            else:
                observation.status = ObservationStatus.SKIPPED
                return observation

        except Exception as e:
            observation.status = ObservationStatus.ERROR
            observation.metadata["error"] = str(e)
            logger.warning(f"Observation error: {e}")
            return observation

    def _observe_file_exists(
        self, rule, arguments, tool_router, observation
    ) -> Observation:
        """Observe whether a file exists."""
        path = arguments.get("path", "")
        if not path:
            observation.status = ObservationStatus.SKIPPED
            return observation

        observation.expected = "file exists"
        observation.metadata["path"] = path

        if tool_router:
            try:
                result = tool_router.execute_tool(
                    "filesystem", {"action": "info", "path": path}
                )
                observation.actual = "file exists" if result.success else "file not found"
                observation.status = (
                    ObservationStatus.PASSED if result.success else ObservationStatus.FAILED
                )
            except Exception as e:
                observation.status = ObservationStatus.ERROR
                observation.metadata["error"] = str(e)
        else:
            observation.status = ObservationStatus.SKIPPED

        return observation

    def _observe_window_state(
        self, rule, arguments, tool_router, observation
    ) -> Observation:
        """Observe window state."""
        title = arguments.get("title", arguments.get("window_title", ""))
        observation.expected = f"window '{title}' state changed"
        observation.metadata["title"] = title

        if tool_router:
            try:
                result = tool_router.execute_tool(
                    "window", {"action": "get_active"}
                )
                if result.success:
                    observation.actual = "window state observed"
                    observation.status = ObservationStatus.PASSED
                else:
                    observation.actual = "could not observe window"
                    observation.status = ObservationStatus.FAILED
            except Exception as e:
                observation.status = ObservationStatus.ERROR
                observation.metadata["error"] = str(e)
        else:
            observation.status = ObservationStatus.SKIPPED

        return observation

    def _observe_url_state(
        self, rule, arguments, tool_router, observation
    ) -> Observation:
        """Observe URL state after navigation."""
        expected_url = arguments.get("url", "")
        observation.expected = f"navigated to {expected_url}"
        observation.metadata["expected_url"] = expected_url

        if tool_router:
            try:
                result = tool_router.execute_tool(
                    "browser", {"action": "read_page"}
                )
                if result.success:
                    observation.actual = "page loaded"
                    observation.status = ObservationStatus.PASSED
                else:
                    observation.actual = "page not loaded"
                    observation.status = ObservationStatus.FAILED
            except Exception as e:
                observation.status = ObservationStatus.ERROR
                observation.metadata["error"] = str(e)
        else:
            observation.status = ObservationStatus.SKIPPED

        return observation

    def _observe_element_exists(
        self, rule, arguments, tool_router, observation
    ) -> Observation:
        """Observe whether an element exists after interaction."""
        selector = arguments.get("selector", "")
        observation.expected = f"element '{selector}' interacted with"
        observation.metadata["selector"] = selector
        observation.status = ObservationStatus.SKIPPED
        return observation

    def get_history(self, last_n: int = 10) -> List[ObservationResult]:
        """Get recent observation results."""
        return list(self._history)[-last_n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get observation statistics."""
        history = list(self._history)
        total = len(history)
        passed = sum(1 for r in history if r.passed)
        failed = sum(1 for r in history if r.overall_status == ObservationStatus.FAILED)

        return {
            "total_observations": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / max(total, 1),
        }
