"""Structured failure recovery for autonomous task execution.

Phase 13 - Observation, Verification & Failure Recovery.

Provides:
- Error categorization
- Recovery strategy selection
- Tool failure recovery
- Permission denial handling
- Timeout recovery
- Verification failure recovery
- Fallback tool selection
- Recovery decision logging
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for recovery decisions."""
    TOOL_FAILURE = "tool_failure"
    PERMISSION_DENIED = "permission_denied"
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"
    UNAVAILABLE_CAPABILITY = "unavailable_capability"
    MISSING_DEPENDENCY = "missing_dependency"
    ENVIRONMENT_FAILURE = "environment_failure"
    VERIFICATION_FAILURE = "verification_failure"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY_SAME = "retry_same"
    RETRY_ALTERNATIVE = "retry_alternative"
    REPLAN = "replan"
    SKIP_STEP = "skip_step"
    ASK_CLARIFICATION = "ask_clarification"
    STOP_SAFE = "stop_safe"
    USE_FALLBACK_TOOL = "use_fallback_tool"


@dataclass
class RecoveryDecision:
    """A decision made by the recovery system."""
    error_category: ErrorCategory
    strategy: RecoveryStrategy
    reasoning: str
    alternative_tool: Optional[str] = None
    fallback_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_category": self.error_category.value,
            "strategy": self.strategy.value,
            "reasoning": self.reasoning,
            "alternative_tool": self.alternative_tool,
            "fallback_data": self.fallback_data,
            "timestamp": self.timestamp,
        }


@dataclass
class FailureContext:
    """Context information about a failure."""
    tool_name: str
    action: str
    error_message: str
    error_type: str = ""
    attempt_number: int = 1
    max_attempts: int = 3
    step_description: str = ""
    previous_failures: List[Dict[str, Any]] = field(default_factory=list)
    task_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "action": self.action,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "step_description": self.step_description,
            "previous_failures": self.previous_failures,
        }


# Tool fallback mappings
TOOL_FALLBACKS: Dict[str, List[str]] = {
    "vision_analyze": ["screen_capture", "image_analyze"],
    "visual_ground": ["vision_analyze"],
    "screen_capture": ["browser"],
    "mouse": ["keyboard"],
    "keyboard": ["mouse"],
    "browser": ["screen_capture"],
    "filesystem": ["python_sandbox"],
    "python_sandbox": ["filesystem"],
    "window": ["screen_capture"],
}


class FailureRecovery:
    """Structured failure recovery with decision-making.

    Supports error categories:
    - tool_failure
    - permission_denied
    - invalid_input
    - timeout
    - unavailable_capability
    - missing_dependency
    - environment_failure
    - verification_failure
    """

    def __init__(self, max_history: int = 100):
        self._history: deque = deque(maxlen=max_history)
        self._tool_failures: Dict[str, int] = {}
        self._recovery_counts: Dict[str, int] = {}

    def classify_error(
        self,
        error_message: str,
        error_type: str = "",
        tool_name: str = "",
    ) -> ErrorCategory:
        """Classify an error into a category."""
        msg_lower = error_message.lower()
        type_lower = error_type.lower()

        if "permission" in msg_lower or "denied" in msg_lower or "not allowed" in msg_lower:
            return ErrorCategory.PERMISSION_DENIED

        if "timeout" in msg_lower or "timed out" in msg_lower:
            return ErrorCategory.TIMEOUT

        if "not found" in msg_lower or "does not exist" in msg_lower or "no such" in msg_lower:
            if tool_name in ("filesystem", "window", "browser"):
                return ErrorCategory.TOOL_FAILURE
            return ErrorCategory.INVALID_INPUT

        if "not available" in msg_lower or "disabled" in msg_lower or "unsupported" in msg_lower:
            return ErrorCategory.UNAVAILABLE_CAPABILITY

        if "import" in msg_lower or "module" in msg_lower or "dependency" in msg_lower:
            return ErrorCategory.MISSING_DEPENDENCY

        if "verification" in msg_lower or "expected" in msg_lower:
            return ErrorCategory.VERIFICATION_FAILURE

        if "connection" in msg_lower or "network" in msg_lower or "os error" in type_lower:
            return ErrorCategory.ENVIRONMENT_FAILURE

        if "invalid" in msg_lower or "invalid" in type_lower or "value" in type_lower:
            return ErrorCategory.INVALID_INPUT

        if "error" in type_lower or "exception" in type_lower or "error" in msg_lower:
            return ErrorCategory.TOOL_FAILURE

        return ErrorCategory.UNKNOWN

    def decide_recovery(
        self,
        context: FailureContext,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RecoveryDecision:
        """Decide on a recovery strategy based on failure context.

        Args:
            context: Information about the failure.
            constraints: Task constraints (prohibited tools, etc.)

        Returns:
            RecoveryDecision with strategy and reasoning.
        """
        constraints = constraints or {}
        prohibited = set(constraints.get("prohibited_tools", []))

        category = self.classify_error(
            error_message=context.error_message,
            error_type=context.error_type,
            tool_name=context.tool_name,
        )

        self._tool_failures[context.tool_name] = (
            self._tool_failures.get(context.tool_name, 0) + 1
        )

        if category == ErrorCategory.PERMISSION_DENIED:
            return self._handle_permission_denied(context, prohibited)

        if category == ErrorCategory.TIMEOUT:
            return self._handle_timeout(context, prohibited)

        if category == ErrorCategory.UNAVAILABLE_CAPABILITY:
            return self._handle_unavailable(context, prohibited)

        if category == ErrorCategory.VERIFICATION_FAILURE:
            return self._handle_verification_failure(context, prohibited)

        if category == ErrorCategory.INVALID_INPUT:
            return self._handle_invalid_input(context, prohibited)

        if category == ErrorCategory.MISSING_DEPENDENCY:
            return self._handle_missing_dependency(context, prohibited)

        if category == ErrorCategory.ENVIRONMENT_FAILURE:
            return self._handle_environment_failure(context, prohibited)

        if category == ErrorCategory.TOOL_FAILURE:
            return self._handle_tool_failure(context, prohibited)

        return RecoveryDecision(
            error_category=category,
            strategy=RecoveryStrategy.STOP_SAFE,
            reasoning=f"Unknown error: {context.error_message}",
        )

    def _handle_permission_denied(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle permission denial."""
        if context.attempt_number >= context.max_attempts:
            return RecoveryDecision(
                error_category=ErrorCategory.PERMISSION_DENIED,
                strategy=RecoveryStrategy.ASK_CLARIFICATION,
                reasoning="Permission denied after multiple attempts, asking user",
            )

        fallback = self._find_alternative_tool(context.tool_name, prohibited)
        if fallback:
            return RecoveryDecision(
                error_category=ErrorCategory.PERMISSION_DENIED,
                strategy=RecoveryStrategy.USE_FALLBACK_TOOL,
                reasoning=f"Permission denied for {context.tool_name}, trying {fallback}",
                alternative_tool=fallback,
            )

        return RecoveryDecision(
            error_category=ErrorCategory.PERMISSION_DENIED,
            strategy=RecoveryStrategy.ASK_CLARIFICATION,
            reasoning="Permission denied and no alternative available",
        )

    def _handle_timeout(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle timeout errors."""
        if context.attempt_number < context.max_attempts:
            return RecoveryDecision(
                error_category=ErrorCategory.TIMEOUT,
                strategy=RecoveryStrategy.RETRY_SAME,
                reasoning=f"Timeout on attempt {context.attempt_number}, retrying",
            )

        fallback = self._find_alternative_tool(context.tool_name, prohibited)
        if fallback:
            return RecoveryDecision(
                error_category=ErrorCategory.TIMEOUT,
                strategy=RecoveryStrategy.USE_FALLBACK_TOOL,
                reasoning=f"Timeout exceeded retries, trying {fallback}",
                alternative_tool=fallback,
            )

        return RecoveryDecision(
            error_category=ErrorCategory.TIMEOUT,
            strategy=RecoveryStrategy.REPLAN,
            reasoning="Timeout and no fallback available, replanning",
        )

    def _handle_unavailable(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle unavailable capabilities."""
        fallback = self._find_alternative_tool(context.tool_name, prohibited)
        if fallback:
            return RecoveryDecision(
                error_category=ErrorCategory.UNAVAILABLE_CAPABILITY,
                strategy=RecoveryStrategy.USE_FALLBACK_TOOL,
                reasoning=f"{context.tool_name} unavailable, using {fallback}",
                alternative_tool=fallback,
            )

        return RecoveryDecision(
            error_category=ErrorCategory.UNAVAILABLE_CAPABILITY,
            strategy=RecoveryStrategy.SKIP_STEP,
            reasoning=f"{context.tool_name} unavailable and no alternative, skipping",
        )

    def _handle_verification_failure(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle verification failures."""
        if context.attempt_number < context.max_attempts:
            return RecoveryDecision(
                error_category=ErrorCategory.VERIFICATION_FAILURE,
                strategy=RecoveryStrategy.RETRY_SAME,
                reasoning=f"Verification failed on attempt {context.attempt_number}, retrying",
            )

        return RecoveryDecision(
            error_category=ErrorCategory.VERIFICATION_FAILURE,
            strategy=RecoveryStrategy.REPLAN,
            reasoning="Verification failed after retries, replanning",
        )

    def _handle_invalid_input(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle invalid input errors."""
        if context.attempt_number < context.max_attempts:
            return RecoveryDecision(
                error_category=ErrorCategory.INVALID_INPUT,
                strategy=RecoveryStrategy.RETRY_SAME,
                reasoning=f"Invalid input on attempt {context.attempt_number}, retrying with adjusted args",
            )

        fallback = self._find_alternative_tool(context.tool_name, prohibited)
        if fallback:
            return RecoveryDecision(
                error_category=ErrorCategory.INVALID_INPUT,
                strategy=RecoveryStrategy.USE_FALLBACK_TOOL,
                reasoning=f"Invalid input persists, trying {fallback}",
                alternative_tool=fallback,
            )

        return RecoveryDecision(
            error_category=ErrorCategory.INVALID_INPUT,
            strategy=RecoveryStrategy.REPLAN,
            reasoning="Invalid input and no alternative, replanning",
        )

    def _handle_missing_dependency(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle missing dependency errors."""
        fallback = self._find_alternative_tool(context.tool_name, prohibited)
        if fallback:
            return RecoveryDecision(
                error_category=ErrorCategory.MISSING_DEPENDENCY,
                strategy=RecoveryStrategy.USE_FALLBACK_TOOL,
                reasoning=f"Missing dependency for {context.tool_name}, using {fallback}",
                alternative_tool=fallback,
            )

        return RecoveryDecision(
            error_category=ErrorCategory.MISSING_DEPENDENCY,
            strategy=RecoveryStrategy.SKIP_STEP,
            reasoning="Missing dependency and no alternative, skipping",
        )

    def _handle_environment_failure(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle environment failures."""
        if context.attempt_number < context.max_attempts:
            return RecoveryDecision(
                error_category=ErrorCategory.ENVIRONMENT_FAILURE,
                strategy=RecoveryStrategy.RETRY_SAME,
                reasoning=f"Environment failure on attempt {context.attempt_number}, retrying",
            )

        return RecoveryDecision(
            error_category=ErrorCategory.ENVIRONMENT_FAILURE,
            strategy=RecoveryStrategy.REPLAN,
            reasoning="Environment failure after retries, replanning",
        )

    def _handle_tool_failure(
        self, context: FailureContext, prohibited: set
    ) -> RecoveryDecision:
        """Handle generic tool failures."""
        if context.attempt_number < context.max_attempts:
            return RecoveryDecision(
                error_category=ErrorCategory.TOOL_FAILURE,
                strategy=RecoveryStrategy.RETRY_SAME,
                reasoning=f"Tool failure on attempt {context.attempt_number}, retrying",
            )

        failures = self._tool_failures.get(context.tool_name, 0)
        if failures >= 3:
            fallback = self._find_alternative_tool(context.tool_name, prohibited)
            if fallback:
                return RecoveryDecision(
                    error_category=ErrorCategory.TOOL_FAILURE,
                    strategy=RecoveryStrategy.USE_FALLBACK_TOOL,
                    reasoning=f"{context.tool_name} failed {failures} times, using {fallback}",
                    alternative_tool=fallback,
                )

        return RecoveryDecision(
            error_category=ErrorCategory.TOOL_FAILURE,
            strategy=RecoveryStrategy.REPLAN,
            reasoning="Tool failure after retries, replanning",
        )

    def _find_alternative_tool(
        self, tool_name: str, prohibited: set
    ) -> Optional[str]:
        """Find an alternative tool from fallback mappings."""
        fallbacks = TOOL_FALLBACKS.get(tool_name, [])
        for fallback in fallbacks:
            if fallback not in prohibited:
                return fallback
        return None

    def record_recovery(self, decision: RecoveryDecision):
        """Record a recovery decision."""
        self._history.append(decision)
        key = decision.strategy.value
        self._recovery_counts[key] = self._recovery_counts.get(key, 0) + 1

    def get_history(self, last_n: int = 10) -> List[RecoveryDecision]:
        """Get recent recovery decisions."""
        return list(self._history)[-last_n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "total_recoveries": len(self._history),
            "by_strategy": dict(self._recovery_counts),
            "tool_failures": dict(self._tool_failures),
        }

    def reset_tool_failures(self, tool_name: Optional[str] = None):
        """Reset failure counts."""
        if tool_name:
            self._tool_failures.pop(tool_name, None)
        else:
            self._tool_failures.clear()
