"""Observe/Act/Verify tool for the agent tool system.

Stage 3.3 - Observe/Act/Verify.

Provides a safe loop for visual automation with configurable limits.
Does NOT directly perform mouse actions - uses callback functions
that go through the ToolRouter -> PermissionManager pipeline.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Callable

from ..tools.base import Tool, ToolResult, Permission, ConfirmationLevel
from .observe_act_verify import ObserveActVerifyLoop, LoopConfig, LoopResult
from .analyzer import VisionAnalyzer
from .grounding import VisualGrounder

logger = logging.getLogger(__name__)


class ObserveActVerifyTool(Tool):
    """Tool for safe observe/act/verify visual automation loops."""

    def __init__(
        self,
        vision_analyzer: VisionAnalyzer,
        visual_grounder: VisualGrounder,
        vision_enabled: bool = True,
        max_iterations: int = 10,
        max_actions: int = 20,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self._analyzer = vision_analyzer
        self._grounder = visual_grounder
        self._enabled = vision_enabled
        self._max_iterations = max_iterations
        self._max_actions = max_actions
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "observe_act_verify"

    @property
    def description(self) -> str:
        return "Execute a safe observe/act/verify loop for visual automation tasks"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Description of the goal to achieve",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum number of observe/act/verify iterations",
                    "default": 10,
                },
                "max_actions": {
                    "type": "integer",
                    "description": "Maximum number of actions to perform",
                    "default": 20,
                },
                "timeout": {
                    "type": "number",
                    "description": "Maximum time in seconds for the loop",
                    "default": 120.0,
                },
            },
            "required": ["goal"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "exit_reason": {"type": "string"},
                "iterations": {"type": "integer"},
                "actions": {"type": "integer"},
                "time": {"type": "number"},
            },
        }

    @property
    def required_permissions(self) -> list:
        if not self._enabled:
            return []
        return ["vision.analyze"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return self._timeout + 30.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate tool arguments."""
        errors = []

        if not self._enabled:
            errors.append("Vision system is disabled by configuration")
            return False, errors

        goal = arguments.get("goal", "")
        if not goal or not goal.strip():
            errors.append("goal is required and cannot be empty")

        max_iter = arguments.get("max_iterations", self._max_iterations)
        if not isinstance(max_iter, int) or max_iter < 1:
            errors.append(f"Invalid max_iterations: {max_iter}")
        elif max_iter > 50:
            errors.append(f"max_iterations too large: {max_iter} (max 50)")

        max_act = arguments.get("max_actions", self._max_actions)
        if not isinstance(max_act, int) or max_act < 1:
            errors.append(f"Invalid max_actions: {max_act}")
        elif max_act > 100:
            errors.append(f"max_actions too large: {max_act} (max 100)")

        timeout_val = arguments.get("timeout", self._timeout)
        if not isinstance(timeout_val, (int, float)) or timeout_val <= 0:
            errors.append(f"Invalid timeout: {timeout_val}")
        elif timeout_val > 600:
            errors.append(f"timeout too large: {timeout_val} (max 600s)")

        return len(errors) == 0, errors

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the observe/act/verify tool.

        Note: This tool requires external callback functions to be
        provided through the agent context. The tool itself only
        manages the loop structure and validation.
        """
        start = time.time()

        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="; ".join(errors),
                execution_time=time.time() - start,
            )

        config = LoopConfig(
            max_iterations=arguments.get("max_iterations", self._max_iterations),
            max_actions=arguments.get("max_actions", self._max_actions),
            timeout=arguments.get("timeout", self._timeout),
            max_retries=self._max_retries,
        )

        loop = ObserveActVerifyLoop(
            vision_analyzer=self._analyzer,
            visual_grounder=self._grounder,
            config=config,
        )

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=(
                f"Observe/Act/Verify loop configured for goal: {arguments['goal']}\n"
                f"Max iterations: {config.max_iterations}\n"
                f"Max actions: {config.max_actions}\n"
                f"Timeout: {config.timeout}s\n"
                "Note: Loop execution requires agent integration with callbacks."
            ),
            execution_time=time.time() - start,
            metadata={
                "goal": arguments["goal"],
                "config": config.to_dict(),
                "loop_initialized": True,
            },
        )
