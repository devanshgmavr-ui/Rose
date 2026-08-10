"""Verification system for task completion."""

import logging
from typing import Optional, List

from .models import Task, Plan
from ..tools.base import ToolRequest, ToolResult
from ..tools.router import ToolRouter

logger = logging.getLogger(__name__)


class Verifier:
    """Verifies task completion based on criteria."""

    def __init__(self, tool_router: Optional[ToolRouter] = None):
        self._tool_router = tool_router

    def verify(self, task: Task) -> bool:
        if not task.plan:
            logger.warning("No plan to verify against")
            return False

        criteria = task.plan.completion_criteria
        if not criteria:
            logger.info("No completion criteria defined")
            return len(task.completed_steps) > 0

        all_passed = True
        for criterion in criteria:
            passed = self._check_criterion(task, criterion)
            if not passed:
                all_passed = False
                logger.info(f"Criterion not met: {criterion}")

        if all_passed:
            self._event_log(task, "verification_passed")
        else:
            self._event_log(task, "verification_failed")

        return all_passed

    def _check_criterion(self, task: Task, criterion: str) -> bool:
        criterion_lower = criterion.lower()

        if "file exists" in criterion_lower or "file created" in criterion_lower:
            return self._verify_file_exists(task, criterion)

        if "output" in criterion_lower or "result" in criterion_lower:
            return self._verify_output(task, criterion)

        if "completed" in criterion_lower or "done" in criterion_lower:
            return len(task.completed_steps) > 0 and len(task.failed_steps) == 0

        return True

    def _verify_file_exists(self, task: Task, criterion: str) -> bool:
        if not self._tool_router:
            return True

        import re
        path_match = re.search(r'["\']?([\w/.\-]+)["\']?', criterion)
        if not path_match:
            return True

        path = path_match.group(1)
        result = self._tool_router.execute_tool(
            "filesystem", {"action": "read", "path": path}
        )
        return result.success

    def _verify_output(self, task: Task, criterion: str) -> bool:
        if not task.result:
            return False
        return len(task.result) > 0

    def _event_log(self, task: Task, event: str):
        logger.info(f"Task {task.task_id}: {event}")

    def verify_step(self, step_result: str, expected: str) -> bool:
        if not expected:
            return True
        if not step_result:
            return False
        expected_lower = expected.lower()
        result_lower = step_result.lower()
        return expected_lower in result_lower or result_lower in expected_lower
