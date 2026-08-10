"""Plan validation for orchestration."""

from typing import List, Tuple, Optional
from .models import Plan, PlanStep


class PlanValidator:
    """Validates plans before execution."""

    def __init__(self, available_tools: Optional[List[str]] = None, max_steps: int = 12):
        self._available_tools = available_tools or ["filesystem", "python_sandbox", "generate"]
        self._max_steps = max_steps

    def validate(self, plan: Plan) -> Tuple[bool, List[str]]:
        errors = []

        if not plan.task_id:
            errors.append("Missing task_id")
        if not plan.objective:
            errors.append("Missing objective")
        if not plan.steps:
            errors.append("Plan has no steps")

        if len(plan.steps) > self._max_steps:
            errors.append(f"Plan exceeds maximum steps ({self._max_steps})")

        step_ids = set()
        for step in plan.steps:
            if step.step_id in step_ids:
                errors.append(f"Duplicate step ID: {step.step_id}")
            step_ids.add(step.step_id)

            if not step.description:
                errors.append(f"Step {step.step_id} has no description")
            if not step.tool_name:
                errors.append(f"Step {step.step_id} has no tool_name")
            elif step.tool_name not in self._available_tools:
                errors.append(
                    f"Step {step.step_id} uses unknown tool: {step.tool_name}"
                )

        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(
                        f"Step {step.step_id} depends on unknown step: {dep}"
                    )

        if self._has_circular_dependencies(plan.steps):
            errors.append("Circular dependency detected")

        return len(errors) == 0, errors

    def _has_circular_dependencies(self, steps: List[PlanStep]) -> bool:
        step_map = {s.step_id: s for s in steps}
        visited = set()
        rec_stack = set()

        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            rec_stack.discard(step_id)
            return False

        for step in steps:
            if step.step_id not in visited:
                if dfs(step.step_id):
                    return True
        return False

    def validate_step_arguments(self, step: PlanStep) -> Tuple[bool, List[str]]:
        errors = []
        if step.tool_name == "filesystem":
            action = step.arguments.get("action")
            if action not in ("list", "read", "write"):
                errors.append(f"Invalid filesystem action: {action}")
            if action in ("read", "write") and not step.arguments.get("path"):
                errors.append(f"Filesystem {action} requires path")
            if action == "write" and "content" not in step.arguments:
                errors.append("Filesystem write requires content")

        elif step.tool_name == "python_sandbox":
            if not step.arguments.get("code"):
                errors.append("Python sandbox requires code")

        return len(errors) == 0, errors
