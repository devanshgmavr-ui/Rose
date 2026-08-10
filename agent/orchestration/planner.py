"""Planner for generating structured plans from user requests."""

import json
import logging
import time
from typing import Optional, Dict, Any, List

from .models import Plan, PlanStep, Task

logger = logging.getLogger(__name__)

PLANNING_PROMPT_TEMPLATE = """You are a task planning assistant. Given a user request, create a structured execution plan.

Available tools:
- filesystem: Read, write, list files in workspace. Actions: list, read, write.
- python_sandbox: Execute Python code in sandbox. Arguments: code (string).
- cli: Execute CLI commands (DISABLED - do not use).

For each step, provide:
- step_id: unique identifier (step_1, step_2, ...)
- description: what this step does
- tool_name: which tool to use (filesystem, python_sandbox, or "generate" for code generation)
- action: tool action (for filesystem: list/read/write)
- arguments: tool arguments
- dependencies: list of step_ids this depends on
- expected_result: what success looks like

Also provide completion_criteria: a list of conditions that must be true for the task to be complete.

IMPORTANT: Return ONLY valid JSON, no other text.

User request: {user_request}

Respond with a JSON object containing:
{{
    "objective": "clear objective description",
    "steps": [
        {{
            "step_id": "step_1",
            "description": "...",
            "tool_name": "...",
            "action": "...",
            "arguments": {{}},
            "dependencies": [],
            "expected_result": "..."
        }}
    ],
    "completion_criteria": ["..."]
}}"""


class Planner:
    """Generates structured plans from user requests using LLM."""

    def __init__(self, llm_provider=None, max_plan_steps: int = 12):
        self._llm_provider = llm_provider
        self._max_plan_steps = max_plan_steps

    def set_llm_provider(self, provider):
        self._llm_provider = provider

    def create_plan(self, user_request: str, task_id: str = "") -> Plan:
        if not task_id:
            task_id = f"task_{int(time.time())}"

        if self._llm_provider is None:
            return self._create_fallback_plan(user_request, task_id)

        prompt = PLANNING_PROMPT_TEMPLATE.format(user_request=user_request)

        try:
            response = self._llm_provider.generate(prompt, max_tokens=2048)
            plan = self._parse_llm_plan(response.text, task_id, user_request)
            if plan:
                return plan
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using fallback")

        return self._create_fallback_plan(user_request, task_id)

    def _parse_llm_plan(self, llm_output: str, task_id: str, user_request: str) -> Optional[Plan]:
        try:
            text = llm_output.strip()
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                text = text[start:end].strip()

            data = json.loads(text)
            return self._build_plan_from_dict(data, task_id, user_request)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM plan: {e}")
            return None

    def _build_plan_from_dict(self, data: dict, task_id: str, user_request: str) -> Optional[Plan]:
        objective = data.get("objective", user_request)
        steps_data = data.get("steps", [])
        criteria = data.get("completion_criteria", [])

        if not steps_data:
            return None

        steps = []
        for s in steps_data:
            step = PlanStep(
                step_id=s.get("step_id", f"step_{len(steps)+1}"),
                description=s.get("description", ""),
                action=s.get("action", ""),
                tool_name=s.get("tool_name", ""),
                arguments=s.get("arguments", {}),
                dependencies=s.get("dependencies", []),
                expected_result=s.get("expected_result", ""),
            )
            steps.append(step)

        if len(steps) > self._max_plan_steps:
            steps = steps[:self._max_plan_steps]

        return Plan(
            task_id=task_id,
            objective=objective,
            steps=steps,
            completion_criteria=criteria,
        )

    def _create_fallback_plan(self, user_request: str, task_id: str) -> Plan:
        steps = [
            PlanStep(
                step_id="step_1",
                description="Process and understand the user request",
                action="analyze",
                tool_name="generate",
                arguments={"request": user_request},
                dependencies=[],
                expected_result="Request understood and broken into actionable items",
            ),
            PlanStep(
                step_id="step_2",
                description="Generate solution based on the request",
                action="generate",
                tool_name="generate",
                arguments={"request": user_request},
                dependencies=["step_1"],
                expected_result="Solution generated",
            ),
            PlanStep(
                step_id="step_3",
                description="Save output to workspace if applicable",
                action="save",
                tool_name="filesystem",
                arguments={"action": "write", "path": "output.txt", "content": "placeholder"},
                dependencies=["step_2"],
                expected_result="Output saved successfully",
            ),
            PlanStep(
                step_id="step_4",
                description="Verify task completion",
                action="verify",
                tool_name="generate",
                arguments={},
                dependencies=["step_3"],
                expected_result="Task verified as complete",
            ),
        ]

        if len(steps) > self._max_plan_steps:
            steps = steps[:self._max_plan_steps]

        return Plan(
            task_id=task_id,
            objective=user_request,
            steps=steps,
            completion_criteria=["Task result is available and valid"],
        )
