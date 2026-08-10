"""Enhanced planner for natural language tool planning.

Stage 4.1 - Natural Language Tool Planning.

Uses tool catalog metadata to generate structured plans
that map natural language requests to tool operations.
"""

import json
import logging
import time
from typing import Optional, Dict, Any, List

from .models import Plan, PlanStep, Task
from .tool_catalog import build_tool_catalog, get_tools_for_request, ToolMetadata

logger = logging.getLogger(__name__)

ENHANCED_PLANNING_PROMPT = """You are a task planning assistant for ROSE, a local AI agent. Given a user request, create a structured execution plan.

Available tools and their capabilities:

{tool_descriptions}

IMPORTANT RULES:
1. Only use tools listed above - never invent unavailable tools
2. Each step must use ONE tool
3. Steps should be ordered by dependencies
4. Include realistic arguments for each tool
5. Mark completion criteria clearly
6. Keep plans concise (max {max_steps} steps)

User request: {user_request}

Respond with a JSON object containing:
{{
    "objective": "clear objective description",
    "steps": [
        {{
            "step_id": "step_1",
            "description": "what this step does",
            "tool_name": "tool name from list above",
            "action": "specific action",
            "arguments": {{}},
            "dependencies": [],
            "expected_result": "what success looks like"
        }}
    ],
    "completion_criteria": ["condition that must be true"]
}}"""


class EnhancedPlanner:
    """Enhanced planner with tool catalog support."""

    def __init__(self, llm_provider=None, max_plan_steps: int = 12):
        self._llm_provider = llm_provider
        self._max_plan_steps = max_plan_steps
        self._catalog = build_tool_catalog()

    def set_llm_provider(self, provider):
        self._llm_provider = provider

    def create_plan(self, user_request: str, task_id: str = "") -> Plan:
        """Create a plan from user request."""
        if not task_id:
            task_id = f"task_{int(time.time())}"

        relevant_tools = get_tools_for_request(user_request, self._catalog)

        if self._llm_provider is None:
            return self._create_tool_based_plan(user_request, task_id, relevant_tools)

        tool_descriptions = self._format_tool_descriptions(relevant_tools)
        prompt = ENHANCED_PLANNING_PROMPT.format(
            tool_descriptions=tool_descriptions,
            max_steps=self._max_plan_steps,
            user_request=user_request,
        )

        try:
            response = self._llm_provider.generate(prompt, max_tokens=2048)
            plan = self._parse_llm_plan(response.text, task_id, user_request)
            if plan:
                return plan
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using tool-based plan")

        return self._create_tool_based_plan(user_request, task_id, relevant_tools)

    def _format_tool_descriptions(self, tools: List[ToolMetadata]) -> str:
        """Format tool metadata for the planning prompt."""
        lines = []
        for tool in tools:
            lines.append(f"Tool: {tool.name}")
            lines.append(f"  Description: {tool.description}")
            lines.append(f"  Actions: {', '.join(tool.actions)}")
            lines.append(f"  Permissions: {', '.join(tool.permissions)}")
            if tool.examples:
                ex = tool.examples[0]
                lines.append(f"  Example: {ex.get('request', 'N/A')}")
            lines.append("")
        return "\n".join(lines)

    def _parse_llm_plan(
        self, llm_output: str, task_id: str, user_request: str
    ) -> Optional[Plan]:
        """Parse LLM output into a Plan."""
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

    def _build_plan_from_dict(
        self, data: dict, task_id: str, user_request: str
    ) -> Optional[Plan]:
        """Build a Plan from parsed dictionary."""
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

    def _create_tool_based_plan(
        self, user_request: str, task_id: str, tools: List[ToolMetadata]
    ) -> Plan:
        """Create a plan based on tool catalog when LLM is unavailable."""
        steps = []

        if any("screenshot" in user_request.lower() for _ in [1]):
            steps.append(PlanStep(
                step_id="step_1",
                description="Capture screenshot of the screen",
                action="capture",
                tool_name="screen_capture",
                arguments={"region": "full"},
                dependencies=[],
                expected_result="Screenshot captured successfully",
            ))

        if any(w in user_request.lower() for w in ["browser", "website", "url", "navigate", "web"]):
            steps.append(PlanStep(
                step_id=f"step_{len(steps)+1}",
                description="Create browser session",
                action="create_session",
                tool_name="browser",
                arguments={},
                dependencies=[],
                expected_result="Browser session created",
            ))

        if any(w in user_request.lower() for w in ["file", "write", "save", "create"]):
            steps.append(PlanStep(
                step_id=f"step_{len(steps)+1}",
                description="Perform file operation",
                action="write",
                tool_name="filesystem",
                arguments={"action": "write", "path": "output.txt", "content": ""},
                dependencies=[],
                expected_result="File operation completed",
            ))

        if any(w in user_request.lower() for w in ["analyze", "look", "understand", "what"]):
            steps.append(PlanStep(
                step_id=f"step_{len(steps)+1}",
                description="Analyze the result",
                action="analyze",
                tool_name="vision_analyze",
                arguments={"action": "analyze", "image_path": "screenshot.png"},
                dependencies=[s.step_id for s in steps],
                expected_result="Analysis complete",
            ))

        if not steps:
            steps = [
                PlanStep(
                    step_id="step_1",
                    description="Understand and process the user request",
                    action="process",
                    tool_name="python_sandbox",
                    arguments={"code": f"# Process: {user_request}\nprint('Processing complete')"},
                    dependencies=[],
                    expected_result="Request processed",
                ),
            ]

        steps.append(PlanStep(
            step_id=f"step_{len(steps)+1}",
            description="Verify task completion",
            action="verify",
            tool_name="system_info",
            arguments={"action": "info"},
            dependencies=[s.step_id for s in steps],
            expected_result="Task verified",
        ))

        if len(steps) > self._max_plan_steps:
            steps = steps[:self._max_plan_steps]

        return Plan(
            task_id=task_id,
            objective=user_request,
            steps=steps,
            completion_criteria=["Task result is available and valid"],
        )
