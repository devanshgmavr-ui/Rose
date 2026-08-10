"""Configurable resource limits for orchestration."""

from dataclasses import dataclass


@dataclass
class OrchestrationLimits:
    max_plan_steps: int = 12
    max_tool_calls: int = 20
    max_replans: int = 3
    max_step_retries: int = 2
    max_task_duration: float = 300.0
    max_repeated_actions: int = 5

    def to_dict(self) -> dict:
        return {
            "max_plan_steps": self.max_plan_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_replans": self.max_replans,
            "max_step_retries": self.max_step_retries,
            "max_task_duration": self.max_task_duration,
            "max_repeated_actions": self.max_repeated_actions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OrchestrationLimits":
        return cls(
            max_plan_steps=data.get("max_plan_steps", 12),
            max_tool_calls=data.get("max_tool_calls", 20),
            max_replans=data.get("max_replans", 3),
            max_step_retries=data.get("max_step_retries", 2),
            max_task_duration=data.get("max_task_duration", 300.0),
            max_repeated_actions=data.get("max_repeated_actions", 5),
        )
