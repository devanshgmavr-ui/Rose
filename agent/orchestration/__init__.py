"""Orchestration system for task planning and execution.

Stage 4.1 - Natural Language Tool Planning.
Phase 9 - Autonomous Tool Selection & Prompt Execution.
"""

from .models import (
    TaskStatus,
    StepStatus,
    Decision,
    PlanStep,
    Plan,
    Task,
)
from .state import StateMachine
from .limits import OrchestrationLimits
from .events import EventType, TaskEvent, EventLogger
from .planner import Planner
from .enhanced_planner import EnhancedPlanner
from .tool_catalog import ToolMetadata, build_tool_catalog, get_tools_for_request
from .tool_selector import IntentClassifier, ToolSelector, ToolMatch
from .validator import PlanValidator
from .executor import TaskExecutor
from .verifier import Verifier
from .persistence import TaskPersistence
from .autonomous import AutonomousTaskManager, TaskProgress, TaskResult

# Phase 9 imports
from .capability_analyzer import CapabilityAnalyzer, Capability, CapabilityAnalysis
from .tool_scorer import ToolScorer, ToolScore, SelectionResult
from .task_state import (
    AutonomousTaskState, TaskPhase, TaskConstraints,
    StepExecution, ExecutionTrace,
)
from .autonomous_loop import AutonomousLoop

__all__ = [
    # Existing
    "TaskStatus",
    "StepStatus",
    "Decision",
    "PlanStep",
    "Plan",
    "Task",
    "StateMachine",
    "OrchestrationLimits",
    "EventType",
    "TaskEvent",
    "EventLogger",
    "Planner",
    "EnhancedPlanner",
    "ToolMetadata",
    "build_tool_catalog",
    "get_tools_for_request",
    "IntentClassifier",
    "ToolSelector",
    "ToolMatch",
    "PlanValidator",
    "TaskExecutor",
    "Verifier",
    "TaskPersistence",
    "AutonomousTaskManager",
    "TaskProgress",
    "TaskResult",
    # Phase 9
    "CapabilityAnalyzer",
    "Capability",
    "CapabilityAnalysis",
    "ToolScorer",
    "ToolScore",
    "SelectionResult",
    "AutonomousTaskState",
    "TaskPhase",
    "TaskConstraints",
    "StepExecution",
    "ExecutionTrace",
    "AutonomousLoop",
]
