"""Unit tests for orchestration system (Stage 1.4)."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.orchestration.models import (
    TaskStatus,
    StepStatus,
    Decision,
    PlanStep,
    Plan,
    Task,
)
from agent.orchestration.state import StateMachine
from agent.orchestration.limits import OrchestrationLimits
from agent.orchestration.events import EventType, TaskEvent, EventLogger
from agent.orchestration.planner import Planner
from agent.orchestration.validator import PlanValidator
from agent.orchestration.executor import TaskExecutor
from agent.orchestration.verifier import Verifier
from agent.orchestration.persistence import TaskPersistence
from agent.tools.base import ToolResult
from agent.tools.router import ToolRouter
from agent.tools.registry import ToolRegistry
from agent.tools.permissions import PermissionManager
from agent.tools.audit import AuditLogger
from agent.tools.filesystem_tool import FilesystemTool


class TestTaskModel:
    """Test Task data model."""

    def test_task_creation(self):
        task = Task(user_request="test request")
        assert task.user_request == "test request"
        assert task.status == TaskStatus.PENDING
        assert task.task_id is not None

    def test_task_to_dict(self):
        task = Task(user_request="hello")
        d = task.to_dict()
        assert d["user_request"] == "hello"
        assert d["status"] == "pending"

    def test_task_from_dict(self):
        original = Task(user_request="test")
        d = original.to_dict()
        restored = Task.from_dict(d)
        assert restored.user_request == original.user_request
        assert restored.status == original.status

    def test_task_status(self):
        task = Task()
        assert task.status == TaskStatus.PENDING
        task.status = TaskStatus.RUNNING
        assert task.status == TaskStatus.RUNNING

    def test_task_with_plan(self):
        plan = Plan(
            task_id="t1",
            objective="test",
            steps=[PlanStep(step_id="s1", description="step 1")],
        )
        task = Task(user_request="test", plan=plan)
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.plan is not None
        assert len(restored.plan.steps) == 1


class TestPlanModel:
    """Test Plan data model."""

    def test_plan_creation(self):
        plan = Plan(task_id="t1", objective="test objective")
        assert plan.task_id == "t1"
        assert plan.objective == "test objective"
        assert len(plan.steps) == 0

    def test_plan_step_creation(self):
        step = PlanStep(
            step_id="step_1",
            description="Do something",
            tool_name="filesystem",
            arguments={"action": "list"},
        )
        assert step.step_id == "step_1"
        assert step.tool_name == "filesystem"

    def test_plan_to_dict(self):
        plan = Plan(
            task_id="t1",
            objective="obj",
            steps=[PlanStep(step_id="s1", description="d")],
            completion_criteria=["c1"],
        )
        d = plan.to_dict()
        assert d["task_id"] == "t1"
        assert len(d["steps"]) == 1

    def test_plan_from_dict(self):
        original = Plan(
            task_id="t1",
            steps=[PlanStep(step_id="s1", description="step")],
        )
        d = original.to_dict()
        restored = Plan.from_dict(d)
        assert restored.task_id == "t1"
        assert len(restored.steps) == 1

    def test_plan_step_dependencies(self):
        step = PlanStep(
            step_id="s2",
            dependencies=["s1"],
        )
        assert "s1" in step.dependencies


class TestStateMachine:
    """Test state machine."""

    def test_initial_state(self):
        sm = StateMachine()
        assert sm.current_status == TaskStatus.PENDING

    def test_valid_transition(self):
        sm = StateMachine()
        ok, err = sm.transition(TaskStatus.PLANNING)
        assert ok is True
        assert sm.current_status == TaskStatus.PLANNING

    def test_invalid_transition(self):
        sm = StateMachine()
        ok, err = sm.transition(TaskStatus.COMPLETED)
        assert ok is False
        assert "Invalid" in err

    def test_multiple_transitions(self):
        sm = StateMachine()
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.RUNNING)
        sm.transition(TaskStatus.COMPLETED)
        assert sm.current_status == TaskStatus.COMPLETED

    def test_terminal_state(self):
        sm = StateMachine()
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.RUNNING)
        sm.transition(TaskStatus.COMPLETED)
        assert sm.is_terminal() is True

    def test_non_terminal_state(self):
        sm = StateMachine()
        sm.transition(TaskStatus.PLANNING)
        assert sm.is_terminal() is False

    def test_history(self):
        sm = StateMachine()
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.RUNNING)
        history = sm.get_history()
        assert len(history) == 2

    def test_replan_transition(self):
        sm = StateMachine()
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.RUNNING)
        ok, err = sm.transition(TaskStatus.PLANNING)
        assert ok is True

    def test_cancel_from_running(self):
        sm = StateMachine()
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.RUNNING)
        ok, err = sm.transition(TaskStatus.CANCELLED)
        assert ok is True

    def test_cannot_transition_from_terminal(self):
        sm = StateMachine()
        sm.transition(TaskStatus.PLANNING)
        sm.transition(TaskStatus.RUNNING)
        sm.transition(TaskStatus.COMPLETED)
        ok, err = sm.transition(TaskStatus.RUNNING)
        assert ok is False


class TestOrchestrationLimits:
    """Test configurable limits."""

    def test_default_limits(self):
        limits = OrchestrationLimits()
        assert limits.max_plan_steps == 12
        assert limits.max_tool_calls == 20
        assert limits.max_replans == 3
        assert limits.max_step_retries == 2
        assert limits.max_task_duration == 300.0

    def test_custom_limits(self):
        limits = OrchestrationLimits(max_plan_steps=5, max_tool_calls=10)
        assert limits.max_plan_steps == 5
        assert limits.max_tool_calls == 10

    def test_to_dict(self):
        limits = OrchestrationLimits()
        d = limits.to_dict()
        assert "max_plan_steps" in d
        assert "max_tool_calls" in d

    def test_from_dict(self):
        d = {"max_plan_steps": 8, "max_tool_calls": 15}
        limits = OrchestrationLimits.from_dict(d)
        assert limits.max_plan_steps == 8
        assert limits.max_tool_calls == 15


class TestEventLogger:
    """Test event logging."""

    def test_log_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            el = EventLogger(log_dir=tmpdir)
            event = TaskEvent(
                event_type=EventType.TASK_CREATED,
                task_id="t1",
                data={"key": "value"},
            )
            result = el.log(event)
            assert result is True

    def test_get_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            el = EventLogger(log_dir=tmpdir)
            for i in range(5):
                el.log(TaskEvent(event_type=EventType.STEP_COMPLETED, task_id="t1"))
            events = el.get_events(task_id="t1")
            assert len(events) == 5

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            el = EventLogger(log_dir=tmpdir)
            el.log(TaskEvent(event_type=EventType.TASK_CREATED, task_id="t1"))
            el.clear()
            events = el.get_events()
            assert len(events) == 0


class TestPlanner:
    """Test planner."""

    def test_fallback_plan(self):
        planner = Planner()
        plan = planner.create_plan("Calculate factorial of 5")
        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.objective == "Calculate factorial of 5"

    def test_plan_has_steps(self):
        planner = Planner()
        plan = planner.create_plan("Create a file")
        assert len(plan.steps) >= 2

    def test_plan_step_ids_unique(self):
        planner = Planner()
        plan = planner.create_plan("Test task")
        ids = [s.step_id for s in plan.steps]
        assert len(ids) == len(set(ids))

    def test_plan_with_max_steps(self):
        planner = Planner(max_plan_steps=3)
        plan = planner.create_plan("complex task")
        assert len(plan.steps) <= 3


class TestPlanValidator:
    """Test plan validation."""

    def test_valid_plan(self):
        validator = PlanValidator()
        plan = Plan(
            task_id="t1",
            objective="test",
            steps=[
                PlanStep(
                    step_id="s1",
                    description="Step 1",
                    tool_name="filesystem",
                    arguments={"action": "list"},
                ),
            ],
        )
        ok, errors = validator.validate(plan)
        assert ok is True
        assert len(errors) == 0

    def test_invalid_plan_no_objective(self):
        validator = PlanValidator()
        plan = Plan(task_id="t1", steps=[PlanStep(step_id="s1", description="d", tool_name="filesystem")])
        ok, errors = validator.validate(plan)
        assert ok is False

    def test_invalid_plan_duplicate_ids(self):
        validator = PlanValidator()
        plan = Plan(
            task_id="t1",
            objective="test",
            steps=[
                PlanStep(step_id="s1", description="d1", tool_name="filesystem"),
                PlanStep(step_id="s1", description="d2", tool_name="filesystem"),
            ],
        )
        ok, errors = validator.validate(plan)
        assert ok is False
        assert any("Duplicate" in e for e in errors)

    def test_invalid_plan_unknown_tool(self):
        validator = PlanValidator()
        plan = Plan(
            task_id="t1",
            objective="test",
            steps=[PlanStep(step_id="s1", description="d", tool_name="unknown_tool")],
        )
        ok, errors = validator.validate(plan)
        assert ok is False
        assert any("unknown tool" in e for e in errors)

    def test_invalid_plan_circular_deps(self):
        validator = PlanValidator()
        plan = Plan(
            task_id="t1",
            objective="test",
            steps=[
                PlanStep(step_id="s1", description="d1", tool_name="filesystem", dependencies=["s2"]),
                PlanStep(step_id="s2", description="d2", tool_name="filesystem", dependencies=["s1"]),
            ],
        )
        ok, errors = validator.validate(plan)
        assert ok is False
        assert any("Circular" in e for e in errors)

    def test_invalid_plan_missing_dep(self):
        validator = PlanValidator()
        plan = Plan(
            task_id="t1",
            objective="test",
            steps=[
                PlanStep(step_id="s1", description="d", tool_name="filesystem", dependencies=["nonexistent"]),
            ],
        )
        ok, errors = validator.validate(plan)
        assert ok is False
        assert any("unknown step" in e for e in errors)

    def test_validate_step_arguments(self):
        validator = PlanValidator()
        step = PlanStep(
            tool_name="filesystem",
            arguments={"action": "read"},
        )
        ok, errors = validator.validate_step_arguments(step)
        assert ok is False
        assert any("path" in e for e in errors)

    def test_max_steps_exceeded(self):
        validator = PlanValidator(max_steps=2)
        plan = Plan(
            task_id="t1",
            objective="test",
            steps=[
                PlanStep(step_id=f"s{i}", description=f"d{i}", tool_name="filesystem")
                for i in range(5)
            ],
        )
        ok, errors = validator.validate(plan)
        assert ok is False
        assert any("maximum" in e.lower() for e in errors)


class TestVerifier:
    """Test verification system."""

    def test_verify_with_no_criteria(self):
        verifier = Verifier()
        task = Task(
            task_id="t1",
            completed_steps=["s1"],
            plan=Plan(task_id="t1", completion_criteria=[]),
        )
        assert verifier.verify(task) is True

    def test_verify_with_completed_steps(self):
        verifier = Verifier()
        task = Task(
            task_id="t1",
            completed_steps=["s1"],
            plan=Plan(task_id="t1", completion_criteria=["Task completed"]),
        )
        assert verifier.verify(task) is True

    def test_verify_step_result(self):
        verifier = Verifier()
        assert verifier.verify_step("success", "success") is True
        assert verifier.verify_step("error", "success") is False
        assert verifier.verify_step("output", "") is True


class TestTaskPersistence:
    """Test task persistence."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPersistence(data_dir=tmpdir)
            task = Task(user_request="test task")
            tp.save_task(task)
            loaded = tp.load_task(task.task_id)
            assert loaded is not None
            assert loaded.user_request == "test task"

    def test_list_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPersistence(data_dir=tmpdir)
            tp.save_task(Task(user_request="task1"))
            tp.save_task(Task(user_request="task2"))
            tasks = tp.list_tasks()
            assert len(tasks) == 2

    def test_delete_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPersistence(data_dir=tmpdir)
            task = Task(user_request="to delete")
            tp.save_task(task)
            tp.delete_task(task.task_id)
            loaded = tp.load_task(task.task_id)
            assert loaded is None

    def test_task_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPersistence(data_dir=tmpdir)
            assert tp.get_task_count() == 0
            tp.save_task(Task(user_request="t1"))
            assert tp.get_task_count() == 1


class TestTaskExecutor:
    """Test task executor."""

    def _make_executor(self, tmpdir):
        registry = ToolRegistry()
        fs_tool = FilesystemTool(workspace_dir=tmpdir)
        registry.register(fs_tool)
        pm = PermissionManager()
        al = AuditLogger(log_dir=tmpdir)
        router = ToolRouter(registry=registry, permission_manager=pm, audit_logger=al)
        limits = OrchestrationLimits(max_tool_calls=10, max_step_retries=1, max_replans=1)
        el = EventLogger(log_dir=tmpdir)
        return TaskExecutor(tool_router=router, limits=limits, event_logger=el)

    def test_execute_simple_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = self._make_executor(tmpdir)
            plan = Plan(
                task_id="t1",
                objective="list files",
                steps=[
                    PlanStep(
                        step_id="s1",
                        description="List workspace files",
                        tool_name="filesystem",
                        arguments={"action": "list"},
                    ),
                ],
            )
            task = Task(task_id="t1", user_request="list files", plan=plan)
            result = executor.execute_task(task)
            assert result.status == TaskStatus.COMPLETED

    def test_execute_task_with_generate_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = self._make_executor(tmpdir)
            plan = Plan(
                task_id="t1",
                objective="generate content",
                steps=[
                    PlanStep(
                        step_id="s1",
                        description="Generate something",
                        tool_name="generate",
                        action="generate",
                    ),
                    PlanStep(
                        step_id="s2",
                        description="Save to file",
                        tool_name="filesystem",
                        arguments={"action": "write", "path": "out.txt", "content": "hello"},
                        dependencies=["s1"],
                    ),
                ],
            )
            task = Task(task_id="t1", user_request="generate and save", plan=plan)
            result = executor.execute_task(task)
            assert result.status == TaskStatus.COMPLETED

    def test_execute_task_no_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = self._make_executor(tmpdir)
            task = Task(task_id="t1", user_request="no plan")
            result = executor.execute_task(task)
            assert result.status == TaskStatus.FAILED
            assert "No plan" in result.error

    def test_execute_task_step_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = self._make_executor(tmpdir)
            plan = Plan(
                task_id="t1",
                objective="read nonexistent",
                steps=[
                    PlanStep(
                        step_id="s1",
                        description="Read file that doesn't exist",
                        tool_name="filesystem",
                        arguments={"action": "read", "path": "nonexistent.txt"},
                    ),
                ],
            )
            task = Task(task_id="t1", user_request="read file", plan=plan)
            result = executor.execute_task(task)
            assert result.status == TaskStatus.FAILED

    def test_cancel_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = self._make_executor(tmpdir)
            task = Task(task_id="t1", user_request="cancel me")
            result = executor.cancel_task(task)
            assert result.status == TaskStatus.CANCELLED


class TestOrchestrationIntegration:
    """Test orchestration components working together."""

    def test_planner_validator_integration(self):
        planner = Planner()
        validator = PlanValidator()
        plan = planner.create_plan("Create a file")
        ok, errors = validator.validate(plan)
        assert ok is True

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            fs_tool = FilesystemTool(workspace_dir=tmpdir)
            registry.register(fs_tool)
            pm = PermissionManager()
            al = AuditLogger(log_dir=tmpdir)
            router = ToolRouter(registry=registry, permission_manager=pm, audit_logger=al)
            limits = OrchestrationLimits(max_tool_calls=5, max_step_retries=1, max_replans=1)
            el = EventLogger(log_dir=tmpdir)
            executor = TaskExecutor(tool_router=router, limits=limits, event_logger=el)
            planner = Planner()
            validator = PlanValidator()

            plan = planner.create_plan("Create a test file in workspace")
            ok, errors = validator.validate(plan)
            assert ok is True

            task = Task(
                task_id="integration_test",
                user_request="Create a test file in workspace",
                plan=plan,
            )
            result = executor.execute_task(task)
            assert result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            assert result.tool_calls >= 0
