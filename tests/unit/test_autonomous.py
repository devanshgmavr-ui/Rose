"""Unit tests for Stage 4.3 Multi-Step Autonomous Tasks."""

import pytest
from unittest.mock import MagicMock, patch

from agent.orchestration.autonomous import AutonomousTaskManager, TaskProgress, TaskResult
from agent.orchestration.models import Task, Plan, PlanStep, TaskStatus
from agent.orchestration.limits import OrchestrationLimits
from agent.tools.router import ToolRouter
from agent.tools.registry import ToolRegistry
from agent.tools.permissions import PermissionManager
from agent.tools.audit import AuditLogger


class TestTaskProgress:
    def test_creation(self):
        progress = TaskProgress(
            task_id="t1",
            status="running",
            objective="Test",
            total_steps=5,
            completed_steps=2,
            failed_steps=0,
            current_step="Step 2",
            elapsed_time=10.0,
            tool_calls=3,
        )
        assert progress.task_id == "t1"
        assert progress.completed_steps == 2

    def test_to_dict(self):
        progress = TaskProgress(
            task_id="t1",
            status="running",
            objective="Test",
            total_steps=5,
            completed_steps=2,
            failed_steps=0,
            current_step="",
            elapsed_time=10.0,
            tool_calls=3,
        )
        d = progress.to_dict()
        assert d["task_id"] == "t1"
        assert d["total_steps"] == 5

    def test_to_text(self):
        progress = TaskProgress(
            task_id="t1",
            status="running",
            objective="Test objective",
            total_steps=5,
            completed_steps=2,
            failed_steps=1,
            current_step="Doing something",
            elapsed_time=15.5,
            tool_calls=4,
        )
        text = progress.to_text()
        assert "t1" in text
        assert "Test objective" in text
        assert "2/5" in text
        assert "15.5s" in text

    def test_to_text_with_error(self):
        progress = TaskProgress(
            task_id="t1",
            status="failed",
            objective="Test",
            total_steps=5,
            completed_steps=0,
            failed_steps=1,
            current_step="",
            elapsed_time=5.0,
            tool_calls=1,
            error="Something failed",
        )
        text = progress.to_text()
        assert "Something failed" in text


class TestTaskResult:
    def test_creation(self):
        result = TaskResult(
            success=True,
            task_id="t1",
            objective="Test",
            result="Done",
        )
        assert result.success is True
        assert result.result == "Done"

    def test_to_dict(self):
        result = TaskResult(
            success=True,
            task_id="t1",
            objective="Test",
            result="Done",
            total_steps=3,
            completed_steps=3,
            total_time=10.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["total_steps"] == 3

    def test_to_text_success(self):
        result = TaskResult(
            success=True,
            task_id="t1",
            objective="Test",
            result="Task completed successfully",
            total_steps=3,
            completed_steps=3,
            total_time=10.0,
        )
        text = result.to_text()
        assert "SUCCESS" in text
        assert "Test" in text

    def test_to_text_failure(self):
        result = TaskResult(
            success=False,
            task_id="t1",
            objective="Test",
            result="",
            error="Failed",
            total_steps=3,
            completed_steps=1,
            failed_steps=1,
            total_time=5.0,
        )
        text = result.to_text()
        assert "FAILED" in text
        assert "Failed" in text


class TestAutonomousTaskManager:
    def _make_manager(self):
        registry = ToolRegistry()
        perm = PermissionManager()
        audit = AuditLogger()
        router = ToolRouter(registry, perm, audit)
        return AutonomousTaskManager(tool_router=router)

    def test_init(self):
        manager = self._make_manager()
        assert manager._tool_router is not None

    def test_set_progress_callback(self):
        manager = self._make_manager()
        callback = MagicMock()
        manager.set_progress_callback(callback)
        assert manager._progress_callback is callback

    def test_cancel(self):
        manager = self._make_manager()
        manager.cancel()
        assert manager._cancelled is True

    def test_execute_simple(self):
        manager = self._make_manager()
        result = manager.execute("List files in workspace")
        assert isinstance(result, TaskResult)
        assert result.task_id.startswith("task_")

    def test_execute_custom_task_id(self):
        manager = self._make_manager()
        result = manager.execute("Do something", task_id="my_task")
        assert result.task_id == "my_task"

    def test_execute_with_progress_callback(self):
        manager = self._make_manager()
        progress_calls = []
        manager.set_progress_callback(lambda p: progress_calls.append(p))
        result = manager.execute("Take a screenshot")
        assert len(progress_calls) > 0

    def test_get_task(self):
        manager = self._make_manager()
        result = manager.execute("Test task", task_id="get_test")
        task = manager.get_task("get_test")
        assert task is None

    def test_get_progress(self):
        manager = self._make_manager()
        progress = manager.get_progress("nonexistent")
        assert progress is None

    def test_list_active_tasks(self):
        manager = self._make_manager()
        tasks = manager.list_active_tasks()
        assert isinstance(tasks, list)

    def test_execute_generates_plan(self):
        manager = self._make_manager()
        result = manager.execute("Open a browser and navigate to a website")
        assert result.total_steps > 0

    def test_execute_failure_recovery(self):
        manager = self._make_manager()
        result = manager.execute("Execute invalid code that will fail")
        assert isinstance(result, TaskResult)


class TestTaskManagerIntegration:
    def _make_manager(self):
        registry = ToolRegistry()
        perm = PermissionManager()
        audit = AuditLogger()
        router = ToolRouter(registry, perm, audit)
        return AutonomousTaskManager(tool_router=router)

    def test_full_task_flow(self):
        manager = self._make_manager()
        progress_log = []
        manager.set_progress_callback(lambda p: progress_log.append(p))

        result = manager.execute("Get system information")
        assert isinstance(result, TaskResult)

    def test_multiple_tasks(self):
        manager = self._make_manager()
        r1 = manager.execute("Task 1", task_id="t1")
        r2 = manager.execute("Task 2", task_id="t2")
        assert r1.task_id == "t1"
        assert r2.task_id == "t2"

    def test_result_serialization(self):
        manager = self._make_manager()
        result = manager.execute("Test")
        d = result.to_dict()
        assert "success" in d
        assert "task_id" in d
        assert "total_time" in d
