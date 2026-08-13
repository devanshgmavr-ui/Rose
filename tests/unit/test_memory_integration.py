"""Phase 14 - Memory Integration & Event Streaming Tests.

Tests memory integration for autonomous execution, task result storage,
tool execution history, and memory-augmented context.
"""

import time
import tempfile
import os
import pytest
from unittest.mock import MagicMock

from agent.orchestration.memory_integration import (
    MemoryIntegration, TaskMemoryRecord, ToolExecutionRecord,
)
from agent.memory.long_term import LongTermMemory
from agent.memory.base import MemoryRecord, MemoryType


class TestTaskMemoryRecord:
    def test_creation(self):
        record = TaskMemoryRecord(
            task_id="task1",
            objective="Open browser",
            result="Browser opened",
            success=True,
            tools_used=["browser"],
            steps_count=2,
            duration=5.0,
        )
        assert record.success is True
        assert record.task_id == "task1"

    def test_to_dict(self):
        record = TaskMemoryRecord(
            task_id="task1",
            objective="Test",
            result="Done",
            success=True,
            tools_used=["filesystem"],
            steps_count=1,
            duration=1.0,
            error="",
        )
        d = record.to_dict()
        assert d["task_id"] == "task1"
        assert d["success"] is True
        assert "tools_used" in d


class TestToolExecutionRecord:
    def test_creation(self):
        record = ToolExecutionRecord(
            tool_name="filesystem",
            action="read",
            success=True,
            duration=0.5,
        )
        assert record.success is True
        assert record.tool_name == "filesystem"

    def test_to_dict(self):
        record = ToolExecutionRecord(
            tool_name="mouse",
            action="click",
            success=False,
            error="Permission denied",
        )
        d = record.to_dict()
        assert d["tool_name"] == "mouse"
        assert d["success"] is False


class TestMemoryIntegrationWithoutMemory:
    def test_init(self):
        integration = MemoryIntegration()
        assert integration.is_available is False

    def test_store_task_result_no_memory(self):
        integration = MemoryIntegration()
        result = integration.store_task_result(
            task_id="t1",
            objective="test",
            result="done",
            success=True,
            tools_used=[],
            steps_count=1,
            duration=1.0,
        )
        assert result is False
        assert len(integration._task_history) == 1

    def test_store_tool_execution_no_memory(self):
        integration = MemoryIntegration()
        result = integration.store_tool_execution(
            tool_name="filesystem",
            action="read",
            success=True,
        )
        assert result is False
        assert len(integration._tool_history) == 1

    def test_retrieve_no_memory(self):
        integration = MemoryIntegration()
        memories = integration.retrieve_relevant_memories("test query")
        assert memories == []

    def test_get_task_history(self):
        integration = MemoryIntegration()
        integration.store_task_result(
            task_id="t1", objective="a", result="r", success=True,
            tools_used=[], steps_count=1, duration=1.0,
        )
        history = integration.get_task_history()
        assert len(history) == 1

    def test_get_tool_history(self):
        integration = MemoryIntegration()
        integration.store_tool_execution("fs", "read", True)
        integration.store_tool_execution("fs", "write", False, error="fail")
        history = integration.get_tool_history()
        assert len(history) == 2

    def test_tool_stats(self):
        integration = MemoryIntegration()
        integration.store_tool_execution("fs", "read", True)
        integration.store_tool_execution("fs", "write", True)
        integration.store_tool_execution("mouse", "click", False, error="denied")
        stats = integration.get_tool_stats()
        assert stats["total_executions"] == 3
        assert stats["success"] == 2
        assert stats["failed"] == 1
        assert "fs" in stats["by_tool"]

    def test_task_stats(self):
        integration = MemoryIntegration()
        integration.store_task_result(
            task_id="t1", objective="a", result="r", success=True,
            tools_used=["fs"], steps_count=2, duration=5.0,
        )
        integration.store_task_result(
            task_id="t2", objective="b", result="err", success=False,
            tools_used=[], steps_count=1, duration=1.0, error="failed",
        )
        stats = integration.get_task_stats()
        assert stats["total_tasks"] == 2
        assert stats["success"] == 1
        assert stats["failed"] == 1
        assert stats["total_duration"] == 6.0

    def test_memory_augmented_context_empty(self):
        integration = MemoryIntegration()
        ctx = integration.get_memory_augmented_context("test")
        assert ctx == ""

    def test_get_tool_history_limit(self):
        integration = MemoryIntegration()
        for i in range(5):
            integration.store_tool_execution("fs", f"action_{i}", True)
        history = integration.get_tool_history(limit=3)
        assert len(history) == 3


class TestMemoryIntegrationWithMemory:
    def test_store_task_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = LongTermMemory(db_path=db_path)
            integration = MemoryIntegration(long_term_memory=memory)

            result = integration.store_task_result(
                task_id="t1",
                objective="Open Notepad",
                result="Notepad opened",
                success=True,
                tools_used=["window", "keyboard"],
                steps_count=3,
                duration=10.5,
                session_id="sess1",
            )
            assert result is True
            assert integration.is_available is True

    def test_store_tool_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = LongTermMemory(db_path=db_path)
            integration = MemoryIntegration(long_term_memory=memory)

            result = integration.store_tool_execution(
                tool_name="filesystem",
                action="write",
                success=True,
                duration=0.3,
            )
            assert result is True

    def test_retrieve_memories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = LongTermMemory(db_path=db_path)
            integration = MemoryIntegration(long_term_memory=memory)

            integration.store_task_result(
                task_id="t1",
                objective="Open Notepad",
                result="Success",
                success=True,
                tools_used=["window"],
                steps_count=2,
                duration=5.0,
            )

            memories = integration.retrieve_relevant_memories("Notepad")
            assert len(memories) >= 1

    def test_memory_augmented_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = LongTermMemory(db_path=db_path)
            integration = MemoryIntegration(long_term_memory=memory)

            integration.store_task_result(
                task_id="t1",
                objective="Open browser and navigate",
                result="Browser opened",
                success=True,
                tools_used=["browser"],
                steps_count=3,
                duration=8.0,
            )

            ctx = integration.get_memory_augmented_context("browser")
            assert "Relevant memories" in ctx

    def test_store_failed_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = LongTermMemory(db_path=db_path)
            integration = MemoryIntegration(long_term_memory=memory)

            result = integration.store_task_result(
                task_id="t2",
                objective="Click button",
                result="Failed",
                success=False,
                tools_used=["mouse"],
                steps_count=1,
                duration=2.0,
                error="Permission denied",
            )
            assert result is True

    def test_store_tool_execution_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = LongTermMemory(db_path=db_path)
            integration = MemoryIntegration(long_term_memory=memory)

            result = integration.store_tool_execution(
                tool_name="mouse",
                action="click",
                success=False,
                duration=0.1,
                error="Blocked by OS",
                context="Trying to click at (100, 200)",
            )
            assert result is True


class TestMemoryIntegrationEdgeCases:
    def test_tool_stats_empty(self):
        integration = MemoryIntegration()
        stats = integration.get_tool_stats()
        assert stats["total_executions"] == 0
        assert stats["success_rate"] == 0

    def test_task_stats_empty(self):
        integration = MemoryIntegration()
        stats = integration.get_task_stats()
        assert stats["total_tasks"] == 0
        assert stats["avg_duration"] == 0

    def test_get_task_history_limit(self):
        integration = MemoryIntegration()
        for i in range(10):
            integration.store_task_result(
                task_id=f"t{i}", objective=f"obj{i}", result=f"res{i}",
                success=True, tools_used=[], steps_count=1, duration=1.0,
            )
        history = integration.get_task_history(limit=5)
        assert len(history) == 5

    def test_memory_augmented_context_long_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = LongTermMemory(db_path=db_path)
            integration = MemoryIntegration(long_term_memory=memory)

            long_result = "x" * 1000
            integration.store_task_result(
                task_id="t1",
                objective="Test with long content",
                result=long_result,
                success=True,
                tools_used=[],
                steps_count=1,
                duration=1.0,
            )

            ctx = integration.get_memory_augmented_context("test")
            assert len(ctx) < 2000
