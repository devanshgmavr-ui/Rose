"""Memory integration for autonomous task execution.

Phase 14 - Memory Integration & Event Streaming Foundation.

Provides:
- Task result storage in long-term memory
- Relevant memory retrieval for context
- Tool execution history tracking
- Memory-augmented planning context
- Task completion memory recording
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from ..memory.long_term import LongTermMemory
from ..memory.base import MemoryRecord, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class TaskMemoryRecord:
    """A memory record for a completed task."""
    task_id: str
    objective: str
    result: str
    success: bool
    tools_used: List[str]
    steps_count: int
    duration: float
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "result": self.result,
            "success": self.success,
            "tools_used": self.tools_used,
            "steps_count": self.steps_count,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "error": self.error,
        }


@dataclass
class ToolExecutionRecord:
    """A memory record for a tool execution."""
    tool_name: str
    action: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    error: str = ""
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "action": self.action,
            "success": self.success,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "error": self.error,
        }


class MemoryIntegration:
    """Integrates memory with autonomous task execution.

    Provides:
    - Task result storage
    - Relevant memory retrieval
    - Tool execution history
    - Memory-augmented context
    """

    def __init__(self, long_term_memory: Optional[LongTermMemory] = None):
        self._memory = long_term_memory
        self._tool_history: List[ToolExecutionRecord] = []
        self._task_history: List[TaskMemoryRecord] = []

    @property
    def is_available(self) -> bool:
        """Check if memory integration is available."""
        return self._memory is not None

    def store_task_result(
        self,
        task_id: str,
        objective: str,
        result: str,
        success: bool,
        tools_used: List[str],
        steps_count: int,
        duration: float,
        session_id: Optional[str] = None,
        error: str = "",
    ) -> bool:
        """Store a task result in long-term memory.

        Args:
            task_id: Unique task identifier.
            objective: Task objective.
            result: Task result or error message.
            success: Whether the task succeeded.
            tools_used: List of tools used during execution.
            steps_count: Number of steps executed.
            duration: Total execution duration.
            session_id: Optional session ID.
            error: Error message if task failed.

        Returns:
            True if stored successfully.
        """
        record = TaskMemoryRecord(
            task_id=task_id,
            objective=objective,
            result=result,
            success=success,
            tools_used=tools_used,
            steps_count=steps_count,
            duration=duration,
            session_id=session_id,
            error=error,
        )
        self._task_history.append(record)

        if not self._memory:
            return False

        try:
            content = (
                f"Task '{objective}' {'succeeded' if success else 'failed'}. "
                f"Result: {result[:500]}. "
                f"Tools used: {', '.join(tools_used) if tools_used else 'none'}. "
                f"Steps: {steps_count}, Duration: {duration:.1f}s."
            )

            memory_record = MemoryRecord(
                content=content,
                memory_type=MemoryType.FACT,
                importance=0.8 if success else 0.6,
                confidence=0.9,
                session_id=session_id,
                metadata={
                    "task_id": task_id,
                    "success": success,
                    "tools_used": tools_used,
                    "steps_count": steps_count,
                    "duration": duration,
                    "error": error,
                },
            )

            return self._memory.store(memory_record)

        except Exception as e:
            logger.warning(f"Failed to store task result in memory: {e}")
            return False

    def store_tool_execution(
        self,
        tool_name: str,
        action: str,
        success: bool,
        duration: float = 0.0,
        error: str = "",
        context: str = "",
    ) -> bool:
        """Store a tool execution in memory.

        Args:
            tool_name: Name of the tool.
            action: Action performed.
            success: Whether the execution succeeded.
            duration: Execution duration.
            error: Error message if failed.
            context: Additional context.

        Returns:
            True if stored successfully.
        """
        record = ToolExecutionRecord(
            tool_name=tool_name,
            action=action,
            success=success,
            duration=duration,
            error=error,
            context=context,
        )
        self._tool_history.append(record)

        if not self._memory:
            return False

        try:
            content = (
                f"Tool '{tool_name}' action '{action}' "
                f"{'succeeded' if success else 'failed'}."
            )
            if error:
                content += f" Error: {error[:200]}."
            if context:
                content += f" Context: {context[:200]}."

            memory_record = MemoryRecord(
                content=content,
                memory_type=MemoryType.FACT,
                importance=0.5 if success else 0.7,
                confidence=0.8,
                metadata={
                    "tool_name": tool_name,
                    "action": action,
                    "success": success,
                    "duration": duration,
                },
            )

            return self._memory.store(memory_record)

        except Exception as e:
            logger.warning(f"Failed to store tool execution in memory: {e}")
            return False

    def retrieve_relevant_memories(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of memory records as dictionaries.
        """
        if not self._memory:
            return []

        try:
            records = self._memory.retrieve(query=query, limit=limit)
            return [r.to_dict() for r in records]
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            return []

    def get_task_history(self, limit: int = 10) -> List[TaskMemoryRecord]:
        """Get recent task execution history."""
        return list(self._task_history)[-limit:]

    def get_tool_history(self, limit: int = 20) -> List[ToolExecutionRecord]:
        """Get recent tool execution history."""
        return list(self._tool_history)[-limit:]

    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        history = list(self._tool_history)
        total = len(history)
        success = sum(1 for r in history if r.success)
        failed = total - success

        by_tool = {}
        for record in history:
            tool = record.tool_name
            if tool not in by_tool:
                by_tool[tool] = {"total": 0, "success": 0, "failed": 0}
            by_tool[tool]["total"] += 1
            if record.success:
                by_tool[tool]["success"] += 1
            else:
                by_tool[tool]["failed"] += 1

        return {
            "total_executions": total,
            "success": success,
            "failed": failed,
            "success_rate": success / max(total, 1),
            "by_tool": by_tool,
        }

    def get_task_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        history = list(self._task_history)
        total = len(history)
        success = sum(1 for r in history if r.success)
        failed = total - success

        total_duration = sum(r.duration for r in history)
        avg_duration = total_duration / max(total, 1)

        return {
            "total_tasks": total,
            "success": success,
            "failed": failed,
            "success_rate": success / max(total, 1),
            "total_duration": total_duration,
            "avg_duration": avg_duration,
        }

    def get_memory_augmented_context(
        self,
        user_prompt: str,
        max_memories: int = 5,
    ) -> str:
        """Get memory-augmented context for planning.

        Args:
            user_prompt: The user's request.
            max_memories: Maximum memories to include.

        Returns:
            Context string with relevant memories.
        """
        memories = self.retrieve_relevant_memories(user_prompt, limit=max_memories)

        if not memories:
            return ""

        context_parts = ["Relevant memories:"]
        for mem in memories:
            content = mem.get("content", "")
            if content:
                context_parts.append(f"- {content[:200]}")

        return "\n".join(context_parts)
