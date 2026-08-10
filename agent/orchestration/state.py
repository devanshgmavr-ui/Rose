"""State machine for task orchestration."""

from typing import Dict, Set, Tuple
from .models import TaskStatus


VALID_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.PLANNING: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.WAITING,
        TaskStatus.VERIFYING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.PLANNING,
        TaskStatus.CANCELLED,
        TaskStatus.TIMEOUT,
    },
    TaskStatus.WAITING: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.VERIFYING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.RUNNING},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.CANCELLED: set(),
    TaskStatus.TIMEOUT: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
}


class StateMachine:
    def __init__(self):
        self._current_status: TaskStatus = TaskStatus.PENDING
        self._history: list = []

    @property
    def current_status(self) -> TaskStatus:
        return self._current_status

    def can_transition(self, new_status: TaskStatus) -> bool:
        return new_status in VALID_TRANSITIONS.get(self._current_status, set())

    def transition(self, new_status: TaskStatus) -> Tuple[bool, str]:
        if not self.can_transition(new_status):
            return False, (
                f"Invalid transition: {self._current_status.value} -> {new_status.value}"
            )
        old = self._current_status
        self._current_status = new_status
        self._history.append((old.value, new_status.value))
        return True, ""

    def get_history(self) -> list:
        return list(self._history)

    def reset(self, status: TaskStatus = TaskStatus.PENDING):
        self._current_status = status
        self._history.clear()

    def is_terminal(self) -> bool:
        return self._current_status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        }
