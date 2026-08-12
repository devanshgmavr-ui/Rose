"""Rose UI - Panels and main application window."""

import time
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from .app import MessageDisplay, InputHandler

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    task_id: str
    objective: str
    status: str = "planning"
    progress: str = ""
    steps_completed: int = 0
    steps_total: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    error: str = ""
    result: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "status": self.status,
            "progress": self.progress,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "result": self.result,
        }


class TaskPanel:
    """Task management panel state."""

    def __init__(self):
        self._tasks: List[TaskInfo] = []
        self._selected_task_id: Optional[str] = None
        self._on_change_callbacks: List[Callable] = []

    def add_task(self, task_id: str, objective: str, status: str = "planning") -> TaskInfo:
        task = TaskInfo(
            task_id=task_id,
            objective=objective,
            status=status,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._tasks.append(task)
        self._notify_change()
        return task

    def update_task(self, task_id: str, **kwargs) -> bool:
        for task in self._tasks:
            if task.task_id == task_id:
                for k, v in kwargs.items():
                    if hasattr(task, k):
                        setattr(task, k, v)
                task.updated_at = time.time()
                self._notify_change()
                return True
        return False

    def remove_task(self, task_id: str) -> bool:
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks if t.task_id != task_id]
        if self._selected_task_id == task_id:
            self._selected_task_id = None
        changed = len(self._tasks) < before
        if changed:
            self._notify_change()
        return changed

    def get_tasks(self) -> List[TaskInfo]:
        return list(self._tasks)

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        for t in self._tasks:
            if t.task_id == task_id:
                return t
        return None

    def select_task(self, task_id: Optional[str]):
        self._selected_task_id = task_id
        self._notify_change()

    def get_selected(self) -> Optional[TaskInfo]:
        if self._selected_task_id:
            return self.get_task(self._selected_task_id)
        return None

    def task_count(self) -> int:
        return len(self._tasks)

    def active_tasks(self) -> List[TaskInfo]:
        return [t for t in self._tasks if t.status in ("planning", "running")]

    def on_change(self, callback: Callable):
        self._on_change_callbacks.append(callback)

    def _notify_change(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass


class SettingsPanel:
    """Settings panel state."""

    def __init__(self):
        self._settings: Dict[str, Any] = {
            "theme": "dark",
            "font_size": 14,
            "auto_scroll": True,
            "show_sidebar": True,
            "max_message_length": 10000,
            "agent_model": "",
            "context_length": 4096,
        }
        self._changed = False
        self._on_change_callbacks: List[Callable] = []

    def get(self, key: str, default=None) -> Any:
        return self._settings.get(key, default)

    def set(self, key: str, value: Any):
        if self._settings.get(key) != value:
            self._settings[key] = value
            self._changed = True
            self._notify_change()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.set(k, v)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._settings)

    def has_changes(self) -> bool:
        return self._changed

    def reset_changes(self):
        self._changed = False

    def on_change(self, callback: Callable):
        self._on_change_callbacks.append(callback)

    def _notify_change(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass


class StatusBar:
    """Status bar state management."""

    def __init__(self):
        self._status: str = "Ready"
        self._agent_status: str = "offline"
        self._progress: float = 0.0
        self._details: Dict[str, Any] = {}
        self._on_change_callbacks: List[Callable] = []

    def set_status(self, status: str):
        self._status = status
        self._notify_change()

    def set_agent_status(self, status: str):
        self._agent_status = status
        self._notify_change()

    def set_progress(self, progress: float):
        self._progress = max(0.0, min(1.0, progress))
        self._notify_change()

    def set_detail(self, key: str, value: Any):
        self._details[key] = value
        self._notify_change()

    def get_status(self) -> str:
        return self._status

    def get_agent_status(self) -> str:
        return self._agent_status

    def get_progress(self) -> float:
        return self._progress

    def get_details(self) -> Dict[str, Any]:
        return dict(self._details)

    def clear_progress(self):
        self._progress = 0.0
        self._notify_change()

    def on_change(self, callback: Callable):
        self._on_change_callbacks.append(callback)

    def _notify_change(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass


class ConfirmationDialog:
    """Confirmation dialog state for dangerous actions."""

    def __init__(self):
        self._pending: List[Dict[str, Any]] = []
        self._on_change_callbacks: List[Callable] = []

    def request_confirmation(
        self, request_id: str, tool_name: str, description: str,
        arguments: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        req = {
            "request_id": request_id,
            "tool_name": tool_name,
            "description": description,
            "arguments": arguments or {},
            "created_at": time.time(),
            "responded": False,
            "approved": False,
        }
        self._pending.append(req)
        self._notify_change()
        return req

    def respond(self, request_id: str, approved: bool) -> bool:
        for req in self._pending:
            if req["request_id"] == request_id and not req["responded"]:
                req["responded"] = True
                req["approved"] = approved
                self._notify_change()
                return True
        return False

    def get_pending(self) -> List[Dict[str, Any]]:
        return [r for r in self._pending if not r["responded"]]

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._pending)

    def clear_responded(self):
        self._pending = [r for r in self._pending if not r["responded"]]

    def pending_count(self) -> int:
        return len(self.get_pending())

    def on_change(self, callback: Callable):
        self._on_change_callbacks.append(callback)

    def _notify_change(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass


class Sidebar:
    """Sidebar state for navigation."""

    def __init__(self):
        self._visible: bool = True
        self._selected_section: str = "chat"
        self._sections: List[str] = ["chat", "tasks", "settings", "history"]
        self._on_change_callbacks: List[Callable] = []

    def toggle_visibility(self) -> bool:
        self._visible = not self._visible
        self._notify_change()
        return self._visible

    def is_visible(self) -> bool:
        return self._visible

    def select_section(self, section: str) -> bool:
        if section in self._sections:
            self._selected_section = section
            self._notify_change()
            return True
        return False

    def get_selected_section(self) -> str:
        return self._selected_section

    def get_sections(self) -> List[str]:
        return list(self._sections)

    def on_change(self, callback: Callable):
        self._on_change_callbacks.append(callback)

    def _notify_change(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass


class RoseUI:
    """Main Rose UI application state manager.

    Coordinates all UI panels and components.
    Connects to ApplicationService for backend operations.
    """

    def __init__(self, app_service=None):
        self._app = app_service
        self._message_display = MessageDisplay()
        self._input_handler = InputHandler()
        self._task_panel = TaskPanel()
        self._settings_panel = SettingsPanel()
        self._status_bar = StatusBar()
        self._confirmation_dialog = ConfirmationDialog()
        self._sidebar = Sidebar()
        self._initialized = False
        self._callbacks: Dict[str, List[Callable]] = {
            "chat": [], "task": [], "settings": [], "status": [],
        }

    @property
    def messages(self) -> MessageDisplay:
        return self._message_display

    @property
    def input(self) -> InputHandler:
        return self._input_handler

    @property
    def tasks(self) -> TaskPanel:
        return self._task_panel

    @property
    def settings(self) -> SettingsPanel:
        return self._settings_panel

    @property
    def status(self) -> StatusBar:
        return self._status_bar

    @property
    def confirmations(self) -> ConfirmationDialog:
        return self._confirmation_dialog

    @property
    def sidebar(self) -> Sidebar:
        return self._sidebar

    def set_app_service(self, app_service):
        self._app = app_service

    def initialize(self) -> bool:
        if self._initialized:
            return True

        self._status_bar.set_status("Initializing...")
        self._status_bar.set_agent_status("connecting")

        if self._app:
            try:
                if self._app.initialize():
                    self._status_bar.set_agent_status("online")
                    self._status_bar.set_status("Ready")
                    self._message_display.add_message(
                        "system", "Rose is ready. How can I help you?"
                    )
                    self._initialized = True
                    return True
                else:
                    self._status_bar.set_agent_status("error")
                    self._status_bar.set_status("Initialization failed")
                    return False
            except Exception as e:
                logger.error(f"UI init error: {e}")
                self._status_bar.set_agent_status("error")
                self._status_bar.set_status(f"Error: {e}")
                return False
        else:
            self._status_bar.set_status("Ready (no backend)")
            self._message_display.add_message(
                "system", "Rose UI initialized (demo mode - no backend connected)"
            )
            self._initialized = True
            return True

    def shutdown(self):
        if self._app:
            self._app.shutdown()
        self._initialized = False
        self._status_bar.set_agent_status("offline")
        self._status_bar.set_status("Shut down")

    def send_message(self, message: str) -> bool:
        valid, error = self._input_handler.validate(message)
        if not valid:
            self._message_display.add_message("error", error)
            return False

        self._message_display.add_message("user", message)
        self._input_handler.set_processing(True)
        self._status_bar.set_status("Processing...")

        if self._app:
            try:
                result = self._app.send_message(message)
                self._input_handler.set_processing(False)

                if result.get("success"):
                    self._message_display.add_message(
                        "assistant", result.get("response", "")
                    )
                    self._status_bar.set_status("Ready")
                else:
                    self._message_display.add_message(
                        "error", f"Error: {result.get('error', 'Unknown error')}"
                    )
                    self._status_bar.set_status("Error")
                return result.get("success", False)
            except Exception as e:
                self._input_handler.set_processing(False)
                self._message_display.add_message("error", str(e))
                self._status_bar.set_status("Error")
                return False
        else:
            self._input_handler.set_processing(False)
            self._message_display.add_message(
                "assistant", f"[Demo] You said: {message}"
            )
            self._status_bar.set_status("Ready")
            return True

    def create_task(self, objective: str) -> Optional[str]:
        if not objective.strip():
            return None

        self._status_bar.set_status(f"Creating task: {objective[:50]}...")

        if self._app:
            try:
                result = self._app.create_task(objective)
                task_id = result.get("task_id", "")
                if task_id:
                    self._task_panel.add_task(task_id, objective, result.get("status", "planning"))
                    self._task_panel.update_task(
                        task_id,
                        steps_total=result.get("steps_total", 0),
                        steps_completed=result.get("steps_completed", 0),
                    )
                self._status_bar.set_status("Ready")
                return task_id
            except Exception as e:
                self._status_bar.set_status(f"Task error: {e}")
                return None
        else:
            import uuid
            task_id = str(uuid.uuid4())[:8]
            self._task_panel.add_task(task_id, objective, "completed")
            self._status_bar.set_status("Ready")
            return task_id

    def cancel_task(self, task_id: str) -> bool:
        if self._app:
            try:
                return self._app.cancel_task(task_id)
            except Exception:
                return False
        return self._task_panel.update_task(task_id, status="cancelled")

    def confirm_action(self, request_id: str, approved: bool):
        self._confirmation_dialog.respond(request_id, approved)
        if self._app:
            try:
                self._app.respond_confirmation(request_id, approved)
            except Exception:
                pass

    def refresh_health(self) -> Dict[str, Any]:
        if self._app:
            try:
                health = self._app.get_health()
                self._status_bar.set_agent_status(
                    "online" if health.get("status") == "healthy" else "degraded"
                )
                return health
            except Exception:
                self._status_bar.set_agent_status("error")
                return {"status": "error"}
        return {"status": "no_backend"}

    def clear_chat(self):
        self._message_display.clear()
        self._message_display.add_message("system", "Chat cleared")

    def get_state(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "message_count": self._message_display.message_count(),
            "task_count": self._task_panel.task_count(),
            "agent_status": self._status_bar.get_agent_status(),
            "pending_confirmations": self._confirmation_dialog.pending_count(),
            "sidebar_visible": self._sidebar.is_visible(),
        }
