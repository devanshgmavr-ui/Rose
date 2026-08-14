"""Unified application service for Rose.

Phase 5 - Unified Local Application Service.

Provides a single stable API layer between UI and backend.
UI never directly manipulates internal Python objects.
All operations go through this service.
"""

import time
import uuid
import logging
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AppEventType(Enum):
    """Application events for UI consumption."""
    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMATION_RESPONDED = "confirmation_responded"
    SCREENSHOT_GENERATED = "screenshot_generated"
    ERROR = "error"
    STATUS_CHANGED = "status_changed"
    SYSTEM_READY = "system_ready"
    SYSTEM_ERROR = "system_error"


@dataclass
class AppEvent:
    """An application event."""
    event_type: AppEventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class AppSession:
    """Application session state."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    title: str = ""
    status: str = "active"
    message_count: int = 0
    last_message_at: float = 0.0
    current_task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "title": self.title,
            "status": self.status,
            "message_count": self.message_count,
            "last_message_at": self.last_message_at,
            "current_task_id": self.current_task_id,
        }


@dataclass
class AppTaskStatus:
    """Task status for UI consumption."""
    task_id: str
    status: str
    objective: str
    progress: str = ""
    steps_completed: int = 0
    steps_total: int = 0
    current_step: str = ""
    error: str = ""
    result: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "objective": self.objective,
            "progress": self.progress,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "current_step": self.current_step,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ConfirmationRequest:
    """A pending confirmation request."""
    request_id: str
    tool_name: str
    action_description: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    responded: bool = False
    approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "action_description": self.action_description,
            "arguments": self.arguments,
            "created_at": self.created_at,
            "responded": self.responded,
            "approved": self.approved,
        }


class ApplicationService:
    """Unified application service.

    Single stable API between UI and Agent backend.
    All UI operations go through this service.
    """

    def __init__(self, agent=None):
        """Initialize the application service.

        Args:
            agent: Optional Agent instance. If None, created during initialize().
        """
        self._agent = agent
        self._sessions: Dict[str, AppSession] = {}
        self._events: deque = deque(maxlen=1000)
        self._event_callbacks: List[Callable[[AppEvent], None]] = []
        self._confirmation_requests: Dict[str, ConfirmationRequest] = {}
        self._lock = threading.Lock()
        self._initialized = False
        self._current_session_id: Optional[str] = None
        self._pending_tasks: Dict[str, AppTaskStatus] = {}

    def initialize(self) -> bool:
        """Initialize the application service and agent."""
        try:
            if self._agent:
                if not self._agent.initialize():
                    self._emit_event(AppEventType.SYSTEM_ERROR, {"error": "Agent initialization failed"})
                    return False
            self._initialized = True
            self._emit_event(AppEventType.SYSTEM_READY, {"status": "initialized"})
            logger.info("ApplicationService initialized")
            return True
        except Exception as e:
            logger.error(f"ApplicationService init failed: {e}")
            self._emit_event(AppEventType.SYSTEM_ERROR, {"error": str(e)})
            return False

    def shutdown(self):
        """Shutdown the application service and agent."""
        if self._agent:
            self._agent.shutdown()
        self._initialized = False
        logger.info("ApplicationService shut down")

    def create_session(self, title: str = "") -> AppSession:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())[:8]
        session = AppSession(session_id=session_id, title=title or f"Session {len(self._sessions) + 1}")
        self._sessions[session_id] = session
        self._current_session_id = session_id

        if self._agent:
            try:
                self._agent.start_new_session(title=title)
            except Exception:
                pass

        return session

    def get_session(self, session_id: str) -> Optional[AppSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self, limit: int = 20) -> List[AppSession]:
        """List recent sessions."""
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )
        return sessions[:limit]

    def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a user message and get a response.

        Args:
            message: The user's message.
            session_id: Optional session ID.

        Returns:
            Dictionary with response data.
        """
        if not self._initialized:
            return {"success": False, "error": "Application not initialized"}

        sid = session_id or self._current_session_id
        if not sid:
            session = self.create_session()
            sid = session.session_id

        self._emit_event(AppEventType.USER_MESSAGE, {
            "session_id": sid,
            "message": message,
        })

        session = self._sessions.get(sid)
        if session:
            session.message_count += 1
            session.last_message_at = time.time()
            if not session.title:
                session.title = message[:50]

        start = time.time()

        try:
            if self._agent:
                response = self._agent.chat(message)
                response_text = response.text
                execution_time = time.time() - start

                self._emit_event(AppEventType.ASSISTANT_RESPONSE, {
                    "session_id": sid,
                    "response": response_text[:500],
                    "execution_time": execution_time,
                })

                return {
                    "success": True,
                    "response": response_text,
                    "session_id": sid,
                    "execution_time": execution_time,
                }
            else:
                return {
                    "success": False,
                    "error": "Agent not available",
                    "session_id": sid,
                }
        except Exception as e:
            logger.error(f"send_message failed: {e}")
            self._emit_event(AppEventType.ERROR, {"error": str(e)})
            return {
                "success": False,
                "error": str(e),
                "session_id": sid,
            }

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if self._agent and self._agent._conversation_manager:
            try:
                messages = self._agent._conversation_manager.get_conversation_history()
                return [
                    {"role": m.role.value if hasattr(m.role, 'value') else str(m.role), "content": m.content}
                    for m in messages[-limit:]
                ]
            except Exception:
                pass
        return []

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool directly."""
        if not self._initialized:
            return {"success": False, "error": "Application not initialized"}

        self._emit_event(AppEventType.TOOL_STARTED, {
            "tool_name": tool_name,
            "arguments": {k: v for k, v in arguments.items() if k != "password"},
        })

        try:
            if self._agent:
                result = self._agent.execute_tool(tool_name, arguments)
                self._emit_event(AppEventType.TOOL_COMPLETED, {
                    "tool_name": tool_name,
                    "success": result.success,
                })
                return result.to_dict()
            return {"success": False, "error": "Agent not available"}
        except Exception as e:
            self._emit_event(AppEventType.ERROR, {"error": str(e)})
            return {"success": False, "error": str(e)}

    def create_task(self, objective: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Create and start an autonomous task."""
        if not self._initialized:
            return {"success": False, "error": "Application not initialized"}

        task_id = str(uuid.uuid4())[:8]
        task_status = AppTaskStatus(
            task_id=task_id,
            status="planning",
            objective=objective,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._pending_tasks[task_id] = task_status

        self._emit_event(AppEventType.TASK_STARTED, {
            "task_id": task_id,
            "objective": objective,
        })

        try:
            if self._agent:
                task = self._agent.execute_task(objective, session_id=session_id)
                task_status.status = task.status.value if hasattr(task.status, 'value') else str(task.status)
                task_status.steps_completed = len(task.completed_steps) if hasattr(task, 'completed_steps') else 0
                task_status.steps_total = len(task.plan.steps) if task.plan else 0
                task_status.updated_at = time.time()

                if task_status.status == "completed":
                    task_status.result = getattr(task, 'result', '') or ""
                    self._emit_event(AppEventType.TASK_COMPLETED, {
                        "task_id": task_id,
                        "status": task_status.status,
                    })
                else:
                    task_status.error = getattr(task, 'error', '') or "Task did not complete"
                    self._emit_event(AppEventType.TASK_FAILED, {
                        "task_id": task_id,
                        "error": task_status.error,
                    })

                return task_status.to_dict()
            return {"success": False, "error": "Agent not available"}
        except Exception as e:
            task_status.status = "failed"
            task_status.error = str(e)
            task_status.updated_at = time.time()
            self._emit_event(AppEventType.TASK_FAILED, {"task_id": task_id, "error": str(e)})
            return task_status.to_dict()

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        task = self._pending_tasks.get(task_id)
        if task:
            return task.to_dict()

        if self._agent:
            try:
                task = self._agent.get_task(task_id)
                if task:
                    return {
                        "task_id": task_id,
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "objective": getattr(task, 'user_request', ''),
                    }
            except Exception:
                pass
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        task = self._pending_tasks.get(task_id)
        if task:
            task.status = "cancelled"
            task.updated_at = time.time()
            self._emit_event(AppEventType.TASK_CANCELLED, {"task_id": task_id})
            return True

        if self._agent:
            try:
                return self._agent.cancel_task(task_id)
            except Exception:
                pass
        return False

    def list_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent tasks."""
        tasks = sorted(
            self._pending_tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )
        return [t.to_dict() for t in tasks[:limit]]

    def respond_confirmation(self, request_id: str, approved: bool) -> bool:
        """Respond to a confirmation request."""
        req = self._confirmation_requests.get(request_id)
        if req:
            req.responded = True
            req.approved = approved
            self._emit_event(AppEventType.CONFIRMATION_RESPONDED, {
                "request_id": request_id,
                "approved": approved,
            })
            return True
        return False

    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        """Get pending confirmation requests."""
        return [
            req.to_dict()
            for req in self._confirmation_requests.values()
            if not req.responded
        ]

    def get_health(self) -> Dict[str, Any]:
        """Get unified health status for all components."""
        health = {
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "timestamp": time.time(),
            "sessions": len(self._sessions),
            "pending_tasks": len(self._pending_tasks),
            "pending_confirmations": len([
                r for r in self._confirmation_requests.values() if not r.responded
            ]),
        }

        if self._agent:
            agent_health = self._agent.health_check()
            health["agent"] = agent_health
        else:
            health["agent"] = {"initialized": False}

        return health

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        if self._agent:
            return self._agent.get_tool_info()
        return []

    def register_event_callback(self, callback: Callable[[AppEvent], None]):
        """Register a callback for application events."""
        self._event_callbacks.append(callback)

    def get_events(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Get recent events."""
        events = list(self._events)
        return [e.to_dict() for e in events[-last_n:]]

    def _emit_event(self, event_type: AppEventType, data: Dict[str, Any] = None):
        """Emit an application event."""
        event = AppEvent(event_type=event_type, data=data or {})
        self._events.append(event)

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")

    def get_app_info(self) -> Dict[str, Any]:
        """Get application information."""
        return {
            "name": "Rose",
            "version": "1.1.0",
            "initialized": self._initialized,
            "sessions": len(self._sessions),
            "tools": len(self.get_tools()),
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities and tool catalog."""
        from ..orchestration.tool_catalog import build_tool_catalog
        from ..orchestration.capability_analyzer import CAPABILITY_DEFINITIONS

        catalog = build_tool_catalog()
        return {
            "capabilities": list(CAPABILITY_DEFINITIONS.keys()),
            "tools": {name: meta.to_dict() for name, meta in catalog.items()},
            "tool_count": len(catalog),
        }

    def get_permissions(self) -> Dict[str, Any]:
        """Get current permission status."""
        if self._agent and self._agent._permission_manager:
            pm = self._agent._permission_manager
            return {
                "permissions": [
                    {"name": p.name if hasattr(p, 'name') else str(p),
                     "status": p.value if hasattr(p, 'value') else str(p)}
                    for p in (pm._permissions.values() if hasattr(pm, '_permissions') else [])
                ],
                "enabled": {
                    "vision": self._agent.config.vision_enabled if self._agent.config else False,
                    "os_control": self._agent.config.os_control_enabled if self._agent.config else False,
                    "browser": self._agent.config.browser_automation_enabled if self._agent.config else False,
                    "mouse": self._agent.config.mouse_control_enabled if self._agent.config else False,
                    "keyboard": self._agent.config.keyboard_control_enabled if self._agent.config else False,
                },
            }
        return {"permissions": [], "enabled": {}}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": self._initialized,
            "timestamp": time.time(),
            "sessions": len(self._sessions),
            "pending_tasks": len(self._pending_tasks),
            "pending_confirmations": len([
                r for r in self._confirmation_requests.values() if not r.responded
            ]),
        }

        if self._agent:
            status["agent"] = self._agent.health_check()
            status["tools"] = {
                "count": self._agent._tool_registry.count() if self._agent._tool_registry else 0,
                "available": self._agent._tool_registry.list_names() if self._agent._tool_registry else [],
            }
            status["media"] = self._agent.get_media_stats()
            status["orchestration"] = self._agent.get_orchestration_stats()
            status["memory"] = self._agent.get_memory_stats()

        return status
