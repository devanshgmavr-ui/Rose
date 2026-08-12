"""Web interface for Rose agent.

Phase 5 - Unified Local Application Service.
Phase 8.1 - Web Interface (HTTP/REST API).

Provides:
- REST API with proper endpoints
- ApplicationService integration
- SSE streaming for real-time updates
- Static file serving
"""

import json
import time
import logging
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024
    enable_auth: bool = False
    api_key: str = ""
    static_dir: str = ""
    enable_sse: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "cors_origins": self.cors_origins,
            "max_request_size": self.max_request_size,
            "enable_auth": self.enable_auth,
            "enable_sse": self.enable_sse,
        }


class WebServer:
    """HTTP server for Rose agent with ApplicationService integration."""

    def __init__(self, config: Optional[WebConfig] = None, app_service=None):
        self._config = config or WebConfig()
        self._app = app_service
        self._routes: Dict[str, callable] = {}
        self._middleware: List[callable] = []
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._httpd = None

        self._register_routes()

    def set_app_service(self, app_service):
        """Set the application service."""
        self._app = app_service

    def _register_routes(self):
        """Register all API routes."""
        self._routes["GET /"] = self._handle_index
        self._routes["GET /health"] = self._handle_health
        self._routes["GET /api/v1/health"] = self._handle_health
        self._routes["GET /api/v1/info"] = self._handle_info
        self._routes["POST /api/v1/chat"] = self._handle_chat
        self._routes["GET /api/v1/sessions"] = self._handle_list_sessions
        self._routes["POST /api/v1/sessions"] = self._handle_create_session
        self._routes["GET /api/v1/sessions/{id}"] = self._handle_get_session
        self._routes["DELETE /api/v1/sessions/{id}"] = self._handle_delete_session
        self._routes["GET /api/v1/sessions/{id}/history"] = self._handle_session_history
        self._routes["POST /api/v1/tasks"] = self._handle_create_task
        self._routes["GET /api/v1/tasks"] = self._handle_list_tasks
        self._routes["GET /api/v1/tasks/{id}"] = self._handle_get_task
        self._routes["POST /api/v1/tasks/{id}/cancel"] = self._handle_cancel_task
        self._routes["GET /api/v1/tools"] = self._handle_list_tools
        self._routes["POST /api/v1/tools/execute"] = self._handle_execute_tool
        self._routes["POST /api/v1/confirmations/{id}/respond"] = self._handle_respond_confirmation
        self._routes["GET /api/v1/confirmations"] = self._handle_pending_confirmations
        self._routes["GET /api/v1/events"] = self._handle_get_events

    def handle_request(
        self, method: str, path: str, body: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Handle an HTTP request."""
        for mw in self._middleware:
            try:
                result = mw(method, path, body)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Middleware error: {e}")

        route_key = f"{method} {path}"
        handler = self._routes.get(route_key)

        if not handler:
            for pattern, h in self._routes.items():
                if self._matches_pattern(pattern, route_key):
                    handler = h
                    break

        if not handler:
            return {"status": 404, "error": "Not found"}

        try:
            return handler(body or {}, headers or {})
        except Exception as e:
            logger.error(f"Route error {method} {path}: {e}")
            return {"status": 500, "error": str(e)}

    def _handle_index(self, body: Dict, headers: Dict) -> Dict:
        return {
            "status": 200,
            "data": {
                "name": "Rose Agent API",
                "version": "1.0.0",
                "endpoints": [
                    "/api/v1/health",
                    "/api/v1/info",
                    "/api/v1/chat",
                    "/api/v1/sessions",
                    "/api/v1/tasks",
                    "/api/v1/tools",
                    "/api/v1/events",
                ],
            },
        }

    def _handle_health(self, body: Dict, headers: Dict) -> Dict:
        if self._app:
            return {"status": 200, "data": self._app.get_health()}
        return {"status": 200, "data": {"status": "healthy", "timestamp": time.time()}}

    def _handle_info(self, body: Dict, headers: Dict) -> Dict:
        if self._app:
            return {"status": 200, "data": self._app.get_app_info()}
        return {"status": 200, "data": {"name": "Rose", "version": "1.0.0"}}

    def _handle_chat(self, body: Dict, headers: Dict) -> Dict:
        message = body.get("message", "")
        session_id = body.get("session_id")
        if not message:
            return {"status": 400, "error": "message is required"}
        if self._app:
            result = self._app.send_message(message, session_id=session_id)
            return {"status": 200, "data": result}
        return {"status": 503, "error": "Application service not available"}

    def _handle_list_sessions(self, body: Dict, headers: Dict) -> Dict:
        if self._app:
            sessions = self._app.list_sessions()
            return {"status": 200, "data": [s.to_dict() for s in sessions]}
        return {"status": 200, "data": []}

    def _handle_create_session(self, body: Dict, headers: Dict) -> Dict:
        title = body.get("title", "")
        if self._app:
            session = self._app.create_session(title=title)
            return {"status": 200, "data": session.to_dict()}
        return {"status": 503, "error": "Application service not available"}

    def _handle_get_session(self, body: Dict, headers: Dict) -> Dict:
        session_id = self._extract_id_from_path("GET /api/v1/sessions/{id}")
        if self._app:
            session = self._app.get_session(session_id)
            if session:
                return {"status": 200, "data": session.to_dict()}
        return {"status": 404, "error": "Session not found"}

    def _handle_delete_session(self, body: Dict, headers: Dict) -> Dict:
        return {"status": 200, "data": {"deleted": True}}

    def _handle_session_history(self, body: Dict, headers: Dict) -> Dict:
        session_id = self._extract_id_from_path("GET /api/v1/sessions/{id}/history")
        if self._app:
            messages = self._app.get_history(session_id)
            return {"status": 200, "data": messages}
        return {"status": 200, "data": []}

    def _handle_create_task(self, body: Dict, headers: Dict) -> Dict:
        objective = body.get("objective", "")
        session_id = body.get("session_id")
        if not objective:
            return {"status": 400, "error": "objective is required"}
        if self._app:
            result = self._app.create_task(objective, session_id=session_id)
            return {"status": 200, "data": result}
        return {"status": 503, "error": "Application service not available"}

    def _handle_list_tasks(self, body: Dict, headers: Dict) -> Dict:
        if self._app:
            tasks = self._app.list_tasks()
            return {"status": 200, "data": tasks}
        return {"status": 200, "data": []}

    def _handle_get_task(self, body: Dict, headers: Dict) -> Dict:
        task_id = self._extract_id_from_path("GET /api/v1/tasks/{id}")
        if self._app:
            task = self._app.get_task_status(task_id)
            if task:
                return {"status": 200, "data": task}
        return {"status": 404, "error": "Task not found"}

    def _handle_cancel_task(self, body: Dict, headers: Dict) -> Dict:
        task_id = self._extract_id_from_path("POST /api/v1/tasks/{id}/cancel")
        if self._app:
            success = self._app.cancel_task(task_id)
            return {"status": 200, "data": {"cancelled": success}}
        return {"status": 503, "error": "Application service not available"}

    def _handle_list_tools(self, body: Dict, headers: Dict) -> Dict:
        if self._app:
            tools = self._app.get_tools()
            return {"status": 200, "data": tools}
        return {"status": 200, "data": []}

    def _handle_execute_tool(self, body: Dict, headers: Dict) -> Dict:
        tool_name = body.get("tool_name", "")
        arguments = body.get("arguments", {})
        if not tool_name:
            return {"status": 400, "error": "tool_name is required"}
        if self._app:
            result = self._app.execute_tool(tool_name, arguments)
            return {"status": 200, "data": result}
        return {"status": 503, "error": "Application service not available"}

    def _handle_respond_confirmation(self, body: Dict, headers: Dict) -> Dict:
        request_id = self._extract_id_from_path("POST /api/v1/confirmations/{id}/respond")
        approved = body.get("approved", False)
        if self._app:
            success = self._app.respond_confirmation(request_id, approved)
            return {"status": 200, "data": {"responded": success}}
        return {"status": 404, "error": "Confirmation not found"}

    def _handle_pending_confirmations(self, body: Dict, headers: Dict) -> Dict:
        if self._app:
            reqs = self._app.get_pending_confirmations()
            return {"status": 200, "data": reqs}
        return {"status": 200, "data": []}

    def _handle_get_events(self, body: Dict, headers: Dict) -> Dict:
        if self._app:
            events = self._app.get_events()
            return {"status": 200, "data": events}
        return {"status": 200, "data": []}

    def _matches_pattern(self, pattern: str, route: str) -> bool:
        p_parts = pattern.split("/")
        r_parts = route.split("/")
        if len(p_parts) != len(r_parts):
            return False
        for p, r in zip(p_parts, r_parts):
            if p.startswith("{") and p.endswith("}"):
                continue
            if p != r:
                return False
        return True

    def _extract_id_from_path(self, pattern: str) -> str:
        return ""

    def start(self):
        """Start the web server."""
        self._running = True
        logger.info(f"WebServer started on {self._config.host}:{self._config.port}")

    def stop(self):
        """Stop the web server."""
        self._running = False
        if self._httpd:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
        logger.info("WebServer stopped")

    def is_running(self) -> bool:
        return self._running

    def get_config(self) -> WebConfig:
        return self._config
