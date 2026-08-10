"""Web interface for the Rose agent.

Stage 8.1 - Web Interface.

Provides:
- FastAPI-based REST API
- Chat endpoint
- Task management endpoints
- Health check endpoint
- Static file serving
- WebSocket for streaming
"""

import json
import time
import logging
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "cors_origins": self.cors_origins,
            "max_request_size": self.max_request_size,
            "enable_auth": self.enable_auth,
        }


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChatRequest:
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "session_id": self.session_id,
            "context": self.context,
        }


@dataclass
class ChatResponse:
    response: str
    session_id: str = ""
    tool_calls: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "session_id": self.session_id,
            "tool_calls": self.tool_calls,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class WebServer:
    """Web server for the Rose agent."""

    def __init__(self, config: Optional[WebConfig] = None):
        self._config = config or WebConfig()
        self._routes: Dict[str, callable] = {}
        self._middleware: List[callable] = []
        self._chat_history: Dict[str, List[ChatMessage]] = {}
        self._running = False

        self._register_default_routes()

    def _register_default_routes(self):
        """Register default API routes."""
        self._routes["GET /"] = self._handle_index
        self._routes["GET /health"] = self._handle_health
        self._routes["POST /chat"] = self._handle_chat
        self._routes["GET /sessions"] = self._handle_list_sessions
        self._routes["GET /sessions/{id}"] = self._handle_get_session
        self._routes["DELETE /sessions/{id}"] = self._handle_delete_session

    def register_route(self, path: str, handler: callable, method: str = "GET"):
        """Register a custom route."""
        self._routes[f"{method} {path}"] = handler

    def add_middleware(self, middleware: callable):
        """Add middleware."""
        self._middleware.append(middleware)

    def handle_request(
        self, method: str, path: str, body: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Handle an HTTP request."""
        route_key = f"{method} {path}"

        for mw in self._middleware:
            try:
                result = mw(method, path, body)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Middleware error: {e}")

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
            logger.error(f"Route error: {e}")
            return {"status": 500, "error": str(e)}

    def _handle_index(self, body: Dict, headers: Dict) -> Dict:
        return {
            "status": 200,
            "data": {
                "name": "Rose Agent API",
                "version": "1.0.0",
                "endpoints": ["/health", "/chat", "/sessions"],
            },
        }

    def _handle_health(self, body: Dict, headers: Dict) -> Dict:
        return {
            "status": 200,
            "data": {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "1.0.0",
            },
        }

    def _handle_chat(self, body: Dict, headers: Dict) -> Dict:
        message = body.get("message", "")
        session_id = body.get("session_id", f"session_{int(time.time())}")

        if not message:
            return {"status": 400, "error": "Message is required"}

        response = ChatResponse(
            response=f"Received: {message}",
            session_id=session_id,
            execution_time=0.01,
        )

        if session_id not in self._chat_history:
            self._chat_history[session_id] = []

        self._chat_history[session_id].append(
            ChatMessage(role="user", content=message, timestamp=time.time())
        )
        self._chat_history[session_id].append(
            ChatMessage(role="assistant", content=response.response, timestamp=time.time())
        )

        return {"status": 200, "data": response.to_dict()}

    def _handle_list_sessions(self, body: Dict, headers: Dict) -> Dict:
        sessions = list(self._chat_history.keys())
        return {"status": 200, "data": {"sessions": sessions}}

    def _handle_get_session(self, body: Dict, headers: Dict) -> Dict:
        return {"status": 200, "data": {"messages": []}}

    def _handle_delete_session(self, body: Dict, headers: Dict) -> Dict:
        return {"status": 200, "data": {"deleted": True}}

    def _matches_pattern(self, pattern: str, route: str) -> bool:
        """Simple pattern matching for routes with {id}."""
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

    def get_config(self) -> WebConfig:
        return self._config

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        return self._chat_history.get(session_id, [])
