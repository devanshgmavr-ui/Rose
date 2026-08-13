"""Phase 12 - Backend API & Event System Tests.

Tests fixed ID extraction, new endpoints (capabilities, permissions,
system status), and ApplicationService API completeness.
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agent.web.server import WebServer, WebConfig
from agent.web.application import (
    ApplicationService,
    AppEventType,
    AppEvent,
    AppSession,
    AppTaskStatus,
    ConfirmationRequest,
)
from agent.web.events import EventBus, SSEHandler, WSEvent


class TestWebConfig:
    def test_defaults(self):
        config = WebConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.enable_sse is True

    def test_to_dict(self):
        config = WebConfig(port=9090)
        d = config.to_dict()
        assert d["port"] == 9090
        assert "host" in d
        assert "enable_sse" in d


class TestWebServerRoutes:
    def test_all_routes_registered(self):
        server = WebServer()
        route_keys = list(server._routes.keys())
        assert "GET /" in route_keys
        assert "GET /health" in route_keys
        assert "GET /api/v1/health" in route_keys
        assert "GET /api/v1/info" in route_keys
        assert "POST /api/v1/chat" in route_keys
        assert "GET /api/v1/sessions" in route_keys
        assert "POST /api/v1/sessions" in route_keys
        assert "GET /api/v1/tasks" in route_keys
        assert "POST /api/v1/tasks" in route_keys
        assert "GET /api/v1/tools" in route_keys
        assert "POST /api/v1/tools/execute" in route_keys
        assert "GET /api/v1/events" in route_keys
        assert "GET /api/v1/capabilities" in route_keys
        assert "GET /api/v1/permissions" in route_keys
        assert "GET /api/v1/system" in route_keys

    def test_404_for_unknown_route(self):
        server = WebServer()
        result = server.handle_request("GET", "/nonexistent")
        assert result["status"] == 404

    def test_health_without_app(self):
        server = WebServer()
        result = server.handle_request("GET", "/health")
        assert result["status"] == 200
        assert "timestamp" in result["data"]

    def test_index_without_app(self):
        server = WebServer()
        result = server.handle_request("GET", "/")
        assert result["status"] == 200
        assert "endpoints" in result["data"]

    def test_capabilities_without_app(self):
        server = WebServer()
        result = server.handle_request("GET", "/api/v1/capabilities")
        assert result["status"] == 200
        assert "capabilities" in result["data"]
        assert "tools" in result["data"]

    def test_permissions_without_app(self):
        server = WebServer()
        result = server.handle_request("GET", "/api/v1/permissions")
        assert result["status"] == 200
        assert "permissions" in result["data"]
        assert "enabled" in result["data"]

    def test_system_status_without_app(self):
        server = WebServer()
        result = server.handle_request("GET", "/api/v1/system")
        assert result["status"] == 200


class TestPathIDExtraction:
    def test_extract_session_id(self):
        server = WebServer()
        server._last_path = "/api/v1/sessions/abc123"
        result = server._extract_id_from_path("GET /api/v1/sessions/{id}")
        assert result == "abc123"

    def test_extract_task_id(self):
        server = WebServer()
        server._last_path = "/api/v1/tasks/task-xyz-789"
        result = server._extract_id_from_path("GET /api/v1/tasks/{id}")
        assert result == "task-xyz-789"

    def test_extract_id_mismatched_lengths(self):
        server = WebServer()
        server._last_path = "/api/v1/sessions"
        result = server._extract_id_from_path("GET /api/v1/sessions/{id}")
        assert result == ""

    def test_extract_id_no_id_segment(self):
        server = WebServer()
        server._last_path = "/api/v1/tools"
        result = server._extract_id_from_path("GET /api/v1/tools")
        assert result == ""


class TestPatternMatching:
    def test_exact_match(self):
        server = WebServer()
        assert server._matches_pattern("GET /health", "GET /health")

    def test_parameter_match(self):
        server = WebServer()
        assert server._matches_pattern("GET /api/v1/sessions/{id}", "GET /api/v1/sessions/abc")

    def test_no_match_different_length(self):
        server = WebServer()
        assert not server._matches_pattern("GET /health", "GET /api/v1/health")

    def test_no_match_different_path(self):
        server = WebServer()
        assert not server._matches_pattern("GET /health", "GET /info")


class TestWebServerWithAppService:
    def test_health_with_app(self):
        app = MagicMock()
        app.get_health.return_value = {"status": "healthy", "initialized": True}
        server = WebServer(app_service=app)
        result = server.handle_request("GET", "/health")
        assert result["status"] == 200
        assert result["data"]["status"] == "healthy"

    def test_chat_without_message(self):
        app = MagicMock()
        server = WebServer(app_service=app)
        result = server.handle_request("POST", "/api/v1/chat", body={})
        assert result["status"] == 400

    def test_chat_with_message(self):
        app = MagicMock()
        app.send_message.return_value = {"success": True, "response": "Hello"}
        server = WebServer(app_service=app)
        result = server.handle_request("POST", "/api/v1/chat", body={"message": "Hi"})
        assert result["status"] == 200
        assert result["data"]["success"] is True

    def test_capabilities_with_app(self):
        app = MagicMock()
        app.get_capabilities.return_value = {
            "capabilities": ["vision_analysis"],
            "tools": {},
            "tool_count": 0,
        }
        server = WebServer(app_service=app)
        result = server.handle_request("GET", "/api/v1/capabilities")
        assert result["status"] == 200
        assert "vision_analysis" in result["data"]["capabilities"]

    def test_permissions_with_app(self):
        app = MagicMock()
        app.get_permissions.return_value = {
            "permissions": [],
            "enabled": {"vision": True, "os_control": False},
        }
        server = WebServer(app_service=app)
        result = server.handle_request("GET", "/api/v1/permissions")
        assert result["status"] == 200
        assert result["data"]["enabled"]["vision"] is True

    def test_system_status_with_app(self):
        app = MagicMock()
        app.get_system_status.return_value = {"initialized": True, "tools": {"count": 5}}
        server = WebServer(app_service=app)
        result = server.handle_request("GET", "/api/v1/system")
        assert result["status"] == 200
        assert result["data"]["initialized"] is True

    def test_session_id_extraction_in_handler(self):
        app = MagicMock()
        session = MagicMock()
        session.to_dict.return_value = {"session_id": "abc123"}
        app.get_session.return_value = session
        server = WebServer(app_service=app)

        result = server.handle_request("GET", "/api/v1/sessions/abc123")
        assert result["status"] == 200

    def test_cancel_task(self):
        app = MagicMock()
        app.cancel_task.return_value = True
        server = WebServer(app_service=app)
        result = server.handle_request("POST", "/api/v1/tasks/task1/cancel")
        assert result["status"] == 200
        assert result["data"]["cancelled"] is True


class TestApplicationServiceEvents:
    def test_event_types(self):
        assert AppEventType.USER_MESSAGE.value == "user_message"
        assert AppEventType.TASK_STARTED.value == "task_started"
        assert AppEventType.TASK_COMPLETED.value == "task_completed"
        assert AppEventType.TOOL_STARTED.value == "tool_started"
        assert AppEventType.CONFIRMATION_REQUIRED.value == "confirmation_required"

    def test_event_creation(self):
        event = AppEvent(
            event_type=AppEventType.TASK_STARTED,
            data={"task_id": "abc"},
        )
        d = event.to_dict()
        assert d["event_type"] == "task_started"
        assert d["data"]["task_id"] == "abc"
        assert "timestamp" in d
        assert "event_id" in d

    def test_event_callback(self):
        service = ApplicationService()
        received = []
        service.register_event_callback(lambda e: received.append(e))
        service._emit_event(AppEventType.SYSTEM_READY, {"status": "ok"})
        assert len(received) == 1
        assert received[0].event_type == AppEventType.SYSTEM_READY


class TestApplicationServiceSession:
    def test_create_session(self):
        service = ApplicationService()
        session = service.create_session("Test Session")
        assert session.session_id != ""
        assert session.title == "Test Session"

    def test_get_session(self):
        service = ApplicationService()
        session = service.create_session("Test")
        found = service.get_session(session.session_id)
        assert found is not None
        assert found.session_id == session.session_id

    def test_list_sessions(self):
        service = ApplicationService()
        service.create_session("S1")
        service.create_session("S2")
        sessions = service.list_sessions()
        assert len(sessions) == 2

    def test_send_message_without_init(self):
        service = ApplicationService()
        result = service.send_message("Hello")
        assert result["success"] is False
        assert "not initialized" in result["error"]


class TestApplicationServiceTools:
    def test_execute_tool_without_init(self):
        service = ApplicationService()
        result = service.execute_tool("filesystem", {"action": "list"})
        assert result["success"] is False

    def test_get_tools_without_agent(self):
        service = ApplicationService()
        tools = service.get_tools()
        assert tools == []


class TestApplicationServiceTask:
    def test_create_task_without_init(self):
        service = ApplicationService()
        result = service.create_task("Do something")
        assert result["success"] is False

    def test_list_tasks_empty(self):
        service = ApplicationService()
        tasks = service.list_tasks()
        assert tasks == []

    def test_get_task_status_missing(self):
        service = ApplicationService()
        result = service.get_task_status("nonexistent")
        assert result is None

    def test_cancel_task_missing(self):
        service = ApplicationService()
        result = service.cancel_task("nonexistent")
        assert result is False


class TestApplicationServiceConfirmations:
    def test_respond_confirmation_missing(self):
        service = ApplicationService()
        result = service.respond_confirmation("nonexistent", True)
        assert result is False

    def test_get_pending_empty(self):
        service = ApplicationService()
        result = service.get_pending_confirmations()
        assert result == []


class TestApplicationServiceHealth:
    def test_health_without_agent(self):
        service = ApplicationService()
        health = service.get_health()
        assert health["status"] == "unhealthy"
        assert health["initialized"] is False

    def test_get_capabilities(self):
        service = ApplicationService()
        caps = service.get_capabilities()
        assert "capabilities" in caps
        assert "tools" in caps
        assert "tool_count" in caps
        assert isinstance(caps["capabilities"], list)

    def test_get_permissions(self):
        service = ApplicationService()
        perms = service.get_permissions()
        assert "permissions" in perms
        assert "enabled" in perms

    def test_get_system_status(self):
        service = ApplicationService()
        status = service.get_system_status()
        assert "initialized" in status
        assert "timestamp" in status


class TestAppSessionDataclass:
    def test_to_dict(self):
        session = AppSession(
            session_id="test123",
            title="My Session",
            status="active",
            message_count=5,
        )
        d = session.to_dict()
        assert d["session_id"] == "test123"
        assert d["title"] == "My Session"
        assert d["message_count"] == 5


class TestAppTaskStatusDataclass:
    def test_to_dict(self):
        task = AppTaskStatus(
            task_id="task1",
            status="running",
            objective="Open browser",
            steps_completed=2,
            steps_total=5,
        )
        d = task.to_dict()
        assert d["task_id"] == "task1"
        assert d["status"] == "running"
        assert d["steps_completed"] == 2


class TestConfirmationRequestDataclass:
    def test_to_dict(self):
        req = ConfirmationRequest(
            request_id="conf1",
            tool_name="mouse",
            action_description="Click at (100, 200)",
            arguments={"x": 100, "y": 200},
        )
        d = req.to_dict()
        assert d["request_id"] == "conf1"
        assert d["tool_name"] == "mouse"
        assert d["responded"] is False


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("test_event", lambda e: received.append(e))
        bus.emit("test_event", {"key": "value"})
        assert len(received) == 1
        assert received[0].event_type == "test_event"

    def test_wildcard_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.emit("any_event", {})
        assert len(received) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe("test", cb)
        bus.unsubscribe("test", cb)
        bus.emit("test", {})
        assert len(received) == 0

    def test_get_history(self):
        bus = EventBus()
        bus.emit("event1", {})
        bus.emit("event2", {})
        history = bus.get_history()
        assert len(history) == 2

    def test_get_history_filtered(self):
        bus = EventBus()
        bus.emit("event1", {})
        bus.emit("event2", {})
        bus.emit("event1", {})
        history = bus.get_history(event_type="event1")
        assert len(history) == 2

    def test_shutdown(self):
        bus = EventBus()
        bus.emit("test", {})
        bus.shutdown()
        assert bus._active is False


class TestSSEHandler:
    def test_create_remove_client(self):
        bus = EventBus()
        handler = SSEHandler(bus)
        q = handler.create_client("client1")
        assert q is not None
        handler.remove_client("client1")

    def test_broadcast_event(self):
        bus = EventBus()
        handler = SSEHandler(bus)
        q = handler.create_client("client1")
        bus.emit("test_event", {"data": "hello"})
        event_json = handler.get_events_for_client("client1", timeout=0.1)
        assert event_json is not None
        assert "test_event" in event_json

    def test_get_events_timeout(self):
        bus = EventBus()
        handler = SSEHandler(bus)
        q = handler.create_client("client1")
        result = handler.get_events_for_client("client1", timeout=0.05)
        assert result is None

    def test_remove_nonexistent_client(self):
        bus = EventBus()
        handler = SSEHandler(bus)
        handler.remove_client("nonexistent")


class TestWSEvent:
    def test_to_json(self):
        event = WSEvent(
            event_type="test",
            data={"key": "value"},
            session_id="sess1",
        )
        import json
        parsed = json.loads(event.to_json())
        assert parsed["type"] == "test"
        assert parsed["data"]["key"] == "value"
        assert parsed["session_id"] == "sess1"
