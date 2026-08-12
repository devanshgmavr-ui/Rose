"""Tests for Stage 8.1 - Web Interface (updated for Phase 5 server)."""

import pytest
import time
from agent.web.server import WebServer, WebConfig


class TestWebConfig:
    def test_defaults(self):
        c = WebConfig()
        assert c.host == "127.0.0.1"
        assert c.port == 8080

    def test_to_dict(self):
        c = WebConfig(port=9090, debug=True)
        d = c.to_dict()
        assert d["port"] == 9090
        assert d["debug"] is True

    def test_custom_values(self):
        c = WebConfig(host="0.0.0.0", port=3000, enable_auth=True)
        assert c.host == "0.0.0.0"
        assert c.port == 3000
        assert c.enable_auth is True


class TestWebServer:
    def test_init(self):
        s = WebServer()
        assert s._running is False
        assert s._app is None

    def test_init_with_config(self):
        cfg = WebConfig(port=9090)
        s = WebServer(config=cfg)
        assert s.get_config().port == 9090

    def test_init_with_app_service(self):
        from unittest.mock import MagicMock
        app = MagicMock()
        s = WebServer(app_service=app)
        assert s._app is app

    def test_set_app_service(self):
        from unittest.mock import MagicMock
        s = WebServer()
        app = MagicMock()
        s.set_app_service(app)
        assert s._app is app

    def test_handle_index(self):
        s = WebServer()
        result = s.handle_request("GET", "/")
        assert result["status"] == 200
        assert result["data"]["name"] == "Rose Agent API"

    def test_handle_health(self):
        s = WebServer()
        result = s.handle_request("GET", "/health")
        assert result["status"] == 200
        assert "status" in result["data"]

    def test_handle_health_v1(self):
        s = WebServer()
        result = s.handle_request("GET", "/api/v1/health")
        assert result["status"] == 200

    def test_handle_info(self):
        s = WebServer()
        result = s.handle_request("GET", "/api/v1/info")
        assert result["status"] == 200
        assert result["data"]["name"] == "Rose"

    def test_handle_chat_no_message(self):
        s = WebServer()
        result = s.handle_request("POST", "/api/v1/chat", body={})
        assert result["status"] == 400

    def test_handle_chat_no_app(self):
        s = WebServer()
        result = s.handle_request("POST", "/api/v1/chat", body={"message": "hi"})
        assert result["status"] == 503

    def test_handle_chat_with_app(self):
        from unittest.mock import MagicMock
        app = MagicMock()
        app.send_message.return_value = {"success": True, "response": "hello"}
        s = WebServer(app_service=app)
        result = s.handle_request("POST", "/api/v1/chat", body={"message": "hi"})
        assert result["status"] == 200
        assert result["data"]["response"] == "hello"

    def test_handle_list_sessions(self):
        s = WebServer()
        result = s.handle_request("GET", "/api/v1/sessions")
        assert result["status"] == 200
        assert isinstance(result["data"], list)

    def test_handle_create_session(self):
        from unittest.mock import MagicMock
        app = MagicMock()
        session = MagicMock()
        session.to_dict.return_value = {"session_id": "abc", "title": "Test"}
        app.create_session.return_value = session
        s = WebServer(app_service=app)
        result = s.handle_request("POST", "/api/v1/sessions", body={"title": "Test"})
        assert result["status"] == 200

    def test_handle_list_tools(self):
        s = WebServer()
        result = s.handle_request("GET", "/api/v1/tools")
        assert result["status"] == 200

    def test_handle_execute_tool_no_name(self):
        s = WebServer()
        result = s.handle_request("POST", "/api/v1/tools/execute", body={})
        assert result["status"] == 400

    def test_handle_execute_tool_no_app(self):
        s = WebServer()
        result = s.handle_request("POST", "/api/v1/tools/execute", body={"tool_name": "shell"})
        assert result["status"] == 503

    def test_handle_list_tasks(self):
        s = WebServer()
        result = s.handle_request("GET", "/api/v1/tasks")
        assert result["status"] == 200

    def test_handle_create_task_no_objective(self):
        s = WebServer()
        result = s.handle_request("POST", "/api/v1/tasks", body={})
        assert result["status"] == 400

    def test_handle_pending_confirmations(self):
        s = WebServer()
        result = s.handle_request("GET", "/api/v1/confirmations")
        assert result["status"] == 200

    def test_handle_get_events(self):
        s = WebServer()
        result = s.handle_request("GET", "/api/v1/events")
        assert result["status"] == 200

    def test_not_found(self):
        s = WebServer()
        result = s.handle_request("GET", "/nonexistent")
        assert result["status"] == 404

    def test_start_stop(self):
        s = WebServer()
        s.start()
        assert s.is_running() is True
        s.stop()
        assert s.is_running() is False
