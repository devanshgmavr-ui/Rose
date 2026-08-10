"""Tests for Stage 8.1 - Web Interface."""

import pytest
import time
from agent.web.server import (
    WebServer, WebConfig, ChatMessage, ChatRequest, ChatResponse,
)


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


class TestChatMessage:
    def test_creation(self):
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_to_dict(self):
        m = ChatMessage(role="assistant", content="hi", timestamp=1.0)
        d = m.to_dict()
        assert d["role"] == "assistant"
        assert d["timestamp"] == 1.0

    def test_from_dict(self):
        d = {"role": "user", "content": "test", "timestamp": 2.0}
        m = ChatMessage.from_dict(d)
        assert m.content == "test"


class TestChatRequest:
    def test_creation(self):
        r = ChatRequest(message="hello")
        assert r.message == "hello"

    def test_to_dict(self):
        r = ChatRequest(message="hi", session_id="s1")
        d = r.to_dict()
        assert d["session_id"] == "s1"


class TestChatResponse:
    def test_creation(self):
        r = ChatResponse(response="hello")
        assert r.response == "hello"

    def test_to_dict(self):
        r = ChatResponse(response="hi", tool_calls=3, execution_time=1.5)
        d = r.to_dict()
        assert d["tool_calls"] == 3
        assert d["execution_time"] == 1.5


class TestWebServer:
    def test_init(self):
        ws = WebServer()
        assert ws._config.port == 8080

    def test_init_custom_config(self):
        ws = WebServer(WebConfig(port=9090))
        assert ws._config.port == 9090

    def test_handle_index(self):
        ws = WebServer()
        result = ws.handle_request("GET", "/")
        assert result["status"] == 200
        assert "Rose Agent API" in result["data"]["name"]

    def test_handle_health(self):
        ws = WebServer()
        result = ws.handle_request("GET", "/health")
        assert result["status"] == 200
        assert result["data"]["status"] == "healthy"

    def test_handle_chat(self):
        ws = WebServer()
        result = ws.handle_request("POST", "/chat", {"message": "hello"})
        assert result["status"] == 200
        assert "Received: hello" in result["data"]["response"]

    def test_handle_chat_empty(self):
        ws = WebServer()
        result = ws.handle_request("POST", "/chat", {"message": ""})
        assert result["status"] == 400

    def test_handle_chat_no_message(self):
        ws = WebServer()
        result = ws.handle_request("POST", "/chat", {})
        assert result["status"] == 400

    def test_handle_chat_with_session(self):
        ws = WebServer()
        result = ws.handle_request(
            "POST", "/chat",
            {"message": "hi", "session_id": "s1"},
        )
        assert result["status"] == 200
        assert result["data"]["session_id"] == "s1"

    def test_handle_list_sessions(self):
        ws = WebServer()
        ws.handle_request("POST", "/chat", {"message": "hi", "session_id": "s1"})
        result = ws.handle_request("GET", "/sessions")
        assert result["status"] == 200
        assert "s1" in result["data"]["sessions"]

    def test_not_found(self):
        ws = WebServer()
        result = ws.handle_request("GET", "/nonexistent")
        assert result["status"] == 404

    def test_register_custom_route(self):
        ws = WebServer()
        ws.register_route("/custom", lambda b, h: {"status": 200, "data": "custom"})
        result = ws.handle_request("GET", "/custom")
        assert result["status"] == 200

    def test_middleware(self):
        ws = WebServer()
        ws.add_middleware(lambda m, p, b: {"status": 200, "data": "intercepted"})
        result = ws.handle_request("GET", "/")
        assert result["data"] == "intercepted"

    def test_chat_history(self):
        ws = WebServer()
        ws.handle_request("POST", "/chat", {"message": "hi", "session_id": "s1"})
        ws.handle_request("POST", "/chat", {"message": "bye", "session_id": "s1"})
        history = ws.get_chat_history("s1")
        assert len(history) == 4
        assert history[0].content == "hi"
        assert history[1].content == "Received: hi"

    def test_pattern_matching(self):
        ws = WebServer()
        assert ws._matches_pattern("/sessions/{id}", "/sessions/123") is True
        assert ws._matches_pattern("/sessions/{id}", "/sessions") is False
        assert ws._matches_pattern("/health", "/health") is True

    def test_get_config(self):
        ws = WebServer(WebConfig(port=7777))
        assert ws.get_config().port == 7777
