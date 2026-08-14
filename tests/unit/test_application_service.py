"""Tests for Phase 5 - Unified Application Service."""

import time
import threading
import pytest
from unittest.mock import MagicMock, patch
from agent.web.application import (
    ApplicationService, AppEvent, AppSession, AppTaskStatus,
    ConfirmationRequest, AppEventType,
)
from agent.web.events import EventBus, SSEHandler, WSEvent


class TestAppEvent:
    def test_event_creation(self):
        event = AppEvent(event_type=AppEventType.USER_MESSAGE, data={"message": "hello"})
        assert event.event_type == AppEventType.USER_MESSAGE
        assert event.data["message"] == "hello"
        assert event.event_id
        assert event.timestamp > 0

    def test_event_to_dict(self):
        event = AppEvent(event_type=AppEventType.ERROR, data={"error": "test"})
        d = event.to_dict()
        assert d["event_type"] == "error"
        assert d["data"]["error"] == "test"
        assert "event_id" in d
        assert "timestamp" in d

    def test_event_auto_id(self):
        e1 = AppEvent(event_type=AppEventType.STATUS_CHANGED)
        e2 = AppEvent(event_type=AppEventType.STATUS_CHANGED)
        assert e1.event_id != e2.event_id

    def test_event_default_data(self):
        event = AppEvent(event_type=AppEventType.SYSTEM_READY)
        assert event.data == {}


class TestAppSession:
    def test_session_creation(self):
        session = AppSession(session_id="abc123", title="Test Session")
        assert session.session_id == "abc123"
        assert session.title == "Test Session"
        assert session.status == "active"
        assert session.message_count == 0

    def test_session_to_dict(self):
        session = AppSession(session_id="s1", title="My Session")
        d = session.to_dict()
        assert d["session_id"] == "s1"
        assert d["title"] == "My Session"
        assert d["status"] == "active"


class TestAppTaskStatus:
    def test_task_status_creation(self):
        task = AppTaskStatus(task_id="t1", status="planning", objective="do stuff")
        assert task.task_id == "t1"
        assert task.status == "planning"
        assert task.objective == "do stuff"
        assert task.steps_completed == 0

    def test_task_status_to_dict(self):
        task = AppTaskStatus(task_id="t2", status="running", objective="run")
        d = task.to_dict()
        assert d["task_id"] == "t2"
        assert d["status"] == "running"
        assert d["objective"] == "run"


class TestConfirmationRequest:
    def test_creation(self):
        req = ConfirmationRequest(
            request_id="r1", tool_name="shell", action_description="run cmd",
        )
        assert req.request_id == "r1"
        assert req.responded is False
        assert req.approved is False

    def test_to_dict(self):
        req = ConfirmationRequest(
            request_id="r2", tool_name="shell", action_description="rm -rf",
            arguments={"command": "rm -rf /"},
        )
        d = req.to_dict()
        assert d["request_id"] == "r2"
        assert d["tool_name"] == "shell"
        assert d["arguments"]["command"] == "rm -rf /"


class TestApplicationService:
    def test_init(self):
        svc = ApplicationService()
        assert not svc._initialized
        assert svc._agent is None

    def test_init_with_agent(self):
        agent = MagicMock()
        svc = ApplicationService(agent=agent)
        assert svc._agent is agent

    def test_initialize_no_agent(self):
        svc = ApplicationService()
        result = svc.initialize()
        assert result is True
        assert svc._initialized is True

    def test_initialize_with_agent(self):
        agent = MagicMock()
        agent.initialize.return_value = True
        svc = ApplicationService(agent=agent)
        result = svc.initialize()
        assert result is True
        agent.initialize.assert_called_once()

    def test_initialize_agent_failure(self):
        agent = MagicMock()
        agent.initialize.return_value = False
        svc = ApplicationService(agent=agent)
        result = svc.initialize()
        assert result is False

    def test_shutdown(self):
        agent = MagicMock()
        svc = ApplicationService(agent=agent)
        svc._initialized = True
        svc.shutdown()
        agent.shutdown.assert_called_once()
        assert svc._initialized is False

    def test_create_session(self):
        svc = ApplicationService()
        session = svc.create_session("Test")
        assert session.session_id
        assert session.title == "Test"
        assert svc._current_session_id == session.session_id

    def test_create_session_auto_title(self):
        svc = ApplicationService()
        session = svc.create_session()
        assert session.title.startswith("Session")

    def test_get_session(self):
        svc = ApplicationService()
        session = svc.create_session("My Session")
        found = svc.get_session(session.session_id)
        assert found is session

    def test_get_session_not_found(self):
        svc = ApplicationService()
        assert svc.get_session("nonexistent") is None

    def test_list_sessions(self):
        svc = ApplicationService()
        svc.create_session("S1")
        svc.create_session("S2")
        sessions = svc.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_limit(self):
        svc = ApplicationService()
        for i in range(10):
            svc.create_session(f"S{i}")
        sessions = svc.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_send_message_not_initialized(self):
        svc = ApplicationService()
        result = svc.send_message("hello")
        assert result["success"] is False
        assert "not initialized" in result["error"].lower()

    def test_send_message_no_agent(self):
        svc = ApplicationService()
        svc._initialized = True
        result = svc.send_message("hello")
        assert result["success"] is False

    def test_send_message_with_agent(self):
        agent = MagicMock()
        response = MagicMock()
        response.text = "Hello back!"
        agent.chat.return_value = response
        svc = ApplicationService(agent=agent)
        svc._initialized = True
        result = svc.send_message("hello")
        assert result["success"] is True
        assert result["response"] == "Hello back!"
        agent.chat.assert_called_once_with("hello")

    def test_send_message_creates_session(self):
        agent = MagicMock()
        response = MagicMock()
        response.text = "hi"
        agent.chat.return_value = response
        svc = ApplicationService(agent=agent)
        svc._initialized = True
        result = svc.send_message("hello")
        assert result["session_id"]

    def test_execute_tool_not_initialized(self):
        svc = ApplicationService()
        result = svc.execute_tool("shell", {"command": "ls"})
        assert result["success"] is False

    def test_execute_tool_with_agent(self):
        agent = MagicMock()
        tool_result = MagicMock()
        tool_result.to_dict.return_value = {"success": True, "output": "done"}
        agent.execute_tool.return_value = tool_result
        svc = ApplicationService(agent=agent)
        svc._initialized = True
        result = svc.execute_tool("shell", {"command": "ls"})
        assert result["success"] is True

    def test_create_task_not_initialized(self):
        svc = ApplicationService()
        result = svc.create_task("do stuff")
        assert result.get("status") == "failed" or "error" in str(result)

    def test_get_task_status(self):
        svc = ApplicationService()
        task = AppTaskStatus(task_id="t1", status="running", objective="test")
        svc._pending_tasks["t1"] = task
        result = svc.get_task_status("t1")
        assert result["task_id"] == "t1"
        assert result["status"] == "running"

    def test_get_task_status_not_found(self):
        svc = ApplicationService()
        assert svc.get_task_status("nonexistent") is None

    def test_cancel_task(self):
        svc = ApplicationService()
        task = AppTaskStatus(task_id="t1", status="running", objective="test")
        svc._pending_tasks["t1"] = task
        result = svc.cancel_task("t1")
        assert result is True
        assert task.status == "cancelled"

    def test_cancel_task_not_found(self):
        svc = ApplicationService()
        assert svc.cancel_task("nonexistent") is False

    def test_list_tasks(self):
        svc = ApplicationService()
        svc._pending_tasks["t1"] = AppTaskStatus(task_id="t1", status="done", objective="a")
        svc._pending_tasks["t2"] = AppTaskStatus(task_id="t2", status="running", objective="b")
        tasks = svc.list_tasks()
        assert len(tasks) == 2

    def test_respond_confirmation(self):
        svc = ApplicationService()
        req = ConfirmationRequest(request_id="r1", tool_name="shell", action_description="run")
        svc._confirmation_requests["r1"] = req
        result = svc.respond_confirmation("r1", approved=True)
        assert result is True
        assert req.approved is True
        assert req.responded is True

    def test_respond_confirmation_not_found(self):
        svc = ApplicationService()
        assert svc.respond_confirmation("nonexistent", approved=True) is False

    def test_get_pending_confirmations(self):
        svc = ApplicationService()
        svc._confirmation_requests["r1"] = ConfirmationRequest(
            request_id="r1", tool_name="s", action_description="d",
        )
        svc._confirmation_requests["r2"] = ConfirmationRequest(
            request_id="r2", tool_name="s", action_description="d", responded=True,
        )
        pending = svc.get_pending_confirmations()
        assert len(pending) == 1
        assert pending[0]["request_id"] == "r1"

    def test_get_health_not_initialized(self):
        svc = ApplicationService()
        health = svc.get_health()
        assert health["status"] == "unhealthy"
        assert health["initialized"] is False

    def test_get_health_initialized(self):
        svc = ApplicationService()
        svc._initialized = True
        health = svc.get_health()
        assert health["status"] == "healthy"
        assert health["initialized"] is True

    def test_get_health_with_agent(self):
        agent = MagicMock()
        agent.health_check.return_value = {"status": "healthy"}
        svc = ApplicationService(agent=agent)
        svc._initialized = True
        health = svc.get_health()
        assert "agent" in health

    def test_get_tools(self):
        svc = ApplicationService()
        assert svc.get_tools() == []

    def test_get_tools_with_agent(self):
        agent = MagicMock()
        agent.get_tool_info.return_value = [{"name": "shell"}]
        svc = ApplicationService(agent=agent)
        tools = svc.get_tools()
        assert len(tools) == 1

    def test_event_callback(self):
        svc = ApplicationService()
        events = []
        svc.register_event_callback(lambda e: events.append(e))
        svc._emit_event(AppEventType.STATUS_CHANGED, {"status": "test"})
        assert len(events) == 1
        assert events[0].event_type == AppEventType.STATUS_CHANGED

    def test_get_events(self):
        svc = ApplicationService()
        svc._emit_event(AppEventType.STATUS_CHANGED, {"status": "test"})
        events = svc.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "status_changed"

    def test_get_app_info(self):
        svc = ApplicationService()
        info = svc.get_app_info()
        assert info["name"] == "Rose"
        assert info["version"] == "1.1.0"

    def test_thread_safety(self):
        svc = ApplicationService()
        errors = []

        def create_sessions():
            try:
                for i in range(20):
                    svc.create_session(f"Thread {threading.current_thread().name} {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sessions) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(svc._sessions) == 100


class TestEventBus:
    def test_init(self):
        bus = EventBus()
        assert bus._active is True

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        event = WSEvent(event_type="test", data={"value": 42})
        bus.publish(event)
        assert len(received) == 1
        assert received[0].data["value"] == 42

    def test_wildcard_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.emit("any_event", {"key": "value"})
        assert len(received) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        callback = lambda e: received.append(e)
        bus.subscribe("test", callback)
        bus.emit("test")
        assert len(received) == 1
        bus.unsubscribe("test", callback)
        bus.emit("test")
        assert len(received) == 1

    def test_get_history(self):
        bus = EventBus()
        bus.emit("e1")
        bus.emit("e2")
        bus.emit("e3")
        history = bus.get_history()
        assert len(history) == 3

    def test_get_history_with_filter(self):
        bus = EventBus()
        bus.emit("e1")
        bus.emit("e2")
        bus.emit("e1")
        history = bus.get_history(event_type="e1")
        assert len(history) == 2

    def test_get_history_limit(self):
        bus = EventBus()
        for i in range(10):
            bus.emit(f"e{i}")
        history = bus.get_history(limit=3)
        assert len(history) == 3

    def test_shutdown(self):
        bus = EventBus()
        bus.shutdown()
        assert bus._active is False
        assert len(bus._subscribers) == 0

    def test_emit_convenience(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        bus.emit("test", {"a": 1}, session_id="s1")
        assert len(received) == 1
        assert received[0].session_id == "s1"

    def test_subscriber_error_handling(self):
        bus = EventBus()
        def bad_callback(e):
            raise ValueError("oops")
        good_received = []
        bus.subscribe("test", bad_callback)
        bus.subscribe("test", lambda e: good_received.append(e))
        bus.emit("test")
        assert len(good_received) == 1

    def test_multiple_subscribers(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("test", lambda e: r1.append(e))
        bus.subscribe("test", lambda e: r2.append(e))
        bus.emit("test")
        assert len(r1) == 1
        assert len(r2) == 1


class TestSSEHandler:
    def test_create_client(self):
        bus = EventBus()
        sse = SSEHandler(bus)
        q = sse.create_client("c1")
        assert q is not None

    def test_broadcast(self):
        bus = EventBus()
        sse = SSEHandler(bus)
        q = sse.create_client("c1")
        bus.emit("test", {"key": "value"})
        event_json = q.get(timeout=1.0)
        assert event_json is not None
        assert "test" in event_json

    def test_remove_client(self):
        bus = EventBus()
        sse = SSEHandler(bus)
        sse.create_client("c1")
        sse.remove_client("c1")
        assert "c1" not in sse._client_queues

    def test_get_events_for_client(self):
        bus = EventBus()
        sse = SSEHandler(bus)
        q = sse.create_client("c1")
        bus.emit("test")
        result = sse.get_events_for_client("c1", timeout=1.0)
        assert result is not None

    def test_get_events_for_unknown_client(self):
        bus = EventBus()
        sse = SSEHandler(bus)
        result = sse.get_events_for_client("unknown", timeout=0.1)
        assert result is None

    def test_multiple_clients(self):
        bus = EventBus()
        sse = SSEHandler(bus)
        q1 = sse.create_client("c1")
        q2 = sse.create_client("c2")
        bus.emit("test")
        assert not q1.empty()
        assert not q2.empty()
