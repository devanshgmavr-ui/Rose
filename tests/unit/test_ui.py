"""Tests for Phase 6 - Rose User Interface."""

import time
import threading
import pytest
from unittest.mock import MagicMock
from agent.ui import (
    RoseUI, UIConfig, UITheme, ChatMessageUI, MessageDisplay, InputHandler,
    TaskPanel, TaskInfo, SettingsPanel, StatusBar, ConfirmationDialog, Sidebar,
)


class TestUIConfig:
    def test_defaults(self):
        cfg = UIConfig()
        assert cfg.theme == UITheme.DARK
        assert cfg.window_width == 1200
        assert cfg.window_height == 800
        assert cfg.sidebar_width == 250
        assert cfg.show_sidebar is True
        assert cfg.show_toolbar is True
        assert cfg.show_statusbar is True
        assert cfg.font_size == 14
        assert cfg.auto_scroll is True

    def test_to_dict(self):
        cfg = UIConfig()
        d = cfg.to_dict()
        assert d["theme"] == "dark"
        assert d["window_width"] == 1200
        assert d["font_size"] == 14


class TestChatMessageUI:
    def test_creation(self):
        msg = ChatMessageUI(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp > 0

    def test_to_dict(self):
        msg = ChatMessageUI(role="assistant", content="hi", message_id="m1")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "hi"
        assert d["message_id"] == "m1"


class TestMessageDisplay:
    def test_add_message(self):
        display = MessageDisplay()
        msg = display.add_message("user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert display.message_count() == 1

    def test_get_messages(self):
        display = MessageDisplay()
        display.add_message("user", "m1")
        display.add_message("assistant", "m2")
        display.add_message("user", "m3")
        msgs = display.get_messages(limit=2)
        assert len(msgs) == 2
        assert msgs[0].content == "m2"

    def test_get_all_messages(self):
        display = MessageDisplay()
        display.add_message("user", "m1")
        display.add_message("user", "m2")
        all_msgs = display.get_all_messages()
        assert len(all_msgs) == 2

    def test_clear(self):
        display = MessageDisplay()
        display.add_message("user", "hello")
        display.clear()
        assert display.message_count() == 0

    def test_max_messages(self):
        display = MessageDisplay(max_messages=3)
        for i in range(5):
            display.add_message("user", f"msg{i}")
        assert display.message_count() == 3
        msgs = display.get_all_messages()
        assert msgs[0].content == "msg2"

    def test_on_message_callback(self):
        display = MessageDisplay()
        received = []
        display.on_message(lambda m: received.append(m))
        display.add_message("user", "test")
        assert len(received) == 1
        assert received[0].content == "test"

    def test_search(self):
        display = MessageDisplay()
        display.add_message("user", "hello world")
        display.add_message("assistant", "goodbye")
        display.add_message("user", "hello again")
        results = display.search("hello")
        assert len(results) == 2

    def test_delete_message(self):
        display = MessageDisplay()
        display.add_message("user", "m1", message_id="id1")
        display.add_message("user", "m2", message_id="id2")
        result = display.delete_message("id1")
        assert result is True
        assert display.message_count() == 1

    def test_delete_nonexistent(self):
        display = MessageDisplay()
        display.add_message("user", "m1", message_id="id1")
        result = display.delete_message("nonexistent")
        assert result is False
        assert display.message_count() == 1


class TestInputHandler:
    def test_validate_empty(self):
        handler = InputHandler()
        valid, error = handler.validate("")
        assert valid is False

    def test_validate_too_long(self):
        handler = InputHandler(max_length=10)
        valid, error = handler.validate("a" * 11)
        assert valid is False
        assert "too long" in error.lower()

    def test_validate_valid(self):
        handler = InputHandler()
        valid, error = handler.validate("hello")
        assert valid is True

    def test_submit(self):
        handler = InputHandler()
        received = []
        handler.on_submit(lambda t: received.append(t))
        result = handler.submit("hello")
        assert result[0] is True
        assert len(received) == 1

    def test_submit_empty(self):
        handler = InputHandler()
        result = handler.submit("")
        assert result[0] is False

    def test_history(self):
        handler = InputHandler()
        handler.submit("cmd1")
        handler.submit("cmd2")
        history = handler.get_history()
        assert history == ["cmd1", "cmd2"]

    def test_navigate_history(self):
        handler = InputHandler()
        handler.submit("cmd1")
        handler.submit("cmd2")
        cmd = handler.navigate_history(-1)
        assert cmd == "cmd2"

    def test_navigate_history_empty(self):
        handler = InputHandler()
        assert handler.navigate_history(-1) is None

    def test_clear_history(self):
        handler = InputHandler()
        handler.submit("cmd1")
        handler.clear_history()
        assert len(handler.get_history()) == 0

    def test_processing_state(self):
        handler = InputHandler()
        assert handler.is_processing() is False
        handler.set_processing(True)
        assert handler.is_processing() is True

    def test_submit_while_processing(self):
        handler = InputHandler()
        handler.set_processing(True)
        result = handler.submit("hello")
        assert result[0] is False
        assert "processing" in result[1].lower()


class TestTaskPanel:
    def test_add_task(self):
        panel = TaskPanel()
        task = panel.add_task("t1", "do stuff")
        assert task.task_id == "t1"
        assert task.objective == "do stuff"
        assert panel.task_count() == 1

    def test_update_task(self):
        panel = TaskPanel()
        panel.add_task("t1", "do stuff")
        result = panel.update_task("t1", status="running", progress="50%")
        assert result is True
        task = panel.get_task("t1")
        assert task.status == "running"
        assert task.progress == "50%"

    def test_update_nonexistent(self):
        panel = TaskPanel()
        result = panel.update_task("nonexistent", status="running")
        assert result is False

    def test_remove_task(self):
        panel = TaskPanel()
        panel.add_task("t1", "stuff")
        result = panel.remove_task("t1")
        assert result is True
        assert panel.task_count() == 0

    def test_remove_nonexistent(self):
        panel = TaskPanel()
        result = panel.remove_task("nonexistent")
        assert result is False

    def test_get_tasks(self):
        panel = TaskPanel()
        panel.add_task("t1", "a")
        panel.add_task("t2", "b")
        tasks = panel.get_tasks()
        assert len(tasks) == 2

    def test_select_task(self):
        panel = TaskPanel()
        panel.add_task("t1", "a")
        panel.add_task("t2", "b")
        panel.select_task("t2")
        selected = panel.get_selected()
        assert selected.task_id == "t2"

    def test_select_nonexistent(self):
        panel = TaskPanel()
        panel.select_task("nonexistent")
        assert panel.get_selected() is None

    def test_active_tasks(self):
        panel = TaskPanel()
        panel.add_task("t1", "a", status="running")
        panel.add_task("t2", "b", status="completed")
        panel.add_task("t3", "c", status="planning")
        active = panel.active_tasks()
        assert len(active) == 2

    def test_on_change_callback(self):
        panel = TaskPanel()
        changes = []
        panel.on_change(lambda: changes.append(True))
        panel.add_task("t1", "a")
        assert len(changes) == 1

    def test_task_info_to_dict(self):
        task = TaskInfo(task_id="t1", objective="test", status="done")
        d = task.to_dict()
        assert d["task_id"] == "t1"
        assert d["status"] == "done"


class TestSettingsPanel:
    def test_get_set(self):
        panel = SettingsPanel()
        panel.set("theme", "light")
        assert panel.get("theme") == "light"

    def test_get_default(self):
        panel = SettingsPanel()
        assert panel.get("nonexistent", "default") == "default"

    def test_has_changes(self):
        panel = SettingsPanel()
        assert panel.has_changes() is False
        panel.set("theme", "light")
        assert panel.has_changes() is True

    def test_reset_changes(self):
        panel = SettingsPanel()
        panel.set("theme", "light")
        panel.reset_changes()
        assert panel.has_changes() is False

    def test_get_all(self):
        panel = SettingsPanel()
        all_settings = panel.get_all()
        assert "theme" in all_settings
        assert "font_size" in all_settings

    def test_update(self):
        panel = SettingsPanel()
        panel.update(theme="blue", font_size=20)
        assert panel.get("theme") == "blue"
        assert panel.get("font_size") == 20

    def test_on_change_callback(self):
        panel = SettingsPanel()
        changes = []
        panel.on_change(lambda: changes.append(True))
        panel.set("theme", "light")
        assert len(changes) == 1

    def test_no_change_same_value(self):
        panel = SettingsPanel()
        panel.set("theme", "dark")
        assert panel.has_changes() is False


class TestStatusBar:
    def test_set_status(self):
        bar = StatusBar()
        bar.set_status("Working...")
        assert bar.get_status() == "Working..."

    def test_set_agent_status(self):
        bar = StatusBar()
        bar.set_agent_status("online")
        assert bar.get_agent_status() == "online"

    def test_set_progress(self):
        bar = StatusBar()
        bar.set_progress(0.5)
        assert bar.get_progress() == 0.5

    def test_progress_clamp(self):
        bar = StatusBar()
        bar.set_progress(1.5)
        assert bar.get_progress() == 1.0
        bar.set_progress(-0.5)
        assert bar.get_progress() == 0.0

    def test_set_detail(self):
        bar = StatusBar()
        bar.set_detail("model", "llama")
        details = bar.get_details()
        assert details["model"] == "llama"

    def test_clear_progress(self):
        bar = StatusBar()
        bar.set_progress(0.7)
        bar.clear_progress()
        assert bar.get_progress() == 0.0

    def test_on_change_callback(self):
        bar = StatusBar()
        changes = []
        bar.on_change(lambda: changes.append(True))
        bar.set_status("test")
        assert len(changes) == 1


class TestConfirmationDialog:
    def test_request_confirmation(self):
        dialog = ConfirmationDialog()
        req = dialog.request_confirmation("r1", "shell", "run command")
        assert req["request_id"] == "r1"
        assert req["responded"] is False

    def test_respond(self):
        dialog = ConfirmationDialog()
        dialog.request_confirmation("r1", "shell", "run cmd")
        result = dialog.respond("r1", approved=True)
        assert result is True
        pending = dialog.get_pending()
        assert len(pending) == 0

    def test_respond_nonexistent(self):
        dialog = ConfirmationDialog()
        result = dialog.respond("nonexistent", approved=True)
        assert result is False

    def test_pending_count(self):
        dialog = ConfirmationDialog()
        dialog.request_confirmation("r1", "s", "d1")
        dialog.request_confirmation("r2", "s", "d2")
        assert dialog.pending_count() == 2
        dialog.respond("r1", approved=True)
        assert dialog.pending_count() == 1

    def test_clear_responded(self):
        dialog = ConfirmationDialog()
        dialog.request_confirmation("r1", "s", "d1")
        dialog.respond("r1", approved=True)
        dialog.clear_responded()
        assert len(dialog.get_all()) == 0

    def test_on_change_callback(self):
        dialog = ConfirmationDialog()
        changes = []
        dialog.on_change(lambda: changes.append(True))
        dialog.request_confirmation("r1", "s", "d")
        assert len(changes) == 1


class TestSidebar:
    def test_toggle_visibility(self):
        sidebar = Sidebar()
        assert sidebar.is_visible() is True
        sidebar.toggle_visibility()
        assert sidebar.is_visible() is False

    def test_select_section(self):
        sidebar = Sidebar()
        result = sidebar.select_section("tasks")
        assert result is True
        assert sidebar.get_selected_section() == "tasks"

    def test_select_invalid_section(self):
        sidebar = Sidebar()
        result = sidebar.select_section("nonexistent")
        assert result is False

    def test_get_sections(self):
        sidebar = Sidebar()
        sections = sidebar.get_sections()
        assert "chat" in sections
        assert "tasks" in sections

    def test_on_change_callback(self):
        sidebar = Sidebar()
        changes = []
        sidebar.on_change(lambda: changes.append(True))
        sidebar.toggle_visibility()
        assert len(changes) == 1


class TestRoseUI:
    def test_init(self):
        ui = RoseUI()
        assert ui._initialized is False
        assert ui._app is None

    def test_init_with_app(self):
        app = MagicMock()
        ui = RoseUI(app_service=app)
        assert ui._app is app

    def test_initialize_no_app(self):
        ui = RoseUI()
        result = ui.initialize()
        assert result is True
        assert ui._initialized is True

    def test_initialize_with_app(self):
        app = MagicMock()
        app.initialize.return_value = True
        ui = RoseUI(app_service=app)
        result = ui.initialize()
        assert result is True

    def test_initialize_app_failure(self):
        app = MagicMock()
        app.initialize.return_value = False
        ui = RoseUI(app_service=app)
        result = ui.initialize()
        assert result is False

    def test_shutdown(self):
        app = MagicMock()
        ui = RoseUI(app_service=app)
        ui._initialized = True
        ui.shutdown()
        app.shutdown.assert_called_once()
        assert ui._initialized is False

    def test_send_message_no_app(self):
        ui = RoseUI()
        ui._initialized = True
        result = ui.send_message("hello")
        assert result is True
        assert ui._message_display.message_count() == 2

    def test_send_message_with_app(self):
        app = MagicMock()
        app.send_message.return_value = {"success": True, "response": "hi back"}
        ui = RoseUI(app_service=app)
        ui._initialized = True
        result = ui.send_message("hello")
        assert result is True
        assert ui._message_display.message_count() == 2

    def test_send_message_empty(self):
        ui = RoseUI()
        ui._initialized = True
        result = ui.send_message("")
        assert result is False

    def test_send_message_app_error(self):
        app = MagicMock()
        app.send_message.return_value = {"success": False, "error": "failed"}
        ui = RoseUI(app_service=app)
        ui._initialized = True
        result = ui.send_message("hello")
        assert result is False

    def test_create_task_no_app(self):
        ui = RoseUI()
        ui._initialized = True
        task_id = ui.create_task("do stuff")
        assert task_id is not None
        assert ui._task_panel.task_count() == 1

    def test_create_task_empty(self):
        ui = RoseUI()
        ui._initialized = True
        task_id = ui.create_task("   ")
        assert task_id is None

    def test_create_task_with_app(self):
        app = MagicMock()
        app.create_task.return_value = {
            "task_id": "t1", "status": "planning",
            "steps_total": 5, "steps_completed": 0,
        }
        ui = RoseUI(app_service=app)
        ui._initialized = True
        task_id = ui.create_task("do stuff")
        assert task_id == "t1"

    def test_cancel_task_no_app(self):
        ui = RoseUI()
        ui._task_panel.add_task("t1", "stuff")
        result = ui.cancel_task("t1")
        assert result is True

    def test_confirm_action(self):
        app = MagicMock()
        ui = RoseUI(app_service=app)
        ui._confirmation_dialog.request_confirmation("r1", "shell", "run")
        ui.confirm_action("r1", approved=True)
        app.respond_confirmation.assert_called_once_with("r1", True)

    def test_refresh_health(self):
        app = MagicMock()
        app.get_health.return_value = {"status": "healthy"}
        ui = RoseUI(app_service=app)
        health = ui.refresh_health()
        assert health["status"] == "healthy"
        assert ui._status_bar.get_agent_status() == "online"

    def test_refresh_health_no_app(self):
        ui = RoseUI()
        health = ui.refresh_health()
        assert health["status"] == "no_backend"

    def test_clear_chat(self):
        ui = RoseUI()
        ui._message_display.add_message("user", "hello")
        ui.clear_chat()
        assert ui._message_display.message_count() == 1

    def test_get_state(self):
        ui = RoseUI()
        ui._initialized = True
        state = ui.get_state()
        assert state["initialized"] is True
        assert state["message_count"] == 0
        assert state["task_count"] == 0

    def test_properties(self):
        ui = RoseUI()
        assert ui.messages is ui._message_display
        assert ui.input is ui._input_handler
        assert ui.tasks is ui._task_panel
        assert ui.settings is ui._settings_panel
        assert ui.status is ui._status_bar
        assert ui.confirmations is ui._confirmation_dialog
        assert ui.sidebar is ui._sidebar

    def test_set_app_service(self):
        ui = RoseUI()
        app = MagicMock()
        ui.set_app_service(app)
        assert ui._app is app

    def test_thread_safety(self):
        ui = RoseUI()
        errors = []

        def add_messages():
            try:
                for i in range(20):
                    ui._message_display.add_message("user", f"msg{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert ui._message_display.message_count() == 100
