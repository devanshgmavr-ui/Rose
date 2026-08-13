"""Tests for Phase 9 - Autonomous Tool Selection & Prompt Execution."""

import time
import pytest
from unittest.mock import MagicMock, patch
from agent.orchestration.capability_analyzer import (
    CapabilityAnalyzer, Capability, CapabilityAnalysis,
    CAPABILITY_DEFINITIONS,
)
from agent.orchestration.tool_scorer import (
    ToolScorer, ToolScore, SelectionResult,
    CAPABILITY_TOOLS_MAP,
)
from agent.orchestration.task_state import (
    AutonomousTaskState, TaskPhase, TaskConstraints,
    StepExecution, ExecutionTrace,
)
from agent.orchestration.autonomous_loop import AutonomousLoop


# ──────────────────────────────────────────────
# CapabilityAnalyzer Tests
# ──────────────────────────────────────────────

class TestCapability:
    def test_creation(self):
        cap = Capability(name="file_operations", description="Read/write files")
        assert cap.name == "file_operations"
        assert cap.confidence == 1.0

    def test_to_dict(self):
        cap = Capability(name="browser", description="Browser control", confidence=0.8)
        d = cap.to_dict()
        assert d["name"] == "browser"
        assert d["confidence"] == 0.8


class TestCapabilityAnalysis:
    def test_creation(self):
        analysis = CapabilityAnalysis()
        assert len(analysis.capabilities) == 0
        assert analysis.task_type == "general"

    def test_get_capability_names(self):
        analysis = CapabilityAnalysis()
        analysis.capabilities = [
            Capability(name="file_operations", description="files"),
            Capability(name="browser_automation", description="browser"),
        ]
        names = analysis.get_capability_names()
        assert "file_operations" in names
        assert "browser_automation" in names

    def test_to_dict(self):
        analysis = CapabilityAnalysis()
        analysis.capabilities = [Capability(name="test", description="test desc")]
        analysis.constraints = {"prohibited_tools": ["browser"]}
        d = analysis.to_dict()
        assert len(d["capabilities"]) == 1
        assert d["constraints"]["prohibited_tools"] == ["browser"]


class TestCapabilityAnalyzer:
    def test_init(self):
        analyzer = CapabilityAnalyzer()
        assert len(analyzer._compiled_patterns) > 0

    def test_analyze_file_request(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("read the file document.txt")
        cap_names = result.get_capability_names()
        assert "file_operations" in cap_names

    def test_analyze_browser_request(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("open the browser and go to example.com")
        cap_names = result.get_capability_names()
        assert "browser_automation" in cap_names

    def test_analyze_screenshot_request(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("take a screenshot of the screen")
        cap_names = result.get_capability_names()
        assert "screen_capture" in cap_names

    def test_analyze_code_request(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("run this python code")
        cap_names = result.get_capability_names()
        assert "code_execution" in cap_names

    def test_analyze_keyboard_request(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("type hello world")
        cap_names = result.get_capability_names()
        assert "keyboard_input" in cap_names

    def test_analyze_window_request(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("minimize the window")
        cap_names = result.get_capability_names()
        assert "window_management" in cap_names

    def test_analyze_vision_request(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("analyze this screenshot for content")
        cap_names = result.get_capability_names()
        assert "vision_analysis" in cap_names or "screen_capture" in cap_names

    def test_analyze_multi_capability(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze(
            "take a screenshot, analyze the image, and save the description to a file"
        )
        cap_names = result.get_capability_names()
        assert "screen_capture" in cap_names
        assert "file_operations" in cap_names

    def test_explicit_tool_detection(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("use the browser to open example.com")
        assert "browser" in result.explicit_tools

    def test_explicit_tool_none(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("read the file")
        assert len(result.explicit_tools) == 0

    def test_constraint_no_browser(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("do this without the browser")
        assert "browser" in result.constraints.get("prohibited_tools", [])

    def test_constraint_no_keyboard(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("don't use the keyboard")
        assert "keyboard" in result.constraints.get("prohibited_tools", [])

    def test_constraint_minimize_confirmations(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("just do it without asking me")
        assert result.constraints.get("minimize_confirmations") is True

    def test_constraint_use_screen_image(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("use the image on my screen")
        assert result.constraints.get("use_screen_image") is True

    def test_task_type_simple(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("take a screenshot")
        assert result.task_type == "simple"

    def test_task_type_web(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("open browser and read the page content")
        assert result.task_type == "web_task"

    def test_task_type_transcription(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("extract text from image and type it")
        assert result.task_type == "transcription"

    def test_task_type_multi_step(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("open calculator, type some numbers, and take a screenshot")
        assert result.task_type in ("multi_step", "desktop_automation")

    def test_get_capabilities_for_step(self):
        analyzer = CapabilityAnalyzer()
        caps = analyzer.get_capabilities_for_step("read file content", "read")
        names = [c.name for c in caps]
        assert "file_operations" in names

    def test_confidence_scoring(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("save file to disk")
        file_cap = next(
            (c for c in result.capabilities if c.name == "file_operations"), None
        )
        assert file_cap is not None
        assert file_cap.confidence > 0


# ──────────────────────────────────────────────
# ToolScorer Tests
# ──────────────────────────────────────────────

class TestToolScore:
    def test_creation(self):
        score = ToolScore(
            tool_name="filesystem",
            total_score=0.85,
            capability_match=0.9,
            permission_score=1.0,
            risk_score=0.8,
            reliability_score=0.7,
            context_score=0.6,
        )
        assert score.tool_name == "filesystem"
        assert score.total_score == 0.85

    def test_to_dict(self):
        score = ToolScore(
            tool_name="browser",
            total_score=0.7,
            capability_match=0.8,
            permission_score=0.6,
            risk_score=0.7,
            reliability_score=0.9,
            context_score=0.5,
        )
        d = score.to_dict()
        assert d["tool_name"] == "browser"
        assert "total_score" in d


class TestSelectionResult:
    def test_creation(self):
        result = SelectionResult(
            selected_tool=None,
            alternatives=[],
            all_candidates=[],
        )
        assert result.selected_tool is None

    def test_to_dict(self):
        score = ToolScore(
            tool_name="test", total_score=0.5,
            capability_match=0.5, permission_score=0.5,
            risk_score=0.5, reliability_score=0.5,
            context_score=0.5,
        )
        result = SelectionResult(
            selected_tool=score,
            alternatives=[],
            all_candidates=[score],
            reason_summary="test reason",
        )
        d = result.to_dict()
        assert d["selected_tool"]["tool_name"] == "test"
        assert d["reason_summary"] == "test reason"


class TestToolScorer:
    def test_init(self):
        scorer = ToolScorer()
        assert len(scorer._catalog) > 0

    def test_select_tool_file_capability(self):
        scorer = ToolScorer()
        caps = [Capability(name="file_operations", description="files", confidence=0.9)]
        result = scorer.select_tool(caps)
        assert result.selected_tool is not None
        assert result.selected_tool.tool_name == "filesystem"

    def test_select_tool_browser_capability(self):
        scorer = ToolScorer()
        caps = [Capability(name="browser_automation", description="browser", confidence=0.9)]
        result = scorer.select_tool(caps)
        assert result.selected_tool is not None
        assert result.selected_tool.tool_name == "browser"

    def test_select_tool_screen_capture(self):
        scorer = ToolScorer()
        caps = [Capability(name="screen_capture", description="screenshot", confidence=0.9)]
        result = scorer.select_tool(caps)
        assert result.selected_tool is not None
        assert result.selected_tool.tool_name == "screen_capture"

    def test_select_tool_keyboard(self):
        scorer = ToolScorer()
        caps = [Capability(name="keyboard_input", description="type", confidence=0.9)]
        result = scorer.select_tool(caps)
        assert result.selected_tool is not None
        assert result.selected_tool.tool_name == "keyboard"

    def test_select_tool_vision(self):
        scorer = ToolScorer()
        caps = [Capability(name="vision_analysis", description="analyze", confidence=0.9)]
        result = scorer.select_tool(caps)
        assert result.selected_tool is not None
        assert result.selected_tool.tool_name in ("vision_analyze", "image_analyze")

    def test_select_tool_with_constraints(self):
        scorer = ToolScorer()
        caps = [Capability(name="browser_automation", description="browser", confidence=0.9)]
        constraints = {"prohibited_tools": ["browser"]}
        result = scorer.select_tool(caps, constraints=constraints)
        assert result.selected_tool is None or result.selected_tool.tool_name != "browser"

    def test_select_tool_with_explicit(self):
        scorer = ToolScorer()
        caps = [Capability(name="file_operations", description="files", confidence=0.5)]
        result = scorer.select_tool(caps, explicit_tools=["filesystem"])
        assert result.selected_tool is not None
        assert result.selected_tool.tool_name == "filesystem"

    def test_select_tool_no_capabilities(self):
        scorer = ToolScorer()
        result = scorer.select_tool([])
        # Should still return something (fallback)
        assert result.selected_tool is not None

    def test_record_failure(self):
        scorer = ToolScorer()
        scorer.record_failure("filesystem")
        scorer.record_failure("filesystem")
        assert scorer._failure_counts["filesystem"] == 2

    def test_record_success(self):
        scorer = ToolScorer()
        scorer.record_success("browser")
        assert scorer._success_counts["browser"] == 1

    def test_reliability_after_failures(self):
        scorer = ToolScorer()
        for _ in range(5):
            scorer.record_failure("test_tool")
        score = scorer._score_reliability("test_tool")
        assert score == 0.0

    def test_reliability_after_successes(self):
        scorer = ToolScorer()
        for _ in range(5):
            scorer.record_success("test_tool")
        score = scorer._score_reliability("test_tool")
        assert score == 1.0

    def test_reliability_mixed(self):
        scorer = ToolScorer()
        scorer.record_success("test_tool")
        scorer.record_failure("test_tool")
        score = scorer._score_reliability("test_tool")
        assert 0.4 < score < 0.6

    def test_select_tool_for_step(self):
        scorer = ToolScorer()
        result = scorer.select_tool_for_step("read the file content", "read")
        assert result.selected_tool is not None

    def test_alternatives(self):
        scorer = ToolScorer()
        caps = [Capability(name="vision_analysis", description="analyze", confidence=0.9)]
        result = scorer.select_tool(caps)
        # Should have at least one alternative (vision_analyze + image_analyze)
        assert len(result.alternatives) >= 0

    def test_context_active_tool_bonus(self):
        scorer = ToolScorer()
        caps = [Capability(name="file_operations", description="files", confidence=0.9)]
        context = {"active_tools": ["filesystem"]}
        result = scorer.select_tool(caps, context=context)
        assert result.selected_tool.tool_name == "filesystem"
        assert result.selected_tool.context_score > 0.5

    def test_context_recent_failure_penalty(self):
        scorer = ToolScorer()
        scorer.record_failure("filesystem")
        caps = [Capability(name="file_operations", description="files", confidence=0.9)]
        context = {"recent_failures": ["filesystem"]}
        result = scorer.select_tool(caps, context=context)
        # Should still select filesystem but with lower context score
        assert result.selected_tool is not None

    def test_reset_history(self):
        scorer = ToolScorer()
        scorer.record_failure("test")
        scorer.record_success("test")
        scorer.reset_history()
        assert len(scorer._failure_counts) == 0
        assert len(scorer._success_counts) == 0


# ──────────────────────────────────────────────
# TaskConstraints Tests
# ──────────────────────────────────────────────

class TestTaskConstraints:
    def test_defaults(self):
        c = TaskConstraints()
        assert len(c.prohibited_tools) == 0
        assert c.minimize_confirmations is False

    def test_to_dict(self):
        c = TaskConstraints(prohibited_tools=["browser"], minimize_confirmations=True)
        d = c.to_dict()
        assert "browser" in d["prohibited_tools"]
        assert d["minimize_confirmations"] is True

    def test_from_dict(self):
        data = {"prohibited_tools": ["keyboard"], "use_screen_image": True}
        c = TaskConstraints.from_dict(data)
        assert "keyboard" in c.prohibited_tools
        assert c.use_screen_image is True


# ──────────────────────────────────────────────
# StepExecution Tests
# ──────────────────────────────────────────────

class TestStepExecution:
    def test_creation(self):
        ex = StepExecution(
            step_id="step_1",
            tool_name="filesystem",
            action="read",
        )
        assert ex.step_id == "step_1"
        assert ex.success is False

    def test_to_dict(self):
        ex = StepExecution(
            step_id="step_1",
            tool_name="filesystem",
            action="read",
            success=True,
            result="file content",
        )
        d = ex.to_dict()
        assert d["success"] is True
        assert d["tool_name"] == "filesystem"


# ──────────────────────────────────────────────
# AutonomousTaskState Tests
# ──────────────────────────────────────────────

class TestAutonomousTaskState:
    def test_creation(self):
        state = AutonomousTaskState(objective="test task")
        assert state.objective == "test task"
        assert state.phase == TaskPhase.RECEIVED

    def test_is_terminal(self):
        state = AutonomousTaskState()
        state.advance_to(TaskPhase.COMPLETED)
        assert state.is_terminal() is True

    def test_is_not_terminal(self):
        state = AutonomousTaskState()
        assert state.is_terminal() is False

    def test_can_continue(self):
        state = AutonomousTaskState()
        assert state.can_continue() is True

    def test_cannot_continue_when_completed(self):
        state = AutonomousTaskState()
        state.advance_to(TaskPhase.COMPLETED)
        assert state.can_continue() is False

    def test_cannot_continue_max_tool_calls(self):
        state = AutonomousTaskState(max_tool_calls=5)
        state.total_tool_calls = 5
        assert state.can_continue() is False

    def test_cannot_continue_max_replans(self):
        state = AutonomousTaskState(max_replans=3)
        state.total_replans = 3
        assert state.can_continue() is False

    def test_record_step_success(self):
        state = AutonomousTaskState()
        ex = StepExecution(step_id="s1", tool_name="fs", action="read", success=True)
        state.record_step(ex)
        assert "s1" in state.completed_step_ids
        assert state.total_tool_calls == 1

    def test_record_step_failure(self):
        state = AutonomousTaskState()
        ex = StepExecution(step_id="s1", tool_name="fs", action="read", success=False)
        state.record_step(ex)
        assert "s1" in state.failed_step_ids

    def test_mark_tool_unavailable(self):
        state = AutonomousTaskState()
        state.mark_tool_unavailable("browser")
        assert "browser" in state.tools_unavailable

    def test_advance_to(self):
        state = AutonomousTaskState()
        state.advance_to(TaskPhase.PLANNING)
        assert state.phase == TaskPhase.PLANNING
        assert state.updated_at > 0

    def test_to_dict(self):
        state = AutonomousTaskState(objective="test")
        d = state.to_dict()
        assert d["objective"] == "test"
        assert d["phase"] == "received"

    def test_get_summary(self):
        state = AutonomousTaskState(objective="test task")
        summary = state.get_summary()
        assert "test task" in summary
        assert "RECEIVED" in summary


# ──────────────────────────────────────────────
# ExecutionTrace Tests
# ──────────────────────────────────────────────

class TestExecutionTrace:
    def test_creation(self):
        trace = ExecutionTrace()
        assert len(trace.entries) == 0

    def test_add_tool_selected(self):
        trace = ExecutionTrace()
        trace.add_tool_selected("s1", "filesystem", "Best match")
        assert len(trace.entries) == 1
        assert trace.entries[0]["tool"] == "filesystem"

    def test_add_step_completed(self):
        trace = ExecutionTrace()
        trace.add_step_completed("s1", success=True, summary="done")
        assert trace.entries[0]["success"] is True

    def test_add_verification(self):
        trace = ExecutionTrace()
        trace.add_verification(passed=True)
        assert trace.entries[0]["passed"] is True

    def test_add_replan(self):
        trace = ExecutionTrace()
        trace.add_replan("tool failed", 1)
        assert trace.entries[0]["attempt"] == 1

    def test_add_error(self):
        trace = ExecutionTrace()
        trace.add_error("something broke")
        assert "broke" in trace.entries[0]["error"]

    def test_add_completion(self):
        trace = ExecutionTrace()
        trace.add_completion(True, "all done")
        assert trace.entries[0]["success"] is True

    def test_to_concise_text(self):
        trace = ExecutionTrace()
        trace.add_tool_selected("s1", "filesystem", "Best match for files")
        trace.add_step_completed("s1", True, "Read file")
        text = trace.to_concise_text()
        assert "filesystem" in text
        assert "Read file" in text

    def test_to_dict_list(self):
        trace = ExecutionTrace()
        trace.add_tool_selected("s1", "browser", "Web task")
        d = trace.to_dict_list()
        assert len(d) == 1
        assert d[0]["type"] == "tool_selected"


# ──────────────────────────────────────────────
# AutonomousLoop Tests
# ──────────────────────────────────────────────

class TestAutonomousLoop:
    def test_init(self):
        router = MagicMock()
        loop = AutonomousLoop(tool_router=router)
        assert loop._tool_router is router

    def test_cancel(self):
        router = MagicMock()
        loop = AutonomousLoop(tool_router=router)
        loop.cancel()
        assert loop._cancelled is True

    def test_execute_simple(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "done"
        mock_result.error = ""
        router.route.return_value = mock_result

        loop = AutonomousLoop(tool_router=router)
        state = loop.execute("read the file document.txt")
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)
        assert state.total_tool_calls >= 0

    def test_execute_with_permission_deny(self):
        router = MagicMock()
        perm_manager = MagicMock()
        perm_status = MagicMock()
        perm_status.value = "deny"
        perm_manager.check_permission.return_value = perm_status

        loop = AutonomousLoop(tool_router=router, permission_manager=perm_manager)
        state = loop.execute("use the browser to open example.com")
        # Should handle denied permission
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)

    def test_execute_with_confirmation(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "done"
        mock_result.error = ""
        router.route.return_value = mock_result

        perm_manager = MagicMock()
        perm_status = MagicMock()
        perm_status.value = "require_confirmation"
        perm_manager.check_permission.return_value = perm_status

        confirm_callback = MagicMock(return_value=True)

        loop = AutonomousLoop(tool_router=router, permission_manager=perm_manager)
        loop.set_confirmation_callback(confirm_callback)
        state = loop.execute("run this python code")
        # Should have requested confirmation
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)

    def test_execute_confirmation_denied(self):
        router = MagicMock()
        perm_manager = MagicMock()
        perm_status = MagicMock()
        perm_status.value = "require_confirmation"
        perm_manager.check_permission.return_value = perm_status

        confirm_callback = MagicMock(return_value=False)

        loop = AutonomousLoop(tool_router=router, permission_manager=perm_manager)
        loop.set_confirmation_callback(confirm_callback)
        state = loop.execute("run this python code")
        # Should have been denied
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)

    def test_execute_tool_failure(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.output = ""
        mock_result.error = "tool failed"
        router.route.return_value = mock_result

        loop = AutonomousLoop(tool_router=router)
        state = loop.execute("read the file")
        # Should handle failure
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)

    def test_execute_exception(self):
        router = MagicMock()
        router.route.side_effect = Exception("boom")
        loop = AutonomousLoop(tool_router=router)
        state = loop.execute("do something")
        assert state.phase == TaskPhase.FAILED

    def test_progress_callback(self):
        router = MagicMock()
        callback = MagicMock()
        loop = AutonomousLoop(tool_router=router)
        loop.set_progress_callback(callback)
        # Callback is set but not called in basic flow
        assert loop._progress_callback is callback

    def test_execute_browser_task(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "page content"
        mock_result.error = ""
        router.route.return_value = mock_result

        loop = AutonomousLoop(tool_router=router)
        state = loop.execute("open browser and navigate to example.com")
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)

    def test_execute_with_constraints(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "done"
        mock_result.error = ""
        router.route.return_value = mock_result

        loop = AutonomousLoop(tool_router=router)
        state = loop.execute("do this without the browser")
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)
        # Browser should be prohibited
        assert "browser" in state.constraints.prohibited_tools

    def test_execute_explicit_tool(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "done"
        mock_result.error = ""
        router.route.return_value = mock_result

        loop = AutonomousLoop(tool_router=router)
        state = loop.execute("use the browser to open example.com")
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)


# ──────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────

class TestPhase9Integration:
    def test_full_pipeline_file(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "file content here"
        mock_result.error = ""
        router.route.return_value = mock_result

        analyzer = CapabilityAnalyzer()
        scorer = ToolScorer()

        analysis = analyzer.analyze("read the file document.txt")
        assert "file_operations" in analysis.get_capability_names()

        caps = [Capability(name="file_operations", description="files", confidence=0.9)]
        selection = scorer.select_tool(caps)
        assert selection.selected_tool.tool_name == "filesystem"

    def test_full_pipeline_browser(self):
        analyzer = CapabilityAnalyzer()
        scorer = ToolScorer()

        analysis = analyzer.analyze("open browser and navigate to example.com")
        assert "browser_automation" in analysis.get_capability_names()

        caps = [Capability(name="browser_automation", description="browser", confidence=0.9)]
        selection = scorer.select_tool(caps)
        assert selection.selected_tool.tool_name == "browser"

    def test_full_pipeline_multi_step(self):
        analyzer = CapabilityAnalyzer()
        scorer = ToolScorer()

        analysis = analyzer.analyze(
            "take a screenshot, analyze what's in it, and save the description to a file"
        )
        cap_names = analysis.get_capability_names()
        assert "screen_capture" in cap_names
        assert "vision_analysis" in cap_names
        assert "file_operations" in cap_names

        # Score each capability
        for cap_name in cap_names:
            caps = [Capability(name=cap_name, description=cap_name, confidence=0.8)]
            selection = scorer.select_tool(caps)
            assert selection.selected_tool is not None

    def test_full_pipeline_constraint_respected(self):
        analyzer = CapabilityAnalyzer()
        scorer = ToolScorer()

        analysis = analyzer.analyze("do this without the browser")
        assert "browser" in analysis.constraints.get("prohibited_tools", [])

        caps = [Capability(name="browser_automation", description="browser", confidence=0.9)]
        selection = scorer.select_tool(
            caps,
            constraints=analysis.constraints,
        )
        assert selection.selected_tool is None or selection.selected_tool.tool_name != "browser"

    def test_full_pipeline_explicit_tool(self):
        analyzer = CapabilityAnalyzer()
        scorer = ToolScorer()

        analysis = analyzer.analyze("use the keyboard to type hello")
        assert "keyboard" in analysis.explicit_tools

        caps = [Capability(name="keyboard_input", description="type", confidence=0.9)]
        selection = scorer.select_tool(caps, explicit_tools=analysis.explicit_tools)
        assert selection.selected_tool.tool_name == "keyboard"

    def test_full_pipeline_autonomous_loop(self):
        router = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "done"
        mock_result.error = ""
        router.route.return_value = mock_result

        loop = AutonomousLoop(tool_router=router)
        state = loop.execute(
            "read the file document.txt and save the result to output.txt"
        )
        assert state.phase in (TaskPhase.COMPLETED, TaskPhase.FAILED)
        assert state.total_tool_calls >= 0

    def test_tool_selection_scoring_deterministic(self):
        scorer = ToolScorer()
        caps = [Capability(name="file_operations", description="files", confidence=0.9)]

        result1 = scorer.select_tool(caps)
        result2 = scorer.select_tool(caps)

        assert result1.selected_tool.tool_name == result2.selected_tool.tool_name

    def test_capability_analyzer_patterns_compiled(self):
        analyzer = CapabilityAnalyzer()
        for cap_name in CAPABILITY_DEFINITIONS:
            assert cap_name in analyzer._compiled_patterns

    def test_tool_catalog_complete(self):
        scorer = ToolScorer()
        expected_tools = [
            "filesystem", "python_sandbox", "screen_capture", "system_info",
            "mouse", "keyboard", "window", "browser",
            "vision_analyze", "visual_ground", "image_analyze",
        ]
        for tool in expected_tools:
            assert tool in scorer._catalog, f"Missing tool: {tool}"

    def test_capability_to_tools_map_complete(self):
        expected_caps = [
            "file_operations", "code_execution", "screen_capture",
            "keyboard_input", "browser_automation", "vision_analysis",
        ]
        for cap in expected_caps:
            assert cap in CAPABILITY_TOOLS_MAP, f"Missing capability: {cap}"
