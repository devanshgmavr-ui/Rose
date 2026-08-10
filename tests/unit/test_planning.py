"""Unit tests for Stage 4.1 Natural Language Tool Planning."""

import pytest

from agent.orchestration.tool_catalog import (
    ToolMetadata, build_tool_catalog, get_tools_for_request,
)
from agent.orchestration.enhanced_planner import EnhancedPlanner
from agent.orchestration.models import Plan, PlanStep


class TestToolMetadata:
    def test_creation(self):
        meta = ToolMetadata(
            name="test_tool",
            description="A test tool",
            category="test",
            actions=["action1"],
            input_schema={},
            output_schema={},
            permissions=["test.perm"],
            confirmation_required=False,
            timeout=10.0,
            failure_modes=["error"],
            examples=[],
        )
        assert meta.name == "test_tool"
        assert meta.category == "test"

    def test_to_dict(self):
        meta = ToolMetadata(
            name="t",
            description="d",
            category="c",
            actions=["a"],
            input_schema={},
            output_schema={},
            permissions=[],
            confirmation_required=False,
            timeout=5.0,
            failure_modes=[],
            examples=[],
        )
        d = meta.to_dict()
        assert d["name"] == "t"
        assert d["description"] == "d"


class TestToolCatalog:
    def test_build_catalog(self):
        catalog = build_tool_catalog()
        assert len(catalog) > 0

    def test_catalog_has_files(self):
        catalog = build_tool_catalog()
        assert "filesystem" in catalog
        assert "python_sandbox" in catalog

    def test_catalog_has_os(self):
        catalog = build_tool_catalog()
        assert "screen_capture" in catalog
        assert "system_info" in catalog
        assert "mouse" in catalog
        assert "keyboard" in catalog
        assert "window" in catalog

    def test_catalog_has_browser(self):
        catalog = build_tool_catalog()
        assert "browser" in catalog

    def test_catalog_has_vision(self):
        catalog = build_tool_catalog()
        assert "vision_analyze" in catalog
        assert "visual_ground" in catalog

    def test_catalog_tool_metadata(self):
        catalog = build_tool_catalog()
        for name, meta in catalog.items():
            assert meta.name == name
            assert len(meta.description) > 0
            assert len(meta.actions) > 0
            assert isinstance(meta.permissions, list)
            assert isinstance(meta.failure_modes, list)

    def test_filesystem_actions(self):
        catalog = build_tool_catalog()
        fs = catalog["filesystem"]
        assert "read" in fs.actions
        assert "write" in fs.actions
        assert "list" in fs.actions

    def test_browser_actions(self):
        catalog = build_tool_catalog()
        browser = catalog["browser"]
        assert "navigate" in browser.actions
        assert "screenshot" in browser.actions
        assert "click" in browser.actions

    def test_vision_actions(self):
        catalog = build_tool_catalog()
        vision = catalog["vision_analyze"]
        assert "analyze" in vision.actions
        assert "describe" in vision.actions


class TestGetToolsForRequest:
    def test_file_request(self):
        tools = get_tools_for_request("Read a file from the workspace")
        names = [t.name for t in tools]
        assert "filesystem" in names

    def test_screenshot_request(self):
        tools = get_tools_for_request("Take a screenshot of the screen")
        names = [t.name for t in tools]
        assert "screen_capture" in names

    def test_browser_request(self):
        tools = get_tools_for_request("Navigate to a website in Chrome")
        names = [t.name for t in tools]
        assert "browser" in names

    def test_code_request(self):
        tools = get_tools_for_request("Execute some Python code")
        names = [t.name for t in tools]
        assert "python_sandbox" in names

    def test_mouse_request(self):
        tools = get_tools_for_request("Click on a button")
        names = [t.name for t in tools]
        assert "mouse" in names

    def test_window_request(self):
        tools = get_tools_for_request("List all open windows")
        names = [t.name for t in tools]
        assert "window" in names

    def test_vision_request(self):
        tools = get_tools_for_request("Analyze this image and tell me what objects are in it")
        names = [t.name for t in tools]
        assert "vision_analyze" in names or "visual_ground" in names or "image_analyze" in names

    def test_fallback(self):
        tools = get_tools_for_request("do something random")
        assert len(tools) > 0

    def test_limit_results(self):
        tools = get_tools_for_request(
            "read a file and take a screenshot and click and type and browse"
        )
        assert len(tools) <= 5


class TestEnhancedPlanner:
    def test_init(self):
        planner = EnhancedPlanner()
        assert planner._max_plan_steps == 12
        assert len(planner._catalog) > 0

    def test_create_plan_no_llm(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Read a file from workspace")
        assert isinstance(plan, Plan)
        assert len(plan.steps) > 0
        assert plan.objective == "Read a file from workspace"

    def test_create_plan_with_task_id(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Test request", task_id="my_task")
        assert plan.task_id == "my_task"

    def test_plan_has_steps(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Take a screenshot")
        assert len(plan.steps) >= 1
        for step in plan.steps:
            assert isinstance(step, PlanStep)
            assert len(step.step_id) > 0
            assert len(step.description) > 0

    def test_plan_max_steps(self):
        planner = EnhancedPlanner(max_plan_steps=3)
        plan = planner.create_plan("Do many things at once with files and browser and vision and mouse and keyboard")
        assert len(plan.steps) <= 3

    def test_plan_browser_request(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Navigate to google.com in the browser")
        tool_names = [s.tool_name for s in plan.steps]
        assert "browser" in tool_names

    def test_plan_screenshot_request(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Take a screenshot of the screen")
        tool_names = [s.tool_name for s in plan.steps]
        assert "screen_capture" in tool_names

    def test_plan_has_completion_criteria(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Do something")
        assert len(plan.completion_criteria) > 0

    def test_plan_step_dependencies(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Analyze a screenshot and describe what you see")
        for step in plan.steps:
            assert isinstance(step.dependencies, list)

    def test_format_tool_descriptions(self):
        planner = EnhancedPlanner()
        from agent.orchestration.tool_catalog import get_tools_for_request
        tools = get_tools_for_request("browser", planner._catalog)
        desc = planner._format_tool_descriptions(tools)
        assert "browser" in desc.lower() or "Tool:" in desc

    def test_create_tool_based_plan_file(self):
        planner = EnhancedPlanner()
        plan = planner.create_plan("Save a file with content")
        assert isinstance(plan, Plan)
        tool_names = [s.tool_name for s in plan.steps]
        assert "filesystem" in tool_names
