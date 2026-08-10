"""Unit tests for Stage 4.2 Automatic Tool Selection."""

import pytest

from agent.orchestration.tool_selector import IntentClassifier, ToolSelector, ToolMatch


class TestToolMatch:
    def test_creation(self):
        match = ToolMatch(
            tool_name="filesystem",
            action="read",
            confidence=0.8,
            reasoning="test",
        )
        assert match.tool_name == "filesystem"
        assert match.confidence == 0.8

    def test_to_dict(self):
        match = ToolMatch(
            tool_name="mouse",
            action="click",
            confidence=0.9,
            reasoning="matched click pattern",
            arguments={"x": 100, "y": 200},
        )
        d = match.to_dict()
        assert d["tool_name"] == "mouse"
        assert d["action"] == "click"
        assert d["arguments"]["x"] == 100


class TestIntentClassifier:
    def test_init(self):
        classifier = IntentClassifier()
        assert len(classifier._compiled) > 0

    def test_screenshot(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Take a screenshot")
        assert len(matches) > 0
        assert matches[0].tool_name == "screen_capture"

    def test_system_info(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Show me system info")
        assert len(matches) > 0
        assert matches[0].tool_name == "system_info"

    def test_file_read(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Read a file")
        assert len(matches) > 0
        assert matches[0].tool_name == "filesystem"

    def test_file_write(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Write a file to disk")
        assert len(matches) > 0
        assert matches[0].tool_name == "filesystem"

    def test_code_execute(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Run Python code")
        assert len(matches) > 0
        assert matches[0].tool_name == "python_sandbox"

    def test_mouse_click(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Click on the button")
        assert len(matches) > 0
        assert matches[0].tool_name == "mouse"

    def test_mouse_move(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Move the mouse")
        assert len(matches) > 0
        assert matches[0].tool_name == "mouse"

    def test_type_text(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Type hello world")
        assert len(matches) > 0
        assert matches[0].tool_name == "keyboard"

    def test_press_key(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Press the enter key")
        assert len(matches) > 0
        assert matches[0].tool_name == "keyboard"
        assert matches[0].arguments.get("key") == "Enter"

    def test_window_list(self):
        classifier = IntentClassifier()
        matches = classifier.classify("List all windows")
        assert len(matches) > 0
        assert matches[0].tool_name == "window"

    def test_window_activate(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Activate the window")
        assert len(matches) > 0
        assert matches[0].tool_name == "window"

    def test_browser_open(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Open a browser")
        assert len(matches) > 0
        assert matches[0].tool_name == "browser"

    def test_browser_navigate(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Navigate to https://google.com")
        assert len(matches) > 0
        assert matches[0].tool_name == "browser"
        assert "google.com" in matches[0].arguments.get("url", "")

    def test_browser_read(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Read the page content")
        assert len(matches) > 0
        assert matches[0].tool_name == "browser"

    def test_vision_analyze(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Analyze this image and describe it")
        assert len(matches) > 0
        assert matches[0].tool_name == "vision_analyze"

    def test_visual_ground(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Find the button")
        assert len(matches) > 0
        assert matches[0].tool_name == "visual_ground"

    def test_search_online(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Search online for Python tutorials")
        assert len(matches) > 0
        assert matches[0].tool_name == "browser"

    def test_available_tools_filter(self):
        classifier = IntentClassifier()
        matches = classifier.classify(
            "Take a screenshot",
            available_tools=["filesystem", "python_sandbox"],
        )
        assert len(matches) == 0

    def test_no_match(self):
        classifier = IntentClassifier()
        matches = classifier.classify("xyzzy foobar baz")
        assert len(matches) == 0

    def test_multiple_matches(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Take a screenshot and read a file")
        assert len(matches) >= 2

    def test_confidence_ordering(self):
        classifier = IntentClassifier()
        matches = classifier.classify("Take a screenshot of the screen")
        assert len(matches) > 0
        for i in range(len(matches) - 1):
            assert matches[i].confidence >= matches[i + 1].confidence


class TestToolSelector:
    def test_init(self):
        selector = ToolSelector()
        assert selector._classifier is not None

    def test_select_screenshot(self):
        selector = ToolSelector()
        match = selector.select("Take a screenshot")
        assert match is not None
        assert match.tool_name == "screen_capture"

    def test_select_no_match(self):
        selector = ToolSelector()
        match = selector.select("xyzzy")
        assert match is None

    def test_select_min_confidence(self):
        selector = ToolSelector()
        match = selector.select("Take a screenshot", min_confidence=0.9)
        assert match is None or match.confidence >= 0.9

    def test_select_available_tools(self):
        selector = ToolSelector()
        match = selector.select(
            "Take a screenshot",
            available_tools=["filesystem"],
        )
        assert match is None

    def test_select_all(self):
        selector = ToolSelector()
        matches = selector.select_all("Take a screenshot and read a file")
        assert len(matches) >= 2

    def test_select_all_min_confidence(self):
        selector = ToolSelector()
        matches = selector.select_all(
            "Take a screenshot",
            min_confidence=0.5,
        )
        for m in matches:
            assert m.confidence >= 0.5

    def test_complex_request(self):
        selector = ToolSelector()
        match = selector.select(
            "Open a browser and navigate to a website"
        )
        assert match is not None

    def test_click_with_coordinates(self):
        selector = ToolSelector()
        match = selector.select("Click on the button")
        assert match is not None
        assert match.tool_name == "mouse"
        assert match.action == "click"
