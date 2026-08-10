"""Unit tests for Stage 3.2 Visual Grounding."""

import os
import tempfile
import pytest
from pathlib import Path

from agent.media.vision import (
    VisionResult, DetectedElement, BoundingBox, VisionConfidence,
)
from agent.media.grounding import (
    VisualGrounder, GroundingResult, GroundedTarget,
    TargetType, GroundingConfidence, Point,
)
from agent.media.grounding_tool import VisualGroundingTool
from agent.media.vision import VisionProvider
from agent.media.analyzer import VisionAnalyzer
from agent.media.storage import MediaStorage


class TestTargetType:
    def test_values(self):
        assert TargetType.BUTTON.value == "button"
        assert TargetType.LINK.value == "link"
        assert TargetType.TEXT_FIELD.value == "text_field"
        assert TargetType.UNKNOWN.value == "unknown"


class TestGroundingConfidence:
    def test_values(self):
        assert GroundingConfidence.HIGH.value == "high"
        assert GroundingConfidence.AMBIGUOUS.value == "ambiguous"
        assert GroundingConfidence.UNFOUND.value == "unfound"


class TestPoint:
    def test_creation(self):
        p = Point(x=100, y=200)
        assert p.x == 100
        assert p.y == 200

    def test_to_dict(self):
        p = Point(x=50, y=75)
        d = p.to_dict()
        assert d["x"] == 50
        assert d["y"] == 75

    def test_from_dict(self):
        p = Point.from_dict({"x": 10, "y": 20})
        assert p.x == 10
        assert p.y == 20


class TestGroundedTarget:
    def test_creation(self):
        target = GroundedTarget(
            description="Submit button",
            target_type=TargetType.BUTTON,
            center=Point(x=100, y=200),
        )
        assert target.description == "Submit button"
        assert target.target_type == TargetType.BUTTON

    def test_to_dict(self):
        target = GroundedTarget(
            description="OK",
            target_type=TargetType.BUTTON,
            center=Point(x=50, y=50),
            confidence=GroundingConfidence.HIGH,
        )
        d = target.to_dict()
        assert d["description"] == "OK"
        assert d["target_type"] == "button"
        assert d["confidence"] == "high"

    def test_from_dict(self):
        target = GroundedTarget.from_dict({
            "description": "Menu",
            "target_type": "menu",
            "center": {"x": 300, "y": 400},
            "confidence": "medium",
        })
        assert target.description == "Menu"
        assert target.target_type == TargetType.MENU
        assert target.center.x == 300

    def test_to_text(self):
        target = GroundedTarget(
            description="Search",
            target_type=TargetType.TEXT_FIELD,
            center=Point(x=200, y=100),
            confidence=GroundingConfidence.HIGH,
        )
        text = target.to_text()
        assert "[BEGIN UNTRUSTED GROUNDING DATA]" in text
        assert "Search" in text


class TestGroundingResult:
    def test_creation(self):
        result = GroundingResult(success=True)
        assert result.success is True
        assert result.targets == []

    def test_to_dict(self):
        result = GroundingResult(success=True, screen_width=1920, screen_height=1080)
        d = result.to_dict()
        assert d["success"] is True
        assert d["screen_width"] == 1920

    def test_to_text(self):
        result = GroundingResult(
            success=True,
            targets=[
                GroundedTarget(
                    description="Submit",
                    target_type=TargetType.BUTTON,
                    center=Point(x=150, y=250),
                    confidence=GroundingConfidence.HIGH,
                ),
            ],
            screen_width=1920,
            screen_height=1080,
        )
        text = result.to_text()
        assert "1 target(s)" in text
        assert "Submit" in text


class TestVisualGrounder:
    def test_init(self):
        grounder = VisualGrounder(screen_width=1920, screen_height=1080)
        assert grounder._screen_width == 1920

    def test_stats(self):
        grounder = VisualGrounder()
        stats = grounder.stats
        assert stats["request_count"] == 0

    def test_ground_empty(self):
        grounder = VisualGrounder()
        vr = VisionResult(success=True, description="Empty image")
        result = grounder.ground(vr)
        assert result.success is True
        assert len(result.targets) == 0

    def test_ground_with_elements(self):
        grounder = VisualGrounder()
        vr = VisionResult(
            success=True,
            description="UI with button",
            detected_elements=[
                DetectedElement(
                    element_type="button",
                    description="Click me",
                    bounding_box=BoundingBox(x=100, y=200, width=80, height=30),
                    confidence=VisionConfidence.HIGH,
                ),
            ],
        )
        result = grounder.ground(vr)
        assert result.success is True
        assert len(result.targets) == 1
        assert result.targets[0].center.x == 140
        assert result.targets[0].center.y == 215

    def test_ground_specific_target(self):
        grounder = VisualGrounder()
        vr = VisionResult(
            success=True,
            description="UI",
            detected_elements=[
                DetectedElement(
                    element_type="button",
                    description="Submit form",
                    bounding_box=BoundingBox(x=10, y=20, width=50, height=25),
                ),
                DetectedElement(
                    element_type="link",
                    description="Cancel operation",
                    bounding_box=BoundingBox(x=100, y=20, width=50, height=25),
                ),
            ],
        )
        result = grounder.ground(vr, target_description="Submit")
        assert result.success is True
        assert len(result.targets) == 1
        assert "Submit" in result.targets[0].description

    def test_ground_not_found(self):
        grounder = VisualGrounder()
        vr = VisionResult(
            success=True,
            description="UI",
            detected_elements=[
                DetectedElement(element_type="button", description="OK"),
            ],
        )
        result = grounder.ground(vr, target_description="Nonexistent")
        assert result.success is True
        assert result.targets[0].confidence == GroundingConfidence.UNFOUND

    def test_ground_failed_vision(self):
        grounder = VisualGrounder()
        vr = VisionResult(success=False, error="Analysis failed")
        result = grounder.ground(vr)
        assert result.success is False

    def test_classify_element_button(self):
        grounder = VisualGrounder()
        elem = DetectedElement(element_type="widget", description="Submit button")
        assert grounder._classify_element(elem) == TargetType.BUTTON

    def test_classify_element_link(self):
        grounder = VisualGrounder()
        elem = DetectedElement(element_type="widget", description="Click this link")
        assert grounder._classify_element(elem) == TargetType.LINK

    def test_classify_element_text_field(self):
        grounder = VisualGrounder()
        elem = DetectedElement(element_type="widget", description="Search text field")
        assert grounder._classify_element(elem) == TargetType.TEXT_FIELD

    def test_clamp_coordinates(self):
        grounder = VisualGrounder(screen_width=800, screen_height=600)
        vr = VisionResult(
            success=True,
            description="Off-screen",
            detected_elements=[
                DetectedElement(
                    element_type="button",
                    description="Edge button",
                    bounding_box=BoundingBox(x=900, y=700, width=50, height=50),
                ),
            ],
        )
        result = grounder.ground(vr)
        target = result.targets[0]
        assert target.center.x <= 799
        assert target.center.y <= 599

    def test_validate_target_valid(self):
        grounder = VisualGrounder(screen_width=1920, screen_height=1080)
        target = GroundedTarget(
            description="Button",
            target_type=TargetType.BUTTON,
            center=Point(x=100, y=200),
            confidence=GroundingConfidence.HIGH,
        )
        is_valid, errors = grounder.validate_target(target)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_target_out_of_bounds(self):
        grounder = VisualGrounder(screen_width=800, screen_height=600)
        target = GroundedTarget(
            description="Button",
            target_type=TargetType.BUTTON,
            center=Point(x=900, y=700),
            confidence=GroundingConfidence.HIGH,
        )
        is_valid, errors = grounder.validate_target(target)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_target_unfound(self):
        grounder = VisualGrounder()
        target = GroundedTarget(
            description="Missing",
            target_type=TargetType.UNKNOWN,
            center=Point(x=100, y=100),
            confidence=GroundingConfidence.UNFOUND,
        )
        is_valid, errors = grounder.validate_target(target)
        assert is_valid is False

    def test_validate_target_ambiguous(self):
        grounder = VisualGrounder()
        target = GroundedTarget(
            description="Ambiguous",
            target_type=TargetType.ELEMENT,
            center=Point(x=100, y=100),
            confidence=GroundingConfidence.AMBIGUOUS,
        )
        is_valid, errors = grounder.validate_target(target)
        assert is_valid is False


class TestVisualGroundingTool:
    def test_init(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        storage = MediaStorage()
        tool = VisualGroundingTool(analyzer, grounder, storage, vision_enabled=True)
        assert tool.name == "visual_ground"

    def test_validate_disabled(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        storage = MediaStorage()
        tool = VisualGroundingTool(analyzer, grounder, storage, vision_enabled=False)
        ok, errors = tool.validate({})
        assert ok is False
        assert "disabled" in errors[0].lower()

    def test_validate_no_path(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        storage = MediaStorage()
        tool = VisualGroundingTool(analyzer, grounder, storage, vision_enabled=True)
        ok, errors = tool.validate({})
        assert ok is False
        assert "image_path" in errors[0].lower()

    def test_execute_ground(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        storage = MediaStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = VisualGroundingTool(analyzer, grounder, storage, vision_enabled=True)
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(img_path)
            result = tool.execute({
                "action": "ground",
                "image_path": img_path,
            })
            assert result.success is True
            assert "GROUNDING" in result.output

    def test_execute_validate(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        storage = MediaStorage()
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = VisualGroundingTool(analyzer, grounder, storage, vision_enabled=True)
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(img_path)
            result = tool.execute({
                "action": "validate",
                "image_path": img_path,
            })
            assert result.success is True
            assert "VALIDATION" in result.output

    def test_permissions(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        storage = MediaStorage()
        tool = VisualGroundingTool(analyzer, grounder, storage, vision_enabled=True)
        assert "vision.analyze" in tool.required_permissions

    def test_disabled_deny(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        storage = MediaStorage()
        tool = VisualGroundingTool(analyzer, grounder, storage, vision_enabled=False)
        assert tool.confirmation_level.value == "deny"
