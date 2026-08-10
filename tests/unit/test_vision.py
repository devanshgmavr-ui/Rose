"""Unit tests for Stage 3.1 Vision Analysis."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.media.base import MediaType, MediaRequest, MediaResult
from agent.media.vision import (
    VisionProvider,
    StubLocalVisionProvider,
    LocalVisionProvider,
    VisionResult,
    DetectedElement,
    BoundingBox,
    VisionConfidence,
)
from agent.media.analyzer import VisionAnalyzer
from agent.media.permissions import register_vision_permissions, VISION_PERMISSIONS
from agent.media.vision_tool import VisionAnalyzeTool
from agent.media.storage import MediaStorage
from agent.tools.permissions import PermissionManager
from agent.tools.base import ConfirmationLevel


class TestVisionConfidence:
    def test_values(self):
        assert VisionConfidence.HIGH.value == "high"
        assert VisionConfidence.MEDIUM.value == "medium"
        assert VisionConfidence.LOW.value == "low"
        assert VisionConfidence.UNKNOWN.value == "unknown"


class TestBoundingBox:
    def test_creation(self):
        bb = BoundingBox(x=10, y=20, width=100, height=50)
        assert bb.x == 10
        assert bb.y == 20
        assert bb.width == 100
        assert bb.height == 50

    def test_to_dict(self):
        bb = BoundingBox(x=10, y=20, width=100, height=50)
        d = bb.to_dict()
        assert d["x"] == 10
        assert d["y"] == 20
        assert d["width"] == 100
        assert d["height"] == 50

    def test_from_dict(self):
        bb = BoundingBox.from_dict({"x": 5, "y": 15, "width": 80, "height": 40})
        assert bb.x == 5
        assert bb.y == 15
        assert bb.width == 80
        assert bb.height == 40

    def test_from_dict_defaults(self):
        bb = BoundingBox.from_dict({})
        assert bb.x == 0
        assert bb.y == 0
        assert bb.width == 0
        assert bb.height == 0


class TestDetectedElement:
    def test_creation(self):
        elem = DetectedElement(
            element_type="button",
            description="Submit button",
            bounding_box=BoundingBox(x=10, y=20, width=100, height=50),
            confidence=VisionConfidence.HIGH,
        )
        assert elem.element_type == "button"
        assert elem.description == "Submit button"
        assert elem.bounding_box is not None
        assert elem.confidence == VisionConfidence.HIGH

    def test_to_dict(self):
        elem = DetectedElement(
            element_type="text",
            description="Hello",
            confidence=VisionConfidence.MEDIUM,
        )
        d = elem.to_dict()
        assert d["element_type"] == "text"
        assert d["description"] == "Hello"
        assert d["confidence"] == "medium"
        assert "bounding_box" not in d

    def test_to_dict_with_bbox(self):
        elem = DetectedElement(
            element_type="icon",
            description="Settings",
            bounding_box=BoundingBox(x=0, y=0, width=32, height=32),
        )
        d = elem.to_dict()
        assert "bounding_box" in d
        assert d["bounding_box"]["width"] == 32

    def test_from_dict(self):
        elem = DetectedElement.from_dict({
            "element_type": "link",
            "description": "Click here",
            "confidence": "low",
        })
        assert elem.element_type == "link"
        assert elem.confidence == VisionConfidence.LOW

    def test_from_dict_with_bbox(self):
        elem = DetectedElement.from_dict({
            "element_type": "button",
            "description": "OK",
            "bounding_box": {"x": 10, "y": 20, "width": 50, "height": 30},
        })
        assert elem.bounding_box is not None
        assert elem.bounding_box.x == 10


class TestVisionResult:
    def test_creation(self):
        result = VisionResult(success=True, description="Test image")
        assert result.success is True
        assert result.description == "Test image"
        assert result.untrusted_content is True

    def test_to_dict(self):
        result = VisionResult(
            success=True,
            description="A screenshot",
            image_width=1920,
            image_height=1080,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["description"] == "A screenshot"
        assert d["image_width"] == 1920
        assert d["untrusted_content"] is True

    def test_to_dict_with_error(self):
        result = VisionResult(success=False, error="Failed")
        d = result.to_dict()
        assert d["error"] == "Failed"

    def test_from_dict(self):
        result = VisionResult.from_dict({
            "success": True,
            "description": "Test",
            "image_width": 800,
            "image_height": 600,
        })
        assert result.success is True
        assert result.image_width == 800

    def test_to_text(self):
        result = VisionResult(
            success=True,
            description="A test image",
            image_width=100,
            image_height=100,
            detected_elements=[
                DetectedElement(
                    element_type="button",
                    description="Submit",
                    bounding_box=BoundingBox(x=10, y=20, width=80, height=30),
                    confidence=VisionConfidence.HIGH,
                ),
            ],
        )
        text = result.to_text()
        assert "[BEGIN UNTRUSTED VISUAL CONTENT]" in text
        assert "[END UNTRUSTED VISUAL CONTENT]" in text
        assert "A test image" in text
        assert "100x100" in text
        assert "button: Submit" in text
        assert "high" in text

    def test_to_text_no_elements(self):
        result = VisionResult(
            success=True,
            description="Empty",
            image_width=50,
            image_height=50,
        )
        text = result.to_text()
        assert "Detected elements" not in text


class TestVisionProvider:
    def test_init(self):
        vp = VisionProvider()
        assert vp.name == "vision"
        assert vp.media_type == MediaType.IMAGE
        assert vp.is_available is True

    def test_stats(self):
        vp = VisionProvider()
        stats = vp.stats
        assert stats["request_count"] == 0
        assert stats["total_time"] == 0.0

    def test_validate_no_path(self):
        vp = VisionProvider()
        req = MediaRequest(media_type=MediaType.IMAGE)
        ok, errors = vp.validate_request(req)
        assert ok is False
        assert "input_path is required" in errors[0]

    def test_validate_nonexistent(self):
        vp = VisionProvider()
        req = MediaRequest(media_type=MediaType.IMAGE, input_path="/nonexistent.png")
        ok, errors = vp.validate_request(req)
        assert ok is False
        assert "not found" in errors[0]

    def test_validate_invalid_format(self):
        vp = VisionProvider()
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            req = MediaRequest(media_type=MediaType.IMAGE, input_path=f.name)
            ok, errors = vp.validate_request(req)
            assert ok is False
            assert "Unsupported" in errors[0]

    def test_validate_empty_file(self):
        vp = VisionProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "empty.png")
            with open(img_path, 'wb') as f:
                pass
            req = MediaRequest(media_type=MediaType.IMAGE, input_path=img_path)
            ok, errors = vp.validate_request(req)
            assert ok is False
            assert "empty" in errors[0]

    def test_process_no_path(self):
        vp = VisionProvider()
        req = MediaRequest(media_type=MediaType.IMAGE)
        result = vp.process(req)
        assert result.success is False

    def test_process_valid_image(self):
        vp = VisionProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(img_path)
            req = MediaRequest(media_type=MediaType.IMAGE, input_path=img_path)
            result = vp.process(req)
            assert result.success is True
            assert result.metadata.get("image_width") == 100


class TestStubVisionProvider:
    def test_init(self):
        svp = StubLocalVisionProvider()
        assert svp.name == "stub_local_vision"
        assert svp.is_available is False

    def test_init_with_model(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            svp = StubLocalVisionProvider(model_path=f.name)
            assert svp.is_available is True

    def test_initialize_no_model(self):
        svp = StubLocalVisionProvider()
        result = svp.initialize()
        assert result is True
        assert svp._initialized is True

    def test_analyze_returns_stub(self):
        svp = StubLocalVisionProvider()
        svp.initialize()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(img_path)
            req = MediaRequest(media_type=MediaType.IMAGE, input_path=img_path)
            result = svp.process(req)
            assert result.success is True
            assert "stub" in result.metadata.get("description", "").lower()


class TestLocalVisionProvider:
    def test_init(self):
        lvp = LocalVisionProvider()
        assert lvp.name == "local_vision"
        assert lvp.is_available is False

    def test_health_check(self):
        lvp = LocalVisionProvider()
        lvp.initialize()
        hc = lvp.health_check()
        assert hc["initialized"] is True
        assert hc["model_loaded"] is False
        assert hc["is_available"] is False

    def test_shutdown(self):
        lvp = LocalVisionProvider()
        lvp.initialize()
        lvp.shutdown()
        assert lvp._model_loaded is False
        assert lvp._initialized is False

    def test_analyze_no_model(self):
        lvp = LocalVisionProvider()
        lvp.initialize()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='green')
            img.save(img_path)
            req = MediaRequest(media_type=MediaType.IMAGE, input_path=img_path)
            result = lvp.process(req)
            assert result.success is True
            assert "no vision model" in result.metadata.get("description", "").lower()


class TestVisionAnalyzer:
    def test_init(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        assert analyzer.is_available is True
        assert analyzer.provider_name == "vision"

    def test_analyze(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='yellow')
            img.save(img_path)
            result = analyzer.analyze(img_path)
            assert result.success is True
            assert result.image_width == 100

    def test_analyze_workspace_boundary(self):
        vp = VisionProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = VisionAnalyzer(vision_provider=vp)
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='white')
            img.save(img_path)
            result = analyzer.analyze(img_path, workspace_root=tmpdir)
            assert result.success is True

    def test_analyze_outside_workspace(self):
        vp = VisionProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = VisionAnalyzer(vision_provider=vp)
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(workspace)
            outside = os.path.join(tmpdir, "outside.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(outside)
            result = analyzer.analyze(outside, workspace_root=workspace)
            assert result.success is False
            assert "workspace" in result.error.lower()

    def test_describe_image(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='purple')
            img.save(img_path)
            text = analyzer.describe_image(img_path)
            assert "[BEGIN UNTRUSTED VISUAL CONTENT]" in text
            assert "[END UNTRUSTED VISUAL CONTENT]" in text

    def test_max_elements_truncation(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp, max_elements=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='cyan')
            img.save(img_path)
            result = analyzer.analyze(img_path)
            assert result.success is True


class TestVisionPermissions:
    def test_permission_constants(self):
        assert "vision.analyze" in VISION_PERMISSIONS
        assert VISION_PERMISSIONS["vision.analyze"] == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_register_enabled(self):
        pm = PermissionManager()
        register_vision_permissions(pm, vision_enabled=True)
        assert pm.has_permission("vision.analyze")
        assert pm.get_confirmation_level("vision.analyze") == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_register_disabled(self):
        pm = PermissionManager()
        register_vision_permissions(pm, vision_enabled=False)
        assert not pm.has_permission("vision.analyze")


class TestVisionAnalyzeTool:
    def test_init(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True)
        assert tool.name == "vision_analyze"
        assert tool.description

    def test_validate_disabled(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=False)
        ok, errors = tool.validate({})
        assert ok is False
        assert "disabled" in errors[0].lower()

    def test_validate_no_path(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True)
        ok, errors = tool.validate({})
        assert ok is False
        assert "image_path" in errors[0].lower()

    def test_validate_nonexistent(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True)
        ok, errors = tool.validate({"image_path": "/nonexistent.png"})
        assert ok is False
        assert "not found" in errors[0].lower()

    def test_validate_invalid_format(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True)
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            ok, errors = tool.validate({"image_path": f.name})
            assert ok is False
            assert "Unsupported" in errors[0]

    def test_execute_analyze(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True, workspace_dir=tempfile.gettempdir())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='orange')
            img.save(img_path)
            result = tool.execute({
                "action": "analyze",
                "image_path": img_path,
            })
            assert result.success is True
            assert "[BEGIN UNTRUSTED VISUAL CONTENT]" in result.output

    def test_execute_describe(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True, workspace_dir=tempfile.gettempdir())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='pink')
            img.save(img_path)
            result = tool.execute({
                "action": "describe",
                "image_path": img_path,
            })
            assert result.success is True
            assert "[BEGIN UNTRUSTED VISUAL CONTENT]" in result.output

    def test_execute_invalid_action(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True)
        result = tool.execute({
            "action": "invalid",
            "image_path": "/nonexistent.png",
        })
        assert result.success is False

    def test_permissions(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True)
        perms = tool.required_permissions
        assert "vision.analyze" in perms

    def test_confirmation_level(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True)
        assert tool.confirmation_level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_disabled_deny(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=False)
        assert tool.confirmation_level == ConfirmationLevel.DENY


class TestVisionIntegration:
    def test_full_flow(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        storage = MediaStorage()
        tool = VisionAnalyzeTool(analyzer, storage, vision_enabled=True, workspace_dir=tempfile.gettempdir())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (200, 200), color='red')
            img.save(img_path)
            
            result = tool.execute({
                "action": "analyze",
                "image_path": img_path,
                "prompt": "What is in this image?",
            })
            
            assert result.success is True
            assert "UNTRUSTED" in result.output
            assert "200x200" in result.output

    def test_analyzer_stats(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (50, 50), color='white')
            img.save(img_path)
            analyzer.analyze(img_path)
            stats = analyzer.stats
            assert stats["request_count"] == 1
            assert stats["total_time"] > 0
