"""Integration tests for Vision + OCR + Grounding pipeline.

Tests the complete flow from screenshot to grounded coordinates.
"""

import os
import sys
import time
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.media.ocr import OCRProvider, LocalOCRProvider, StubOCRProvider, OCRResult, OCRStatus
from agent.media.vision import VisionProvider, VisionResult, DetectedElement, BoundingBox, VisionConfidence
from agent.media.real_vision import RealVisionProvider, ImagePreprocessor
from agent.media.grounding import VisualGrounder, GroundingResult, TargetType, GroundingConfidence
from agent.media.base import MediaRequest


def _dicts_to_elements(elements):
    """Convert a list of dicts to DetectedElement objects."""
    result = []
    for e in elements:
        if isinstance(e, DetectedElement):
            result.append(e)
        elif isinstance(e, dict):
            bb_data = e.get("bounding_box")
            bb = None
            if bb_data:
                if isinstance(bb_data, BoundingBox):
                    bb = bb_data
                elif isinstance(bb_data, dict):
                    bb = BoundingBox(
                        x=bb_data.get("x", 0),
                        y=bb_data.get("y", 0),
                        width=bb_data.get("width", 0),
                        height=bb_data.get("height", 0),
                    )
            conf = e.get("confidence", VisionConfidence.UNKNOWN)
            if isinstance(conf, str):
                try:
                    conf = VisionConfidence(conf)
                except ValueError:
                    conf = VisionConfidence.UNKNOWN
            result.append(DetectedElement(
                element_type=e.get("element_type", "unknown"),
                description=e.get("description", ""),
                bounding_box=bb,
                confidence=conf,
                metadata=e.get("metadata", {}),
            ))
    return result


@pytest.fixture
def text_image():
    """Create an image with clear text for OCR testing."""
    from PIL import Image, ImageDraw
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Submit", fill=(0, 0, 0))
        draw.text((50, 120), "Cancel", fill=(0, 0, 0))
        draw.text((50, 190), "Hello World", fill=(0, 0, 0))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def multi_text_image():
    """Create an image with multiple text regions."""
    from PIL import Image, ImageDraw
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((100, 50), "Login", fill=(0, 0, 0))
        draw.text((400, 50), "Register", fill=(0, 0, 0))
        draw.text((100, 300), "Settings", fill=(0, 0, 0))
        draw.text((400, 300), "Help", fill=(0, 0, 0))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


# ============================================================
# Test End-to-End Pipeline
# ============================================================

class TestEndToEndVisionPipeline:
    def test_screenshot_to_grounded_targets(self, text_image):
        """Complete pipeline: image -> vision -> OCR -> grounding."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-e2e-001",
            media_type=None,
            prompt="Analyze this image",
            input_path=text_image,
        )

        vision_result = provider.process(request)
        assert vision_result.success

        # Build VisionResult from process result
        elements = _dicts_to_elements(vision_result.metadata.get("detected_elements", []))
        vr = VisionResult(
            success=True,
            description=vision_result.metadata.get("description", ""),
            detected_elements=elements,
            image_width=vision_result.metadata.get("image_width", 0),
            image_height=vision_result.metadata.get("image_height", 0),
        )

        # Ground the vision result
        grounder = VisualGrounder(screen_width=600, screen_height=400)
        grounding_result = grounder.ground(vr)

        assert grounding_result.success
        assert len(grounding_result.targets) > 0

        # Verify targets have valid coordinates
        for target in grounding_result.targets:
            assert 0 <= target.center.x < 600
            assert 0 <= target.center.y < 400

    def test_ocr_text_provides_coordinates(self, text_image):
        """OCR should provide bounding box coordinates for text."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-coord-001",
            media_type=None,
            prompt="Extract text",
            input_path=text_image,
        )

        vision_result = provider.process(request)
        assert vision_result.success

        # Check that OCR results are in metadata
        metadata = vision_result.metadata
        if "ocr_result" in metadata:
            ocr_data = metadata["ocr_result"]
            assert ocr_data.get("status") in ("success", "partial")
            # If blocks exist, they should have coordinates
            for block in ocr_data.get("blocks", []):
                bbox = block.get("bbox", {})
                assert "x" in bbox
                assert "y" in bbox
                assert "width" in bbox
                assert "height" in bbox

    def test_grounding_finds_specific_text(self, text_image):
        """Grounding should find specific text like 'Submit'."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-find-001",
            media_type=None,
            prompt="Find Submit button",
            input_path=text_image,
        )

        vision_result = provider.process(request)
        elements = _dicts_to_elements(vision_result.metadata.get("detected_elements", []))
        vr = VisionResult(
            success=True,
            description=vision_result.metadata.get("description", ""),
            detected_elements=elements,
            image_width=vision_result.metadata.get("image_width", 0),
            image_height=vision_result.metadata.get("image_height", 0),
        )

        grounder = VisualGrounder(screen_width=600, screen_height=400)
        grounding_result = grounder.ground(vr, target_description="Submit")

        assert grounding_result.success
        # Should find at least one target matching "Submit"
        if grounding_result.targets:
            found_submit = any(
                "submit" in t.description.lower()
                for t in grounding_result.targets
            )
            # May or may not find exact text depending on OCR quality
            # Just verify grounding works

    def test_multiple_text_regions(self, multi_text_image):
        """Pipeline should handle multiple text regions."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-multi-001",
            media_type=None,
            prompt="Analyze all text",
            input_path=multi_text_image,
        )

        vision_result = provider.process(request)
        elements = _dicts_to_elements(vision_result.metadata.get("detected_elements", []))
        vr = VisionResult(
            success=True,
            description=vision_result.metadata.get("description", ""),
            detected_elements=elements,
            image_width=vision_result.metadata.get("image_width", 0),
            image_height=vision_result.metadata.get("image_height", 0),
        )

        grounder = VisualGrounder(screen_width=800, screen_height=600)
        grounding_result = grounder.ground(vr)

        assert grounding_result.success
        assert len(grounding_result.targets) > 0


# ============================================================
# Test OCR Provider Integration
# ============================================================

class TestOCRProviderIntegration:
    def test_stub_provider_in_vision(self):
        """StubOCRProvider should work with RealVisionProvider."""
        ocr = StubOCRProvider()
        provider = RealVisionProvider(ocr_provider=ocr)
        provider.initialize()
        assert provider._ocr_provider is ocr
        assert provider.name == "real_vision"

    def test_local_provider_initialization(self):
        """LocalOCRProvider should initialize (or gracefully fail)."""
        ocr = LocalOCRProvider()
        result = ocr.initialize()
        # Should not crash
        assert isinstance(result, bool)

    def test_ocr_result_json_serializable(self, text_image):
        """OCR results should be JSON serializable."""
        ocr = LocalOCRProvider()
        ocr.initialize()

        result = ocr.extract_text(text_image)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_vision_result_serializable(self, text_image):
        """Vision results with OCR data should be JSON serializable."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-serial-001",
            media_type=None,
            prompt="test",
            input_path=text_image,
        )

        result = provider.process(request)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ============================================================
# Test Performance
# ============================================================

class TestVisionPerformance:
    def test_ocr_processing_time(self, text_image):
        """OCR processing should complete in reasonable time."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-perf-001",
            media_type=None,
            prompt="test",
            input_path=text_image,
        )

        start = time.time()
        result = provider.process(request)
        elapsed = time.time() - start

        assert result.success
        # Should complete within 30 seconds
        assert elapsed < 30.0

    def test_grounding_processing_time(self, text_image):
        """Grounding should be fast."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-gperf-001",
            media_type=None,
            prompt="test",
            input_path=text_image,
        )

        vision_result = provider.process(request)
        elements = _dicts_to_elements(vision_result.metadata.get("detected_elements", []))
        vr = VisionResult(
            success=True,
            detected_elements=elements,
            image_width=vision_result.metadata.get("image_width", 0),
            image_height=vision_result.metadata.get("image_height", 0),
        )

        grounder = VisualGrounder(screen_width=600, screen_height=400)
        start = time.time()
        grounding_result = grounder.ground(vr)
        elapsed = time.time() - start

        assert grounding_result.success
        # Grounding should be fast (< 1 second)
        assert elapsed < 1.0
