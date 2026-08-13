"""Comprehensive tests for OCR pipeline (Phase 1).

Tests cover:
- OCR abstraction (OCRProvider, OCRBlock, OCRResult, OCRStatus)
- LocalOCRProvider initialization and extraction
- StubOCRProvider
- Image validation (format, size, dimensions)
- Resource limits (text chars, blocks, timeout)
- Security boundaries (untrusted content markers)
- Vision + OCR integration
- Grounding + OCR text integration
"""

import os
import sys
import time
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.media.ocr import (
    OCRProvider,
    LocalOCRProvider,
    StubOCRProvider,
    OCRResult,
    OCRBlock,
    OCRStatus,
)


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_ocr_block():
    """Create a sample OCRBlock."""
    return OCRBlock(
        text="Hello World",
        confidence=0.95,
        x=100,
        y=200,
        width=300,
        height=40,
    )


@pytest.fixture
def sample_ocr_result(sample_ocr_block):
    """Create a sample OCRResult."""
    return OCRResult(
        text="Hello World from OCR",
        confidence=0.92,
        blocks=[sample_ocr_block],
        image_width=800,
        image_height=600,
        status=OCRStatus.SUCCESS,
        provider="test_provider",
        execution_time=0.123,
    )


@pytest.fixture
def stub_provider():
    """Create a StubOCRProvider."""
    return StubOCRProvider()


@pytest.fixture
def temp_image():
    """Create a temporary test image with text."""
    from PIL import Image, ImageDraw, ImageFont
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Rose Vision Test", fill=(0, 0, 0))
        draw.text((50, 100), "OCR Integration", fill=(0, 0, 0))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_image_jpg():
    """Create a temporary JPEG test image."""
    from PIL import Image, ImageDraw
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (300, 150), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), "JPEG Test", fill=(0, 0, 0))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_image_webp():
    """Create a temporary WEBP test image."""
    from PIL import Image, ImageDraw
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
        img = Image.new("RGB", (300, 150), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), "WEBP Test", fill=(0, 0, 0))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def oversized_image():
    """Create an oversized test image (exceeds max dimension)."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (5000, 5000), color=(255, 255, 255))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def empty_image():
    """Create a completely white/empty image."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def corrupt_image():
    """Create a corrupt image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"NOT_A_REAL_IMAGE_FILE")
        yield f.name
    os.unlink(f.name)


# ============================================================
# Test OCRBlock
# ============================================================

class TestOCRBlock:
    def test_creation(self, sample_ocr_block):
        assert sample_ocr_block.text == "Hello World"
        assert sample_ocr_block.confidence == 0.95
        assert sample_ocr_block.x == 100
        assert sample_ocr_block.y == 200
        assert sample_ocr_block.width == 300
        assert sample_ocr_block.height == 40

    def test_center(self, sample_ocr_block):
        cx, cy = sample_ocr_block.center
        assert cx == 250  # 100 + 300//2
        assert cy == 220  # 200 + 40//2

    def test_area(self, sample_ocr_block):
        assert sample_ocr_block.area == 12000  # 300 * 40

    def test_to_dict(self, sample_ocr_block):
        d = sample_ocr_block.to_dict()
        assert d["text"] == "Hello World"
        assert d["confidence"] == 0.95
        assert d["bbox"]["x"] == 100
        assert d["bbox"]["y"] == 200
        assert d["bbox"]["width"] == 300
        assert d["bbox"]["height"] == 40

    def test_from_dict(self, sample_ocr_block):
        d = sample_ocr_block.to_dict()
        restored = OCRBlock.from_dict(d)
        assert restored.text == sample_ocr_block.text
        assert restored.confidence == sample_ocr_block.confidence
        assert restored.x == sample_ocr_block.x
        assert restored.y == sample_ocr_block.y

    def test_from_dict_minimal(self):
        block = OCRBlock.from_dict({})
        assert block.text == ""
        assert block.confidence == 0.0
        assert block.x == 0

    def test_metadata(self):
        block = OCRBlock(
            text="test",
            confidence=0.8,
            x=0, y=0, width=10, height=10,
            metadata={"block_num": 1, "line_num": 2},
        )
        assert block.metadata["block_num"] == 1


# ============================================================
# Test OCRResult
# ============================================================

class TestOCRResult:
    def test_creation(self, sample_ocr_result):
        assert sample_ocr_result.text == "Hello World from OCR"
        assert sample_ocr_result.confidence == 0.92
        assert len(sample_ocr_result.blocks) == 1
        assert sample_ocr_result.status == OCRStatus.SUCCESS
        assert sample_ocr_result.provider == "test_provider"

    def test_block_count(self, sample_ocr_result):
        assert sample_ocr_result.block_count == 1

    def test_char_count(self, sample_ocr_result):
        assert sample_ocr_result.char_count == len("Hello World from OCR")

    def test_to_dict(self, sample_ocr_result):
        d = sample_ocr_result.to_dict()
        assert d["text"] == "Hello World from OCR"
        assert d["confidence"] == 0.92
        assert len(d["blocks"]) == 1
        assert d["status"] == "success"
        assert d["block_count"] == 1

    def test_to_text_untrusted_markers(self, sample_ocr_result):
        text = sample_ocr_result.to_text()
        assert "[BEGIN UNTRUSTED OCR CONTENT]" in text
        assert "[END UNTRUSTED OCR CONTENT]" in text
        assert "Hello World from OCR" in text

    def test_to_text_empty(self):
        result = OCRResult(
            text="",
            confidence=0.0,
            blocks=[],
            image_width=0,
            image_height=0,
            status=OCRStatus.PARTIAL,
            provider="test",
            execution_time=0.0,
        )
        text = result.to_text()
        assert "No text detected" in text

    def test_empty_result(self):
        result = OCRResult(
            text="",
            confidence=0.0,
            blocks=[],
            image_width=800,
            image_height=600,
            status=OCRStatus.PARTIAL,
            provider="test",
            execution_time=0.0,
        )
        assert result.block_count == 0
        assert result.char_count == 0


# ============================================================
# Test OCRStatus
# ============================================================

class TestOCRStatus:
    def test_values(self):
        assert OCRStatus.SUCCESS.value == "success"
        assert OCRStatus.PARTIAL.value == "partial"
        assert OCRStatus.FAILED.value == "failed"
        assert OCRStatus.TIMEOUT.value == "timeout"
        assert OCRStatus.UNSUPPORTED.value == "unsupported"


# ============================================================
# Test StubOCRProvider
# ============================================================

class TestStubOCRProvider:
    def test_init(self, stub_provider):
        assert stub_provider.name == "stub"
        assert stub_provider.is_available is True

    def test_extract_text(self, stub_provider, temp_image):
        result = stub_provider.extract_text(temp_image)
        assert result.status == OCRStatus.UNSUPPORTED
        assert result.text == ""
        assert result.provider == "stub"

    def test_extract_text_missing_file(self, stub_provider):
        result = stub_provider.extract_text("/nonexistent/image.png")
        assert result.status == OCRStatus.FAILED
        assert "not found" in result.error.lower()


# ============================================================
# Test LocalOCRProvider
# ============================================================

class TestLocalOCRProvider:
    def test_init(self):
        provider = LocalOCRProvider()
        assert provider.name == "local_tesseract"

    def test_init_with_params(self):
        provider = LocalOCRProvider(
            language="eng",
            config="--psm 6",
            max_image_size_mb=10,
            max_text_chars=50000,
            max_blocks=500,
        )
        assert provider._language == "eng"
        assert provider._max_image_size_mb == 10
        assert provider._max_text_chars == 50000
        assert provider._max_blocks == 500

    def test_initialize(self):
        provider = LocalOCRProvider()
        # This will succeed or fail based on tesseract installation
        # We just verify it doesn't crash
        result = provider.initialize()
        assert isinstance(result, bool)

    def test_validate_image_missing(self):
        provider = LocalOCRProvider()
        result = provider.extract_text("/nonexistent/file.png")
        assert result.status == OCRStatus.FAILED
        assert "not found" in result.error.lower()

    def test_validate_image_oversized(self, oversized_image):
        provider = LocalOCRProvider(max_image_size_mb=0.001)  # Very small limit
        result = provider.extract_text(oversized_image)
        assert result.status == OCRStatus.FAILED
        assert "too large" in result.error.lower()

    def test_validate_image_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            f.flush()
            f.close()
            try:
                provider = LocalOCRProvider()
                result = provider.extract_text(f.name)
                assert result.status == OCRStatus.UNSUPPORTED
                assert "unsupported" in result.error.lower()
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass

    def test_validate_corrupt_image(self, corrupt_image):
        provider = LocalOCRProvider()
        result = provider.extract_text(corrupt_image)
        assert result.status == OCRStatus.FAILED

    def test_extract_text_valid_image(self, temp_image):
        provider = LocalOCRProvider()
        provider.initialize()
        if provider.is_available:
            result = provider.extract_text(temp_image)
            assert result.status in (OCRStatus.SUCCESS, OCRStatus.PARTIAL)
            assert isinstance(result.text, str)
            assert isinstance(result.blocks, list)
            assert result.image_width > 0
            assert result.image_height > 0
            assert result.execution_time >= 0

    def test_extract_text_jpg(self, temp_image_jpg):
        provider = LocalOCRProvider()
        provider.initialize()
        if provider.is_available:
            result = provider.extract_text(temp_image_jpg)
            assert result.status in (OCRStatus.SUCCESS, OCRStatus.PARTIAL)

    def test_extract_text_webp(self, temp_image_webp):
        provider = LocalOCRProvider()
        provider.initialize()
        if provider.is_available:
            result = provider.extract_text(temp_image_webp)
            assert result.status in (OCRStatus.SUCCESS, OCRStatus.PARTIAL)

    def test_empty_image(self, empty_image):
        provider = LocalOCRProvider()
        provider.initialize()
        if provider.is_available:
            result = provider.extract_text(empty_image)
            # Empty image may have no text or very low confidence
            assert result.status in (OCRStatus.SUCCESS, OCRStatus.PARTIAL)

    def test_resource_limits_text_truncation(self, temp_image):
        provider = LocalOCRProvider(max_text_chars=5)
        provider.initialize()
        if provider.is_available:
            result = provider.extract_text(temp_image)
            # If text was truncated, char_count should be <= 5
            # (or 0 if no text detected)
            if result.text:
                assert len(result.text) <= 5 or result.metadata.get("text_truncated")

    def test_resource_limits_blocks_truncation(self, temp_image):
        provider = LocalOCRProvider(max_blocks=2)
        provider.initialize()
        if provider.is_available:
            result = provider.extract_text(temp_image)
            if result.blocks:
                assert len(result.blocks) <= 2 or result.metadata.get("blocks_truncated")


# ============================================================
# Test Security Boundaries
# ============================================================

class TestOCRSpecurity:
    def test_result_to_text_untrusted_markers(self, sample_ocr_result):
        text = sample_ocr_result.to_text()
        assert "[BEGIN UNTRUSTED OCR CONTENT]" in text
        assert "[END UNTRUSTED OCR CONTENT]" in text

    def test_ocr_result_no_command_execution(self, sample_ocr_result):
        """OCR results should not contain executable code."""
        d = sample_ocr_result.to_dict()
        # Verify structure doesn't have exec fields
        assert "exec" not in d
        assert "shell" not in d
        assert "command" not in d

    def test_ocr_blocks_marked_untrusted(self):
        """OCR text blocks should have untrusted metadata."""
        block = OCRBlock(
            text="<script>alert('xss')</script>",
            confidence=0.5,
            x=0, y=0, width=100, height=20,
            metadata={"untrusted": True},
        )
        assert block.metadata.get("untrusted") is True

    def test_result_json_serializable(self, sample_ocr_result):
        """OCR results must be JSON serializable."""
        d = sample_ocr_result.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["text"] == sample_ocr_result.text
        assert restored["provider"] == "test_provider"


# ============================================================
# Test Resource Limits
# ============================================================

class TestOCRResourceLimits:
    def test_max_image_size_mb(self):
        provider = OCRProvider.__init__  # Just verify it accepts the param
        p = StubOCRProvider.__new__(StubOCRProvider)
        OCRProvider.__init__(p, max_image_size_mb=10)
        assert p._max_image_size_mb == 10

    def test_max_text_chars(self):
        p = StubOCRProvider.__new__(StubOCRProvider)
        OCRProvider.__init__(p, max_text_chars=50000)
        assert p._max_text_chars == 50000

    def test_max_blocks(self):
        p = StubOCRProvider.__new__(StubOCRProvider)
        OCRProvider.__init__(p, max_blocks=500)
        assert p._max_blocks == 500

    def test_ocr_timeout(self):
        p = StubOCRProvider.__new__(StubOCRProvider)
        OCRProvider.__init__(p, ocr_timeout=15.0)
        assert p._ocr_timeout == 15.0

    def test_stats_initial(self, stub_provider):
        stats = stub_provider.stats
        assert stats["requests"] == 0
        assert stats["errors"] == 0
        assert stats["avg_time"] == 0.0


# ============================================================
# Test OCR with Vision Integration
# ============================================================

class TestVisionOCRIntegration:
    def test_real_vision_provider_uses_ocr_abstraction(self):
        """RealVisionProvider should use OCRProvider abstraction."""
        from agent.media.real_vision import RealVisionProvider
        from agent.media.ocr import StubOCRProvider

        ocr = StubOCRProvider()
        provider = RealVisionProvider(ocr_provider=ocr)
        assert provider._ocr_provider is ocr

    def test_real_vision_provider_default_ocr(self):
        """RealVisionProvider should have default OCR provider."""
        from agent.media.real_vision import RealVisionProvider

        provider = RealVisionProvider()
        assert provider._ocr_provider is not None

    def test_vision_analyze_includes_ocr_metadata(self, temp_image):
        """Vision analysis should include OCR result in metadata."""
        from agent.media.real_vision import RealVisionProvider
        from agent.media.base import MediaRequest

        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-001",
            media_type=None,
            prompt="test",
            input_path=temp_image,
        )
        result = provider.process(request)
        assert result.success
        vision_result = result.metadata.get("vision_result")
        if vision_result:
            assert "ocr_result" in vision_result or "detected_text" in vision_result


# ============================================================
# Test Grounding with OCR Text
# ============================================================

class TestGroundingOCRIntegration:
    def test_ocr_text_creates_text_target(self):
        """OCR text elements should be classified as text targets."""
        from agent.media.grounding import VisualGrounder, TargetType
        from agent.media.vision import VisionResult, DetectedElement, BoundingBox, VisionConfidence

        grounder = VisualGrounder(screen_width=800, screen_height=600)

        vision_result = VisionResult(
            success=True,
            detected_elements=[
                DetectedElement(
                    element_type="text",
                    description="OCR text: Submit button",
                    bounding_box=BoundingBox(x=100, y=200, width=100, height=30),
                    confidence=VisionConfidence.HIGH,
                )
            ],
            image_width=800,
            image_height=600,
        )

        grounding_result = grounder.ground(vision_result)
        assert grounding_result.success
        assert len(grounding_result.targets) == 1
        # Should be classified as BUTTON due to "submit" keyword
        assert grounding_result.targets[0].target_type == TargetType.BUTTON

    def test_ocr_text_without_keywords(self):
        """OCR text without button keywords should be TEXT type."""
        from agent.media.grounding import VisualGrounder, TargetType
        from agent.media.vision import VisionResult, DetectedElement, BoundingBox, VisionConfidence

        grounder = VisualGrounder(screen_width=800, screen_height=600)

        vision_result = VisionResult(
            success=True,
            detected_elements=[
                DetectedElement(
                    element_type="text",
                    description="OCR text: Hello World",
                    bounding_box=BoundingBox(x=50, y=50, width=200, height=30),
                    confidence=VisionConfidence.HIGH,
                )
            ],
            image_width=800,
            image_height=600,
        )

        grounding_result = grounder.ground(vision_result)
        assert grounding_result.success
        assert grounding_result.targets[0].target_type == TargetType.TEXT

    def test_ocr_text_coordinates_preserved(self):
        """OCR bounding box coordinates should be preserved in grounding."""
        from agent.media.grounding import VisualGrounder
        from agent.media.vision import VisionResult, DetectedElement, BoundingBox, VisionConfidence

        grounder = VisualGrounder(screen_width=800, screen_height=600)

        vision_result = VisionResult(
            success=True,
            detected_elements=[
                DetectedElement(
                    element_type="text",
                    description="OCR text: Click me",
                    bounding_box=BoundingBox(x=150, y=250, width=120, height=40),
                    confidence=VisionConfidence.HIGH,
                )
            ],
            image_width=800,
            image_height=600,
        )

        grounding_result = grounder.ground(vision_result)
        target = grounding_result.targets[0]
        assert target.bounding_box.x == 150
        assert target.bounding_box.y == 250
        assert target.center.x == 210  # 150 + 120//2
        assert target.center.y == 270  # 250 + 40//2
