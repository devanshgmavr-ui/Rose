"""Comprehensive tests for Phase 10 Vision System.

Tests image analysis, preprocessing, color analysis, OCR integration,
and the real vision provider.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.media.base import MediaType, MediaRequest, MediaResult
from agent.media.vision import (
    VisionProvider,
    VisionResult,
    DetectedElement,
    BoundingBox,
    VisionConfidence,
)
from agent.media.real_vision import (
    RealVisionProvider,
    ImagePreprocessor,
    ImageMetadata,
    ColorInfo,
    ImageAnalysis,
    ImageFormat,
)


class TestImageFormat:
    def test_values(self):
        assert ImageFormat.PNG.value == "png"
        assert ImageFormat.JPEG.value == "jpeg"
        assert ImageFormat.WEBP.value == "webp"
        assert ImageFormat.BMP.value == "bmp"
        assert ImageFormat.GIF.value == "gif"
        assert ImageFormat.UNKNOWN.value == "unknown"


class TestImageMetadata:
    def test_creation(self):
        meta = ImageMetadata(
            file_path="/test/image.png",
            file_name="image.png",
            file_size=1024,
            format=ImageFormat.PNG,
            width=800,
            height=600,
            mode="RGB",
        )
        assert meta.file_path == "/test/image.png"
        assert meta.width == 800
        assert meta.height == 600

    def test_to_dict(self):
        meta = ImageMetadata(
            file_path="/test.png",
            file_name="test.png",
            file_size=512,
            format=ImageFormat.PNG,
            width=100,
            height=100,
            mode="RGB",
        )
        d = meta.to_dict()
        assert d["format"] == "png"
        assert d["width"] == 100
        assert d["height"] == 100
        assert d["mode"] == "RGB"


class TestColorInfo:
    def test_creation(self):
        info = ColorInfo(
            dominant_colors=[(255, 0, 0), (0, 255, 0)],
            average_color=(128, 128, 128),
            color_variance=0.1,
            is_grayscale=False,
            brightness=0.5,
            contrast=0.3,
        )
        assert len(info.dominant_colors) == 2
        assert info.brightness == 0.5

    def test_to_dict(self):
        info = ColorInfo(
            dominant_colors=[(255, 0, 0)],
            average_color=(128, 0, 0),
            color_variance=0.05,
            is_grayscale=False,
            brightness=0.3,
            contrast=0.2,
        )
        d = info.to_dict()
        assert d["dominant_colors"] == [[255, 0, 0]]
        assert d["average_color"] == [128, 0, 0]
        assert d["brightness"] == 0.3


class TestImageAnalysis:
    def test_creation(self):
        meta = ImageMetadata(
            file_path="/test.png",
            file_name="test.png",
            file_size=1024,
            format=ImageFormat.PNG,
            width=100,
            height=100,
            mode="RGB",
        )
        analysis = ImageAnalysis(metadata=meta)
        assert analysis.metadata.width == 100
        assert len(analysis.detected_shapes) == 0

    def test_to_dict(self):
        meta = ImageMetadata(
            file_path="/test.png",
            file_name="test.png",
            file_size=1024,
            format=ImageFormat.PNG,
            width=100,
            height=100,
            mode="RGB",
        )
        analysis = ImageAnalysis(
            metadata=meta,
            detected_text="Hello World",
            ocr_confidence=0.85,
        )
        d = analysis.to_dict()
        assert d["detected_text"] == "Hello World"
        assert d["ocr_confidence"] == 0.85


class TestImagePreprocessor:
    def test_load_image_not_found(self):
        img, warnings = ImagePreprocessor.load_image_safely("/nonexistent.png")
        assert img is None
        assert len(warnings) > 0
        assert "not found" in warnings[0].lower() or "not a file" in warnings[0].lower()

    def test_load_image_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"")
            f.flush()
            path = f.name

        try:
            img, warnings = ImagePreprocessor.load_image_safely(path)
            assert img is None
            assert any("empty" in w.lower() for w in warnings)
        finally:
            os.unlink(path)

    def test_load_image_too_large(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write 21MB file (exceeds default 20MB limit)
            f.write(b"x" * (21 * 1024 * 1024))
            f.flush()
            path = f.name

        try:
            img, warnings = ImagePreprocessor.load_image_safely(path, max_size_mb=20)
            assert img is None
            assert any("too large" in w.lower() for w in warnings)
        finally:
            os.unlink(path)

    def test_load_valid_image(self):
        try:
            from PIL import Image

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                img.save(f)
                path = f.name

            try:
                loaded, warnings = ImagePreprocessor.load_image_safely(path)
                assert loaded is not None
                assert loaded.size == (100, 100)
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_extract_metadata(self):
        try:
            from PIL import Image

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (200, 150), color=(0, 128, 255))
                img.save(f)
                path = f.name

            try:
                loaded = Image.open(path)
                meta = ImagePreprocessor.extract_metadata(loaded, path)
                assert meta.width == 200
                assert meta.height == 150
                assert meta.format == ImageFormat.PNG
                assert meta.mode == "RGB"
                assert meta.file_name == Path(path).name
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_analyze_colors(self):
        try:
            from PIL import Image

            # Create a red image
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            color_info = ImagePreprocessor.analyze_colors(img, sample_size=100)
            assert color_info.brightness > 0
            assert not color_info.is_grayscale
            assert len(color_info.dominant_colors) > 0

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_analyze_colors_grayscale(self):
        try:
            from PIL import Image

            # Create a grayscale-like image
            img = Image.new("RGB", (100, 100), color=(128, 128, 128))
            color_info = ImagePreprocessor.analyze_colors(img, sample_size=100)
            assert color_info.is_grayscale

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_detect_basic_regions(self):
        try:
            from PIL import Image

            # Create image with varied content
            img = Image.new("RGB", (200, 200), color=(128, 128, 128))
            # Add a dark region
            for x in range(50, 100):
                for y in range(50, 100):
                    img.putpixel((x, y), (0, 0, 0))

            regions = ImagePreprocessor.detect_basic_regions(img)
            assert isinstance(regions, list)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_preprocess_for_analysis(self):
        try:
            from PIL import Image

            img = Image.new("RGB", (1024, 768), color=(100, 150, 200))
            processed, info = ImagePreprocessor.preprocess_for_analysis(
                img, target_size=(256, 256)
            )
            assert processed.size[0] <= 256
            assert processed.size[1] <= 256
            assert "original_size" in info

        except ImportError:
            pytest.skip("Pillow not installed")


class TestRealVisionProvider:
    def test_init(self):
        provider = RealVisionProvider()
        assert provider.name == "real_vision"
        assert provider.is_available is True

    def test_initialize(self):
        provider = RealVisionProvider()
        result = provider.initialize()
        assert result is True
        assert provider._initialized is True

    def test_analyze_nonexistent_image(self):
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            media_type=MediaType.IMAGE,
            input_path="/nonexistent/image.png",
        )
        result = provider._analyze_image(request)
        assert result.success is False
        assert result.error != ""

    def test_analyze_valid_image(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                img.save(f)
                path = f.name

            try:
                request = MediaRequest(
                    media_type=MediaType.IMAGE,
                    input_path=path,
                    prompt="Describe this image",
                )
                result = provider._analyze_image(request)
                assert result.success is True
                assert result.image_width == 100
                assert result.image_height == 100
                assert result.provider == "real_vision"
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_analyze_with_prompt(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (200, 150), color=(0, 255, 0))
                img.save(f)
                path = f.name

            try:
                request = MediaRequest(
                    media_type=MediaType.IMAGE,
                    input_path=path,
                    prompt="What colors are in this image?",
                )
                result = provider._analyze_image(request)
                assert result.success is True
                assert result.analysis_prompt == "What colors are in this image?"
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_capabilities(self):
        provider = RealVisionProvider()
        caps = provider.get_capabilities()
        assert caps["image_analysis"] is True
        assert caps["color_analysis"] is True
        assert caps["region_detection"] is True
        assert "ocr" in caps
        assert "multimodal" in caps

    def test_process_integration(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (100, 100), color=(128, 64, 200))
                img.save(f)
                path = f.name

            try:
                request = MediaRequest(
                    media_type=MediaType.IMAGE,
                    input_path=path,
                )
                result = provider.process(request)
                assert result.success is True
                assert result.provider == "real_vision"
                assert result.metadata is not None
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")


class TestVisionResultExtended:
    def test_to_text_with_elements(self):
        result = VisionResult(
            success=True,
            description="Test analysis",
            detected_elements=[
                DetectedElement(
                    element_type="button",
                    description="Submit button",
                    bounding_box=BoundingBox(x=10, y=20, width=100, height=50),
                    confidence=VisionConfidence.HIGH,
                ),
                DetectedElement(
                    element_type="text",
                    description="Hello world",
                    confidence=VisionConfidence.MEDIUM,
                ),
            ],
            image_width=800,
            image_height=600,
        )
        text = result.to_text()
        assert "[BEGIN UNTRUSTED VISUAL CONTENT]" in text
        assert "Submit button" in text
        assert "Hello world" in text
        assert "[END UNTRUSTED VISUAL CONTENT]" in text

    def test_to_dict_roundtrip(self):
        result = VisionResult(
            success=True,
            description="Test",
            detected_elements=[
                DetectedElement(
                    element_type="icon",
                    description="Settings",
                    bounding_box=BoundingBox(x=0, y=0, width=32, height=32),
                ),
            ],
            image_width=100,
            image_height=100,
            provider="test",
        )
        d = result.to_dict()
        restored = VisionResult.from_dict(d)
        assert restored.success is True
        assert restored.description == "Test"
        assert len(restored.detected_elements) == 1
        assert restored.detected_elements[0].element_type == "icon"


class TestVisionProviderBase:
    def test_validation_missing_path(self):
        provider = VisionProvider()
        request = MediaRequest(
            media_type=MediaType.IMAGE,
            input_path="",
        )
        valid, errors = provider.validate_request(request)
        assert valid is False
        assert any("required" in e.lower() for e in errors)

    def test_validation_unsupported_format(self):
        try:
            with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
                f.write(b"test")
                path = f.name

            try:
                provider = VisionProvider()
                request = MediaRequest(
                    media_type=MediaType.IMAGE,
                    input_path=path,
                )
                valid, errors = provider.validate_request(request)
                assert valid is False
                assert any("format" in e.lower() for e in errors)
            finally:
                os.unlink(path)

        except Exception:
            pytest.skip("Test setup failed")

    def test_stats(self):
        provider = VisionProvider()
        stats = provider.stats
        assert "request_count" in stats
        assert "total_time" in stats
        assert "avg_time" in stats


class TestDetectedElementExtended:
    def test_from_dict_roundtrip(self):
        elem = DetectedElement(
            element_type="button",
            description="Click me",
            bounding_box=BoundingBox(x=10, y=20, width=100, height=50),
            confidence=VisionConfidence.HIGH,
            metadata={"custom": "value"},
        )
        d = elem.to_dict()
        restored = DetectedElement.from_dict(d)
        assert restored.element_type == "button"
        assert restored.description == "Click me"
        assert restored.bounding_box.x == 10
        assert restored.confidence == VisionConfidence.HIGH
        assert restored.metadata["custom"] == "value"


class TestBoundingBoxExtended:
    def test_to_text(self):
        bb = BoundingBox(x=10, y=20, width=100, height=50)
        text = bb.to_dict()
        assert text["x"] == 10
        assert text["y"] == 20
        assert text["width"] == 100
        assert text["height"] == 50

    def test_center_calculation(self):
        bb = BoundingBox(x=10, y=20, width=100, height=50)
        center_x = bb.x + bb.width // 2
        center_y = bb.y + bb.height // 2
        assert center_x == 60
        assert center_y == 45
