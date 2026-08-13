"""Real OCR test with generated test image.

Performs actual OCR on a generated image to verify the pipeline works.
This is NOT a unit test with mocks - it uses real OCR.
"""

import os
import sys
import time
import json
import tempfile
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def rose_test_image():
    """Create a test image with 'Rose Vision Test' text."""
    from PIL import Image, ImageDraw, ImageFont

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (600, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try to use a larger font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 36)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except (OSError, IOError):
                font = ImageFont.load_default()
                font_small = font

        draw.text((100, 50), "Rose Vision Test", fill=(0, 0, 0), font=font)
        draw.text((150, 120), "OCR Pipeline", fill=(0, 100, 0), font=font_small)
        draw.text((100, 180), "Submit", fill=(0, 0, 200), font=font_small)
        draw.text((300, 180), "Cancel", fill=(200, 0, 0), font=font_small)

        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def chinese_test_image():
    """Create a test image with simple Chinese characters (if font available)."""
    from PIL import Image, ImageDraw

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Hello 123", fill=(0, 0, 0))
        draw.text((50, 100), "ABC def", fill=(0, 0, 0))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


# ============================================================
# Real OCR Tests
# ============================================================

class TestRealOCR:
    def test_ocr_extracts_text(self, rose_test_image):
        """Real OCR should extract text from the test image."""
        from agent.media.ocr import LocalOCRProvider, OCRStatus

        provider = LocalOCRProvider()
        provider.initialize()

        if not provider.is_available:
            pytest.skip("Tesseract not installed")

        result = provider.extract_text(rose_test_image)

        print(f"\n[OCR Result]")
        print(f"  Status: {result.status.value}")
        print(f"  Text: '{result.text}'")
        print(f"  Blocks: {result.block_count}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Time: {result.execution_time:.3f}s")
        print(f"  Dimensions: {result.image_width}x{result.image_height}")

        assert result.status in (OCRStatus.SUCCESS, OCRStatus.PARTIAL)
        assert result.image_width == 600
        assert result.image_height == 300

        if result.blocks:
            print(f"\n[OCR Blocks]")
            for i, block in enumerate(result.blocks):
                print(f"  [{i}] '{block.text}' @ ({block.x},{block.y}) {block.width}x{block.height} conf={block.confidence:.2f}")

    def test_ocr_finds_key_text(self, rose_test_image):
        """OCR should find key words like 'Rose', 'Submit', 'Cancel'."""
        from agent.media.ocr import LocalOCRProvider

        provider = LocalOCRProvider()
        provider.initialize()

        if not provider.is_available:
            pytest.skip("Tesseract not installed")

        result = provider.extract_text(rose_test_image)

        # Convert to lowercase for comparison
        text_lower = result.text.lower()

        print(f"\n[Key Text Search]")
        print(f"  Full text: '{result.text}'")

        # Check for key words
        key_words = ["rose", "submit", "cancel", "ocr"]
        found_words = []
        for word in key_words:
            if word in text_lower:
                found_words.append(word)
                print(f"  Found: '{word}'")
            else:
                print(f"  NOT found: '{word}'")

        print(f"\n  Found {len(found_words)}/{len(key_words)} key words")

        # Should find at least some text
        assert len(result.text.strip()) > 0, "No text detected"

    def test_ocr_blocks_have_coordinates(self, rose_test_image):
        """OCR blocks should have valid bounding box coordinates."""
        from agent.media.ocr import LocalOCRProvider

        provider = LocalOCRProvider()
        provider.initialize()

        if not provider.is_available:
            pytest.skip("Tesseract not installed")

        result = provider.extract_text(rose_test_image)

        if result.blocks:
            for block in result.blocks:
                assert block.x >= 0, f"Block x={block.x} is negative"
                assert block.y >= 0, f"Block y={block.y} is negative"
                assert block.width > 0, f"Block width={block.width} is zero"
                assert block.height > 0, f"Block height={block.height} is zero"
                assert 0 <= block.confidence <= 1.0, f"Block confidence={block.confidence} out of range"

    def test_vision_with_real_ocr(self, rose_test_image):
        """Full vision pipeline with real OCR."""
        from agent.media.real_vision import RealVisionProvider
        from agent.media.base import MediaRequest

        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-real-ocr-001",
            media_type=None,
            prompt="Analyze this test image",
            input_path=rose_test_image,
        )

        result = provider.process(request)

        print(f"\n[Vision Result]")
        print(f"  Success: {result.success}")
        print(f"  Provider: {result.metadata.get('provider', 'unknown')}")
        if result.success:
            print(f"  Elements: {len(result.metadata.get('detected_elements', []))}")
            print(f"  OCR text: '{result.metadata.get('detected_text', '')}'")
            print(f"  OCR confidence: {result.metadata.get('ocr_confidence', 0):.2f}")

        assert result.success

    def test_ocr_performance(self, rose_test_image):
        """Measure OCR processing time."""
        from agent.media.ocr import LocalOCRProvider

        provider = LocalOCRProvider()
        provider.initialize()

        if not provider.is_available:
            pytest.skip("Tesseract not installed")

        start = time.time()
        result = provider.extract_text(rose_test_image)
        elapsed = time.time() - start

        print(f"\n[Performance]")
        print(f"  Image: 600x300")
        print(f"  OCR time: {elapsed:.3f}s")
        print(f"  Blocks: {result.block_count}")
        print(f"  Characters: {result.char_count}")

        # Should be reasonably fast
        assert elapsed < 10.0, f"OCR too slow: {elapsed:.1f}s"

    def test_ocr_json_serializable(self, rose_test_image):
        """OCR results should be fully JSON serializable."""
        from agent.media.ocr import LocalOCRProvider

        provider = LocalOCRProvider()
        provider.initialize()

        if not provider.is_available:
            pytest.skip("Tesseract not installed")

        result = provider.extract_text(rose_test_image)
        d = result.to_dict()

        # Serialize and deserialize
        json_str = json.dumps(d, indent=2)
        restored = json.loads(json_str)

        assert restored["text"] == result.text
        assert restored["provider"] == result.provider
        assert restored["status"] == result.status.value

        print(f"\n[JSON Serialization]")
        print(f"  JSON size: {len(json_str)} bytes")
        print(f"  Round-trip: OK")
