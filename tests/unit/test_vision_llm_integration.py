"""Integration tests for Vision → LLM pipeline.

Tests the complete flow: screenshot → vision → OCR → grounding → context → LLM message.
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

from agent.media.multimodal import (
    VisionContextBuilder,
    VisionSummaryContent,
    MultimodalMessage,
    TextContent,
    OCRContent,
    GroundingContent,
)
from agent.media.real_vision import RealVisionProvider
from agent.media.grounding import VisualGrounder
from agent.media.base import MediaRequest
from agent.media.vision import VisionResult, DetectedElement, BoundingBox, VisionConfidence


@pytest.fixture
def text_image():
    """Create an image with clear text for testing."""
    from PIL import Image, ImageDraw
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Submit", fill=(0, 0, 0))
        draw.text((50, 120), "Cancel", fill=(0, 0, 0))
        draw.text((50, 200), "Hello World", fill=(0, 0, 0))
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


def _dicts_to_elements(elements):
    """Convert dicts to DetectedElement objects."""
    result = []
    for e in elements:
        if isinstance(e, DetectedElement):
            result.append(e)
        elif isinstance(e, dict):
            bb_data = e.get("bounding_box")
            bb = None
            if bb_data and isinstance(bb_data, dict):
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


# ============================================================
# Test Vision → Context Builder Pipeline
# ============================================================

class TestVisionToContextPipeline:
    def test_screenshot_to_context(self, text_image):
        """Full pipeline: image → vision → context for LLM."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-ctx-001",
            media_type=None,
            prompt="Analyze this image",
            input_path=text_image,
        )

        vision_result = provider.process(request)
        assert vision_result.success

        elements = _dicts_to_elements(vision_result.metadata.get("detected_elements", []))
        vr = VisionResult(
            success=True,
            description=vision_result.metadata.get("description", ""),
            detected_elements=elements,
            image_width=vision_result.metadata.get("image_width", 0),
            image_height=vision_result.metadata.get("image_height", 0),
        )

        grounder = VisualGrounder(screen_width=600, screen_height=400)
        grounding_result = grounder.ground(vr)

        builder = VisionContextBuilder()
        summary = builder.build_from_vision_result(
            vr,
            grounding_result=grounding_result,
            image_path=text_image,
        )

        assert summary.image_path == text_image
        assert summary.screen_width == 600
        assert summary.screen_height == 400
        assert summary.grounded_targets is not None

    def test_context_to_llm_messages(self, text_image):
        """Vision context should produce valid LLM messages."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-llm-001",
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
        grounding_result = grounder.ground(vr)

        builder = VisionContextBuilder()
        summary = builder.build_from_vision_result(vr, grounding_result, text_image)

        messages = builder.build_context_for_llm(
            summary,
            user_query="What is on the screen?",
        )

        # Messages should be valid for LLM
        assert isinstance(messages, list)
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("system", "user", "assistant")

        # Should contain visual context
        all_content = " ".join(m["content"] for m in messages)
        assert "UNTRUSTED" in all_content

    def test_autonomous_task_context(self, text_image):
        """Autonomous context should include action definitions."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-auto-001",
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
        grounding_result = grounder.ground(vr)

        builder = VisionContextBuilder()
        summary = builder.build_from_vision_result(vr, grounding_result, text_image)

        messages = builder.build_autonomous_context(
            summary,
            task_objective="Click the Submit button",
            previous_actions=["took screenshot"],
            retry_count=0,
        )

        assert isinstance(messages, list)
        all_content = " ".join(m["content"] for m in messages)
        assert "Click the Submit button" in all_content
        assert "action" in all_content.lower()

    def test_vision_summary_serializable(self, text_image):
        """VisionSummaryContent should be JSON serializable."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-serial-001",
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

        builder = VisionContextBuilder()
        summary = builder.build_from_vision_result(vr, image_path=text_image)

        d = summary.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

        restored = VisionSummaryContent.from_dict(json.loads(json_str))
        assert restored.image_path == text_image

    def test_multimodal_message_with_ocr(self, text_image):
        """MultimodalMessage with OCR content should produce valid LLM text."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-msg-001",
            media_type=None,
            prompt="test",
            input_path=text_image,
        )

        vision_result = provider.process(request)
        ocr_text = vision_result.metadata.get("detected_text", "")
        ocr_conf = vision_result.metadata.get("ocr_confidence", 0.0)

        msg = MultimodalMessage(
            role="user",
            content_parts=[
                TextContent(text="What do you see?"),
                OCRContent(
                    text=ocr_text,
                    confidence=ocr_conf,
                    image_width=vision_result.metadata.get("image_width", 0),
                    image_height=vision_result.metadata.get("image_height", 0),
                ),
            ],
        )

        llm_msg = msg.to_llm_message()
        assert llm_msg["role"] == "user"
        assert "What do you see?" in llm_msg["content"]
        # Should have UNTRUSTED markers if OCR text exists
        if ocr_text:
            assert "UNTRUSTED" in llm_msg["content"]


# ============================================================
# Test Performance
# ============================================================

class TestVisionPipelinePerformance:
    def test_full_pipeline_time(self, text_image):
        """Full pipeline should complete in reasonable time."""
        provider = RealVisionProvider()
        provider.initialize()

        request = MediaRequest(
            request_id="test-perf-001",
            media_type=None,
            prompt="test",
            input_path=text_image,
        )

        start = time.time()

        vision_result = provider.process(request)
        elements = _dicts_to_elements(vision_result.metadata.get("detected_elements", []))
        vr = VisionResult(
            success=True,
            detected_elements=elements,
            image_width=vision_result.metadata.get("image_width", 0),
            image_height=vision_result.metadata.get("image_height", 0),
        )

        grounder = VisualGrounder(screen_width=600, screen_height=400)
        grounding_result = grounder.ground(vr)

        builder = VisionContextBuilder()
        summary = builder.build_from_vision_result(vr, grounding_result, text_image)
        messages = builder.build_context_for_llm(summary, user_query="test")

        elapsed = time.time() - start

        # Should complete within 5 seconds
        assert elapsed < 5.0
        assert len(messages) > 0
