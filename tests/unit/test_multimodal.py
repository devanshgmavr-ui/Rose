"""Comprehensive tests for multimodal message types and VisionContextBuilder.

Tests cover:
- ContentPart types (TextContent, ImageContent, OCRContent, GroundingContent, VisionSummaryContent)
- MultimodalMessage
- VisionContextBuilder
- Serialization/deserialization
- Security boundaries
- LLM context building
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.media.multimodal import (
    ContentType,
    ContentPart,
    TextContent,
    ImageContent,
    OCRContent,
    GroundingContent,
    VisionSummaryContent,
    MultimodalMessage,
    VisionContextBuilder,
    create_content_part,
    serialize_content_parts,
    deserialize_content_parts,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_text_content():
    return TextContent(text="Hello World")


@pytest.fixture
def sample_image_content():
    return ImageContent(
        image_path="/workspace/screenshots/test.png",
        description="Screenshot of desktop",
        width=1920,
        height=1080,
        format="PNG",
    )


@pytest.fixture
def sample_ocr_content():
    return OCRContent(
        text="Submit Cancel Login",
        confidence=0.92,
        blocks=[
            {"text": "Submit", "confidence": 0.95, "bbox": {"x": 100, "y": 200, "width": 80, "height": 30}},
            {"text": "Cancel", "confidence": 0.93, "bbox": {"x": 200, "y": 200, "width": 80, "height": 30}},
            {"text": "Login", "confidence": 0.88, "bbox": {"x": 300, "y": 200, "width": 60, "height": 30}},
        ],
        image_width=1920,
        image_height=1080,
    )


@pytest.fixture
def sample_grounding_content():
    return GroundingContent(
        targets=[
            {
                "description": "Submit button",
                "target_type": "button",
                "center": {"x": 140, "y": 215},
                "bounding_box": {"x": 100, "y": 200, "width": 80, "height": 30},
                "confidence": "high",
            },
            {
                "description": "Cancel button",
                "target_type": "button",
                "center": {"x": 240, "y": 215},
                "bounding_box": {"x": 200, "y": 200, "width": 80, "height": 30},
                "confidence": "medium",
            },
        ],
        screen_width=1920,
        screen_height=1080,
    )


@pytest.fixture
def sample_vision_summary():
    return VisionSummaryContent(
        image_path="/workspace/screenshots/test.png",
        ocr_text="Submit Cancel Login",
        ocr_confidence=0.92,
        grounded_targets=[
            {
                "description": "Submit button",
                "target_type": "button",
                "center": {"x": 140, "y": 215},
            },
        ],
        image_description="Screen with login form",
        screen_width=1920,
        screen_height=1080,
        analysis_time=0.25,
    )


@pytest.fixture
def sample_multimodal_message(sample_text_content, sample_ocr_content):
    return MultimodalMessage(
        role="user",
        content_parts=[sample_text_content, sample_ocr_content],
        metadata={"source": "test"},
    )


# ============================================================
# Test TextContent
# ============================================================

class TestTextContent:
    def test_creation(self, sample_text_content):
        assert sample_text_content.content_type == ContentType.TEXT
        assert sample_text_content.text == "Hello World"

    def test_to_text(self, sample_text_content):
        assert sample_text_content.to_text() == "Hello World"

    def test_to_dict(self, sample_text_content):
        d = sample_text_content.to_dict()
        assert d["content_type"] == "text"
        assert d["text"] == "Hello World"

    def test_from_dict(self, sample_text_content):
        d = sample_text_content.to_dict()
        restored = TextContent.from_dict(d)
        assert restored.text == "Hello World"

    def test_empty(self):
        c = TextContent(text="")
        assert c.to_text() == ""


# ============================================================
# Test ImageContent
# ============================================================

class TestImageContent:
    def test_creation(self, sample_image_content):
        assert sample_image_content.content_type == ContentType.IMAGE
        assert sample_image_content.image_path == "/workspace/screenshots/test.png"
        assert sample_image_content.width == 1920
        assert sample_image_content.height == 1080

    def test_to_text(self, sample_image_content):
        text = sample_image_content.to_text()
        assert "/workspace/screenshots/test.png" in text
        assert "1920x1080" in text
        assert "PNG" in text

    def test_to_dict(self, sample_image_content):
        d = sample_image_content.to_dict()
        assert d["content_type"] == "image"
        assert d["image_path"] == "/workspace/screenshots/test.png"
        assert d["width"] == 1920

    def test_from_dict(self, sample_image_content):
        d = sample_image_content.to_dict()
        restored = ImageContent.from_dict(d)
        assert restored.image_path == sample_image_content.image_path
        assert restored.width == 1920

    def test_minimal(self):
        c = ImageContent(image_path="test.png")
        assert c.content_type == ContentType.IMAGE
        assert c.width == 0


# ============================================================
# Test OCRContent
# ============================================================

class TestOCRContent:
    def test_creation(self, sample_ocr_content):
        assert sample_ocr_content.content_type == ContentType.OCR
        assert sample_ocr_content.text == "Submit Cancel Login"
        assert sample_ocr_content.confidence == 0.92
        assert len(sample_ocr_content.blocks) == 3

    def test_to_text_untrusted_markers(self, sample_ocr_content):
        text = sample_ocr_content.to_text()
        assert "[BEGIN UNTRUSTED VISION CONTENT]" in text
        assert "[END UNTRUSTED VISION CONTENT]" in text
        assert "Submit Cancel Login" in text

    def test_to_text_blocks(self, sample_ocr_content):
        text = sample_ocr_content.to_text()
        assert "Submit" in text
        assert "Cancel" in text
        assert "100,200" in text

    def test_to_dict(self, sample_ocr_content):
        d = sample_ocr_content.to_dict()
        assert d["content_type"] == "ocr"
        assert d["text"] == "Submit Cancel Login"
        assert d["confidence"] == 0.92
        assert len(d["blocks"]) == 3

    def test_from_dict(self, sample_ocr_content):
        d = sample_ocr_content.to_dict()
        restored = OCRContent.from_dict(d)
        assert restored.text == sample_ocr_content.text
        assert restored.confidence == 0.92

    def test_empty(self):
        c = OCRContent(text="")
        text = c.to_text()
        assert "No text detected" in text


# ============================================================
# Test GroundingContent
# ============================================================

class TestGroundingContent:
    def test_creation(self, sample_grounding_content):
        assert sample_grounding_content.content_type == ContentType.GROUNDING
        assert len(sample_grounding_content.targets) == 2
        assert sample_grounding_content.screen_width == 1920

    def test_to_text_untrusted_markers(self, sample_grounding_content):
        text = sample_grounding_content.to_text()
        assert "[BEGIN UNTRUSTED GROUNDING DATA]" in text
        assert "[END UNTRUSTED GROUNDING DATA]" in text

    def test_to_text_targets(self, sample_grounding_content):
        text = sample_grounding_content.to_text()
        assert "Submit button" in text
        assert "140,215" in text
        assert "button" in text

    def test_to_dict(self, sample_grounding_content):
        d = sample_grounding_content.to_dict()
        assert d["content_type"] == "grounding"
        assert len(d["targets"]) == 2
        assert d["screen_width"] == 1920

    def test_from_dict(self, sample_grounding_content):
        d = sample_grounding_content.to_dict()
        restored = GroundingContent.from_dict(d)
        assert len(restored.targets) == 2

    def test_empty(self):
        c = GroundingContent()
        text = c.to_text()
        assert "No visual targets" in text


# ============================================================
# Test VisionSummaryContent
# ============================================================

class TestVisionSummaryContent:
    def test_creation(self, sample_vision_summary):
        assert sample_vision_summary.content_type == ContentType.VISION_SUMMARY
        assert sample_vision_summary.image_path == "/workspace/screenshots/test.png"
        assert sample_vision_summary.ocr_confidence == 0.92

    def test_to_text_untrusted_markers(self, sample_vision_summary):
        text = sample_vision_summary.to_text()
        assert "[BEGIN UNTRUSTED VISUAL SUMMARY]" in text
        assert "[END UNTRUSTED VISUAL SUMMARY]" in text

    def test_to_text_content(self, sample_vision_summary):
        text = sample_vision_summary.to_text()
        assert "1920x1080" in text
        assert "Submit Cancel Login" in text
        assert "Submit button" in text

    def test_to_dict(self, sample_vision_summary):
        d = sample_vision_summary.to_dict()
        assert d["content_type"] == "vision_summary"
        assert d["image_path"] == "/workspace/screenshots/test.png"
        assert d["ocr_confidence"] == 0.92

    def test_from_dict(self, sample_vision_summary):
        d = sample_vision_summary.to_dict()
        restored = VisionSummaryContent.from_dict(d)
        assert restored.image_path == sample_vision_summary.image_path
        assert restored.ocr_confidence == 0.92


# ============================================================
# Test MultimodalMessage
# ============================================================

class TestMultimodalMessage:
    def test_creation(self, sample_multimodal_message):
        assert sample_multimodal_message.role == "user"
        assert len(sample_multimodal_message.content_parts) == 2
        assert sample_multimodal_message.metadata["source"] == "test"

    def test_has_images(self, sample_multimodal_message):
        assert sample_multimodal_message.has_images is False

    def test_has_images_true(self):
        msg = MultimodalMessage(role="user", content_parts=[ImageContent(image_path="test.png")])
        assert msg.has_images is True

    def test_has_ocr(self, sample_multimodal_message):
        assert sample_multimodal_message.has_ocr is True

    def test_to_text(self, sample_multimodal_message):
        text = sample_multimodal_message.to_text()
        assert "Hello World" in text
        assert "Submit Cancel Login" in text

    def test_to_llm_message(self, sample_multimodal_message):
        llm_msg = sample_multimodal_message.to_llm_message()
        assert llm_msg["role"] == "user"
        assert "Hello World" in llm_msg["content"]
        assert "Submit Cancel Login" in llm_msg["content"]

    def test_to_dict(self, sample_multimodal_message):
        d = sample_multimodal_message.to_dict()
        assert d["role"] == "user"
        assert len(d["content_parts"]) == 2

    def test_from_dict(self, sample_multimodal_message):
        d = sample_multimodal_message.to_dict()
        restored = MultimodalMessage.from_dict(d)
        assert restored.role == "user"
        assert len(restored.content_parts) == 2

    def test_serialize_deserialize(self, sample_multimodal_message):
        d = sample_multimodal_message.to_dict()
        json_str = json.dumps(d)
        restored = MultimodalMessage.from_dict(json.loads(json_str))
        assert restored.role == "user"


# ============================================================
# Test VisionContextBuilder
# ============================================================

class TestVisionContextBuilder:
    def test_init(self):
        builder = VisionContextBuilder()
        assert builder._max_ocr_chars == 2000
        assert builder._max_targets == 15

    def test_stats(self):
        builder = VisionContextBuilder()
        assert builder.stats["build_count"] == 0
        assert builder.stats["avg_time"] == 0.0

    def test_build_from_vision_result_mock(self):
        """Test with mock VisionResult."""
        builder = VisionContextBuilder()

        mock_vr = MagicMock()
        mock_vr.description = "Screen analysis"
        mock_vr.image_width = 1920
        mock_vr.image_height = 1080
        mock_vr.detected_elements = []
        mock_vr.metadata = {
            "detected_text": "Hello World",
            "ocr_confidence": 0.95,
            "metadata": {"width": 1920, "height": 1080},
        }

        summary = builder.build_from_vision_result(mock_vr, image_path="test.png")
        assert summary.image_path == "test.png"
        assert summary.ocr_text == "Hello World"
        assert summary.ocr_confidence == 0.95
        assert summary.screen_width == 1920

    def test_build_context_for_llm(self, sample_vision_summary):
        builder = VisionContextBuilder()
        messages = builder.build_context_for_llm(
            sample_vision_summary,
            user_query="What do you see?",
        )
        assert len(messages) >= 2
        # Should have system message with visual context
        visual_msgs = [m for m in messages if "UNTRUSTED" in m.get("content", "")]
        assert len(visual_msgs) > 0
        # Should have user query
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "What do you see?" in user_msgs[0]["content"]

    def test_build_autonomous_context(self, sample_vision_summary):
        builder = VisionContextBuilder()
        messages = builder.build_autonomous_context(
            sample_vision_summary,
            task_objective="Click the Submit button",
        )
        assert len(messages) >= 3
        # Should have system prompt with action definitions
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) >= 2
        # Should have user task
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert "Click the Submit button" in user_msgs[0]["content"]

    def test_build_autonomous_context_with_retries(self, sample_vision_summary):
        builder = VisionContextBuilder()
        messages = builder.build_autonomous_context(
            sample_vision_summary,
            task_objective="Click Submit",
            retry_count=2,
        )
        # Should mention retry
        all_text = " ".join(m["content"] for m in messages)
        assert "Retry" in all_text or "retry" in all_text


# ============================================================
# Test Content Factory
# ============================================================

class TestContentFactory:
    def test_create_text(self):
        part = create_content_part({"content_type": "text", "text": "hello"})
        assert isinstance(part, TextContent)
        assert part.text == "hello"

    def test_create_image(self):
        part = create_content_part({"content_type": "image", "image_path": "test.png"})
        assert isinstance(part, ImageContent)

    def test_create_ocr(self):
        part = create_content_part({"content_type": "ocr", "text": "hello"})
        assert isinstance(part, OCRContent)

    def test_create_grounding(self):
        part = create_content_part({"content_type": "grounding", "targets": []})
        assert isinstance(part, GroundingContent)

    def test_create_vision_summary(self):
        part = create_content_part({"content_type": "vision_summary"})
        assert isinstance(part, VisionSummaryContent)

    def test_create_unknown_defaults_to_text(self):
        part = create_content_part({"content_type": "unknown"})
        assert isinstance(part, TextContent)

    def test_serialize_deserialize_parts(self):
        parts = [
            TextContent(text="hello"),
            OCRContent(text="world", confidence=0.9),
        ]
        json_str = serialize_content_parts(parts)
        restored = deserialize_content_parts(json_str)
        assert len(restored) == 2
        assert isinstance(restored[0], TextContent)
        assert isinstance(restored[1], OCRContent)


# ============================================================
# Test Security
# ============================================================

class TestMultimodalSecurity:
    def test_ocr_content_untrusted_markers(self, sample_ocr_content):
        text = sample_ocr_content.to_text()
        assert "[BEGIN UNTRUSTED VISION CONTENT]" in text
        assert "[END UNTRUSTED VISION CONTENT]" in text

    def test_grounding_untrusted_markers(self, sample_grounding_content):
        text = sample_grounding_content.to_text()
        assert "[BEGIN UNTRUSTED GROUNDING DATA]" in text
        assert "[END UNTRUSTED GROUNDING DATA]" in text

    def test_vision_summary_untrusted_markers(self, sample_vision_summary):
        text = sample_vision_summary.to_text()
        assert "[BEGIN UNTRUSTED VISUAL SUMMARY]" in text
        assert "[END UNTRUSTED VISUAL SUMMARY]" in text

    def test_no_execution_paths_in_content(self):
        """Content types should not have exec/shell/command fields."""
        for cls, kwargs in [
            (TextContent, {"text": "test"}),
            (ImageContent, {"image_path": "test.png"}),
            (OCRContent, {"text": "test"}),
            (GroundingContent, {}),
            (VisionSummaryContent, {}),
        ]:
            obj = cls(**kwargs)
            d = obj.to_dict()
            assert "exec" not in d
            assert "shell" not in d
            assert "command" not in d

    def test_message_serializable(self, sample_multimodal_message):
        """Messages must be JSON serializable."""
        d = sample_multimodal_message.to_dict()
        json_str = json.dumps(d)
        restored = MultimodalMessage.from_dict(json.loads(json_str))
        assert restored.role == "user"

    def test_image_path_no_file_access(self):
        """ImageContent should store path but not provide file access methods."""
        ic = ImageContent(image_path="/etc/passwd")
        # Should only have to_text and to_dict, no read_file or similar
        assert hasattr(ic, "to_text")
        assert hasattr(ic, "to_dict")
        assert not hasattr(ic, "read_file")
        assert not hasattr(ic, "load_image")
