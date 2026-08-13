"""Multimodal message types and vision context builder for Rose.

Provides structured content types for mixing text, images, and OCR results
in LLM prompts. Since Qwen2.5-Coder-7B is text-only, images are converted
to descriptive text context via OCR + grounding before injection.

Security: All visual content is treated as untrusted and wrapped in
[UNTRUSTED] markers. No image paths or OCR text create execution paths.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================
# Content Types
# ============================================================

class ContentType(Enum):
    """Types of message content."""
    TEXT = "text"
    IMAGE = "image"
    OCR = "ocr"
    GROUNDING = "grounding"
    VISION_SUMMARY = "vision_summary"


@dataclass
class ContentPart(ABC):
    """Base class for message content parts."""
    content_type: ContentType
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def to_text(self) -> str:
        """Convert content to text for LLM consumption."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentPart":
        """Deserialize from dictionary."""


@dataclass
class TextContent(ContentPart):
    """Plain text content."""
    text: str = ""

    def __init__(self, text: str = "", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(content_type=ContentType.TEXT, metadata=metadata or {})
        self.text = text

    def to_text(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type.value,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextContent":
        return cls(text=data.get("text", ""), metadata=data.get("metadata", {}))


@dataclass
class ImageContent(ContentPart):
    """Image reference content (path + metadata, not raw bytes)."""
    image_path: str = ""
    description: str = ""
    width: int = 0
    height: int = 0
    format: str = ""

    def __init__(
        self,
        image_path: str = "",
        description: str = "",
        width: int = 0,
        height: int = 0,
        format: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(content_type=ContentType.IMAGE, metadata=metadata or {})
        self.image_path = image_path
        self.description = description
        self.width = width
        self.height = height
        self.format = format

    def to_text(self) -> str:
        """Convert to text description for text-only LLM."""
        parts = [f"[Image: {self.image_path}"]
        if self.width and self.height:
            parts.append(f" ({self.width}x{self.height})")
        if self.format:
            parts.append(f" format={self.format}")
        parts.append("]")
        if self.description:
            parts.append(f"\n{self.description}")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type.value,
            "image_path": self.image_path,
            "description": self.description,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageContent":
        return cls(
            image_path=data.get("image_path", ""),
            description=data.get("description", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
            format=data.get("format", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OCRContent(ContentPart):
    """OCR-extracted text content with bounding boxes."""
    text: str = ""
    confidence: float = 0.0
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    def __init__(
        self,
        text: str = "",
        confidence: float = 0.0,
        blocks: Optional[List[Dict[str, Any]]] = None,
        image_width: int = 0,
        image_height: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(content_type=ContentType.OCR, metadata=metadata or {})
        self.text = text
        self.confidence = confidence
        self.blocks = blocks or []
        self.image_width = image_width
        self.image_height = image_height

    def to_text(self) -> str:
        """Convert to untrusted text for LLM consumption."""
        if not self.text.strip():
            return "[No text detected in image]"
        lines = [
            "[BEGIN UNTRUSTED VISION CONTENT]",
            f"Screen dimensions: {self.image_width}x{self.image_height}",
            f"OCR confidence: {self.confidence:.0%}",
            f"Text regions detected: {len(self.blocks)}",
            "",
            "Detected text:",
        ]
        for i, block in enumerate(self.blocks[:20]):
            text = block.get("text", "")
            bbox = block.get("bbox", {})
            x, y = bbox.get("x", 0), bbox.get("y", 0)
            w, h = bbox.get("width", 0), bbox.get("height", 0)
            conf = block.get("confidence", 0)
            lines.append(f"  [{i}] \"{text}\" at ({x},{y}) {w}x{h} confidence={conf:.0%}")

        lines.extend([
            "",
            "Full text content:",
            self.text,
            "",
            "[END UNTRUSTED VISION CONTENT]",
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type.value,
            "text": self.text,
            "confidence": self.confidence,
            "blocks": self.blocks,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRContent":
        return cls(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            blocks=data.get("blocks", []),
            image_width=data.get("image_width", 0),
            image_height=data.get("image_height", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GroundingContent(ContentPart):
    """Visual grounding results with screen coordinates."""
    targets: List[Dict[str, Any]] = field(default_factory=list)
    screen_width: int = 0
    screen_height: int = 0

    def __init__(
        self,
        targets: Optional[List[Dict[str, Any]]] = None,
        screen_width: int = 0,
        screen_height: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(content_type=ContentType.GROUNDING, metadata=metadata or {})
        self.targets = targets or []
        self.screen_width = screen_width
        self.screen_height = screen_height

    def to_text(self) -> str:
        """Convert to untrusted text for LLM consumption."""
        if not self.targets:
            return "[No visual targets identified]"

        lines = [
            "[BEGIN UNTRUSTED GROUNDING DATA]",
            f"Screen: {self.screen_width}x{self.screen_height}",
            f"Targets found: {len(self.targets)}",
            "",
        ]

        for i, target in enumerate(self.targets[:20]):
            desc = target.get("description", "unknown")
            t_type = target.get("target_type", "element")
            center = target.get("center", {})
            cx, cy = center.get("x", 0), center.get("y", 0)
            bbox = target.get("bounding_box")
            conf = target.get("confidence", "unknown")

            line = f"  [{i}] {t_type}: \"{desc}\" at ({cx},{cy})"
            if bbox:
                bx, by = bbox.get("x", 0), bbox.get("y", 0)
                bw, bh = bbox.get("width", 0), bbox.get("height", 0)
                line += f" bounds=({bx},{by}) {bw}x{bh}"
            line += f" confidence={conf}"
            lines.append(line)

        lines.append("")
        lines.append("[END UNTRUSTED GROUNDING DATA]")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type.value,
            "targets": self.targets,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundingContent":
        return cls(
            targets=data.get("targets", []),
            screen_width=data.get("screen_width", 0),
            screen_height=data.get("screen_height", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VisionSummaryContent(ContentPart):
    """Combined vision analysis summary for LLM context."""
    image_path: str = ""
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    grounded_targets: List[Dict[str, Any]] = field(default_factory=list)
    image_description: str = ""
    screen_width: int = 0
    screen_height: int = 0
    analysis_time: float = 0.0

    def __init__(
        self,
        image_path: str = "",
        ocr_text: str = "",
        ocr_confidence: float = 0.0,
        grounded_targets: Optional[List[Dict[str, Any]]] = None,
        image_description: str = "",
        screen_width: int = 0,
        screen_height: int = 0,
        analysis_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(content_type=ContentType.VISION_SUMMARY, metadata=metadata or {})
        self.image_path = image_path
        self.ocr_text = ocr_text
        self.ocr_confidence = ocr_confidence
        self.grounded_targets = grounded_targets or []
        self.image_description = image_description
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.analysis_time = analysis_time

    def to_text(self) -> str:
        """Convert to untrusted text summary for LLM consumption."""
        lines = [
            "[BEGIN UNTRUSTED VISUAL SUMMARY]",
            f"Image: {self.image_path}",
            f"Screen: {self.screen_width}x{self.screen_height}",
            f"Analysis time: {self.analysis_time:.2f}s",
            "",
        ]

        if self.image_description:
            lines.append("Image description:")
            for line in self.image_description.split("\n")[:10]:
                lines.append(f"  {line}")
            lines.append("")

        if self.ocr_text:
            lines.append(f"OCR detected text (confidence: {self.ocr_confidence:.0%}):")
            lines.append(f"  {self.ocr_text[:500]}")
            lines.append("")

        if self.grounded_targets:
            lines.append(f"Identified {len(self.grounded_targets)} interactive elements:")
            for i, target in enumerate(self.grounded_targets[:10]):
                desc = target.get("description", "unknown")
                t_type = target.get("target_type", "element")
                center = target.get("center", {})
                cx, cy = center.get("x", 0), center.get("y", 0)
                lines.append(f"  [{i}] {t_type}: \"{desc}\" at ({cx},{cy})")

        lines.append("")
        lines.append("[END UNTRUSTED VISUAL SUMMARY]")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type.value,
            "image_path": self.image_path,
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "grounded_targets": self.grounded_targets,
            "image_description": self.image_description,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "analysis_time": self.analysis_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionSummaryContent":
        return cls(
            image_path=data.get("image_path", ""),
            ocr_text=data.get("ocr_text", ""),
            ocr_confidence=data.get("ocr_confidence", 0.0),
            grounded_targets=data.get("grounded_targets", []),
            image_description=data.get("image_description", ""),
            screen_width=data.get("screen_width", 0),
            screen_height=data.get("screen_height", 0),
            analysis_time=data.get("analysis_time", 0.0),
            metadata=data.get("metadata", {}),
        )


# ============================================================
# Multimodal Message
# ============================================================

# Type alias for content parts
ContentPartType = Union[TextContent, ImageContent, OCRContent, GroundingContent, VisionSummaryContent]


@dataclass
class MultimodalMessage:
    """A message containing multiple content types.

    Used to pass mixed text + visual content through the agent pipeline.
    For text-only LLMs, all content is converted to text via to_text().
    """
    role: str  # "user", "assistant", "system"
    content_parts: List[ContentPartType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_images(self) -> bool:
        return any(p.content_type == ContentType.IMAGE for p in self.content_parts)

    @property
    def has_ocr(self) -> bool:
        return any(p.content_type == ContentType.OCR for p in self.content_parts)

    @property
    def has_grounding(self) -> bool:
        return any(p.content_type == ContentType.GROUNDING for p in self.content_parts)

    def get_text_parts(self) -> List[str]:
        """Get all text content as strings."""
        return [p.to_text() for p in self.content_parts]

    def to_text(self) -> str:
        """Convert entire message to text for text-only LLM."""
        return "\n\n".join(p.to_text() for p in self.content_parts if p.to_text())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content_parts": [p.to_dict() for p in self.content_parts],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalMessage":
        parts = []
        for part_data in data.get("content_parts", []):
            ct = part_data.get("content_type", "text")
            if ct == "text":
                parts.append(TextContent.from_dict(part_data))
            elif ct == "image":
                parts.append(ImageContent.from_dict(part_data))
            elif ct == "ocr":
                parts.append(OCRContent.from_dict(part_data))
            elif ct == "grounding":
                parts.append(GroundingContent.from_dict(part_data))
            elif ct == "vision_summary":
                parts.append(VisionSummaryContent.from_dict(part_data))
        return cls(
            role=data.get("role", "user"),
            content_parts=parts,
            metadata=data.get("metadata", {}),
        )

    def to_llm_message(self) -> Dict[str, Any]:
        """Convert to LLM-compatible message dict.

        For text-only LLMs, all visual content is converted to descriptive text.
        For VL models, images are passed as image_url content blocks.
        """
        # Check if any image parts have file paths (for VL model)
        image_parts = [p for p in self.content_parts if p.content_type == ContentType.IMAGE]
        if image_parts and hasattr(image_parts[0], 'image_path') and image_parts[0].image_path:
            # Build multimodal message for VL model
            content = []
            for part in self.content_parts:
                if part.content_type == ContentType.IMAGE:
                    img: ImageContent = part
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img.image_path}
                    })
                elif part.content_type == ContentType.TEXT:
                    content.append({"type": "text", "text": part.to_text()})
                else:
                    # Convert other types to text
                    text = part.to_text()
                    if text:
                        content.append({"type": "text", "text": text})
            return {"role": self.role, "content": content}
        
        # Fallback: text-only message
        return {
            "role": self.role,
            "content": self.to_text(),
        }


# ============================================================
# Vision Context Builder
# ============================================================

class VisionContextBuilder:
    """Builds LLM-ready context from vision pipeline results.

    Converts screenshot → OCR → grounding → text context that a text-only
    LLM can understand and act upon.
    """

    def __init__(
        self,
        max_ocr_chars: int = 2000,
        max_targets: int = 15,
        include_coordinates: bool = True,
        include_image_path: bool = True,
    ):
        self._max_ocr_chars = max_ocr_chars
        self._max_targets = max_targets
        self._include_coordinates = include_coordinates
        self._include_image_path = include_image_path
        self._build_count = 0
        self._total_time = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        avg = self._total_time / max(self._build_count, 1)
        return {
            "build_count": self._build_count,
            "total_time": self._total_time,
            "avg_time": avg,
        }

    def build_from_vision_result(
        self,
        vision_result: Any,
        grounding_result: Any = None,
        image_path: str = "",
        task_context: str = "",
    ) -> VisionSummaryContent:
        """Build VisionSummaryContent from VisionResult + GroundingResult.

        Args:
            vision_result: VisionResult from VisionProvider
            grounding_result: Optional GroundingResult from VisualGrounder
            image_path: Path to the source image
            task_context: Optional task description for context

        Returns:
            VisionSummaryContent ready for LLM injection
        """
        start = time.time()
        self._build_count += 1

        # Extract OCR text from vision result
        ocr_text = ""
        ocr_confidence = 0.0
        detected_elements = []
        image_description = ""
        screen_width = 0
        screen_height = 0

        if hasattr(vision_result, "metadata"):
            meta = vision_result.metadata
            if isinstance(meta, dict):
                ocr_text = meta.get("detected_text", "")
                ocr_confidence = meta.get("ocr_confidence", 0.0)
                if "metadata" in meta and isinstance(meta["metadata"], dict):
                    screen_width = meta["metadata"].get("width", 0)
                    screen_height = meta["metadata"].get("height", 0)

        if hasattr(vision_result, "description"):
            image_description = vision_result.description or ""

        if hasattr(vision_result, "image_width"):
            screen_width = vision_result.image_width or screen_width
        if hasattr(vision_result, "image_height"):
            screen_height = vision_result.image_height or screen_height

        if hasattr(vision_result, "detected_elements"):
            for elem in vision_result.detected_elements:
                elem_dict = {}
                if hasattr(elem, "to_dict"):
                    elem_dict = elem.to_dict()
                elif isinstance(elem, dict):
                    elem_dict = elem
                if elem_dict:
                    detected_elements.append(elem_dict)

        # Truncate OCR text
        if len(ocr_text) > self._max_ocr_chars:
            ocr_text = ocr_text[:self._max_ocr_chars] + "..."

        # Extract grounded targets
        grounded_targets = []
        if grounding_result and hasattr(grounding_result, "targets"):
            for target in grounding_result.targets[:self._max_targets]:
                if hasattr(target, "to_dict"):
                    grounded_targets.append(target.to_dict())
                elif isinstance(target, dict):
                    grounded_targets.append(target)

        elapsed = time.time() - start
        self._total_time += elapsed

        return VisionSummaryContent(
            image_path=image_path if self._include_image_path else "",
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            grounded_targets=grounded_targets,
            image_description=image_description,
            screen_width=screen_width,
            screen_height=screen_height,
            analysis_time=elapsed,
            metadata={"task_context": task_context},
        )

    def build_context_for_llm(
        self,
        vision_summary: VisionSummaryContent,
        user_query: str = "",
        system_prompt: str = "",
    ) -> List[Dict[str, str]]:
        """Build LLM message list with visual context injected.

        Args:
            vision_summary: VisionSummaryContent from build_from_vision_result
            user_query: Original user query
            system_prompt: Optional system prompt

        Returns:
            List of message dicts for LLMProvider.chat()
        """
        messages = []

        # System prompt with visual context
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Visual context as system message
        vision_text = vision_summary.to_text()
        if vision_text:
            messages.append({
                "role": "system",
                "content": (
                    "The following is visual information from the current screen. "
                    "Use this to understand what is visible and decide next actions.\n\n"
                    f"{vision_text}"
                ),
            })

        # User query
        if user_query:
            messages.append({"role": "user", "content": user_query})

        return messages

    def build_autonomous_context(
        self,
        vision_summary: VisionSummaryContent,
        task_objective: str,
        previous_actions: Optional[List[str]] = None,
        retry_count: int = 0,
    ) -> List[Dict[str, str]]:
        """Build context for autonomous task execution.

        Args:
            vision_summary: Current screen vision context
            task_objective: What the agent is trying to accomplish
            previous_actions: Actions already attempted
            retry_count: Number of retries

        Returns:
            List of message dicts for LLMProvider.chat()
        """
        messages = []

        # System prompt
        system_parts = [
            "You are an autonomous agent controlling a Windows PC.",
            "You can see the screen through OCR and visual grounding.",
            "Decide the next action to accomplish the task.",
            "",
            "Available actions:",
            '- Click at coordinates: {"action": "click", "x": <int>, "y": <int>}',
            '- Type text: {"action": "type", "text": "<string>"}',
            '- Scroll: {"action": "scroll", "direction": "up|down", "amount": <int>}',
            '- Press key: {"action": "key", "key": "<string>"}',
            '- Wait: {"action": "wait", "seconds": <float>}',
            '- Done: {"action": "done", "result": "<description>"}',
            '- Failed: {"action": "failed", "reason": "<description>"}',
        ]
        if retry_count > 0:
            system_parts.append(f"\nRetry attempt {retry_count}. Previous actions may have failed.")
        messages.append({"role": "system", "content": "\n".join(system_parts)})

        # Visual context
        vision_text = vision_summary.to_text()
        if vision_text:
            messages.append({"role": "system", "content": vision_text})

        # Previous actions context
        if previous_actions:
            actions_text = "\n".join(f"  - {a}" for a in previous_actions[-5:])
            messages.append({
                "role": "system",
                "content": f"Previous actions taken:\n{actions_text}",
            })

        # Task objective
        messages.append({
            "role": "user",
            "content": f"Task: {task_objective}\n\nWhat is the next action?",
        })

        return messages

    def build_vl_context_for_llm(
        self,
        image_path: str,
        user_query: str,
        vision_summary: Optional[VisionSummaryContent] = None,
        system_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """Build LLM message list with native image input for VL models.

        Passes the image directly to the VL model instead of converting
        to text descriptions. Falls back to text if vision_summary provided.

        Args:
            image_path: Path to the screenshot/image file
            user_query: User's question about the image
            vision_summary: Optional pre-computed vision summary for text fallback
            system_prompt: Optional system prompt

        Returns:
            List of message dicts with image_url content blocks
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build multimodal content with image
        content = [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": user_query},
        ]

        messages.append({"role": "user", "content": content})
        return messages

    def build_vl_autonomous_context(
        self,
        image_path: str,
        task_objective: str,
        previous_actions: Optional[List[str]] = None,
        retry_count: int = 0,
    ) -> List[Dict[str, Any]]:
        """Build autonomous context with native image for VL models.

        Passes the screenshot directly to the VL model for visual reasoning.

        Args:
            image_path: Path to the screenshot
            task_objective: What the agent is trying to accomplish
            previous_actions: Actions already attempted
            retry_count: Number of retries

        Returns:
            List of message dicts with image_url content blocks
        """
        messages = []

        # System prompt
        system_parts = [
            "You are an autonomous agent controlling a Windows PC.",
            "You can see the screen directly through screenshots.",
            "Analyze the screenshot and decide the next action.",
            "",
            "Available actions:",
            '- Click at coordinates: {"action": "click", "x": <int>, "y": <int>}',
            '- Type text: {"action": "type", "text": "<string>"}',
            '- Scroll: {"action": "scroll", "direction": "up|down", "amount": <int>}',
            '- Press key: {"action": "key", "key": "<string>"}',
            '- Wait: {"action": "wait", "seconds": <float>}',
            '- Done: {"action": "done", "result": "<description>"}',
            '- Failed: {"action": "failed", "reason": "<description>"}',
        ]
        if retry_count > 0:
            system_parts.append(f"\nRetry attempt {retry_count}. Previous actions may have failed.")
        messages.append({"role": "system", "content": "\n".join(system_parts)})

        # Previous actions context
        if previous_actions:
            actions_text = "\n".join(f"  - {a}" for a in previous_actions[-5:])
            messages.append({
                "role": "system",
                "content": f"Previous actions taken:\n{actions_text}",
            })

        # Build multimodal content with screenshot
        content = [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": f"Task: {task_objective}\n\nAnalyze the screenshot and decide the next action."},
        ]
        messages.append({"role": "user", "content": content})

        return messages


# ============================================================
# Content Factory
# ============================================================

def create_content_part(data: Dict[str, Any]) -> ContentPartType:
    """Factory function to create ContentPart from dict."""
    ct = data.get("content_type", "text")
    if ct == "text":
        return TextContent.from_dict(data)
    elif ct == "image":
        return ImageContent.from_dict(data)
    elif ct == "ocr":
        return OCRContent.from_dict(data)
    elif ct == "grounding":
        return GroundingContent.from_dict(data)
    elif ct == "vision_summary":
        return VisionSummaryContent.from_dict(data)
    else:
        return TextContent.from_dict(data)


def serialize_content_parts(parts: List[ContentPartType]) -> str:
    """Serialize content parts to JSON string."""
    return json.dumps([p.to_dict() for p in parts])


def deserialize_content_parts(data: str) -> List[ContentPartType]:
    """Deserialize content parts from JSON string."""
    raw = json.loads(data)
    return [create_content_part(d) for d in raw]
