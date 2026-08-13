"""Visual grounding for translating vision into actionable coordinates.

Stage 3.2 - Visual Grounding.

Translates vision analysis results into structured targets
with coordinates, bounding boxes, and confidence scores.
Does NOT directly perform mouse actions - targets go through
the Planner -> ToolRouter -> MouseTool pipeline.
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from .vision import VisionResult, DetectedElement, BoundingBox, VisionConfidence

logger = logging.getLogger(__name__)


class TargetType(Enum):
    """Types of visual targets."""
    BUTTON = "button"
    LINK = "link"
    TEXT_FIELD = "text_field"
    ICON = "icon"
    MENU = "menu"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    TAB = "tab"
    IMAGE = "image"
    TEXT = "text"
    WINDOW = "window"
    ELEMENT = "element"
    UNKNOWN = "unknown"


class GroundingConfidence(Enum):
    """Confidence levels for grounding results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    UNFOUND = "unfound"


@dataclass
class Point:
    """A 2D point."""
    x: int
    y: int

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Point":
        return cls(x=data.get("x", 0), y=data.get("y", 0))


@dataclass
class GroundedTarget:
    """A grounded target with coordinates and metadata."""
    description: str
    target_type: TargetType
    center: Point
    bounding_box: Optional[BoundingBox] = None
    confidence: GroundingConfidence = GroundingConfidence.UNKNOWN
    source_image: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "description": self.description,
            "target_type": self.target_type.value,
            "center": self.center.to_dict(),
            "confidence": self.confidence.value,
            "source_image": self.source_image,
            "stale": self.stale,
            "metadata": self.metadata,
        }
        if self.bounding_box:
            result["bounding_box"] = self.bounding_box.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundedTarget":
        bb_data = data.get("bounding_box")
        bb = BoundingBox.from_dict(bb_data) if bb_data else None
        return cls(
            description=data.get("description", ""),
            target_type=TargetType(data.get("target_type", "unknown")),
            center=Point.from_dict(data.get("center", {})),
            bounding_box=bb,
            confidence=GroundingConfidence(data.get("confidence", "unknown")),
            source_image=data.get("source_image", ""),
            stale=data.get("stale", False),
            metadata=data.get("metadata", {}),
        )

    def to_text(self) -> str:
        """Format target as untrusted text."""
        bb_info = ""
        if self.bounding_box:
            bb = self.bounding_box
            bb_info = f" (bounds: {bb.x},{bb.y} {bb.width}x{bb.height})"
        return (
            f"[BEGIN UNTRUSTED GROUNDING DATA]\n"
            f"Target: {self.description}\n"
            f"Type: {self.target_type.value}\n"
            f"Center: ({self.center.x}, {self.center.y}){bb_info}\n"
            f"Confidence: {self.confidence.value}\n"
            f"Source: {self.source_image}\n"
            f"[END UNTRUSTED GROUNDING DATA]"
        )


@dataclass
class GroundingResult:
    """Result of visual grounding."""
    success: bool
    targets: List[GroundedTarget] = field(default_factory=list)
    screen_width: int = 0
    screen_height: int = 0
    source_image: str = ""
    execution_time: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "targets": [t.to_dict() for t in self.targets],
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "source_image": self.source_image,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }
        if self.error:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundingResult":
        targets = [GroundedTarget.from_dict(t) for t in data.get("targets", [])]
        return cls(
            success=data.get("success", False),
            targets=targets,
            screen_width=data.get("screen_width", 0),
            screen_height=data.get("screen_height", 0),
            source_image=data.get("source_image", ""),
            execution_time=data.get("execution_time", 0.0),
            error=data.get("error", ""),
            metadata=data.get("metadata", {}),
        )

    def to_text(self) -> str:
        """Format result as untrusted text."""
        lines = ["[BEGIN UNTRUSTED GROUNDING RESULTS]"]
        lines.append(f"Found {len(self.targets)} target(s)")
        lines.append(f"Screen: {self.screen_width}x{self.screen_height}")
        for i, target in enumerate(self.targets):
            lines.append(f"\n[{i}] {target.description}")
            lines.append(f"    Type: {target.target_type.value}")
            lines.append(
                f"    Center: ({target.center.x}, {target.center.y})"
            )
            if target.bounding_box:
                bb = target.bounding_box
                lines.append(
                    f"    Bounds: ({bb.x},{bb.y}) {bb.width}x{bb.height}"
                )
            lines.append(f"    Confidence: {target.confidence.value}")
        lines.append("[END UNTRUSTED GROUNDING RESULTS]")
        return "\n".join(lines)


class VisualGrounder:
    """Translates vision results into grounded targets."""

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        confidence_threshold: float = 0.3,
        stale_timeout: float = 30.0,
    ):
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._confidence_threshold = confidence_threshold
        self._stale_timeout = stale_timeout
        self._request_count = 0
        self._total_time = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "request_count": self._request_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(self._request_count, 1),
        }

    def ground(
        self,
        vision_result: VisionResult,
        target_description: Optional[str] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
    ) -> GroundingResult:
        """Ground vision results into actionable targets.

        Args:
            vision_result: Result from vision analysis.
            target_description: Optional specific target to find.
            screen_width: Override screen width.
            screen_height: Override screen height.

        Returns:
            GroundingResult with grounded targets.
        """
        start = time.time()
        self._request_count += 1

        sw = screen_width or self._screen_width
        sh = screen_height or self._screen_height

        if not vision_result.success:
            elapsed = time.time() - start
            self._total_time += elapsed
            return GroundingResult(
                success=False,
                error=vision_result.error or "Vision analysis failed",
                screen_width=sw,
                screen_height=sh,
                execution_time=elapsed,
            )

        targets = []
        for elem in vision_result.detected_elements:
            target = self._ground_element(
                elem, vision_result, sw, sh
            )
            if target:
                if target_description:
                    if self._matches_description(
                        target.description, target_description
                    ):
                        targets.append(target)
                else:
                    targets.append(target)

        if target_description and not targets:
            targets.append(GroundedTarget(
                description=target_description,
                target_type=TargetType.UNKNOWN,
                center=Point(x=sw // 2, y=sh // 2),
                confidence=GroundingConfidence.UNFOUND,
                source_image=vision_result.metadata.get("file_name", ""),
                stale=True,
                metadata={"note": "Target not found in vision results"},
            ))

        elapsed = time.time() - start
        self._total_time += elapsed

        return GroundingResult(
            success=True,
            targets=targets,
            screen_width=sw,
            screen_height=sh,
            source_image=vision_result.metadata.get("file_name", ""),
            execution_time=elapsed,
            metadata={
                "total_elements": len(vision_result.detected_elements),
                "matched_targets": len(targets),
            },
        )

    def _ground_element(
        self,
        element: DetectedElement,
        vision_result: VisionResult,
        screen_width: int,
        screen_height: int,
    ) -> Optional[GroundedTarget]:
        """Ground a single detected element."""
        target_type = self._classify_element(element)

        if element.bounding_box:
            bb = element.bounding_box
            center_x = bb.x + bb.width // 2
            center_y = bb.y + bb.height // 2
        else:
            center_x = screen_width // 2
            center_y = screen_height // 2

        center_x = max(0, min(center_x, screen_width - 1))
        center_y = max(0, min(center_y, screen_height - 1))

        confidence = self._map_confidence(element.confidence)

        return GroundedTarget(
            description=element.description,
            target_type=target_type,
            center=Point(x=center_x, y=center_y),
            bounding_box=element.bounding_box,
            confidence=confidence,
            source_image=vision_result.metadata.get("file_name", ""),
            metadata=element.metadata,
        )

    def _classify_element(self, element: DetectedElement) -> TargetType:
        """Classify element type from description."""
        desc_lower = element.description.lower()
        type_lower = element.element_type.lower()

        # OCR text elements - classify by content patterns
        if "ocr text" in desc_lower or type_lower == "text":
            if any(w in desc_lower for w in ["button", "submit", "ok", "cancel", "sign in", "log in"]):
                return TargetType.BUTTON
            if any(w in desc_lower for w in ["link", "click here"]):
                return TargetType.LINK
            if any(w in desc_lower for w in ["search", "find"]):
                return TargetType.TEXT_FIELD
            return TargetType.TEXT

        if any(w in desc_lower for w in ["button", "submit", "ok", "cancel"]):
            return TargetType.BUTTON
        if any(w in desc_lower for w in ["link", "href"]):
            return TargetType.LINK
        if any(w in desc_lower for w in ["text field", "input", "search"]):
            return TargetType.TEXT_FIELD
        if any(w in desc_lower for w in ["icon", "image"]):
            return TargetType.ICON
        if any(w in desc_lower for w in ["menu", "dropdown"]):
            return TargetType.MENU
        if any(w in desc_lower for w in ["checkbox"]):
            return TargetType.CHECKBOX
        if any(w in desc_lower for w in ["tab"]):
            return TargetType.TAB
        if "button" in type_lower:
            return TargetType.BUTTON
        if "link" in type_lower:
            return TargetType.LINK
        if "text" in type_lower:
            return TargetType.TEXT

        return TargetType.ELEMENT

    def _map_confidence(self, vision_confidence: VisionConfidence) -> GroundingConfidence:
        """Map vision confidence to grounding confidence."""
        mapping = {
            VisionConfidence.HIGH: GroundingConfidence.HIGH,
            VisionConfidence.MEDIUM: GroundingConfidence.MEDIUM,
            VisionConfidence.LOW: GroundingConfidence.LOW,
            VisionConfidence.UNKNOWN: GroundingConfidence.UNKNOWN,
        }
        return mapping.get(vision_confidence, GroundingConfidence.UNKNOWN)

    def _matches_description(
        self, target_desc: str, query: str
    ) -> bool:
        """Check if target matches query description."""
        target_lower = target_desc.lower()
        query_lower = query.lower()
        return query_lower in target_lower or target_lower in query_lower

    def validate_target(
        self, target: GroundedTarget
    ) -> Tuple[bool, List[str]]:
        """Validate a grounded target for safety.

        Args:
            target: The target to validate.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []

        if target.center.x < 0 or target.center.x >= self._screen_width:
            errors.append(
                f"X coordinate {target.center.x} outside screen bounds "
                f"(0-{self._screen_width - 1})"
            )

        if target.center.y < 0 or target.center.y >= self._screen_height:
            errors.append(
                f"Y coordinate {target.center.y} outside screen bounds "
                f"(0-{self._screen_height - 1})"
            )

        if target.confidence == GroundingConfidence.UNFOUND:
            errors.append("Target was not found in vision results")

        if target.confidence == GroundingConfidence.AMBIGUOUS:
            errors.append("Target is ambiguous - multiple matches possible")

        if target.stale:
            errors.append("Target data may be stale")

        if target.confidence.value in ["low", "unknown"]:
            errors.append(
                f"Low confidence ({target.confidence.value}) - "
                "action may be unsafe"
            )

        return len(errors) == 0, errors
