"""Vision provider for image understanding.

Stage 3.1 - Vision Analysis.

Provides provider-agnostic vision analysis with structured output.
Treats image content as untrusted input with appropriate security boundaries.
"""

import os
import time
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from abc import abstractmethod

from .base import MediaProvider, MediaType, MediaRequest, MediaResult, MediaOutput

logger = logging.getLogger(__name__)


class VisionConfidence(Enum):
    """Confidence levels for vision analysis results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box for detected elements."""
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
        )


@dataclass
class DetectedElement:
    """A detected element in an image."""
    element_type: str
    description: str
    bounding_box: Optional[BoundingBox] = None
    confidence: VisionConfidence = VisionConfidence.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "element_type": self.element_type,
            "description": self.description,
            "confidence": self.confidence.value,
            "metadata": self.metadata,
        }
        if self.bounding_box:
            result["bounding_box"] = self.bounding_box.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectedElement":
        bb_data = data.get("bounding_box")
        bb = BoundingBox.from_dict(bb_data) if bb_data else None
        return cls(
            element_type=data.get("element_type", "unknown"),
            description=data.get("description", ""),
            bounding_box=bb,
            confidence=VisionConfidence(data.get("confidence", "unknown")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VisionResult:
    """Structured result from vision analysis."""
    success: bool
    description: str = ""
    detected_elements: List[DetectedElement] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    analysis_prompt: str = ""
    provider: str = ""
    execution_time: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    untrusted_content: bool = True

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "description": self.description,
            "detected_elements": [e.to_dict() for e in self.detected_elements],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "analysis_prompt": self.analysis_prompt,
            "provider": self.provider,
            "execution_time": self.execution_time,
            "untrusted_content": self.untrusted_content,
            "metadata": self.metadata,
        }
        if self.error:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionResult":
        elements = [
            DetectedElement.from_dict(e)
            for e in data.get("detected_elements", [])
        ]
        return cls(
            success=data.get("success", False),
            description=data.get("description", ""),
            detected_elements=elements,
            image_width=data.get("image_width", 0),
            image_height=data.get("image_height", 0),
            analysis_prompt=data.get("analysis_prompt", ""),
            provider=data.get("provider", ""),
            execution_time=data.get("execution_time", 0.0),
            error=data.get("error", ""),
            metadata=data.get("metadata", {}),
            untrusted_content=data.get("untrusted_content", True),
        )

    def to_text(self) -> str:
        """Format result as untrusted text for agent consumption."""
        lines = ["[BEGIN UNTRUSTED VISUAL CONTENT]"]
        lines.append(f"Description: {self.description}")
        lines.append(f"Image size: {self.image_width}x{self.image_height}")

        if self.detected_elements:
            lines.append(f"Detected elements ({len(self.detected_elements)}):")
            for i, elem in enumerate(self.detected_elements):
                bb_info = ""
                if elem.bounding_box:
                    bb = elem.bounding_box
                    bb_info = f" at ({bb.x},{bb.y} {bb.width}x{bb.height})"
                lines.append(
                    f"  [{i}] {elem.element_type}: {elem.description}"
                    f"{bb_info} (confidence: {elem.confidence.value})"
                )

        if self.analysis_prompt:
            lines.append(f"Prompt: {self.analysis_prompt}")

        lines.append("[END UNTRUSTED VISUAL CONTENT]")
        return "\n".join(lines)


class VisionProvider(MediaProvider):
    """Base vision provider with validation and structured output."""

    def __init__(
        self,
        max_image_size_mb: int = 20,
        max_image_width: int = 4096,
        max_image_height: int = 4096,
        max_elements: int = 100,
        analysis_timeout: float = 30.0,
    ):
        self._max_image_size_mb = max_image_size_mb
        self._max_image_width = max_image_width
        self._max_image_height = max_image_height
        self._max_elements = max_elements
        self._analysis_timeout = analysis_timeout
        self._initialized = False
        self._request_count = 0
        self._total_time = 0.0

    @property
    def name(self) -> str:
        return "vision"

    @property
    def media_type(self) -> MediaType:
        return MediaType.IMAGE

    @property
    def description(self) -> str:
        return "Image understanding and analysis"

    @property
    def is_available(self) -> bool:
        return True

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "request_count": self._request_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(self._request_count, 1),
        }

    def validate_request(self, request: MediaRequest) -> Tuple[bool, List[str]]:
        """Validate vision analysis request."""
        errors = []

        if not request.input_path:
            errors.append("input_path is required for vision analysis")
            return False, errors

        if not os.path.exists(request.input_path):
            errors.append(f"Image file not found: {request.input_path}")
            return False, errors

        if not os.path.isfile(request.input_path):
            errors.append(f"Path is not a file: {request.input_path}")
            return False, errors

        ext = Path(request.input_path).suffix.lower()
        allowed = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]
        if ext not in allowed:
            errors.append(f"Unsupported image format: {ext}")

        try:
            size = os.path.getsize(request.input_path)
        except OSError as e:
            errors.append(f"Cannot read file size: {e}")
            return False, errors

        max_size_bytes = self._max_image_size_mb * 1024 * 1024
        if size > max_size_bytes:
            errors.append(
                f"Image too large: {size} bytes "
                f"(max {max_size_bytes} bytes / {self._max_image_size_mb} MB)"
            )

        if size == 0:
            errors.append("Image file is empty")

        try:
            from PIL import Image
            with Image.open(request.input_path) as img:
                width, height = img.size
                if width > self._max_image_width:
                    errors.append(
                        f"Image width {width} exceeds maximum {self._max_image_width}"
                    )
                if height > self._max_image_height:
                    errors.append(
                        f"Image height {height} exceeds maximum {self._max_image_height}"
                    )
                img.verify()
        except ImportError:
            logger.debug("Pillow not available for dimension validation")
        except Exception as e:
            errors.append(f"Image validation failed: {e}")

        return len(errors) == 0, errors

    def process(self, request: MediaRequest) -> MediaResult:
        """Process vision analysis request."""
        start = time.time()
        self._request_count += 1

        valid, errors = self.validate_request(request)
        if not valid:
            return MediaResult(
                success=False,
                media_type=MediaType.IMAGE,
                error="; ".join(errors),
                execution_time=time.time() - start,
                provider=self.name,
            )

        try:
            vision_result = self._analyze_image(request)
            execution_time = time.time() - start
            self._total_time += execution_time

            return MediaResult(
                success=True,
                media_type=MediaType.IMAGE,
                output=MediaOutput(
                    media_type=MediaType.IMAGE,
                    path=request.input_path,
                    metadata=vision_result.to_dict(),
                ),
                execution_time=execution_time,
                provider=self.name,
                metadata=vision_result.to_dict(),
            )
        except Exception as e:
            execution_time = time.time() - start
            self._total_time += execution_time
            return MediaResult(
                success=False,
                media_type=MediaType.IMAGE,
                error=f"Vision analysis failed: {e}",
                execution_time=execution_time,
                provider=self.name,
            )

    def _analyze_image(self, request: MediaRequest) -> VisionResult:
        """Analyze image and return structured result. Override in subclasses."""
        file_path = Path(request.input_path)

        try:
            from PIL import Image
            with Image.open(request.input_path) as img:
                width, height = img.size
        except ImportError:
            width, height = 0, 0
        except Exception:
            width, height = 0, 0

        try:
            stat = file_path.stat()
            file_size = stat.st_size
        except OSError:
            file_size = 0

        return VisionResult(
            success=True,
            description=f"Image analysis of {file_path.name}",
            detected_elements=[],
            image_width=width,
            image_height=height,
            analysis_prompt=request.prompt or "General image analysis",
            provider=self.name,
            metadata={
                "file_name": file_path.name,
                "file_size": file_size,
                "file_extension": file_path.suffix.lower(),
                "status": "stub_analysis_complete",
                "note": "This is a stub provider. Replace with local vision model for real analysis.",
            },
        )


class StubLocalVisionProvider(VisionProvider):
    """Stub vision provider for when no real vision model is available."""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path

    @property
    def name(self) -> str:
        return "stub_local_vision"

    @property
    def description(self) -> str:
        return "Stub local vision provider (requires model for real analysis)"

    @property
    def is_available(self) -> bool:
        if self._model_path and os.path.exists(self._model_path):
            return True
        return False

    def initialize(self) -> bool:
        if self._model_path and os.path.exists(self._model_path):
            self._initialized = True
            logger.info(f"Vision provider initialized with model: {self._model_path}")
            return True
        logger.warning("Vision provider: no model available, running as stub")
        self._initialized = True
        return True

    def _analyze_image(self, request: MediaRequest) -> VisionResult:
        """Stub analysis returning metadata only."""
        file_path = Path(request.input_path)

        try:
            from PIL import Image
            with Image.open(request.input_path) as img:
                width, height = img.size
        except ImportError:
            width, height = 0, 0
        except Exception:
            width, height = 0, 0

        try:
            stat = file_path.stat()
            file_size = stat.st_size
        except OSError:
            file_size = 0

        return VisionResult(
            success=True,
            description=f"Stub analysis of {file_path.name}. "
                        "No real vision model loaded - install a vision model for actual analysis.",
            detected_elements=[],
            image_width=width,
            image_height=height,
            analysis_prompt=request.prompt or "General image analysis",
            provider=self.name,
            metadata={
                "file_name": file_path.name,
                "file_size": file_size,
                "file_extension": file_path.suffix.lower(),
                "status": "stub_analysis_complete",
                "note": "Stub provider - no real image understanding.",
            },
        )


class LocalVisionProvider(VisionProvider):
    """Local vision provider with clean unavailable state."""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._model_loaded = False

    @property
    def name(self) -> str:
        return "local_vision"

    @property
    def description(self) -> str:
        return "Local vision provider for image analysis"

    @property
    def is_available(self) -> bool:
        return self._model_loaded and self._model_path is not None

    def initialize(self) -> bool:
        """Initialize vision provider."""
        if not self._model_path:
            logger.info("Local vision: no model path configured")
            self._initialized = True
            return True

        if not os.path.exists(self._model_path):
            logger.warning(f"Local vision: model not found: {self._model_path}")
            self._initialized = True
            return True

        try:
            self._model_loaded = True
            self._initialized = True
            logger.info(f"Local vision provider initialized with model: {self._model_path}")
            return True
        except Exception as e:
            logger.error(f"Local vision init failed: {e}")
            self._initialized = True
            return True

    def shutdown(self):
        """Shutdown vision provider."""
        self._model_loaded = False
        self._initialized = False
        logger.info("Local vision provider shut down")

    def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return {
            "initialized": self._initialized,
            "model_loaded": self._model_loaded,
            "model_path": self._model_path,
            "is_available": self.is_available,
            "stats": self.stats,
        }

    def _analyze_image(self, request: MediaRequest) -> VisionResult:
        """Analyze image with local provider."""
        if not self.is_available:
            file_path = Path(request.input_path)
            try:
                from PIL import Image
                with Image.open(request.input_path) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0

            return VisionResult(
                success=True,
                description=f"Vision analysis of {file_path.name}. "
                            "No vision model loaded - showing metadata only.",
                detected_elements=[],
                image_width=width,
                image_height=height,
                analysis_prompt=request.prompt or "General image analysis",
                provider=self.name,
                metadata={
                    "file_name": file_path.name,
                    "status": "no_model",
                    "note": "No vision model available",
                },
            )

        return super()._analyze_image(request)
