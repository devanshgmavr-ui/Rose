"""Multimodal media system for the local AI agent.

Stage 3.1 - Vision Analysis.
"""

from .base import (
    MediaType,
    MediaFormat,
    MediaInput,
    MediaOutput,
    MediaRequest,
    MediaResult,
    MediaProvider,
    MEDIA_TYPE_EXTENSIONS,
    MEDIA_MIME_TYPES,
)
from .storage import MediaStorage
from .router import MediaRouter
from .vision import (
    VisionProvider,
    StubLocalVisionProvider,
    LocalVisionProvider,
    VisionResult,
    DetectedElement,
    BoundingBox,
    VisionConfidence,
)
from .analyzer import VisionAnalyzer
from .permissions import register_vision_permissions, VISION_PERMISSIONS
from .vision_tool import VisionAnalyzeTool
from .grounding import (
    VisualGrounder,
    GroundingResult,
    GroundedTarget,
    TargetType,
    GroundingConfidence,
    Point,
)
from .grounding_tool import VisualGroundingTool
from .image_gen import ImageGenProvider, StubLocalImageGenProvider
from .video_gen import VideoGenProvider, StubLocalVideoGenProvider

__all__ = [
    "MediaType",
    "MediaFormat",
    "MediaInput",
    "MediaOutput",
    "MediaRequest",
    "MediaResult",
    "MediaProvider",
    "MediaStorage",
    "MediaRouter",
    "VisionProvider",
    "StubLocalVisionProvider",
    "LocalVisionProvider",
    "VisionResult",
    "DetectedElement",
    "BoundingBox",
    "VisionConfidence",
    "VisionAnalyzer",
    "register_vision_permissions",
    "VISION_PERMISSIONS",
    "VisionAnalyzeTool",
    "VisualGrounder",
    "GroundingResult",
    "GroundedTarget",
    "TargetType",
    "GroundingConfidence",
    "Point",
    "VisualGroundingTool",
    "ImageGenProvider",
    "StubLocalImageGenProvider",
    "VideoGenProvider",
    "StubLocalVideoGenProvider",
    "MEDIA_TYPE_EXTENSIONS",
    "MEDIA_MIME_TYPES",
]
