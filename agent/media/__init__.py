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
from .observe_act_verify import ObserveActVerifyLoop, LoopConfig, LoopResult, LoopState, LoopExitReason
from .oav_tool import ObserveActVerifyTool
from .image_gen import ImageGenProvider, StubLocalImageGenProvider
from .video_gen import VideoGenProvider, StubLocalVideoGenProvider
from .real_vision import (
    RealVisionProvider,
    ImagePreprocessor,
    ImageMetadata,
    ColorInfo,
    ImageAnalysis,
    ImageFormat,
)
from .ocr import (
    OCRProvider,
    LocalOCRProvider,
    StubOCRProvider,
    OCRResult,
    OCRBlock,
    OCRStatus,
)
from .multimodal import (
    MultimodalMessage,
    TextContent,
    ImageContent,
    OCRContent,
    GroundingContent,
    VisionSummaryContent,
    VisionContextBuilder,
    ContentType,
    ContentPart,
    create_content_part,
    serialize_content_parts,
    deserialize_content_parts,
)
from .screen_understanding import (
    ScreenUnderstandingProvider,
    ScreenUnderstanding,
    ScreenQuery,
)
from .vision_pipeline import (
    VisionPipeline,
    VisionPipelineResult,
    VisionMode,
)
from .vision_decision import (
    VisionDecisionSystem,
    VisionDecision,
    VisionSource,
    VisionRequirement,
)
from .multimodal_pipeline import (
    MultimodalRequestPipeline,
    MultimodalRequest,
    RequestType,
)

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
    "ObserveActVerifyLoop",
    "LoopConfig",
    "LoopResult",
    "LoopState",
    "LoopExitReason",
    "ObserveActVerifyTool",
    "ImageGenProvider",
    "StubLocalImageGenProvider",
    "VideoGenProvider",
    "StubLocalVideoGenProvider",
    "RealVisionProvider",
    "ImagePreprocessor",
    "ImageMetadata",
    "ColorInfo",
    "ImageAnalysis",
    "ImageFormat",
    "OCRProvider",
    "LocalOCRProvider",
    "StubOCRProvider",
    "OCRResult",
    "OCRBlock",
    "OCRStatus",
    "MultimodalMessage",
    "TextContent",
    "ImageContent",
    "OCRContent",
    "GroundingContent",
    "VisionSummaryContent",
    "VisionContextBuilder",
    "ContentType",
    "ContentPart",
    "create_content_part",
    "serialize_content_parts",
    "deserialize_content_parts",
    "ScreenUnderstandingProvider",
    "ScreenUnderstanding",
    "ScreenQuery",
    "VisionPipeline",
    "VisionPipelineResult",
    "VisionMode",
    "VisionDecisionSystem",
    "VisionDecision",
    "VisionSource",
    "VisionRequirement",
    "MultimodalRequestPipeline",
    "MultimodalRequest",
    "RequestType",
    "MEDIA_TYPE_EXTENSIONS",
    "MEDIA_MIME_TYPES",
]
