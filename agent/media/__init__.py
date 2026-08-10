"""Multimodal media system for the local AI agent."""

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
from .vision import VisionProvider, StubLocalVisionProvider
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
    "ImageGenProvider",
    "StubLocalImageGenProvider",
    "VideoGenProvider",
    "StubLocalVideoGenProvider",
    "MEDIA_TYPE_EXTENSIONS",
    "MEDIA_MIME_TYPES",
]
