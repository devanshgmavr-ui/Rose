"""Base abstractions for the multimodal media system."""

import uuid
import time
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MediaFormat(Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    BMP = "bmp"
    GIF = "gif"
    MP4 = "mp4"
    WEBM = "webm"
    AVI = "avi"
    WAV = "wav"
    MP3 = "mp3"


MEDIA_TYPE_EXTENSIONS: Dict[MediaType, List[str]] = {
    MediaType.IMAGE: [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"],
    MediaType.VIDEO: [".mp4", ".webm", ".avi", ".mov"],
    MediaType.AUDIO: [".wav", ".mp3", ".ogg", ".flac"],
}

MEDIA_MIME_TYPES: Dict[str, MediaType] = {
    "image/png": MediaType.IMAGE,
    "image/jpeg": MediaType.IMAGE,
    "image/webp": MediaType.IMAGE,
    "image/bmp": MediaType.IMAGE,
    "image/gif": MediaType.IMAGE,
    "video/mp4": MediaType.VIDEO,
    "video/webm": MediaType.VIDEO,
    "video/avi": MediaType.VIDEO,
    "video/quicktime": MediaType.VIDEO,
    "audio/wav": MediaType.AUDIO,
    "audio/mpeg": MediaType.AUDIO,
}


@dataclass
class MediaInput:
    media_type: MediaType
    path: Optional[str] = None
    data: Optional[bytes] = None
    prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type.value,
            "path": self.path,
            "prompt": self.prompt,
            "has_data": self.data is not None,
            "data_size": len(self.data) if self.data else 0,
            "metadata": self.metadata,
        }


@dataclass
class MediaOutput:
    media_type: MediaType
    path: str
    format: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    file_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type.value,
            "path": self.path,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "file_size": self.file_size,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaOutput":
        return cls(
            media_type=MediaType(data["media_type"]),
            path=data["path"],
            format=data.get("format", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
            duration=data.get("duration", 0.0),
            file_size=data.get("file_size", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MediaRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    media_type: MediaType = MediaType.IMAGE
    prompt: str = ""
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    width: int = 0
    height: int = 0
    seed: Optional[int] = None
    num_frames: int = 1
    duration: float = 5.0
    format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "media_type": self.media_type.value,
            "prompt": self.prompt,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "num_frames": self.num_frames,
            "duration": self.duration,
            "format": self.format,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaRequest":
        return cls(
            request_id=data.get("request_id", str(uuid.uuid4())[:8]),
            media_type=MediaType(data.get("media_type", "image")),
            prompt=data.get("prompt", ""),
            input_path=data.get("input_path"),
            output_path=data.get("output_path"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            seed=data.get("seed"),
            num_frames=data.get("num_frames", 1),
            duration=data.get("duration", 5.0),
            format=data.get("format", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MediaResult:
    success: bool
    media_type: MediaType = MediaType.IMAGE
    output: Optional[MediaOutput] = None
    error: str = ""
    execution_time: float = 0.0
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "media_type": self.media_type.value,
            "error": self.error,
            "execution_time": self.execution_time,
            "provider": self.provider,
            "metadata": self.metadata,
        }
        if self.output:
            result["output"] = self.output.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaResult":
        output = None
        if "output" in data and data["output"]:
            output = MediaOutput.from_dict(data["output"])
        return cls(
            success=data["success"],
            media_type=MediaType(data.get("media_type", "image")),
            output=output,
            error=data.get("error", ""),
            execution_time=data.get("execution_time", 0.0),
            provider=data.get("provider", ""),
            metadata=data.get("metadata", {}),
        )


class MediaProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def media_type(self) -> MediaType:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def is_available(self) -> bool:
        return True

    @abstractmethod
    def validate_request(self, request: MediaRequest) -> Tuple[bool, List[str]]:
        pass

    @abstractmethod
    def process(self, request: MediaRequest) -> MediaResult:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "media_type": self.media_type.value,
            "description": self.description,
            "is_available": self.is_available,
        }
