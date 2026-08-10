"""Video generation provider abstraction."""

import os
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from .base import MediaProvider, MediaType, MediaRequest, MediaResult, MediaOutput

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_WIDTH = 256
DEFAULT_VIDEO_HEIGHT = 256
DEFAULT_DURATION = 5.0
MAX_DURATION = 30.0
MAX_FRAMES = 150
MAX_DIMENSION = 1024


class VideoGenProvider(MediaProvider):
    @property
    def name(self) -> str:
        return "video_generate"

    @property
    def media_type(self) -> MediaType:
        return MediaType.VIDEO

    @property
    def description(self) -> str:
        return "Video generation from text prompts"

    def validate_request(self, request: MediaRequest) -> Tuple[bool, List[str]]:
        errors = []

        if not request.prompt or not request.prompt.strip():
            errors.append("prompt is required for video generation")

        if request.width < 0:
            errors.append(f"Invalid width: {request.width}")
        elif request.width > MAX_DIMENSION:
            errors.append(f"Width exceeds maximum: {request.width} > {MAX_DIMENSION}")

        if request.height < 0:
            errors.append(f"Invalid height: {request.height}")
        elif request.height > MAX_DIMENSION:
            errors.append(f"Height exceeds maximum: {request.height} > {MAX_DIMENSION}")

        if request.duration <= 0:
            errors.append(f"Duration must be positive: {request.duration}")
        elif request.duration > MAX_DURATION:
            errors.append(f"Duration exceeds maximum: {request.duration}s > {MAX_DURATION}s")

        if request.num_frames < 1:
            errors.append(f"num_frames must be at least 1: {request.num_frames}")
        elif request.num_frames > MAX_FRAMES:
            errors.append(f"num_frames exceeds maximum: {request.num_frames} > {MAX_FRAMES}")

        return len(errors) == 0, errors

    def process(self, request: MediaRequest) -> MediaResult:
        start = time.time()

        valid, errors = self.validate_request(request)
        if not valid:
            return MediaResult(
                success=False,
                media_type=MediaType.VIDEO,
                error="; ".join(errors),
                execution_time=time.time() - start,
                provider=self.name,
            )

        try:
            result = self._generate_video(request)
            execution_time = time.time() - start

            return MediaResult(
                success=True,
                media_type=MediaType.VIDEO,
                output=result,
                execution_time=execution_time,
                provider=self.name,
                metadata={
                    "prompt": request.prompt,
                    "width": request.width or DEFAULT_VIDEO_WIDTH,
                    "height": request.height or DEFAULT_VIDEO_HEIGHT,
                    "duration": request.duration,
                    "num_frames": request.num_frames,
                },
            )
        except Exception as e:
            return MediaResult(
                success=False,
                media_type=MediaType.VIDEO,
                error=f"Video generation failed: {e}",
                execution_time=time.time() - start,
                provider=self.name,
            )

    def _generate_video(self, request: MediaRequest) -> MediaOutput:
        output_path = request.output_path
        if not output_path:
            filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
            output_path = str(Path("workspace") / "media" / "videos" / filename)

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        self._create_stub_video(output_path, request)

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        return MediaOutput(
            media_type=MediaType.VIDEO,
            path=output_path,
            format=Path(output_path).suffix.lstrip(".") or "mp4",
            width=request.width or DEFAULT_VIDEO_WIDTH,
            height=request.height or DEFAULT_VIDEO_HEIGHT,
            duration=request.duration or DEFAULT_DURATION,
            file_size=file_size,
            metadata={
                "prompt": request.prompt,
                "num_frames": request.num_frames,
                "generation_method": "stub",
            },
        )

    def _create_stub_video(self, output_path: str, request: MediaRequest) -> None:
        ftyp_box = b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41'

        mdat_content = b'\x00' * 1024
        mdat_box = b'\x00\x00\x00\x08' + b'mdat' + mdat_content

        moov_content = b'\x00' * 512
        moov_box = b'\x00\x00\x00\x08' + b'moov' + moov_content

        with open(output_path, 'wb') as f:
            f.write(ftyp_box)
            f.write(mdat_box)
            f.write(moov_box)


class StubLocalVideoGenProvider(VideoGenProvider):
    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._initialized = False

    @property
    def name(self) -> str:
        return "stub_local_video_gen"

    @property
    def description(self) -> str:
        return "Stub local video generation (requires model for real generation)"

    @property
    def is_available(self) -> bool:
        return os.path.exists(self._model_path) if self._model_path else False

    def initialize(self) -> bool:
        if self._model_path and os.path.exists(self._model_path):
            self._initialized = True
            logger.info(f"Video gen provider initialized with model: {self._model_path}")
            return True
        logger.warning("Video gen provider: no model, running as stub")
        self._initialized = True
        return True
