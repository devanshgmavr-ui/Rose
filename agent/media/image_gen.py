"""Image generation provider abstraction."""

import os
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from .base import MediaProvider, MediaType, MediaRequest, MediaResult, MediaOutput

logger = logging.getLogger(__name__)

SUPPORTED_DIMENSIONS = [
    (256, 256), (512, 512), (768, 768), (1024, 1024),
    (512, 768), (768, 512), (768, 1024), (1024, 768),
]
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
MAX_DIMENSION = 2048
MAX_SEED = 2**32 - 1


class ImageGenProvider(MediaProvider):
    @property
    def name(self) -> str:
        return "image_generate"

    @property
    def media_type(self) -> MediaType:
        return MediaType.IMAGE

    @property
    def description(self) -> str:
        return "Image generation from text prompts"

    def validate_request(self, request: MediaRequest) -> Tuple[bool, List[str]]:
        errors = []

        if not request.prompt or not request.prompt.strip():
            errors.append("prompt is required for image generation")

        if request.width < 0:
            errors.append(f"Invalid width: {request.width}")
        elif request.width > MAX_DIMENSION:
            errors.append(f"Width exceeds maximum: {request.width} > {MAX_DIMENSION}")

        if request.height < 0:
            errors.append(f"Invalid height: {request.height}")
        elif request.height > MAX_DIMENSION:
            errors.append(f"Height exceeds maximum: {request.height} > {MAX_DIMENSION}")

        if request.seed is not None and (request.seed < 0 or request.seed > MAX_SEED):
            errors.append(f"Seed must be between 0 and {MAX_SEED}")

        return len(errors) == 0, errors

    def process(self, request: MediaRequest) -> MediaResult:
        start = time.time()

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
            result = self._generate_image(request)
            execution_time = time.time() - start

            return MediaResult(
                success=True,
                media_type=MediaType.IMAGE,
                output=result,
                execution_time=execution_time,
                provider=self.name,
                metadata={
                    "prompt": request.prompt,
                    "width": request.width or DEFAULT_WIDTH,
                    "height": request.height or DEFAULT_HEIGHT,
                    "seed": request.seed,
                },
            )
        except Exception as e:
            return MediaResult(
                success=False,
                media_type=MediaType.IMAGE,
                error=f"Image generation failed: {e}",
                execution_time=time.time() - start,
                provider=self.name,
            )

    def _generate_image(self, request: MediaRequest) -> MediaOutput:
        output_path = request.output_path
        if not output_path:
            filename = f"generated_{uuid.uuid4().hex[:8]}.png"
            output_path = str(Path("workspace") / "media" / "images" / filename)

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        self._create_stub_image(output_path, request)

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        return MediaOutput(
            media_type=MediaType.IMAGE,
            path=output_path,
            format=Path(output_path).suffix.lstrip(".") or "png",
            width=request.width or DEFAULT_WIDTH,
            height=request.height or DEFAULT_HEIGHT,
            file_size=file_size,
            metadata={
                "prompt": request.prompt,
                "seed": request.seed,
                "generation_method": "stub",
            },
        )

    def _create_stub_image(self, output_path: str, request: MediaRequest) -> None:
        width = request.width or DEFAULT_WIDTH
        height = request.height or DEFAULT_HEIGHT

        png_header = b'\x89PNG\r\n\x1a\n'

        ihdr_data = (
            width.to_bytes(4, 'big') +
            height.to_bytes(4, 'big') +
            b'\x08\x02\x00\x00\x00'
        )
        ihdr_crc = b'\x00' * 4

        raw_data = b''
        for y in range(min(height, 8)):
            raw_data += b'\x00' + bytes([128, 128, 128] * (width // 3 + 1))[:width * 3]

        compressed = b'\x00' + raw_data

        idat_crc = b'\x00' * 4

        with open(output_path, 'wb') as f:
            f.write(png_header)

            ihdr_chunk = b'IHDR' + ihdr_data
            f.write(len(ihdr_data).to_bytes(4, 'big'))
            f.write(ihdr_chunk)
            f.write(ihdr_crc)

            idat_chunk = b'IDAT' + compressed
            f.write(len(compressed).to_bytes(4, 'big'))
            f.write(idat_chunk)
            f.write(idat_crc)

            iend_chunk = b'IEND'
            f.write(b'\x00\x00\x00\x00')
            f.write(iend_chunk)
            f.write(b'\x00' * 4)


class StubLocalImageGenProvider(ImageGenProvider):
    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._initialized = False

    @property
    def name(self) -> str:
        return "stub_local_image_gen"

    @property
    def description(self) -> str:
        return "Stub local image generation (requires model for real generation)"

    @property
    def is_available(self) -> bool:
        return os.path.exists(self._model_path) if self._model_path else False

    def initialize(self) -> bool:
        if self._model_path and os.path.exists(self._model_path):
            self._initialized = True
            logger.info(f"Image gen provider initialized with model: {self._model_path}")
            return True
        logger.warning("Image gen provider: no model, running as stub")
        self._initialized = True
        return True
