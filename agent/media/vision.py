"""Vision provider for image understanding."""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from .base import MediaProvider, MediaType, MediaRequest, MediaResult, MediaOutput

logger = logging.getLogger(__name__)


class VisionProvider(MediaProvider):
    @property
    def name(self) -> str:
        return "vision"

    @property
    def media_type(self) -> MediaType:
        return MediaType.IMAGE

    @property
    def description(self) -> str:
        return "Image understanding and analysis"

    def validate_request(self, request: MediaRequest) -> Tuple[bool, List[str]]:
        errors = []
        if not request.input_path:
            errors.append("input_path is required for vision analysis")
        elif not os.path.exists(request.input_path):
            errors.append(f"Image file not found: {request.input_path}")
        elif not os.path.isfile(request.input_path):
            errors.append(f"Path is not a file: {request.input_path}")
        else:
            ext = Path(request.input_path).suffix.lower()
            allowed = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]
            if ext not in allowed:
                errors.append(f"Unsupported image format: {ext}")

            size = os.path.getsize(request.input_path)
            max_size = 20 * 1024 * 1024
            if size > max_size:
                errors.append(f"Image too large: {size} bytes (max {max_size})")

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
            result_data = self._analyze_image(request)
            execution_time = time.time() - start

            return MediaResult(
                success=True,
                media_type=MediaType.IMAGE,
                output=MediaOutput(
                    media_type=MediaType.IMAGE,
                    path=request.input_path,
                    metadata=result_data,
                ),
                execution_time=execution_time,
                provider=self.name,
                metadata=result_data,
            )
        except Exception as e:
            return MediaResult(
                success=False,
                media_type=MediaType.IMAGE,
                error=f"Vision analysis failed: {e}",
                execution_time=time.time() - start,
                provider=self.name,
            )

    def _analyze_image(self, request: MediaRequest) -> dict:
        file_path = Path(request.input_path)
        stat = file_path.stat()

        result = {
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "file_extension": file_path.suffix.lower(),
            "description": f"Image analysis of {file_path.name}",
            "objects_detected": [],
            "analysis_prompt": request.prompt or "General image analysis",
            "status": "stub_analysis_complete",
            "note": "This is a stub provider. Replace with local vision model for real analysis.",
        }

        return result


class StubLocalVisionProvider(VisionProvider):
    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._initialized = False

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
