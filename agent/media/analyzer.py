"""Vision analyzer for structured image understanding.

Stage 3.1 - Vision Analysis.

Wraps vision providers with additional analysis capabilities
and enforces security boundaries for untrusted visual content.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .vision import (
    VisionProvider,
    VisionResult,
    DetectedElement,
    BoundingBox,
    VisionConfidence,
)
from .storage import MediaStorage

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Structured vision analysis with security boundaries."""

    def __init__(
        self,
        vision_provider: VisionProvider,
        media_storage: Optional[MediaStorage] = None,
        max_elements: int = 100,
        analysis_timeout: float = 30.0,
    ):
        self._provider = vision_provider
        self._storage = media_storage
        self._max_elements = max_elements
        self._analysis_timeout = analysis_timeout
        self._request_count = 0
        self._total_time = 0.0

    @property
    def is_available(self) -> bool:
        """Check if analyzer is available."""
        return self._provider is not None and self._provider.is_available

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider.name if self._provider else "none"

    @property
    def stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "available": self.is_available,
            "provider": self.provider_name,
            "request_count": self._request_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(self._request_count, 1),
        }

    def analyze(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        workspace_root: Optional[str] = None,
    ) -> VisionResult:
        """Analyze an image with security validation.

        Args:
            image_path: Path to the image file.
            prompt: Optional analysis prompt.
            workspace_root: Optional workspace root for path validation.

        Returns:
            VisionResult with structured analysis.
        """
        start = time.time()
        self._request_count += 1

        if not self._provider:
            return VisionResult(
                success=False,
                error="No vision provider available",
                execution_time=time.time() - start,
            )

        if not self._provider.is_available:
            return VisionResult(
                success=False,
                error=f"Vision provider {self._provider.name} is not available",
                execution_time=time.time() - start,
            )

        if workspace_root:
            abs_path = Path(image_path).resolve()
            workspace = Path(workspace_root).resolve()
            try:
                abs_path.relative_to(workspace)
            except ValueError:
                return VisionResult(
                    success=False,
                    error="Image path is outside workspace boundary",
                    execution_time=time.time() - start,
                )

        from .base import MediaRequest, MediaType
        request = MediaRequest(
            media_type=MediaType.IMAGE,
            input_path=image_path,
            prompt=prompt or "General image analysis",
        )

        result = self._provider.process(request)
        elapsed = time.time() - start
        self._total_time += elapsed

        if not result.success:
            return VisionResult(
                success=False,
                error=result.error,
                execution_time=elapsed,
                provider=self._provider.name,
            )

        metadata = result.metadata
        vision_result = VisionResult.from_dict(metadata)
        vision_result.execution_time = elapsed
        vision_result.provider = self._provider.name

        if len(vision_result.detected_elements) > self._max_elements:
            vision_result.detected_elements = (
                vision_result.detected_elements[:self._max_elements]
            )
            vision_result.metadata["truncated_elements"] = True

        return vision_result

    def analyze_screenshot(
        self,
        screenshot_path: str,
        prompt: Optional[str] = None,
    ) -> VisionResult:
        """Analyze a screenshot with appropriate handling.

        Args:
            screenshot_path: Path to the screenshot file.
            prompt: Optional analysis prompt.

        Returns:
            VisionResult with analysis.
        """
        return self.analyze(
            image_path=screenshot_path,
            prompt=prompt or "Analyze this screenshot",
        )

    def describe_image(self, image_path: str) -> str:
        """Get a text description of an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Text description wrapped in untrusted content markers.
        """
        result = self.analyze(image_path)
        return result.to_text()
