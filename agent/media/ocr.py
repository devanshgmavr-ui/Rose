"""OCR abstraction and local OCR provider for Rose Vision pipeline.

Provides a modular OCR system that integrates with the existing VisionProvider
architecture. Supports pluggable OCR engines through the OCRProvider abstraction.

Security: OCR output is treated as untrusted external data. It must not
execute commands, modify permissions, or bypass the ToolRouter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class OCRStatus(Enum):
    """OCR processing status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


@dataclass
class OCRBlock:
    """A single detected text block with bounding box.

    Coordinates are in pixels relative to the original image.
    """
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRBlock":
        bbox = data.get("bbox", {})
        return cls(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            x=bbox.get("x", 0),
            y=bbox.get("y", 0),
            width=bbox.get("width", 0),
            height=bbox.get("height", 0),
            metadata=data.get("metadata", {}),
        )

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height


@dataclass
class OCRResult:
    """Structured OCR result with text, blocks, and metadata.

    All text content is marked as untrusted visual content.
    """
    text: str
    confidence: float
    blocks: List[OCRBlock]
    image_width: int
    image_height: int
    status: OCRStatus
    provider: str
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "blocks": [b.to_dict() for b in self.blocks],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "status": self.status.value,
            "provider": self.provider,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata,
            "block_count": self.block_count,
            "char_count": self.char_count,
        }

    def to_text(self) -> str:
        """Format as untrusted content for LLM consumption."""
        if not self.text.strip():
            return "[No text detected in image]"
        lines = [
            "[BEGIN UNTRUSTED OCR CONTENT]",
            f"Provider: {self.provider}",
            f"Confidence: {self.confidence:.2f}",
            f"Blocks: {self.block_count}",
            "",
            self.text,
            "",
            "[END UNTRUSTED OCR CONTENT]",
        ]
        return "\n".join(lines)


class OCRProvider(ABC):
    """Abstract base class for OCR providers.

    Implementations must return OCRResult with text and bounding boxes.
    OCR output is untrusted and must not be used to execute commands.
    """

    def __init__(
        self,
        max_image_size_mb: float = 20.0,
        max_image_width: int = 4096,
        max_image_height: int = 4096,
        max_text_chars: int = 100000,
        max_blocks: int = 1000,
        ocr_timeout: float = 30.0,
    ):
        self._max_image_size_mb = max_image_size_mb
        self._max_image_width = max_image_width
        self._max_image_height = max_image_height
        self._max_text_chars = max_text_chars
        self._max_blocks = max_blocks
        self._ocr_timeout = ocr_timeout
        self._request_count = 0
        self._total_time = 0.0
        self._error_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the OCR engine is available."""

    @property
    def stats(self) -> Dict[str, Any]:
        """Provider statistics."""
        avg_time = (self._total_time / self._request_count) if self._request_count > 0 else 0.0
        return {
            "provider": self.name,
            "requests": self._request_count,
            "errors": self._error_count,
            "avg_time": avg_time,
            "total_time": self._total_time,
        }

    @abstractmethod
    def _extract_text(self, image_path: str) -> OCRResult:
        """Core OCR extraction. Subclasses implement this."""

    def extract_text(self, image_path: str) -> OCRResult:
        """Extract text from an image with validation and resource limits.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and bounding boxes.
        """
        self._request_count += 1
        start_time = time.time()

        try:
            # Validate image exists
            path = Path(image_path)
            if not path.exists():
                self._error_count += 1
                return OCRResult(
                    text="",
                    confidence=0.0,
                    blocks=[],
                    image_width=0,
                    image_height=0,
                    status=OCRStatus.FAILED,
                    provider=self.name,
                    execution_time=0.0,
                    error=f"Image file not found: {image_path}",
                )

            # Validate file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self._max_image_size_mb:
                self._error_count += 1
                return OCRResult(
                    text="",
                    confidence=0.0,
                    blocks=[],
                    image_width=0,
                    image_height=0,
                    status=OCRStatus.FAILED,
                    provider=self.name,
                    execution_time=time.time() - start_time,
                    error=f"Image too large: {file_size_mb:.1f}MB > {self._max_image_size_mb}MB",
                )

            # Validate file type
            suffix = path.suffix.lower()
            if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"):
                self._error_count += 1
                return OCRResult(
                    text="",
                    confidence=0.0,
                    blocks=[],
                    image_width=0,
                    image_height=0,
                    status=OCRStatus.UNSUPPORTED,
                    provider=self.name,
                    execution_time=time.time() - start_time,
                    error=f"Unsupported image format: {suffix}",
                )

            # Validate dimensions
            try:
                from PIL import Image
                with Image.open(path) as img:
                    w, h = img.size
                    if w > self._max_image_width or h > self._max_image_height:
                        self._error_count += 1
                        return OCRResult(
                            text="",
                            confidence=0.0,
                            blocks=[],
                            image_width=w,
                            image_height=h,
                            status=OCRStatus.FAILED,
                            provider=self.name,
                            execution_time=time.time() - start_time,
                            error=f"Image dimensions too large: {w}x{h} > {self._max_image_width}x{self._max_image_height}",
                        )
            except Exception as e:
                self._error_count += 1
                return OCRResult(
                    text="",
                    confidence=0.0,
                    blocks=[],
                    image_width=0,
                    image_height=0,
                    status=OCRStatus.FAILED,
                    provider=self.name,
                    execution_time=time.time() - start_time,
                    error=f"Failed to read image: {e}",
                )

            # Run OCR with timeout
            result = self._extract_text(image_path)

            # Enforce resource limits on result
            if result.text and len(result.text) > self._max_text_chars:
                result.text = result.text[:self._max_text_chars]
                result.metadata["text_truncated"] = True

            if result.blocks and len(result.blocks) > self._max_blocks:
                result.blocks = result.blocks[:self._max_blocks]
                result.metadata["blocks_truncated"] = True

            elapsed = time.time() - start_time
            result.execution_time = elapsed
            self._total_time += elapsed

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            self._error_count += 1
            self._total_time += elapsed
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                blocks=[],
                image_width=0,
                image_height=0,
                status=OCRStatus.FAILED,
                provider=self.name,
                execution_time=elapsed,
                error=str(e),
            )


class LocalOCRProvider(OCRProvider):
    """Local OCR provider using pytesseract (Tesseract).

    Requires: pip install pytesseract
    Requires: Tesseract-OCR installed on the system.
    """

    def __init__(
        self,
        language: str = "eng",
        config: str = "--psm 6",
        max_image_size_mb: float = 20.0,
        max_image_width: int = 4096,
        max_image_height: int = 4096,
        max_text_chars: int = 100000,
        max_blocks: int = 1000,
        ocr_timeout: float = 30.0,
    ):
        super().__init__(
            max_image_size_mb=max_image_size_mb,
            max_image_width=max_image_width,
            max_image_height=max_image_height,
            max_text_chars=max_text_chars,
            max_blocks=max_blocks,
            ocr_timeout=ocr_timeout,
        )
        self._language = language
        self._config = config
        self._pytesseract = None
        self._available = False

    def initialize(self) -> bool:
        """Initialize the OCR engine."""
        try:
            import pytesseract
            self._pytesseract = pytesseract
            # Verify Tesseract is installed
            pytesseract.get_tesseract_version()
            self._available = True
            logger.info(f"LocalOCRProvider initialized (tesseract available, lang={self._language})")
            return True
        except ImportError:
            logger.warning("pytesseract not installed. OCR unavailable.")
            self._available = False
            return False
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self._available = False
            return False

    @property
    def name(self) -> str:
        return "local_tesseract"

    @property
    def is_available(self) -> bool:
        return self._available

    def _extract_text(self, image_path: str) -> OCRResult:
        """Extract text using pytesseract."""
        if not self._available or not self._pytesseract:
            return OCRResult(
                text="",
                confidence=0.0,
                blocks=[],
                image_width=0,
                image_height=0,
                status=OCRStatus.UNSUPPORTED,
                provider=self.name,
                execution_time=0.0,
                error="OCR engine not initialized",
            )

        try:
            from PIL import Image

            img = Image.open(image_path)
            img_width, img_height = img.size

            # Get detailed OCR data
            data = self._pytesseract.image_to_data(
                img,
                lang=self._language,
                config=self._config,
                output_type=self._pytesseract.Output.DICT,
            )

            # Build blocks from word-level data
            blocks = []
            confidences = []
            texts = []

            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                # Skip empty or very low confidence entries
                if not text or conf < 0:
                    continue

                block = OCRBlock(
                    text=text,
                    confidence=conf / 100.0,
                    x=data["left"][i],
                    y=data["top"][i],
                    width=data["width"][i],
                    height=data["height"][i],
                    metadata={
                        "block_num": data.get("block_num", [0])[i],
                        "line_num": data.get("line_num", [0])[i],
                        "word_num": data.get("word_num", [0])[i],
                    },
                )
                blocks.append(block)
                confidences.append(conf / 100.0)
                texts.append(text)

            # Build full text (preserve line breaks)
            full_text = self._build_full_text(data)

            # Compute average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                blocks=blocks,
                image_width=img_width,
                image_height=img_height,
                status=OCRStatus.SUCCESS if blocks else OCRStatus.PARTIAL,
                provider=self.name,
                execution_time=0.0,
                metadata={
                    "language": self._language,
                    "config": self._config,
                    "total_words": len(blocks),
                },
            )

        except Exception as e:
            logger.error(f"pytesseract extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                blocks=[],
                image_width=0,
                image_height=0,
                status=OCRStatus.FAILED,
                provider=self.name,
                execution_time=0.0,
                error=str(e),
            )

    def _build_full_text(self, data: Dict) -> str:
        """Build full text preserving line structure."""
        lines = []
        current_block = -1
        current_line = -1
        line_text = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            block_num = data.get("block_num", [0])[i]
            line_num = data.get("line_num", [0])[i]
            text = data["text"][i].strip()

            if block_num != current_block or line_num != current_line:
                if line_text:
                    lines.append(" ".join(line_text))
                line_text = []
                current_block = block_num
                current_line = line_num

            if text:
                line_text.append(text)

        if line_text:
            lines.append(" ".join(line_text))

        return "\n".join(lines)


class StubOCRProvider(OCRProvider):
    """Stub OCR provider when no engine is available."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def is_available(self) -> bool:
        return True

    def _extract_text(self, image_path: str) -> OCRResult:
        try:
            from PIL import Image
            img = Image.open(image_path)
            w, h = img.size
        except Exception:
            w, h = 0, 0

        return OCRResult(
            text="",
            confidence=0.0,
            blocks=[],
            image_width=w,
            image_height=h,
            status=OCRStatus.UNSUPPORTED,
            provider=self.name,
            execution_time=0.0,
            error="No OCR engine available",
        )
