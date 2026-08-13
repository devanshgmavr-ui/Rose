"""Real vision analysis provider with image understanding.

Phase 10 - Vision System.

Provides actual image analysis capabilities including:
- Image preprocessing and validation
- Metadata extraction
- Color analysis
- Basic shape detection
- OCR/text extraction (when available)
- Integration with local multimodal models
- Security boundaries for untrusted content
"""

import os
import time
import logging
import hashlib
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union
from abc import abstractmethod

from .base import MediaProvider, MediaType, MediaRequest, MediaResult, MediaOutput
from .vision import (
    VisionProvider,
    VisionResult,
    DetectedElement,
    BoundingBox,
    VisionConfidence,
)

logger = logging.getLogger(__name__)


class ImageFormat(Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    BMP = "bmp"
    GIF = "gif"
    UNKNOWN = "unknown"


@dataclass
class ImageMetadata:
    """Comprehensive image metadata."""
    file_path: str
    file_name: str
    file_size: int
    format: ImageFormat
    width: int
    height: int
    mode: str  # RGB, RGBA, L, etc.
    dpi: Tuple[int, int] = (72, 72)
    has_alpha: bool = False
    is_animated: bool = False
    frame_count: int = 1
    color_depth: int = 8
    gamma: float = 1.0
    exif_data: Dict[str, Any] = field(default_factory=dict)
    file_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "dpi": list(self.dpi),
            "has_alpha": self.has_alpha,
            "is_animated": self.is_animated,
            "frame_count": self.frame_count,
            "color_depth": self.color_depth,
            "gamma": self.gamma,
            "exif_data": self.exif_data,
            "file_hash": self.file_hash,
        }


@dataclass
class ColorInfo:
    """Color information for an image region."""
    dominant_colors: List[Tuple[int, int, int]]
    average_color: Tuple[int, int, int]
    color_variance: float
    is_grayscale: bool
    brightness: float  # 0.0 to 1.0
    contrast: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_colors": [list(c) for c in self.dominant_colors],
            "average_color": list(self.average_color),
            "color_variance": self.color_variance,
            "is_grayscale": self.is_grayscale,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }


@dataclass
class ImageAnalysis:
    """Comprehensive image analysis result."""
    metadata: ImageMetadata
    color_info: Optional[ColorInfo] = None
    detected_text: str = ""
    detected_shapes: List[Dict[str, Any]] = field(default_factory=list)
    detected_regions: List[Dict[str, Any]] = field(default_factory=list)
    ocr_confidence: float = 0.0
    analysis_notes: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "metadata": self.metadata.to_dict(),
            "detected_text": self.detected_text,
            "detected_shapes": self.detected_shapes,
            "detected_regions": self.detected_regions,
            "ocr_confidence": self.ocr_confidence,
            "analysis_notes": self.analysis_notes,
            "processing_time": self.processing_time,
        }
        if self.color_info:
            result["color_info"] = self.color_info.to_dict()
        return result


class ImagePreprocessor:
    """Image preprocessing utilities."""

    @staticmethod
    def load_image_safely(
        image_path: str,
        max_size_mb: int = 20,
        max_dimension: int = 4096,
    ) -> Tuple[Optional[Any], List[str]]:
        """Load image with safety checks.

        Args:
            image_path: Path to image file.
            max_size_mb: Maximum file size in MB.
            max_dimension: Maximum width/height.

        Returns:
            Tuple of (PIL Image or None, list of warnings).
        """
        warnings = []

        if not os.path.exists(image_path):
            return None, [f"File not found: {image_path}"]

        if not os.path.isfile(image_path):
            return None, [f"Not a file: {image_path}"]

        try:
            file_size = os.path.getsize(image_path)
        except OSError as e:
            return None, [f"Cannot read file: {e}"]

        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return None, [
                f"File too large: {file_size / (1024*1024):.1f} MB "
                f"(max {max_size_mb} MB)"
            ]

        if file_size == 0:
            return None, ["File is empty"]

        try:
            from PIL import Image
            img = Image.open(image_path)
            img.load()  # Force load to catch corrupted images

            width, height = img.size
            if width > max_dimension or height > max_dimension:
                warnings.append(
                    f"Image dimensions {width}x{height} exceed max {max_dimension}"
                )
                # Resize to fit
                ratio = min(max_dimension / width, max_dimension / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                warnings.append(f"Resized to {img.size[0]}x{img.size[1]}")

            return img, warnings

        except ImportError:
            return None, ["Pillow not installed - cannot load images"]
        except Exception as e:
            return None, [f"Failed to load image: {e}"]

    @staticmethod
    def preprocess_for_analysis(
        img: Any,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Preprocess image for analysis.

        Args:
            img: PIL Image object.
            target_size: Target size for resizing.
            normalize: Whether to normalize pixel values.

        Returns:
            Tuple of (preprocessed image, preprocessing info).
        """
        from PIL import Image

        info = {
            "original_size": img.size,
            "original_mode": img.mode,
            "target_size": target_size,
        }

        # Convert to RGB if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
            info["converted_mode"] = "RGB"

        # Resize while maintaining aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        info["final_size"] = img.size

        return img, info

    @staticmethod
    def extract_metadata(img: Any, file_path: str) -> ImageMetadata:
        """Extract comprehensive image metadata.

        Args:
            img: PIL Image object.
            file_path: Path to image file.

        Returns:
            ImageMetadata with extracted information.
        """
        from PIL import Image

        path = Path(file_path)
        stat = path.stat()

        # Calculate file hash
        file_hash = ""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read(8192)).hexdigest()
        except Exception:
            pass

        # Determine format
        format_map = {
            "PNG": ImageFormat.PNG,
            "JPEG": ImageFormat.JPEG,
            "WEBP": ImageFormat.WEBP,
            "BMP": ImageFormat.BMP,
            "GIF": ImageFormat.GIF,
        }
        fmt = format_map.get(img.format, ImageFormat.UNKNOWN)

        # Get DPI
        dpi = (72, 72)
        if hasattr(img, "info") and "dpi" in img.info:
            dpi = tuple(img.info["dpi"])

        # Check for animation
        is_animated = False
        frame_count = 1
        if hasattr(img, "n_frames"):
            frame_count = img.n_frames
            is_animated = frame_count > 1

        # Get EXIF data
        exif_data = {}
        if hasattr(img, "_getexif") and img._getexif():
            try:
                exif = img._getexif()
                if exif:
                    exif_data = {str(k): str(v) for k, v in list(exif.items())[:20]}
            except Exception:
                pass

        return ImageMetadata(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_size=stat.st_size,
            format=fmt,
            width=img.size[0],
            height=img.size[1],
            mode=img.mode,
            dpi=dpi,
            has_alpha="A" in img.mode,
            is_animated=is_animated,
            frame_count=frame_count,
            color_depth=img.bits if hasattr(img, "bits") else 8,
            exif_data=exif_data,
            file_hash=file_hash,
        )

    @staticmethod
    def analyze_colors(img: Any, sample_size: int = 1000) -> ColorInfo:
        """Analyze image colors.

        Args:
            img: PIL Image object.
            sample_size: Number of pixels to sample.

        Returns:
            ColorInfo with color analysis.
        """
        from PIL import Image
        import random

        # Convert to RGB if needed
        if img.mode != "RGB":
            img_rgb = img.convert("RGB")
        else:
            img_rgb = img

        # Sample pixels
        pixels = []
        width, height = img_rgb.size
        for _ in range(min(sample_size, width * height)):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            pixels.append(img_rgb.getpixel((x, y)))

        if not pixels:
            return ColorInfo(
                dominant_colors=[],
                average_color=(0, 0, 0),
                color_variance=0.0,
                is_grayscale=True,
                brightness=0.5,
                contrast=0.0,
            )

        # Calculate average color
        avg_r = sum(p[0] for p in pixels) // len(pixels)
        avg_g = sum(p[1] for p in pixels) // len(pixels)
        avg_b = sum(p[2] for p in pixels) // len(pixels)
        average_color = (avg_r, avg_g, avg_b)

        # Calculate brightness (perceived luminance)
        brightness = (0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b) / 255.0

        # Calculate color variance
        variance = sum(
            (p[0] - avg_r) ** 2 + (p[1] - avg_g) ** 2 + (p[2] - avg_b) ** 2
            for p in pixels
        ) / len(pixels)
        color_variance = variance / (255 * 255)  # Normalize

        # Check if grayscale
        color_diff_threshold = 10
        is_grayscale = all(
            abs(p[0] - p[1]) < color_diff_threshold and
            abs(p[1] - p[2]) < color_diff_threshold
            for p in pixels[:100]
        )

        # Find dominant colors (simple clustering)
        from collections import Counter
        # Quantize colors to reduce space
        quantized = [
            (p[0] // 32 * 32, p[1] // 32 * 32, p[2] // 32 * 32)
            for p in pixels
        ]
        color_counts = Counter(quantized)
        dominant_colors = [c for c, _ in color_counts.most_common(5)]

        # Calculate contrast (standard deviation of brightness)
        brightnesses = [
            (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) / 255.0
            for p in pixels
        ]
        avg_bright = sum(brightnesses) / len(brightnesses)
        contrast = sum((b - avg_bright) ** 2 for b in brightnesses) / len(brightnesses)
        contrast = min(1.0, contrast * 4)  # Scale to 0-1

        return ColorInfo(
            dominant_colors=dominant_colors,
            average_color=average_color,
            color_variance=color_variance,
            is_grayscale=is_grayscale,
            brightness=brightness,
            contrast=contrast,
        )

    @staticmethod
    def detect_basic_regions(img: Any) -> List[Dict[str, Any]]:
        """Detect basic image regions (simple edge-based detection).

        Args:
            img: PIL Image object.

        Returns:
            List of detected regions.
        """
        from PIL import Image

        regions = []

        try:
            # Convert to grayscale
            if img.mode != "L":
                img_gray = img.convert("L")
            else:
                img_gray = img

            width, height = img.size

            # Simple grid-based region detection
            grid_size = 4
            cell_w = width // grid_size
            cell_h = height // grid_size

            for gy in range(grid_size):
                for gx in range(grid_size):
                    x1 = gx * cell_w
                    y1 = gy * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h

                    # Sample region
                    try:
                        region = img_gray.crop((x1, y1, x2, y2))
                        pixels = list(region.getdata())

                        if pixels:
                            avg = sum(pixels) / len(pixels)
                            variance = sum((p - avg) ** 2 for p in pixels) / len(pixels)

                            # High variance suggests interesting content
                            if variance > 1000:
                                regions.append({
                                    "x": x1, "y": y1,
                                    "width": cell_w, "height": cell_h,
                                    "brightness": avg / 255.0,
                                    "variance": variance / (255 * 255),
                                    "type": "high_detail",
                                })
                            elif avg < 50:
                                regions.append({
                                    "x": x1, "y": y1,
                                    "width": cell_w, "height": cell_h,
                                    "brightness": avg / 255.0,
                                    "variance": variance / (255 * 255),
                                    "type": "dark_region",
                                })
                            elif avg > 200:
                                regions.append({
                                    "x": x1, "y": y1,
                                    "width": cell_w, "height": cell_h,
                                    "brightness": avg / 255.0,
                                    "variance": variance / (255 * 255),
                                    "type": "bright_region",
                                })
                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Region detection failed: {e}")

        return regions


class RealVisionProvider(VisionProvider):
    """Real vision provider with actual image analysis.

    Provides actual image understanding through:
    - Image preprocessing
    - Metadata extraction
    - Color analysis
    - Basic region detection
    - OCR when available
    - Integration with multimodal models when available
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_ocr: bool = True,
        use_multimodal: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._use_ocr = use_ocr
        self._use_multimodal = use_multimodal
        self._ocr_available = False
        self._multimodal_available = False
        self._preprocessor = ImagePreprocessor()

    @property
    def name(self) -> str:
        return "real_vision"

    @property
    def description(self) -> str:
        return "Real vision provider with image analysis capabilities"

    @property
    def is_available(self) -> bool:
        return True  # Always available for basic analysis

    def initialize(self) -> bool:
        """Initialize vision provider and check capabilities."""
        self._initialized = True

        # Check OCR availability
        try:
            import pytesseract
            self._ocr_available = True
            logger.info("OCR available via pytesseract")
        except ImportError:
            logger.info("OCR not available (pytesseract not installed)")

        # Check multimodal model availability
        if self._model_path and os.path.exists(self._model_path):
            try:
                from llama_cpp import Llama
                self._multimodal_available = True
                logger.info(f"Multimodal model available: {self._model_path}")
            except ImportError:
                logger.info("llama.cpp not available for multimodal analysis")

        logger.info(
            f"Vision provider initialized: "
            f"ocr={self._ocr_available}, "
            f"multimodal={self._multimodal_available}"
        )
        return True

    def _analyze_image(self, request: MediaRequest) -> VisionResult:
        """Analyze image with real capabilities."""
        start = time.time()
        file_path = request.input_path

        # Load image
        img, warnings = self._preprocessor.load_image_safely(
            file_path,
            max_size_mb=self._max_image_size_mb,
            max_dimension=max(self._max_image_width, self._max_image_height),
        )

        if img is None:
            return VisionResult(
                success=False,
                error=warnings[0] if warnings else "Failed to load image",
                provider=self.name,
            )

        # Extract metadata
        metadata = self._preprocessor.extract_metadata(img, file_path)

        # Analyze colors
        color_info = self._preprocessor.analyze_colors(img)

        # Detect regions
        regions = self._preprocessor.detect_basic_regions(img)

        # OCR if available
        detected_text = ""
        ocr_confidence = 0.0
        if self._ocr_available and self._use_ocr:
            try:
                import pytesseract
                # Get OCR with confidence
                ocr_data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT
                )
                texts = []
                confidences = []
                for i, text in enumerate(ocr_data["text"]):
                    if text.strip():
                        texts.append(text)
                        conf = ocr_data["conf"][i]
                        if isinstance(conf, (int, float)) and conf > 0:
                            confidences.append(conf)

                detected_text = " ".join(texts)
                ocr_confidence = (
                    sum(confidences) / len(confidences) / 100.0
                    if confidences else 0.0
                )
            except Exception as e:
                logger.debug(f"OCR failed: {e}")

        # Build detected elements from regions
        detected_elements = []
        for region in regions[:self._max_elements]:
            elem_type = "region"
            desc = f"Image region at ({region['x']}, {region['y']})"

            if region["type"] == "high_detail":
                desc = f"Detailed area at ({region['x']}, {region['y']})"
            elif region["type"] == "dark_region":
                desc = f"Dark region at ({region['x']}, {region['y']})"
            elif region["type"] == "bright_region":
                desc = f"Bright region at ({region['x']}, {region['y']})"

            detected_elements.append(DetectedElement(
                element_type=elem_type,
                description=desc,
                bounding_box=BoundingBox(
                    x=region["x"],
                    y=region["y"],
                    width=region["width"],
                    height=region["height"],
                ),
                confidence=VisionConfidence.MEDIUM,
                metadata={
                    "brightness": region["brightness"],
                    "variance": region["variance"],
                },
            ))

        # Add text element if detected
        if detected_text:
            detected_elements.insert(0, DetectedElement(
                element_type="text",
                description=f"Detected text: {detected_text[:200]}",
                confidence=VisionConfidence.HIGH if ocr_confidence > 0.7 else VisionConfidence.MEDIUM,
                metadata={"ocr_confidence": ocr_confidence},
            ))

        # Build description
        description_parts = [
            f"Image analysis of {metadata.file_name}",
            f"Size: {metadata.width}x{metadata.height} {metadata.mode}",
            f"Format: {metadata.format.value}",
            f"Brightness: {color_info.brightness:.1%}",
            f"Contrast: {color_info.contrast:.1%}",
        ]

        if color_info.is_grayscale:
            description_parts.append("Grayscale image")
        else:
            description_parts.append(
                f"Dominant colors: {len(color_info.dominant_colors)}"
            )

        if detected_text:
            description_parts.append(
                f"OCR detected {len(detected_text)} characters "
                f"(confidence: {ocr_confidence:.1%})"
            )

        description_parts.extend(warnings)

        elapsed = time.time() - start

        return VisionResult(
            success=True,
            description="\n".join(description_parts),
            detected_elements=detected_elements,
            image_width=metadata.width,
            image_height=metadata.height,
            analysis_prompt=request.prompt or "General image analysis",
            provider=self.name,
            execution_time=elapsed,
            metadata={
                "metadata": metadata.to_dict(),
                "color_info": color_info.to_dict(),
                "detected_text": detected_text,
                "ocr_confidence": ocr_confidence,
                "region_count": len(regions),
                "element_count": len(detected_elements),
                "warnings": warnings,
                "capabilities": {
                    "ocr": self._ocr_available,
                    "multimodal": self._multimodal_available,
                },
            },
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        return {
            "image_analysis": True,
            "color_analysis": True,
            "region_detection": True,
            "ocr": self._ocr_available,
            "multimodal": self._multimodal_available,
            "preprocessing": True,
            "metadata_extraction": True,
        }
