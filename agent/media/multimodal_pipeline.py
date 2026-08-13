"""Multimodal Request Pipeline for Rose.

Stage C - Clean multimodal request abstraction.

Provides:
- Text-only requests
- Image-only requests
- Text + image requests
- Screenshot + instruction requests
- Browser screenshot + instruction requests
- Screen observation + instruction requests
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Types of multimodal requests."""
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    TEXT_AND_IMAGE = "text_and_image"
    SCREENSHOT_AND_INSTRUCTION = "screenshot_and_instruction"
    BROWSER_SCREENSHOT_AND_INSTRUCTION = "browser_screenshot_and_instruction"
    SCREEN_OBSERVATION = "screen_observation"


@dataclass
class MultimodalRequest:
    """A structured multimodal request."""
    request_type: RequestType
    text: str = ""
    image_path: Optional[str] = None
    instruction: str = ""
    source: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_type": self.request_type.value,
            "text": self.text,
            "image_path": self.image_path,
            "instruction": self.instruction,
            "source": self.source,
            "metadata": self.metadata,
        }


class MultimodalRequestPipeline:
    """Processes and routes multimodal requests.
    
    Converts raw user input into structured MultimodalRequest objects
    that can be routed to the appropriate processing pipeline.
    """

    def __init__(self):
        self._stats: Dict[str, int] = {}

    def classify_request(
        self,
        user_input: str,
        has_image: bool = False,
        image_path: Optional[str] = None,
        has_screenshot: bool = False,
        screenshot_path: Optional[str] = None,
        has_browser: bool = False,
    ) -> MultimodalRequest:
        """Classify a user request into the appropriate multimodal type.
        
        Args:
            user_input: Raw user text input
            has_image: Whether user provided an image
            image_path: Path to user-provided image
            has_screenshot: Whether a screenshot is available
            screenshot_path: Path to available screenshot
            has_browser: Whether a browser session is active
            
        Returns:
            MultimodalRequest with classified type
        """
        self._stats["total"] = self._stats.get("total", 0) + 1
        
        # User provided an image
        if has_image and image_path:
            self._stats["text_and_image"] = self._stats.get("text_and_image", 0) + 1
            return MultimodalRequest(
                request_type=RequestType.TEXT_AND_IMAGE,
                text=user_input,
                image_path=image_path,
                source="user",
            )
        
        # Screenshot available and request seems visual
        if has_screenshot and screenshot_path:
            if self._needs_visual_context(user_input):
                self._stats["screenshot_and_instruction"] = self._stats.get("screenshot_and_instruction", 0) + 1
                return MultimodalRequest(
                    request_type=RequestType.SCREENSHOT_AND_INSTRUCTION,
                    text=user_input,
                    instruction=user_input,
                    image_path=screenshot_path,
                    source="screen",
                )
        
        # Browser available and request seems browser-related
        if has_browser and self._is_browser_request(user_input):
            self._stats["browser_screenshot_and_instruction"] = self._stats.get("browser_screenshot_and_instruction", 0) + 1
            return MultimodalRequest(
                request_type=RequestType.BROWSER_SCREENSHOT_AND_INSTRUCTION,
                text=user_input,
                instruction=user_input,
                source="browser",
            )
        
        # Default: text-only
        self._stats["text_only"] = self._stats.get("text_only", 0) + 1
        return MultimodalRequest(
            request_type=RequestType.TEXT_ONLY,
            text=user_input,
            source="user",
        )

    def _needs_visual_context(self, text: str) -> bool:
        """Determine if a request needs visual context."""
        text_lower = text.lower()
        visual_indicators = [
            "screen", "desktop", "display", "visible", "see", "look",
            "what's on", "what is on", "showing", "button", "icon",
            "find", "click", "locate", "where is", "position",
            "describe what", "what do you see", "screenshot",
            "blue", "red", "green", "color",
        ]
        return any(indicator in text_lower for indicator in visual_indicators)

    def _is_browser_request(self, text: str) -> bool:
        """Determine if a request is browser-related."""
        text_lower = text.lower()
        browser_indicators = [
            "website", "web page", "browser", "url", "http",
            "page", "link", "navigate", "open example",
            "tab", "site", "internet",
        ]
        return any(indicator in text_lower for indicator in browser_indicators)

    def get_stats(self) -> Dict[str, int]:
        """Get request classification statistics."""
        return self._stats.copy()
