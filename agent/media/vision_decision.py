"""Vision Decision System for Rose.

Stage D - Determines when visual input is required.

Provides:
- Deterministic vision requirement decisions
- Source selection (screen, browser, user image, none)
- Unnecessary vision avoidance
- Testable decision logic
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VisionSource(Enum):
    """Possible sources for visual input."""
    SCREEN = "screen"
    BROWSER = "browser"
    USER_IMAGE = "user_image"
    EXISTING_MEDIA = "existing_media"
    NONE = "none"


class VisionRequirement(Enum):
    """Whether visual context is required."""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    NOT_NEEDED = "not_needed"


@dataclass
class VisionDecision:
    """Decision about whether vision is needed."""
    requirement: VisionRequirement
    source: VisionSource
    reasoning: str
    needs_grounding: bool = False
    needs_screenshot: bool = False
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement": self.requirement.value,
            "source": self.source.value,
            "reasoning": self.reasoning,
            "needs_grounding": self.needs_grounding,
            "needs_screenshot": self.needs_screenshot,
            "confidence": self.confidence,
        }


# Keywords that indicate vision is needed
SCREEN_KEYWORDS = {
    "screen", "desktop", "display", "monitor", "visible", "see",
    "what's on", "what is on", "looking at", "showing", "current screen",
    "screenshot", "capture", "look at", "describe what", "what do you see",
}

VISUAL_UI_KEYWORDS = {
    "button", "icon", "menu", "window", "dialog", "form", "text field",
    "checkbox", "dropdown", "slider", "tab", "panel", "toolbar",
    "find the", "click the", "locate", "where is", "position",
    "blue", "red", "green", "yellow", "color", "colour",
    "top left", "bottom right", "center", "corner", "edge",
}

BROWSER_VISUAL_KEYWORDS = {
    "page look", "page appearance", "describe the page", "what does the page",
    "page design", "layout", "visual", "screenshot of", "capture the page",
    "what's on the page", "page look like",
}

SCREENSHOT_KEYWORDS = {
    "screenshot", "capture", "take a picture", "snap",
}

NO_VISION_KEYWORDS = {
    "explain", "define", "what is", "what are", "how to", "how do",
    "tell me about", "describe in words", "write", "create", "generate",
    "code", "function", "class", "variable", "list", "dict",
    "open example.com", "navigate to", "go to",
    "type", "press", "enter",
}


class VisionDecisionSystem:
    """Determines when visual input is required.
    
    Uses keyword matching and context analysis to decide:
    - Whether visual context is needed
    - What source to use
    - Whether grounding is needed
    - Whether a screenshot should be taken
    """

    def __init__(self):
        self._history: List[VisionDecision] = []

    def decide(
        self,
        user_request: str,
        has_existing_image: bool = False,
        has_browser_session: bool = False,
        has_screen_capture: bool = False,
    ) -> VisionDecision:
        """Decide whether vision is required for a user request.
        
        Args:
            user_request: The user's natural language request
            has_existing_image: Whether an image was already provided
            has_browser_session: Whether a browser session is active
            has_screen_capture: Whether a recent screen capture exists
            
        Returns:
            VisionDecision with requirement, source, and reasoning
        """
        request_lower = user_request.lower().strip()
        
        # Check if user provided an image
        if has_existing_image:
            decision = VisionDecision(
                requirement=VisionRequirement.REQUIRED,
                source=VisionSource.EXISTING_MEDIA,
                reasoning="User provided an image for analysis",
                needs_grounding=False,
                needs_screenshot=False,
            )
            self._history.append(decision)
            return decision
        
        # Check for explicit screenshot requests FIRST (before text-only)
        if self._mentions_screenshot(request_lower):
            decision = VisionDecision(
                requirement=VisionRequirement.REQUIRED,
                source=VisionSource.SCREEN,
                reasoning="User explicitly requested screenshot or screen capture",
                needs_grounding=False,
                needs_screenshot=True,
            )
            self._history.append(decision)
            return decision
        
        # Check for screen-related questions FIRST (before text-only)
        if self._mentions_screen(request_lower):
            decision = VisionDecision(
                requirement=VisionRequirement.REQUIRED,
                source=VisionSource.SCREEN,
                reasoning="User asked about screen/desktop content",
                needs_grounding=False,
                needs_screenshot=True,
            )
            self._history.append(decision)
            return decision
        
        # Check for visual UI questions (find button, locate element, etc.)
        if self._mentions_visual_ui(request_lower):
            decision = VisionDecision(
                requirement=VisionRequirement.REQUIRED,
                source=VisionSource.SCREEN,
                reasoning="User asked about visual UI elements (buttons, icons, positions)",
                needs_grounding=True,
                needs_screenshot=True,
            )
            self._history.append(decision)
            return decision
        
        # Check for browser visual questions
        if self._mentions_browser_visual(request_lower):
            if has_browser_session:
                decision = VisionDecision(
                    requirement=VisionRequirement.REQUIRED,
                    source=VisionSource.BROWSER,
                    reasoning="User asked about browser page appearance",
                    needs_grounding=False,
                    needs_screenshot=True,
                )
            else:
                decision = VisionDecision(
                    requirement=VisionRequirement.RECOMMENDED,
                    source=VisionSource.SCREEN,
                    reasoning="User asked about browser but no session active; using screen",
                    needs_grounding=False,
                    needs_screenshot=True,
                )
            self._history.append(decision)
            return decision
        
        # Check for color-related queries (might need vision)
        if self._mentions_colors(request_lower):
            decision = VisionDecision(
                requirement=VisionRequirement.REQUIRED,
                source=VisionSource.SCREEN,
                reasoning="User asked about colors, needs visual analysis",
                needs_grounding=False,
                needs_screenshot=True,
            )
            self._history.append(decision)
            return decision
        
        # Check for explicitly no-vision requests (after visual checks)
        if self._is_text_only_request(request_lower):
            decision = VisionDecision(
                requirement=VisionRequirement.NOT_NEEDED,
                source=VisionSource.NONE,
                reasoning="Text-only request, no visual input needed",
                needs_grounding=False,
                needs_screenshot=False,
            )
            self._history.append(decision)
            return decision
        
        # Default: text-only
        decision = VisionDecision(
            requirement=VisionRequirement.NOT_NEEDED,
            source=VisionSource.NONE,
            reasoning="No visual input indicators detected",
            needs_grounding=False,
            needs_screenshot=False,
        )
        self._history.append(decision)
        return decision

    def _is_text_only_request(self, request: str) -> bool:
        """Check if request is purely text-based."""
        # Strong text-only indicators
        for keyword in NO_VISION_KEYWORDS:
            if keyword in request:
                return True
        return False

    def _mentions_screenshot(self, request: str) -> bool:
        """Check if request mentions screenshots."""
        for keyword in SCREENSHOT_KEYWORDS:
            if keyword in request:
                return True
        return False

    def _mentions_screen(self, request: str) -> bool:
        """Check if request mentions screen/desktop content."""
        for keyword in SCREEN_KEYWORDS:
            if keyword in request:
                return True
        return False

    def _mentions_visual_ui(self, request: str) -> bool:
        """Check if request mentions visual UI elements."""
        for keyword in VISUAL_UI_KEYWORDS:
            if keyword in request:
                return True
        return False

    def _mentions_browser_visual(self, request: str) -> bool:
        """Check if request asks about browser page appearance."""
        for keyword in BROWSER_VISUAL_KEYWORDS:
            if keyword in request:
                return True
        return False

    def _mentions_colors(self, request: str) -> bool:
        """Check if request mentions colors."""
        color_words = {"color", "colour", "red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white"}
        for word in color_words:
            if word in request:
                return True
        return False

    def get_history(self) -> List[VisionDecision]:
        """Get decision history."""
        return self._history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get decision statistics."""
        total = len(self._history)
        if total == 0:
            return {"total": 0, "required": 0, "not_needed": 0}
        
        required = sum(1 for d in self._history if d.requirement == VisionRequirement.REQUIRED)
        not_needed = sum(1 for d in self._history if d.requirement == VisionRequirement.NOT_NEEDED)
        
        return {
            "total": total,
            "required": required,
            "not_needed": not_needed,
            "required_pct": required / total * 100,
        }
