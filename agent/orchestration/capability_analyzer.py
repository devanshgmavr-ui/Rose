"""Capability analyzer for autonomous tool selection.

Phase 9 - Autonomous Capability Analysis.

Analyzes user prompts to determine required capabilities,
then maps capabilities to registered tools.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Capability:
    """A required capability derived from user intent."""
    name: str
    description: str
    confidence: float = 1.0
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
            "keywords": self.keywords,
        }


@dataclass
class CapabilityAnalysis:
    """Result of analyzing a user prompt for required capabilities."""
    capabilities: List[Capability] = field(default_factory=list)
    explicit_tools: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capabilities": [c.to_dict() for c in self.capabilities],
            "explicit_tools": self.explicit_tools,
            "constraints": self.constraints,
            "task_type": self.task_type,
        }

    def get_capability_names(self) -> List[str]:
        return [c.name for c in self.capabilities]


# Capability definitions mapped to keyword patterns
CAPABILITY_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "file_operations": {
        "description": "Read, write, copy, move, delete files",
        "keywords": [
            "file", "read file", "write file", "save", "create file",
            "copy file", "move file", "delete file", "list files",
            "directory", "folder", "search files", "rename",
        ],
    },
    "code_execution": {
        "description": "Execute Python code in sandbox",
        "keywords": [
            "run code", "execute", "python", "script", "calculate",
            "compute", "program", "run python",
        ],
    },
    "screen_capture": {
        "description": "Capture desktop screenshots",
        "keywords": [
            "screenshot", "screen capture", "capture screen",
            "take picture", "grab screen", "screen shot",
        ],
    },
    "system_information": {
        "description": "Get system info (CPU, memory, OS)",
        "keywords": [
            "system info", "computer info", "hardware", "cpu",
            "memory", "ram", "disk space", "os info",
        ],
    },
    "mouse_control": {
        "description": "Move mouse and click",
        "keywords": [
            "click", "mouse", "cursor", "move mouse", "scroll",
            "tap", "pointer",
        ],
    },
    "keyboard_input": {
        "description": "Type text and press keys",
        "keywords": [
            "type", "keyboard", "press key", "press enter",
            "hotkey", "shortcut", "text input", "input text",
        ],
    },
    "window_management": {
        "description": "Manage windows (list, activate, minimize, etc.)",
        "keywords": [
            "window", "minimize", "maximize", "restore",
            "activate window", "focus window", "close window",
            "list windows", "switch to",
        ],
    },
    "browser_automation": {
        "description": "Control browser sessions and navigate",
        "keywords": [
            "browser", "chrome", "firefox", "edge", "navigate",
            "website", "web page", "url", "browse", "open link",
            "search online", "go to http",
        ],
    },
    "browser_reading": {
        "description": "Read and extract content from web pages",
        "keywords": [
            "read page", "page content", "extract text",
            "webpage content", "what's on the page",
        ],
    },
    "browser_interaction": {
        "description": "Click, fill, select elements on web pages",
        "keywords": [
            "click button", "fill form", "select option",
            "interact with page", "press button on",
        ],
    },
    "vision_analysis": {
        "description": "Analyze images to understand content",
        "keywords": [
            "analyze", "what's in", "describe",
            "look at", "understand", "interpret", "read",
            "visual", "ocr", "extract text",
            "screenshot analysis", "picture content",
        ],
    },
    "visual_grounding": {
        "description": "Find UI elements in images by coordinates",
        "keywords": [
            "find button", "find element", "locate",
            "coordinates", "where is",
            "click on element", "ui element",
        ],
    },
    "app_launch": {
        "description": "Launch desktop applications",
        "keywords": [
            "open app", "launch", "start program", "open notepad",
            "open calculator", "run application",
        ],
    },
    "text_transcription": {
        "description": "Extract text from images and type it",
        "keywords": [
            "transcribe", "ocr", "extract text from image",
            "type text from", "copy text from image",
        ],
    },
    "verification": {
        "description": "Verify task completion",
        "keywords": [
            "verify", "check", "confirm", "validate",
            "make sure", "ensure",
        ],
    },
}


class CapabilityAnalyzer:
    """Analyzes user prompts to determine required capabilities.

    Does NOT select tools directly. Instead determines WHAT
    capabilities are needed, leaving tool selection to the scorer.
    """

    def __init__(self):
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for cap_name, cap_def in CAPABILITY_DEFINITIONS.items():
            patterns = []
            for keyword in cap_def["keywords"]:
                escaped = re.escape(keyword)
                patterns.append(re.compile(escaped, re.IGNORECASE))
            self._compiled_patterns[cap_name] = patterns

    def analyze(self, user_prompt: str) -> CapabilityAnalysis:
        """Analyze a user prompt for required capabilities.

        Args:
            user_prompt: The user's natural language request.

        Returns:
            CapabilityAnalysis with detected capabilities and constraints.
        """
        analysis = CapabilityAnalysis()
        prompt_lower = user_prompt.lower()

        # Detect explicit tool requests
        analysis.explicit_tools = self._detect_explicit_tools(prompt_lower)

        # Detect constraints
        analysis.constraints = self._detect_constraints(prompt_lower)

        # Detect capabilities
        analysis.capabilities = self._detect_capabilities(prompt_lower)

        # Determine task type
        analysis.task_type = self._classify_task_type(analysis)

        return analysis

    def _detect_capabilities(self, prompt_lower: str) -> List[Capability]:
        """Detect required capabilities from the prompt."""
        detected = []

        for cap_name, patterns in self._compiled_patterns.items():
            confidence = 0.0
            matched_keywords = []

            for i, pattern in enumerate(patterns):
                if pattern.search(prompt_lower):
                    confidence += 1.0 / (i + 1)
                    matched_keywords.append(
                        CAPABILITY_DEFINITIONS[cap_name]["keywords"][i]
                    )

            if confidence > 0:
                confidence = min(confidence / 2.0, 1.0)
                detected.append(Capability(
                    name=cap_name,
                    description=CAPABILITY_DEFINITIONS[cap_name]["description"],
                    confidence=confidence,
                    keywords=matched_keywords,
                ))

        detected.sort(key=lambda c: c.confidence, reverse=True)
        return detected

    def _detect_explicit_tools(self, prompt_lower: str) -> List[str]:
        """Detect if user explicitly requested specific tools."""
        explicit = []
        tool_patterns = {
            "browser": [r"use\s+(?:the\s+)?browser", r"open\s+(?:a\s+)?browser"],
            "filesystem": [r"use\s+(?:the\s+)?file(?:system)?", r"save\s+to\s+(?:a\s+)?file"],
            "keyboard": [r"use\s+(?:the\s+)?keyboard", r"type\s+(?:it\s+)?(?:with|using)"],
            "mouse": [r"use\s+(?:the\s+)?mouse", r"click\s+(?:with|using)"],
            "vision_analyze": [r"use\s+(?:the\s+)?vision", r"use\s+(?:image\s+)?(?:analysis|ocr)"],
            "screen_capture": [r"use\s+(?:the\s+)?screenshot", r"take\s+(?:a\s+)?screenshot"],
            "python_sandbox": [r"use\s+(?:the\s+)?(?:python|code|sandbox)"],
            "window": [r"use\s+(?:the\s+)?window"],
            "launch_app": [r"use\s+(?:the\s+)?(?:launch|app)"],
        }

        for tool_name, patterns in tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    explicit.append(tool_name)
                    break

        return explicit

    def _detect_constraints(self, prompt_lower: str) -> Dict[str, Any]:
        """Detect user constraints from the prompt."""
        constraints = {}

        # "without browser" / "don't use browser"
        no_browser = [
            r"without\s+(?:the\s+)?browser",
            r"don'?t\s+use\s+(?:the\s+)?browser",
            r"no\s+browser",
            r"skip\s+(?:the\s+)?browser",
        ]
        for pattern in no_browser:
            if re.search(pattern, prompt_lower):
                constraints["prohibited_tools"] = ["browser"]
                break

        # "without keyboard" / "don't use keyboard"
        no_keyboard = [
            r"without\s+(?:the\s+)?keyboard",
            r"don'?t\s+use\s+(?:the\s+)?keyboard",
            r"no\s+keyboard",
        ]
        for pattern in no_keyboard:
            if re.search(pattern, prompt_lower):
                if "prohibited_tools" not in constraints:
                    constraints["prohibited_tools"] = []
                constraints["prohibited_tools"].append("keyboard")
                break

        # "only modify X"
        only_modify = re.search(r"only\s+(?:modify|edit|change)\s+(\S+)", prompt_lower)
        if only_modify:
            constraints["allowed_files"] = [only_modify.group(1)]

        # "don't ask me"
        no_confirm = [
            r"don'?t\s+ask\s+me",
            r"no\s+confirm(?:ation)?",
            r"without\s+asking",
            r"just\s+do\s+it",
        ]
        for pattern in no_confirm:
            if re.search(pattern, prompt_lower):
                constraints["minimize_confirmations"] = True
                break

        # "use the image on my screen"
        screen_image = re.search(
            r"use\s+(?:the\s+)?image\s+(?:on\s+)?(?:my\s+)?screen",
            prompt_lower,
        )
        if screen_image:
            constraints["use_screen_image"] = True

        return constraints

    def _classify_task_type(self, analysis: CapabilityAnalysis) -> str:
        """Classify the overall task type based on capabilities."""
        cap_names = set(analysis.get_capability_names())

        if "text_transcription" in cap_names:
            return "transcription"
        if "browser_automation" in cap_names or "browser_reading" in cap_names:
            return "web_task"
        if "vision_analysis" in cap_names and "keyboard_input" in cap_names:
            return "visual_to_text"
        if "file_operations" in cap_names and "code_execution" in cap_names:
            return "development"
        if "window_management" in cap_names and "keyboard_input" in cap_names:
            return "desktop_automation"
        if len(cap_names) <= 1:
            return "simple"
        return "multi_step"

    def get_capabilities_for_step(
        self,
        step_description: str,
        step_action: str = "",
    ) -> List[Capability]:
        """Get capabilities needed for a specific step.

        Used for step-level tool selection during execution.
        """
        combined = f"{step_description} {step_action}".lower()
        return self._detect_capabilities(combined)
