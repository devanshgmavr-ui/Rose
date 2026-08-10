"""Automatic tool selection based on user intent.

Stage 4.2 - Automatic Tool Selection.

Allows ROSE to select the appropriate tool based on natural
language user intent. The user does not need to know internal
tool names - the selector maps intent to tools.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolMatch:
    """A matched tool with confidence and reasoning."""
    tool_name: str
    action: str
    confidence: float
    reasoning: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "arguments": self.arguments,
        }


class IntentClassifier:
    """Classifies user intent into tool selections."""

    INTENT_PATTERNS = {
        "screenshot": {
            "patterns": [
                r"screenshot",
                r"screen\s*capture",
                r"capture\s*screen",
                r"take\s*(a\s*)?picture",
                r"grab\s*screen",
            ],
            "tool": "screen_capture",
            "action": "capture",
            "default_args": {"region": "full"},
        },
        "system_info": {
            "patterns": [
                r"system\s*info",
                r"computer\s*info",
                r"hardware",
                r"cpu",
                r"memory",
                r"ram",
                r"disk\s*space",
            ],
            "tool": "system_info",
            "action": "info",
            "default_args": {},
        },
        "file_read": {
            "patterns": [
                r"read\s*(a\s*)?file",
                r"open\s*(a\s*)?file",
                r"show\s*file",
                r"list\s*files?",
                r"show\s*(me\s*)?(the\s*)?(directory|folder|files)",
            ],
            "tool": "filesystem",
            "action": "read",
            "default_args": {},
        },
        "file_write": {
            "patterns": [
                r"write\s*(a\s*)?file",
                r"save\s*(to\s*)?(a\s*)?file",
                r"create\s*(a\s*)?file",
                r"save\s*output",
            ],
            "tool": "filesystem",
            "action": "write",
            "default_args": {},
        },
        "code_execute": {
            "patterns": [
                r"run\s*(a\s*)?code",
                r"execute\s*(code|python|script)",
                r"run\s*(a\s*)?python",
                r"calculate",
                r"compute",
            ],
            "tool": "python_sandbox",
            "action": "execute",
            "default_args": {},
        },
        "mouse_click": {
            "patterns": [
                r"click",
                r"tap",
                r"press\s*(on|at)",
            ],
            "tool": "mouse",
            "action": "click",
            "default_args": {},
        },
        "mouse_move": {
            "patterns": [
                r"move\s*(the\s*)?mouse",
                r"move\s*cursor",
                r"position\s*cursor",
            ],
            "tool": "mouse",
            "action": "move",
            "default_args": {},
        },
        "type_text": {
            "patterns": [
                r"type",
                r"enter\s*text",
                r"input\s*text",
                r"keyboard\s*type",
            ],
            "tool": "keyboard",
            "action": "type",
            "default_args": {},
        },
        "press_key": {
            "patterns": [
                r"press\s*(the\s*)?(\w+)\s*key",
                r"press\s*enter",
                r"press\s*escape",
                r"press\s*tab",
                r"hit\s*(the\s*)?(\w+)",
            ],
            "tool": "keyboard",
            "action": "press",
            "default_args": {},
        },
        "window_list": {
            "patterns": [
                r"list\s*windows?",
                r"show\s*windows?",
                r"what\s*windows?",
                r"all\s*windows?",
            ],
            "tool": "window",
            "action": "list",
            "default_args": {},
        },
        "window_activate": {
            "patterns": [
                r"activate\s*(the\s*)?window",
                r"focus\s*(the\s*)?window",
                r"bring\s*(to\s*front|forward)",
                r"switch\s*to",
            ],
            "tool": "window",
            "action": "activate",
            "default_args": {},
        },
        "browser_open": {
            "patterns": [
                r"open\s*(a\s*)?browser",
                r"start\s*(a\s*)?browser",
                r"launch\s*(a\s*)?browser",
            ],
            "tool": "browser",
            "action": "create_session",
            "default_args": {},
        },
        "browser_navigate": {
            "patterns": [
                r"navigate\s*to",
                r"go\s*to\s*(https?://|www\.)",
                r"open\s*(https?://|www\.)",
                r"browse\s*to",
                r"visit\s*(a\s*)?website",
            ],
            "tool": "browser",
            "action": "navigate",
            "default_args": {},
        },
        "browser_read": {
            "patterns": [
                r"read\s*(the\s*)?(page|website|webpage)",
                r"get\s*(the\s*)?(page\s*)?content",
                r"extract\s*text",
                r"what('s|\s+is)\s*(on\s*)?(the\s*)?page",
            ],
            "tool": "browser",
            "action": "read_page",
            "default_args": {},
        },
        "browser_screenshot": {
            "patterns": [
                r"browser\s*screenshot",
                r"screenshot\s*(of\s*)?(the\s*)?(page|browser|website)",
                r"capture\s*(the\s*)?page",
            ],
            "tool": "browser",
            "action": "screenshot",
            "default_args": {},
        },
        "vision_analyze": {
            "patterns": [
                r"analyze\s*(this\s*)?(image|screenshot|picture|photo)",
                r"what('s|\s+is)\s*in\s*(this|the)\s*(image|screenshot|picture)",
                r"describe\s*(this|the)\s*(image|screenshot|picture)",
                r"understand\s*(this|the)\s*(image|screenshot)",
                r"look\s*at\s*(this|the)\s*(image|screenshot)",
            ],
            "tool": "vision_analyze",
            "action": "analyze",
            "default_args": {},
        },
        "visual_ground": {
            "patterns": [
                r"find\s*(the\s*)?(button|link|element|icon)",
                r"locate\s*(the\s*)?(button|link|element|icon)",
                r"where\s*is\s*(the\s*)?(button|link|element)",
                r"coordinates\s*of",
            ],
            "tool": "visual_ground",
            "action": "ground",
            "default_args": {},
        },
        "search_online": {
            "patterns": [
                r"search\s*(for\s*)?(the\s*)?web",
                r"google\s*(for\s*)?",
                r"look\s*up\s*online",
                r"search\s*online",
            ],
            "tool": "browser",
            "action": "navigate",
            "default_args": {"url": "https://www.google.com"},
        },
    }

    def __init__(self):
        self._compiled = {}
        for intent_name, intent_data in self.INTENT_PATTERNS.items():
            compiled = []
            for pattern in intent_data["patterns"]:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            self._compiled[intent_name] = compiled

    def classify(
        self, user_request: str, available_tools: Optional[List[str]] = None
    ) -> List[ToolMatch]:
        """Classify user intent and return matching tools.

        Args:
            user_request: The user's natural language request.
            available_tools: Optional list of available tool names.

        Returns:
            List of ToolMatch objects sorted by confidence.
        """
        matches = []
        request_lower = user_request.lower()

        for intent_name, intent_data in self.INTENT_PATTERNS.items():
            if available_tools and intent_data["tool"] not in available_tools:
                continue

            confidence = 0.0
            matched_patterns = []

            for i, pattern in enumerate(self._compiled[intent_name]):
                if pattern.search(request_lower):
                    confidence += 1.0 / (i + 1)
                    matched_patterns.append(intent_data["patterns"][i])

            if confidence > 0:
                confidence = min(confidence / 2.0, 1.0)

                args = dict(intent_data["default_args"])

                if intent_name == "press_key":
                    key_match = re.search(
                        r"press\s*(?:the\s*)?(\w+)\s*key",
                        request_lower,
                    )
                    if key_match:
                        args["key"] = key_match.group(1).title()

                if intent_name in ("file_read", "file_write"):
                    path_match = re.search(
                        r"(?:file|path)\s*[:=]\s*(\S+)",
                        request_lower,
                    )
                    if path_match:
                        args["path"] = path_match.group(1)

                if intent_name == "vision_analyze":
                    path_match = re.search(
                        r"(?:image|screenshot|picture|photo)\s*[:=]\s*(\S+)",
                        request_lower,
                    )
                    if path_match:
                        args["image_path"] = path_match.group(1)

                if intent_name == "browser_navigate":
                    url_match = re.search(
                        r"(https?://[^\s]+|www\.[^\s]+)",
                        request_lower,
                    )
                    if url_match:
                        args["url"] = url_match.group(1)
                        if args["url"].startswith("www."):
                            args["url"] = "https://" + args["url"]

                matches.append(ToolMatch(
                    tool_name=intent_data["tool"],
                    action=intent_data["action"],
                    confidence=confidence,
                    reasoning=f"Matched patterns: {matched_patterns}",
                    arguments=args,
                ))

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches


class ToolSelector:
    """Selects the best tool for a user request."""

    def __init__(self):
        self._classifier = IntentClassifier()

    def select(
        self,
        user_request: str,
        available_tools: Optional[List[str]] = None,
        min_confidence: float = 0.1,
    ) -> Optional[ToolMatch]:
        """Select the best tool for a user request.

        Args:
            user_request: The user's natural language request.
            available_tools: Optional list of available tool names.
            min_confidence: Minimum confidence threshold.

        Returns:
            Best ToolMatch or None if no match found.
        """
        matches = self._classifier.classify(user_request, available_tools)

        if not matches:
            return None

        if matches[0].confidence < min_confidence:
            return None

        return matches[0]

    def select_all(
        self,
        user_request: str,
        available_tools: Optional[List[str]] = None,
        min_confidence: float = 0.1,
    ) -> List[ToolMatch]:
        """Select all matching tools above confidence threshold."""
        matches = self._classifier.classify(user_request, available_tools)
        return [m for m in matches if m.confidence >= min_confidence]
