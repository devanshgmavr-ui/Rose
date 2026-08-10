"""Enhanced tool catalog for natural language planning.

Stage 4.1 - Natural Language Tool Planning.

Provides rich tool metadata for the planner including:
- Tool descriptions
- Argument schemas
- Permission metadata
- Expected outputs
- Failure information
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Rich metadata for a tool."""
    name: str
    description: str
    category: str
    actions: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    permissions: List[str]
    confirmation_required: bool
    timeout: float
    failure_modes: List[str]
    examples: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "actions": self.actions,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": self.permissions,
            "confirmation_required": self.confirmation_required,
            "timeout": self.timeout,
            "failure_modes": self.failure_modes,
            "examples": self.examples,
        }


def build_tool_catalog() -> Dict[str, ToolMetadata]:
    """Build a catalog of all available tools with rich metadata."""
    catalog = {}

    catalog["filesystem"] = ToolMetadata(
        name="filesystem",
        description="Read, write, list, and manage files in the workspace",
        category="filesystem",
        actions=["list", "read", "write", "copy", "move", "delete", "search", "info"],
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "read", "write", "copy", "move", "delete", "search", "info"]},
                "path": {"type": "string"},
                "content": {"type": "string"},
                "destination": {"type": "string"},
                "pattern": {"type": "string"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "files": {"type": "array"},
                "success": {"type": "boolean"},
            },
        },
        permissions=["filesystem.read", "filesystem.write"],
        confirmation_required=False,
        timeout=30.0,
        failure_modes=["file not found", "permission denied", "disk full"],
        examples=[
            {"request": "List files in workspace", "action": "list", "path": "."},
            {"request": "Read a file", "action": "read", "path": "document.txt"},
            {"request": "Write a file", "action": "write", "path": "output.txt", "content": "data"},
        ],
    )

    catalog["python_sandbox"] = ToolMetadata(
        name="python_sandbox",
        description="Execute Python code safely in a sandboxed environment",
        category="code",
        actions=["execute"],
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {"type": "string"},
                "error": {"type": "string"},
                "success": {"type": "boolean"},
            },
        },
        permissions=["code.execute"],
        confirmation_required=True,
        timeout=60.0,
        failure_modes=["syntax error", "runtime error", "timeout", "import error"],
        examples=[
            {"request": "Calculate something", "code": "print(2 + 2)"},
            {"request": "Process data", "code": "import json; print(json.dumps([1,2,3]))"},
        ],
    )

    catalog["screen_capture"] = ToolMetadata(
        name="screen_capture",
        description="Capture screenshots of the desktop",
        category="os_control",
        actions=["capture"],
        input_schema={
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": ["full", "active_window", "region"]},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
            },
        },
        permissions=["os.screen_capture"],
        confirmation_required=False,
        timeout=10.0,
        failure_modes=["capture failed", "permission denied"],
        examples=[
            {"request": "Take a screenshot", "region": "full"},
            {"request": "Capture active window", "region": "active_window"},
        ],
    )

    catalog["system_info"] = ToolMetadata(
        name="system_info",
        description="Get system information (OS, CPU, memory, screen)",
        category="os_control",
        actions=["info"],
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {
                "os": {"type": "string"},
                "cpu": {"type": "string"},
                "memory": {"type": "object"},
                "screen": {"type": "object"},
            },
        },
        permissions=["os.system_info"],
        confirmation_required=False,
        timeout=5.0,
        failure_modes=[],
        examples=[{"request": "Get system info"}],
    )

    catalog["mouse"] = ToolMetadata(
        name="mouse",
        description="Control mouse movement and clicks",
        category="os_control",
        actions=["move", "click", "double_click", "right_click", "scroll"],
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["move", "click", "double_click", "right_click", "scroll"]},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "scroll_amount": {"type": "integer"},
            },
            "required": ["action", "x", "y"],
        },
        output_schema={"type": "object", "properties": {"success": {"type": "boolean"}}},
        permissions=["os.mouse"],
        confirmation_required=True,
        timeout=5.0,
        failure_modes=["out of bounds", "action blocked", "timeout"],
        examples=[
            {"request": "Click at coordinates", "action": "click", "x": 100, "y": 200},
            {"request": "Move mouse", "action": "move", "x": 500, "y": 300},
        ],
    )

    catalog["keyboard"] = ToolMetadata(
        name="keyboard",
        description="Type text and press keys",
        category="os_control",
        actions=["type", "press", "hotkey"],
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["type", "press", "hotkey"]},
                "text": {"type": "string"},
                "key": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["action"],
        },
        output_schema={"type": "object", "properties": {"success": {"type": "boolean"}}},
        permissions=["os.keyboard"],
        confirmation_required=True,
        timeout=5.0,
        failure_modes=["text too long", "blocked shortcut", "timeout"],
        examples=[
            {"request": "Type text", "action": "type", "text": "Hello"},
            {"request": "Press Enter", "action": "press", "key": "Enter"},
        ],
    )

    catalog["window"] = ToolMetadata(
        name="window",
        description="Manage windows (list, activate, minimize, maximize, close)",
        category="os_control",
        actions=["list", "get_active", "activate", "minimize", "restore", "maximize", "close", "move", "resize"],
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "window_id": {"type": "string"},
                "title": {"type": "string"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
            },
            "required": ["action"],
        },
        output_schema={"type": "object", "properties": {"windows": {"type": "array"}, "success": {"type": "boolean"}}},
        permissions=["os.window"],
        confirmation_required=False,
        timeout=10.0,
        failure_modes=["window not found", "permission denied", "protected window"],
        examples=[
            {"request": "List windows", "action": "list"},
            {"request": "Get active window", "action": "get_active"},
            {"request": "Activate a window", "action": "activate", "title": "Calculator"},
        ],
    )

    catalog["browser"] = ToolMetadata(
        name="browser",
        description="Control browser sessions, navigate, read pages, interact with elements, take screenshots",
        category="browser",
        actions=[
            "create_session", "list_sessions", "close_session",
            "navigate", "read_page", "inspect", "click", "fill", "select", "press", "wait",
            "screenshot",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "session_id": {"type": "string"},
                "url": {"type": "string"},
                "selector": {"type": "string"},
                "text": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["action"],
        },
        output_schema={"type": "object", "properties": {"result": {"type": "string"}, "success": {"type": "boolean"}}},
        permissions=[
            "browser.session", "browser.navigation", "browser.page_read",
            "browser.inspect", "browser.interact", "browser.screenshot",
        ],
        confirmation_required=True,
        timeout=30.0,
        failure_modes=[
            "session not found", "navigation failed", "element not found",
            "interaction failed", "timeout",
        ],
        examples=[
            {"request": "Open a browser", "action": "create_session"},
            {"request": "Navigate to a URL", "action": "navigate", "url": "https://example.com"},
            {"request": "Take a screenshot", "action": "screenshot"},
        ],
    )

    catalog["vision_analyze"] = ToolMetadata(
        name="vision_analyze",
        description="Analyze images and screenshots to understand visual content",
        category="vision",
        actions=["analyze", "describe"],
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["analyze", "describe"]},
                "image_path": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["image_path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "detected_elements": {"type": "array"},
                "image_width": {"type": "integer"},
                "image_height": {"type": "integer"},
            },
        },
        permissions=["vision.analyze"],
        confirmation_required=True,
        timeout=60.0,
        failure_modes=["file not found", "unsupported format", "image too large", "provider unavailable"],
        examples=[
            {"request": "Analyze this screenshot", "action": "analyze", "image_path": "screenshot.png"},
            {"request": "Describe an image", "action": "describe", "image_path": "photo.jpg"},
        ],
    )

    catalog["visual_ground"] = ToolMetadata(
        name="visual_ground",
        description="Ground vision results into actionable coordinates and targets",
        category="vision",
        actions=["ground", "validate"],
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["ground", "validate"]},
                "image_path": {"type": "string"},
                "target": {"type": "string"},
                "screen_width": {"type": "integer"},
                "screen_height": {"type": "integer"},
            },
            "required": ["image_path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "targets": {"type": "array"},
                "target_count": {"type": "integer"},
            },
        },
        permissions=["vision.analyze"],
        confirmation_required=True,
        timeout=60.0,
        failure_modes=["no targets found", "low confidence", "outside screen bounds"],
        examples=[
            {"request": "Find the submit button", "action": "ground", "target": "submit"},
            {"request": "Validate coordinates", "action": "validate", "image_path": "screenshot.png"},
        ],
    )

    catalog["image_analyze"] = ToolMetadata(
        name="image_analyze",
        description="Analyze an image to understand its content and objects",
        category="media",
        actions=["analyze"],
        input_schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["image_path"],
        },
        output_schema={"type": "object", "properties": {"description": {"type": "string"}}},
        permissions=["filesystem.read"],
        confirmation_required=False,
        timeout=60.0,
        failure_modes=["file not found", "unsupported format"],
        examples=[{"request": "What's in this image?", "image_path": "photo.png"}],
    )

    return catalog


def get_tools_for_request(
    user_request: str,
    catalog: Optional[Dict[str, ToolMetadata]] = None,
) -> List[ToolMetadata]:
    """Select relevant tools based on user request.

    Args:
        user_request: The user's natural language request.
        catalog: Optional pre-built catalog.

    Returns:
        List of potentially relevant ToolMetadata objects.
    """
    if catalog is None:
        catalog = build_tool_catalog()

    request_lower = user_request.lower()
    relevant = []

    keyword_map = {
        "filesystem": ["file", "read", "write", "save", "create", "list", "directory", "folder", "copy", "move", "delete", "rename", "search"],
        "python_sandbox": ["code", "python", "execute", "script", "program", "calculate", "compute"],
        "screen_capture": ["screenshot", "screen", "capture", "photo", "picture", "image of screen"],
        "system_info": ["system", "info", "cpu", "memory", "disk", "os", "hardware"],
        "mouse": ["click", "mouse", "cursor", "pointer", "move mouse", "scroll"],
        "keyboard": ["type", "keyboard", "press", "key", "hotkey", "shortcut", "text input"],
        "window": ["window", "minimize", "maximize", "restore", "activate", "focus", "close window"],
        "browser": ["browser", "chrome", "firefox", "edge", "navigate", "website", "web page", "url", "search online", "browse"],
        "vision_analyze": ["analyze image", "what's in", "describe image", "look at", "understand image", "visual"],
        "visual_ground": ["find button", "locate", "coordinates", "where is", "find element", "click on"],
        "image_analyze": ["image", "picture", "photo", "analyze picture"],
    }

    for tool_name, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in request_lower:
                if tool_name in catalog:
                    relevant.append(catalog[tool_name])
                break

    if not relevant:
        for tool_name, meta in catalog.items():
            relevant.append(meta)

    return relevant[:5]
