"""Tool registry for managing available tools."""

from typing import Dict, List, Optional, Any
from .base import Tool


class ToolRegistry:
    """Registry for managing tool instances."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> bool:
        if tool.name in self._tools:
            return False
        self._tools[tool.name] = tool
        return True

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_dict() for tool in self._tools.values()]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def count(self) -> int:
        return len(self._tools)

    def clear(self):
        self._tools.clear()
