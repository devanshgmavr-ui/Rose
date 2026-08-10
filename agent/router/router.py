"""Message routing for the local agent."""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages the agent can handle."""
    TEXT = "text"
    COMMAND = "command"
    QUERY = "query"
    SYSTEM = "system"


class Router:
    """Routes messages to appropriate handlers.
    
    This router will be expanded in later stages to handle
    different input/output types.
    """
    
    def __init__(self):
        """Initialize the router."""
        self._handlers = {}
    
    def register_handler(self, message_type: MessageType, handler):
        """Register a handler for a message type."""
        self._handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type.value}")
    
    def route(self, message_type: MessageType, content: str, **kwargs) -> Any:
        """Route a message to its handler."""
        if message_type not in self._handlers:
            logger.warning(f"No handler for message type: {message_type}")
            return None
        
        handler = self._handlers[message_type]
        return handler(content, **kwargs)
