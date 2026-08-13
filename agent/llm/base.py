"""Abstract base class for LLM providers.

Supports text-only and multimodal (Vision-Language) inference.
Providers declare their vision capabilities, and the rest of Rose
depends on this interface rather than Qwen-specific APIs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum


class LLMProviderType(Enum):
    """Types of LLM providers."""
    LOCAL = "local"
    CLOUD = "cloud"
    API = "api"


class VisionCapability(Enum):
    """Vision capability levels for LLM providers."""
    NONE = "none"           # Text-only, no image understanding
    BASIC = "basic"         # Can process single images
    MULTIPLE = "multiple"   # Can process multiple images
    NATIVE = "native"       # Full native vision understanding (e.g., Qwen2.5-VL)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    model: str
    tokens_used: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.tokens_prompt + self.tokens_completion


@dataclass
class ImageInput:
    """An image input for multimodal inference.
    
    Supports file paths, base64 data URIs, and raw bytes.
    """
    source: str  # file path, data URI, or URL
    media_type: str = "image/png"  # MIME type
    description: Optional[str] = None  # Optional description for the image
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, file_path: str, description: Optional[str] = None) -> "ImageInput":
        """Create an ImageInput from a file path."""
        from pathlib import Path
        path = Path(file_path)
        suffix = path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif"}
        return cls(source=str(path), media_type=mime_map.get(suffix, "image/png"),
                   description=description)
    
    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png",
                    description: Optional[str] = None) -> "ImageInput":
        """Create an ImageInput from base64 data (without data URI prefix)."""
        return cls(source=f"data:{media_type};base64,{data}", media_type=media_type,
                   description=description)
    
    def to_llm_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible image_url format for chat completion."""
        return {"type": "image_url", "image_url": {"url": self.source}}


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider_type: LLMProviderType = LLMProviderType.LOCAL
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    context_length: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 0
    n_batch: int = 512
    verbose: bool = False
    # Vision-Language model support
    mmproj_path: Optional[str] = None  # Path to mmproj F16 file for VL models
    logits_all: bool = False  # Required for some VL models
    # Vision configuration
    vision_capability: VisionCapability = VisionCapability.NONE
    max_images: int = 1  # Maximum images per request


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers must implement these methods.
    This allows the agent to work with different providers
    (local, cloud, API) without modification.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM provider.
        
        Args:
            config: Configuration for the provider.
        """
        self.config = config
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the provider and load model.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            LLMResponse containing the generated text.
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Generate a response from a conversation.
        
        For multimodal providers, messages can include image content:
        [{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}]
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
                      Content can be string (text) or list (multimodal).
            **kwargs: Additional generation parameters.
            
        Returns:
            LLMResponse containing the generated text.
        """
        pass
    
    def chat_with_images(self, text: str, images: List[ImageInput],
                         system_prompt: Optional[str] = None,
                         **kwargs) -> LLMResponse:
        """Convenience method for multimodal chat with images.
        
        Builds the appropriate message format and calls chat().
        Providers that don't support images will raise NotImplementedError.
        
        Args:
            text: The text prompt.
            images: List of ImageInput objects.
            system_prompt: Optional system prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            LLMResponse containing the generated text.
            
        Raises:
            NotImplementedError: If provider doesn't support vision.
        """
        if not self.supports_vision:
            raise NotImplementedError(
                f"Provider {self.config.provider_type.value} does not support vision. "
                "Use chat() with text-only messages instead."
            )
        
        # Build multimodal message content
        content: List[Dict[str, Any]] = []
        
        # Add images first (before text, as per VL model convention)
        for img in images[:self.config.max_images]:
            content.append(img.to_llm_format())
        
        # Add text
        content.append({"type": "text", "text": text})
        
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        return self.chat(messages, **kwargs)
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the provider.
        
        Returns:
            Dictionary with health status information.
        """
        pass
    
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        pass
    
    @abstractmethod
    def unload(self) -> bool:
        """Unload the model to free resources.
        
        Returns:
            True if unload successful, False otherwise.
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._is_initialized
    
    @property
    def supports_vision(self) -> bool:
        """Check if provider supports vision/multimodal input."""
        return self.config.vision_capability != VisionCapability.NONE
    
    @property
    def vision_capability(self) -> VisionCapability:
        """Get the vision capability level."""
        return self.config.vision_capability
    
    @property
    def max_images_per_request(self) -> int:
        """Get maximum images per request."""
        return self.config.max_images
