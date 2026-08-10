"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class LLMProviderType(Enum):
    """Types of LLM providers."""
    LOCAL = "local"
    CLOUD = "cloud"
    API = "api"


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
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from a conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters.
            
        Returns:
            LLMResponse containing the generated text.
        """
        pass
    
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
