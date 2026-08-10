"""Base worker interface for future GPU management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum


class WorkerStatus(Enum):
    """Status of a worker."""
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class GPUInfo:
    """Information about GPU resources."""
    name: str
    total_memory: int  # bytes
    used_memory: int  # bytes
    free_memory: int  # bytes


class Worker(ABC):
    """Abstract base class for GPU workers.
    
    This interface will be used in later stages for
    managing GPU memory and model loading.
    """
    
    @property
    @abstractmethod
    def status(self) -> WorkerStatus:
        """Worker status."""
        pass
    
    @abstractmethod
    def get_gpu_info(self) -> GPUInfo:
        """Get GPU information."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load a model into GPU memory."""
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        """Unload model from GPU memory."""
        pass
