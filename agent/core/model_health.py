"""Model Health Status for Rose.

Stage B - Single reliable contract for Qwen2.5-VL model health.

Provides:
- Model name and paths
- Existence validation
- Runtime availability
- Vision capability status
- GPU availability
- Backend status
- Actionable error messages
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelHealthStatus:
    """Comprehensive model health status."""
    model_name: str = ""
    model_path: str = ""
    mmproj_path: str = ""
    model_exists: bool = False
    mmproj_exists: bool = False
    model_size_mb: float = 0.0
    mmproj_size_mb: float = 0.0
    runtime_available: bool = False
    vision_available: bool = False
    gpu_available: bool = False
    gpu_layers: int = 0
    context_length: int = 0
    backend_status: str = "unknown"
    initialization_time: float = 0.0
    last_error: str = ""
    is_loaded: bool = False
    vision_capability: str = "none"
    max_images: int = 0
    mmproj_validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "model_exists": self.model_exists,
            "mmproj_exists": self.mmproj_exists,
            "model_size_mb": round(self.model_size_mb, 2),
            "mmproj_size_mb": round(self.mmproj_size_mb, 2),
            "runtime_available": self.runtime_available,
            "vision_available": self.vision_available,
            "gpu_available": self.gpu_available,
            "gpu_layers": self.gpu_layers,
            "context_length": self.context_length,
            "backend_status": self.backend_status,
            "initialization_time": round(self.initialization_time, 3),
            "last_error": self.last_error,
            "is_loaded": self.is_loaded,
            "vision_capability": self.vision_capability,
            "max_images": self.max_images,
            "mmproj_validated": self.mmproj_validated,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if model is in a healthy state."""
        return (
            self.model_exists
            and self.runtime_available
            and self.is_loaded
            and not self.last_error
        )

    @property
    def status_summary(self) -> str:
        """Get a human-readable status summary."""
        if self.is_healthy:
            vision_info = f", Vision: {self.vision_capability}" if self.vision_available else ""
            gpu_info = f", GPU: {self.gpu_layers} layers" if self.gpu_available else ", CPU"
            return f"READY ({self.model_name}{vision_info}{gpu_info})"
        elif self.last_error:
            return f"ERROR: {self.last_error}"
        elif not self.model_exists:
            return "NOT FOUND: Model file missing"
        elif not self.runtime_available:
            return "UNAVAILABLE: Runtime not loaded"
        else:
            return "UNKNOWN"


class ModelHealthChecker:
    """Checks and reports model health status."""

    def __init__(self, config=None):
        self._config = config
        self._status = ModelHealthStatus()
        self._check_history: list = []

    def check_health(self, llm_provider=None) -> ModelHealthStatus:
        """Perform a comprehensive health check.
        
        Args:
            llm_provider: Optional LLM provider to check
            
        Returns:
            ModelHealthStatus with current health information
        """
        start = time.time()
        self._status = ModelHealthStatus()
        
        # Load config if available
        if self._config:
            self._status.model_name = getattr(self._config, 'model_name', '')
            self._status.model_path = str(getattr(self._config, 'model_path', ''))
            self._status.mmproj_path = str(getattr(self._config, 'mmproj_path', ''))
            self._status.gpu_layers = getattr(self._config, 'llm_gpu_layers', 0)
            self._status.context_length = getattr(self._config, 'model_context_length', 0)
        
        # Validate model file exists
        if self._status.model_path:
            model_path = Path(self._status.model_path)
            if not model_path.is_absolute():
                model_path = Path.cwd() / model_path
            self._status.model_exists = model_path.exists()
            if self._status.model_exists:
                self._status.model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Validate mmproj file exists
        if self._status.mmproj_path:
            mmproj_path = Path(self._status.mmproj_path)
            if not mmproj_path.is_absolute():
                mmproj_path = Path.cwd() / mmproj_path
            self._status.mmproj_exists = mmproj_path.exists()
            if self._status.mmproj_exists:
                self._status.mmproj_size_mb = mmproj_path.stat().st_size / (1024 * 1024)
                # Validate mmproj is a valid file (basic size check)
                if self._status.mmproj_size_mb > 100:
                    self._status.mmproj_validated = True
                else:
                    self._status.last_error = f"mmproj file too small ({self._status.mmproj_size_mb:.1f} MB)"
        
        # Check runtime availability
        try:
            import llama_cpp
            self._status.runtime_available = True
            self._status.backend_status = "llama-cpp-python available"
        except ImportError:
            self._status.runtime_available = False
            self._status.backend_status = "llama-cpp-python not installed"
        
        # Check GPU availability
        self._status.gpu_available = self._check_gpu()
        
        # Check LLM provider health
        if llm_provider:
            try:
                provider_health = llm_provider.health_check()
                self._status.is_loaded = provider_health.get("initialized", False)
                self._status.vision_available = provider_health.get("vision_capable", False)
                self._status.vision_capability = provider_health.get("vision_capability", "none")
                
                if self._status.vision_available and self._status.mmproj_exists:
                    self._status.max_images = 4
            except Exception as e:
                self._status.last_error = f"Provider health check failed: {str(e)}"
        
        self._status.initialization_time = time.time() - start
        self._check_history.append(self._status.to_dict())
        
        return self._status

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def get_status(self) -> ModelHealthStatus:
        """Get current health status."""
        return self._status

    def get_check_history(self) -> list:
        """Get history of health checks."""
        return self._check_history.copy()
