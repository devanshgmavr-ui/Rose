"""Local LLM provider using llama-cpp-python.

Supports both text-only models and Vision-Language (VL) models
like Qwen2.5-VL via Llava16ChatHandler.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import LLMProvider, LLMConfig, LLMResponse, LLMProviderType, VisionCapability, ImageInput

logger = logging.getLogger(__name__)


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using llama-cpp-python for GGUF models.
    
    Supports text-only models and Vision-Language (VL) models.
    When mmproj_path is provided, uses Llava16ChatHandler for
    multimodal inference (e.g., Qwen2.5-VL).
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the local LLM provider.
        
        Args:
            config: Configuration for the local provider.
        """
        super().__init__(config)
        self._llama = None
        self._model_path: Optional[Path] = None
        self._mmproj_path: Optional[Path] = None
        self._is_vl_model = False
    
    def initialize(self) -> bool:
        """Initialize the local LLM and load the model.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            from llama_cpp import Llama
            
            # Validate model path
            if not self.config.model_path:
                logger.error("No model path specified")
                return False
            
            self._model_path = Path(self.config.model_path)
            if not self._model_path.exists():
                logger.error(f"Model file not found: {self._model_path}")
                return False
            
            # Check for vision model (mmproj)
            if self.config.mmproj_path:
                self._mmproj_path = Path(self.config.mmproj_path)
                if self._mmproj_path.exists():
                    self._is_vl_model = True
                    logger.info(f"Vision model detected, mmproj: {self._mmproj_path}")
                else:
                    logger.warning(f"mmproj file not found: {self._mmproj_path}, falling back to text-only")
                    self._mmproj_path = None
            
            logger.info(f"Loading model: {self._model_path}")
            logger.info(f"Context length: {self.config.context_length}")
            logger.info(f"GPU layers: {self.config.n_gpu_layers}")
            
            # Load the model with optional VL support
            if self._is_vl_model and self._mmproj_path:
                from llama_cpp.llama_chat_format import Llava16ChatHandler
                
                logger.info("Initializing with Llava16ChatHandler for VL model")
                chat_handler = Llava16ChatHandler(
                    clip_model_path=str(self._mmproj_path),
                    verbose=self.config.verbose,
                )
                self._llama = Llama(
                    model_path=str(self._model_path),
                    chat_handler=chat_handler,
                    n_ctx=self.config.context_length,
                    n_gpu_layers=self.config.n_gpu_layers,
                    n_batch=self.config.n_batch,
                    verbose=self.config.verbose,
                    logits_all=True,  # Required for VL models
                )
                # Auto-detect vision capability
                if self.config.vision_capability == VisionCapability.NONE:
                    self.config.vision_capability = VisionCapability.MULTIPLE
                    self.config.max_images = 4
                    logger.info("Auto-detected vision capability: MULTIPLE")
            else:
                self._llama = Llama(
                    model_path=str(self._model_path),
                    n_ctx=self.config.context_length,
                    n_gpu_layers=self.config.n_gpu_layers,
                    n_batch=self.config.n_batch,
                    verbose=self.config.verbose,
                    embedding=True,
                )
            
            self._is_initialized = True
            model_type = "Vision-Language (Qwen2.5-VL)" if self._is_vl_model else "Text-only"
            logger.info(f"Model loaded successfully ({model_type})")
            return True
            
        except ImportError as e:
            logger.error(f"llama-cpp-python not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            LLMResponse containing the generated text.
            
        Raises:
            RuntimeError: If provider not initialized.
        """
        if not self._is_initialized or self._llama is None:
            raise RuntimeError("LLM provider not initialized. Call initialize() first.")
        
        # Merge config with overrides
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        repeat_penalty = kwargs.get("repeat_penalty", self.config.repeat_penalty)
        
        start_time = time.time()
        
        try:
            response = self._llama(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                echo=False,
            )
            
            elapsed = time.time() - start_time
            
            # Extract response text
            text = response["choices"][0]["text"]
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            tokens_prompt = response.get("usage", {}).get("prompt_tokens", 0)
            tokens_completion = response.get("usage", {}).get("completion_tokens", 0)
            finish_reason = response["choices"][0].get("finish_reason", "stop")
            
            logger.debug(f"Generation completed in {elapsed:.2f}s, {tokens_used} tokens")
            
            return LLMResponse(
                text=text,
                model=self.config.model_name or "unknown",
                tokens_used=tokens_used,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                finish_reason=finish_reason,
                metadata={"elapsed_seconds": elapsed}
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Generate a response from a conversation.
        
        For VL models, messages can include image content:
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
            
        Raises:
            RuntimeError: If provider not initialized.
        """
        if not self._is_initialized or self._llama is None:
            raise RuntimeError("LLM provider not initialized. Call initialize() first.")
        
        # Merge config with overrides
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        repeat_penalty = kwargs.get("repeat_penalty", self.config.repeat_penalty)
        
        start_time = time.time()
        
        try:
            response = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )
            
            elapsed = time.time() - start_time
            
            # Extract response text
            text = response["choices"][0]["message"]["content"]
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            tokens_prompt = response.get("usage", {}).get("prompt_tokens", 0)
            tokens_completion = response.get("usage", {}).get("completion_tokens", 0)
            finish_reason = response["choices"][0].get("finish_reason", "stop")
            
            logger.debug(f"Chat completed in {elapsed:.2f}s, {tokens_used} tokens")
            
            return LLMResponse(
                text=text,
                model=self.config.model_name or "unknown",
                tokens_used=tokens_used,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                finish_reason=finish_reason,
                metadata={"elapsed_seconds": elapsed, "vl_model": self._is_vl_model}
            )
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the local LLM.
        
        Returns:
            Dictionary with health status information.
        """
        status = {
            "provider": "local",
            "initialized": self._is_initialized,
            "model_path": str(self._model_path) if self._model_path else None,
            "model_exists": self._model_path.exists() if self._model_path else False,
            "cuda_available": False,
            "gpu_info": None,
            "vision_capable": self._is_vl_model,
            "vision_capability": self.config.vision_capability.value if self._is_vl_model else "none",
            "mmproj_path": str(self._mmproj_path) if self._mmproj_path else None,
            "mmproj_exists": self._mmproj_path.exists() if self._mmproj_path else False,
        }
        
        # Check CUDA availability via nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 2:
                    status["cuda_available"] = True
                    status["gpu_info"] = {
                        "name": parts[0].strip(),
                        "memory_total_mb": int(float(parts[1].strip())),
                    }
        except Exception:
            pass
        
        # Check llama-cpp-python CUDA support
        try:
            from llama_cpp import Llama
            status["llama_cpp_installed"] = True
        except Exception:
            status["llama_cpp_installed"] = False
        
        # Check for ggml-cuda.dll
        try:
            import llama_cpp
            lib_dir = Path(llama_cpp.__file__).parent / "lib"
            cuda_dll = lib_dir / "ggml-cuda.dll"
            status["cuda_backend"] = cuda_dll.exists()
        except Exception:
            status["cuda_backend"] = False
        
        return status
    
    def model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        if not self._is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "model_name": self.config.model_name,
            "model_path": str(self._model_path),
            "mmproj_path": str(self._mmproj_path) if self._mmproj_path else None,
            "is_vl_model": self._is_vl_model,
            "vision_capability": self.config.vision_capability.value,
            "max_images": self.config.max_images,
            "context_length": self.config.context_length,
            "gpu_layers": self.config.n_gpu_layers,
            "status": "loaded",
        }
    
    def unload(self) -> bool:
        """Unload the model to free resources.
        
        Returns:
            True if unload successful, False otherwise.
        """
        try:
            if self._llama is not None:
                del self._llama
                self._llama = None
            
            self._is_initialized = False
            logger.info("Model unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False
