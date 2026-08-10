"""Local LLM provider using llama-cpp-python."""

import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import LLMProvider, LLMConfig, LLMResponse, LLMProviderType

logger = logging.getLogger(__name__)


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using llama-cpp-python for GGUF models.
    
    This provider loads and runs GGUF models locally using
    llama-cpp-python with optional CUDA GPU acceleration.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the local LLM provider.
        
        Args:
            config: Configuration for the local provider.
        """
        super().__init__(config)
        self._llama = None
        self._model_path: Optional[Path] = None
    
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
            
            logger.info(f"Loading model: {self._model_path}")
            logger.info(f"Context length: {self.config.context_length}")
            logger.info(f"GPU layers: {self.config.n_gpu_layers}")
            
            # Load the model
            self._llama = Llama(
                model_path=str(self._model_path),
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                verbose=self.config.verbose,
                embedding=True,  # Enable embeddings for future use
            )
            
            self._is_initialized = True
            logger.info("Model loaded successfully")
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
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from a conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
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
                metadata={"elapsed_seconds": elapsed}
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
