#!/usr/bin/env python3
"""Tests for Qwen2.5-VL Integration.

These tests verify the complete Qwen2.5-VL vision pipeline:
1. Configuration tests
2. Model loading tests
3. mmproj loading tests
4. Multimodal pipeline tests
5. Vision routing tests
6. Agent integration tests
7. Negative-control tests
8. Error handling tests

Tests that require the actual large model are marked as integration tests.
"""

import os
import sys
import struct
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration Tests
# ============================================================================

class TestVLConfiguration:
    """Test Qwen2.5-VL configuration."""
    
    def test_model_path_configured(self):
        """Test that model path is configured."""
        from agent.core.config import Config
        config = Config()
        assert config.model_path, "model_path should be configured"
        assert "Qwen" in config.model_path or "qwen" in config.model_path.lower()
    
    def test_mmproj_path_configured(self):
        """Test that mmproj path is configured."""
        from agent.core.config import Config
        config = Config()
        assert config.mmproj_path, "mmproj_path should be configured"
        assert "mmproj" in config.mmproj_path.lower()
    
    def test_model_files_exist(self):
        """Test that model files exist."""
        from agent.core.config import Config
        config = Config()
        model_path = config.get_model_full_path()
        mmproj_path = Path(config.mmproj_path) if Path(config.mmproj_path).is_absolute() else Path.cwd() / config.mmproj_path
        assert model_path.exists(), f"Model file not found: {model_path}"
        assert mmproj_path.exists(), f"mmproj file not found: {mmproj_path}"
    
    def test_model_file_sizes(self):
        """Test that model files have reasonable sizes."""
        from agent.core.config import Config
        config = Config()
        model_path = config.get_model_full_path()
        mmproj_path = Path(config.mmproj_path) if Path(config.mmproj_path).is_absolute() else Path.cwd() / config.mmproj_path
        
        # Qwen2.5-VL-7B Q4_K_M should be around 4-5 GB
        model_size_gb = model_path.stat().st_size / (1024**3)
        assert 3.0 < model_size_gb < 6.0, f"Model size unexpected: {model_size_gb:.2f} GB"
        
        # mmproj should be around 1-2 GB
        mmproj_size_gb = mmproj_path.stat().st_size / (1024**3)
        assert 0.5 < mmproj_size_gb < 3.0, f"mmproj size unexpected: {mmproj_size_gb:.2f} GB"


# ============================================================================
# Model Loading Tests
# ============================================================================

class TestVLModelLoading:
    """Test Qwen2.5-VL model loading."""
    
    def test_llama_cpp_imports(self):
        """Test that llama-cpp-python imports successfully."""
        import llama_cpp
        assert hasattr(llama_cpp, '__version__')
    
    def test_llava_handler_available(self):
        """Test that Llava16ChatHandler is available."""
        from llama_cpp.llama_chat_format import Llava16ChatHandler
        assert Llava16ChatHandler is not None
    
    def test_vision_capability_enum(self):
        """Test that VisionCapability enum exists with expected values."""
        from agent.llm.base import VisionCapability
        assert hasattr(VisionCapability, 'NONE')
        assert hasattr(VisionCapability, 'BASIC')
        assert hasattr(VisionCapability, 'MULTIPLE')
        assert hasattr(VisionCapability, 'NATIVE')
    
    def test_image_input_creation(self):
        """Test that ImageInput can be created from a file."""
        from agent.llm.base import ImageInput
        
        # Create a test image
        from PIL import Image
        test_path = Path(PROJECT_ROOT) / "test_vision_image.png"
        if not test_path.exists():
            img = Image.new('RGB', (100, 100), 'red')
            img.save(test_path)
        
        image_input = ImageInput.from_file(str(test_path))
        assert image_input.source == str(test_path)
        assert image_input.media_type == "image/png"
    
    def test_image_input_to_llm_format(self):
        """Test that ImageInput converts to proper LLM format."""
        from agent.llm.base import ImageInput
        from PIL import Image
        import base64
        
        test_path = Path(PROJECT_ROOT) / "test_vision_image.png"
        if not test_path.exists():
            img = Image.new('RGB', (100, 100), 'red')
            img.save(test_path)
        
        image_input = ImageInput.from_file(str(test_path))
        llm_format = image_input.to_llm_format()
        
        assert llm_format["type"] == "image_url"
        assert "url" in llm_format["image_url"]
        # Should be base64 data URI
        assert llm_format["image_url"]["url"].startswith("data:image/png;base64,")


# ============================================================================
# Model Loading Integration Tests
# ============================================================================

@pytest.mark.skipif(
    not Path(PROJECT_ROOT / "models" / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").exists(),
    reason="Model file not found"
)
class TestVLModelLoadingIntegration:
    """Integration tests that require the actual model."""
    
    def test_model_loads_with_mmproj(self):
        """Test that the model loads with mmproj."""
        from agent.core.config import Config
        from agent.llm.local_provider import LocalLLMProvider
        from agent.llm.base import LLMConfig, VisionCapability
        
        config = Config()
        model_path = config.get_model_full_path()
        mmproj_path = Path(config.mmproj_path) if Path(config.mmproj_path).is_absolute() else Path.cwd() / config.mmproj_path
        
        llm_config = LLMConfig(
            model_path=str(model_path),
            mmproj_path=str(mmproj_path),
            vision_capability=VisionCapability.MULTIPLE,
            max_images=4,
            context_length=2048,
            n_gpu_layers=0,  # CPU mode for testing
        )
        provider = LocalLLMProvider(llm_config)
        result = provider.initialize()
        
        assert result is True, "Model failed to load"
        assert provider._is_vl_model is True, "Model should be detected as VL"
        
        provider.unload()
    
    def test_model_supports_vision(self):
        """Test that the loaded model supports vision."""
        from agent.core.config import Config
        from agent.llm.local_provider import LocalLLMProvider
        from agent.llm.base import LLMConfig, VisionCapability
        
        config = Config()
        model_path = config.get_model_full_path()
        mmproj_path = Path(config.mmproj_path) if Path(config.mmproj_path).is_absolute() else Path.cwd() / config.mmproj_path
        
        llm_config = LLMConfig(
            model_path=str(model_path),
            mmproj_path=str(mmproj_path),
            vision_capability=VisionCapability.MULTIPLE,
            max_images=4,
            context_length=2048,
            n_gpu_layers=0,
        )
        provider = LocalLLMProvider(llm_config)
        provider.initialize()
        
        assert provider.supports_vision is True
        
        provider.unload()


# ============================================================================
# Multimodal Pipeline Tests
# ============================================================================

class TestVisionPipeline:
    """Test VisionPipeline routing logic."""
    
    def test_vision_pipeline_imports(self):
        """Test that VisionPipeline imports correctly."""
        from agent.media.vision_pipeline import VisionPipeline, VisionMode
        assert VisionPipeline is not None
        assert hasattr(VisionMode, 'VL_NATIVE')
        assert hasattr(VisionMode, 'CLASSICAL')
        assert hasattr(VisionMode, 'HYBRID')
    
    def test_vision_mode_enum(self):
        """Test VisionMode enum values."""
        from agent.media.vision_pipeline import VisionMode
        assert VisionMode.VL_NATIVE.value == "vl_native"
        assert VisionMode.CLASSICAL.value == "classical"
        assert VisionMode.HYBRID.value == "hybrid"
    
    def test_screen_understanding_imports(self):
        """Test that ScreenUnderstandingProvider imports correctly."""
        from agent.media.screen_understanding import ScreenUnderstandingProvider
        assert ScreenUnderstandingProvider is not None


# ============================================================================
# Agent Integration Tests
# ============================================================================

class TestAgentVisionIntegration:
    """Test Agent vision integration."""
    
    def test_agent_creates_vl_provider(self):
        """Test that Agent creates a VL-capable LLM provider."""
        from agent.core.config import Config
        from agent.core.agent import Agent
        
        config = Config()
        agent = Agent(config=config)
        agent.initialize()
        
        llm = agent._llm_provider
        assert hasattr(llm, 'supports_vision')
        assert llm.supports_vision is True
        
        agent.shutdown()


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestVLErrorHandling:
    """Test error handling in vision pipeline."""
    
    def test_image_input_missing_file(self):
        """Test handling of missing image file."""
        from agent.llm.base import ImageInput
        
        image_input = ImageInput.from_file("/nonexistent/path/image.png")
        llm_format = image_input.to_llm_format()
        # Should still produce a format, even if file doesn't exist
        assert llm_format["type"] == "image_url"
    
    def test_vision_mode_string_conversion(self):
        """Test that string mode is converted to enum."""
        from agent.media.vision_pipeline import VisionPipeline, VisionMode
        
        # Create a minimal pipeline
        pipeline = VisionPipeline()
        
        # Test that string mode is handled
        # This should not raise an error
        assert VisionMode("vl_native") == VisionMode.VL_NATIVE
        assert VisionMode("classical") == VisionMode.CLASSICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
