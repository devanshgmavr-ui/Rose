"""Unit tests for configuration module."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.core.config import Config


class TestConfig:
    """Test configuration loading and defaults."""
    
    def test_default_config(self):
        """Test default configuration values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dir=Path(tmpdir))
            
            assert config.project_name == "Rose"
            assert config.version == "0.1.0"
            assert config.stage == "1.1"
            assert config.llm_provider == "local"
            assert config.llm_temperature == 0.7
            assert config.llm_max_tokens == 2048
    
    def test_config_to_dict(self):
        """Test configuration dictionary conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dir=Path(tmpdir))
            config_dict = config.to_dict()
            
            assert "project" in config_dict
            assert "model" in config_dict
            assert "llm" in config_dict
            assert "logging" in config_dict
            assert "paths" in config_dict
    
    def test_model_path_resolution(self):
        """Test model path resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dir=Path(tmpdir))
            model_path = config.get_model_full_path()
            
            assert isinstance(model_path, Path)
            assert "models" in str(model_path)
    
    @patch.dict(os.environ, {"LLM_TEMPERATURE": "0.5"})
    def test_env_override(self):
        """Test environment variable override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dir=Path(tmpdir))
            
            # Environment variable should override default
            assert config.llm_temperature == 0.5
    
    def test_directories_created(self):
        """Test that required directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(config_dir=Path(tmpdir))
            
            # Check directories exist
            assert (Path(tmpdir) / "models").exists()
            assert (Path(tmpdir) / "data").exists()
            assert (Path(tmpdir) / "outputs").exists()
            assert (Path(tmpdir) / "logs").exists()
