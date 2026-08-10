"""Configuration management for the local agent."""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Main configuration class for the local agent.
    
    Loads configuration from:
    1. Environment variables (.env file)
    2. YAML configuration files
    3. Default values
    """
    
    # Project settings
    project_name: str = "Rose"
    version: str = "0.1.0"
    stage: str = "1.1"
    
    # Paths (relative to project root)
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    model_dir: Path = field(default_factory=lambda: Path("./models"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    config_dir: Path = field(default_factory=lambda: Path("./configs"))
    
    # Model settings
    model_path: str = "./models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    model_name: str = "qwen2.5-coder-7b-instruct"
    model_context_length: int = 4096
    
    # LLM settings
    llm_provider: str = "local"
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_max_tokens: int = 2048
    llm_repeat_penalty: float = 1.1
    llm_gpu_layers: int = 28
    llm_batch_size: int = 512
    llm_verbose: bool = False
    llm_timeout: int = 120
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "./logs/agent.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Agent settings
    agent_verbose: bool = False
    agent_max_history: int = 50
    agent_require_confirmation: bool = True
    agent_log_commands: bool = True
    
    # Orchestration settings
    max_plan_steps: int = 12
    max_tool_calls: int = 20
    max_replans: int = 3
    max_step_retries: int = 2
    max_task_duration: float = 300.0
    
    # OS control settings
    os_control_enabled: bool = True
    screen_capture_enabled: bool = True
    max_screenshot_width: int = 4096
    max_screenshot_height: int = 4096
    max_screenshot_size_mb: int = 20
    system_info_enabled: bool = True
    mouse_control_enabled: bool = False
    keyboard_control_enabled: bool = False
    window_control_enabled: bool = False
    max_mouse_actions_per_request: int = 20
    max_keyboard_actions_per_request: int = 20
    max_typed_text_length: int = 1000
    mouse_action_timeout: float = 5.0
    keyboard_action_timeout: float = 5.0
    max_scroll_amount: int = 10
    
    def __post_init__(self):
        """Post-initialization hook for dataclass."""
        # Load environment variables
        load_dotenv()
        
        # Set base directory from config_dir if provided
        # (config_dir is already set by dataclass init)
        
        # Load configuration from files
        self._load_yaml_config()
        self._load_env_config()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration.
        
        Args:
            config_dir: Optional custom config directory path.
        """
        # Set defaults for all fields
        self.project_name = "Rose"
        self.version = "0.1.0"
        self.stage = "1.1"
        self.base_dir = Path.cwd()
        self.model_dir = Path("./models")
        self.data_dir = Path("./data")
        self.output_dir = Path("./outputs")
        self.log_dir = Path("./logs")
        self.config_dir = Path("./configs")
        self.model_path = "./models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        self.model_name = "qwen2.5-coder-7b-instruct"
        self.model_context_length = 4096
        self.llm_provider = "local"
        self.llm_temperature = 0.7
        self.llm_top_p = 0.9
        self.llm_max_tokens = 2048
        self.llm_repeat_penalty = 1.1
        self.llm_gpu_layers = 0
        self.llm_batch_size = 512
        self.llm_verbose = False
        self.llm_timeout = 120
        self.log_level = "INFO"
        self.log_file = "./logs/agent.log"
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.agent_verbose = False
        self.agent_max_history = 50
        self.agent_require_confirmation = True
        self.agent_log_commands = True
        self.max_plan_steps = 12
        self.max_tool_calls = 20
        self.max_replans = 3
        self.max_step_retries = 2
        self.max_task_duration = 300.0
        
        # OS control settings
        self.os_control_enabled = True
        self.screen_capture_enabled = True
        self.max_screenshot_width = 4096
        self.max_screenshot_height = 4096
        self.max_screenshot_size_mb = 20
        self.system_info_enabled = True
        self.mouse_control_enabled = False
        self.keyboard_control_enabled = False
        self.window_control_enabled = False
        self.max_mouse_actions_per_request = 20
        self.max_keyboard_actions_per_request = 20
        self.max_typed_text_length = 1000
        self.mouse_action_timeout = 5.0
        self.keyboard_action_timeout = 5.0
        self.max_scroll_amount = 10
        
        # Set base directory
        if config_dir:
            self.base_dir = config_dir
        else:
            self.base_dir = Path(__file__).parent.parent.parent
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration from files
        self._load_yaml_config()
        self._load_env_config()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _load_yaml_config(self):
        """Load configuration from YAML files."""
        config_file = self.base_dir / "configs" / "settings.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    self._apply_config(config_data)
                    logger.debug(f"Loaded config from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load YAML config: {e}")
    
    def _apply_config(self, config_data: Dict[str, Any]):
        """Apply configuration data to this instance."""
        # Model settings
        if "model" in config_data:
            model = config_data["model"]
            if "path" in model:
                self.model_path = model["path"]
            if "name" in model:
                self.model_name = model["name"]
            if "context_length" in model:
                self.model_context_length = model["context_length"]
            if "n_gpu_layers" in model:
                self.llm_gpu_layers = model["n_gpu_layers"]
            if "n_batch" in model:
                self.llm_batch_size = model["n_batch"]
            if "temperature" in model:
                self.llm_temperature = model["temperature"]
            if "top_p" in model:
                self.llm_top_p = model["top_p"]
            if "max_tokens" in model:
                self.llm_max_tokens = model["max_tokens"]
            if "repeat_penalty" in model:
                self.llm_repeat_penalty = model["repeat_penalty"]
        
        # LLM settings
        if "llm" in config_data:
            llm = config_data["llm"]
            if "provider" in llm:
                self.llm_provider = llm["provider"]
            if "timeout" in llm:
                self.llm_timeout = llm["timeout"]
        
        # Logging settings
        if "logging" in config_data:
            logging_config = config_data["logging"]
            if "level" in logging_config:
                self.log_level = logging_config["level"]
            if "file" in logging_config:
                self.log_file = logging_config["file"]
            if "format" in logging_config:
                self.log_format = logging_config["format"]
        
        # Agent settings
        if "agent" in config_data:
            agent = config_data["agent"]
            if "verbose" in agent:
                self.agent_verbose = agent["verbose"]
            if "max_history" in agent:
                self.agent_max_history = agent["max_history"]
            if "require_confirmation" in agent:
                self.agent_require_confirmation = agent["require_confirmation"]
            if "log_commands" in agent:
                self.agent_log_commands = agent["log_commands"]
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        # Override with environment variables if present
        self.model_path = os.getenv("MODEL_PATH", self.model_path)
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.model_context_length = int(os.getenv("MODEL_CONTEXT_LENGTH", str(self.model_context_length)))
        
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", str(self.llm_temperature)))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(self.llm_max_tokens)))
        self.llm_gpu_layers = int(os.getenv("LLM_GPU_LAYERS", str(self.llm_gpu_layers)))
        self.llm_verbose = os.getenv("LLM_VERBOSE", "false").lower() == "true"
        
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.log_file = os.getenv("LOG_FILE", self.log_file)
        
        self.max_plan_steps = int(os.getenv("MAX_PLAN_STEPS", str(self.max_plan_steps)))
        self.max_tool_calls = int(os.getenv("MAX_TOOL_CALLS", str(self.max_tool_calls)))
        self.max_replans = int(os.getenv("MAX_REPLANS", str(self.max_replans)))
        self.max_step_retries = int(os.getenv("MAX_STEP_RETRIES", str(self.max_step_retries)))
        self.max_task_duration = float(os.getenv("MAX_TASK_DURATION", str(self.max_task_duration)))
        
        # OS control settings
        self.os_control_enabled = os.getenv("OS_CONTROL_ENABLED", "true").lower() == "true"
        self.screen_capture_enabled = os.getenv("SCREEN_CAPTURE_ENABLED", "true").lower() == "true"
        self.max_screenshot_width = int(os.getenv("MAX_SCREEN_WIDTH", str(self.max_screenshot_width)))
        self.max_screenshot_height = int(os.getenv("MAX_SCREEN_HEIGHT", str(self.max_screenshot_height)))
        self.max_screenshot_size_mb = int(os.getenv("MAX_SCREENSHOT_SIZE_MB", str(self.max_screenshot_size_mb)))
        self.system_info_enabled = os.getenv("SYSTEM_INFO_ENABLED", "true").lower() == "true"
        self.mouse_control_enabled = os.getenv("MOUSE_CONTROL_ENABLED", "false").lower() == "true"
        self.keyboard_control_enabled = os.getenv("KEYBOARD_CONTROL_ENABLED", "false").lower() == "true"
        self.window_control_enabled = os.getenv("WINDOW_CONTROL_ENABLED", "false").lower() == "true"
        self.max_mouse_actions_per_request = int(os.getenv("MAX_MOUSE_ACTIONS_PER_REQUEST", str(self.max_mouse_actions_per_request)))
        self.max_keyboard_actions_per_request = int(os.getenv("MAX_KEYBOARD_ACTIONS_PER_REQUEST", str(self.max_keyboard_actions_per_request)))
        self.max_typed_text_length = int(os.getenv("MAX_TYPED_TEXT_LENGTH", str(self.max_typed_text_length)))
        self.mouse_action_timeout = float(os.getenv("MOUSE_ACTION_TIMEOUT", str(self.mouse_action_timeout)))
        self.keyboard_action_timeout = float(os.getenv("KEYBOARD_ACTION_TIMEOUT", str(self.keyboard_action_timeout)))
        self.max_scroll_amount = int(os.getenv("MAX_SCROLL_AMOUNT", str(self.max_scroll_amount)))
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.model_dir,
            self.data_dir,
            self.output_dir,
            self.log_dir,
            self.config_dir,
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_full_path(self) -> Path:
        """Get the full path to the model file.
        
        Returns:
            Full path to the model file.
        """
        return self.base_dir / self.model_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            "project": {
                "name": self.project_name,
                "version": self.version,
                "stage": self.stage,
            },
            "model": {
                "path": self.model_path,
                "name": self.model_name,
                "context_length": self.model_context_length,
            },
            "llm": {
                "provider": self.llm_provider,
                "temperature": self.llm_temperature,
                "top_p": self.llm_top_p,
                "max_tokens": self.llm_max_tokens,
                "gpu_layers": self.llm_gpu_layers,
                "batch_size": self.llm_batch_size,
                "verbose": self.llm_verbose,
                "timeout": self.llm_timeout,
            },
            "logging": {
                "level": self.log_level,
                "file": self.log_file,
                "format": self.log_format,
            },
            "agent": {
                "verbose": self.agent_verbose,
                "max_history": self.agent_max_history,
                "require_confirmation": self.agent_require_confirmation,
                "log_commands": self.agent_log_commands,
            },
            "paths": {
                "base": str(self.base_dir),
                "model": str(self.model_dir),
                "data": str(self.data_dir),
                "output": str(self.output_dir),
                "log": str(self.log_dir),
            },
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(project={self.project_name}, model={self.model_name})"
