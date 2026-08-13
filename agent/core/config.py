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
    model_path: str = "./models/Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    model_name: str = "qwen2.5-vl-7b-instruct"
    model_context_length: int = 4096
    mmproj_path: str = "./models/mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"
    
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
    
    # Vision-Language settings
    vision_enabled: bool = True
    vision_max_images: int = 4
    vision_max_image_size_mb: int = 20
    vision_max_image_width: int = 4096
    vision_max_image_height: int = 4096
    vision_grounding_enabled: bool = True
    vision_screen_understanding: bool = True
    vision_browser_integration: bool = True
    
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
    window_close_enabled: bool = False
    window_move_enabled: bool = False
    window_resize_enabled: bool = False
    min_window_width: int = 100
    min_window_height: int = 100
    max_window_width: int = 4096
    max_window_height: int = 4096
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
        self.model_path = "./models/Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        self.model_name = "qwen2.5-vl-7b-instruct"
        self.model_context_length = 4096
        self.mmproj_path = "./models/mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"
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
        
        # Browser automation settings
        self.browser_automation_enabled = False
        self.browser_headless = True
        self.browser_max_sessions = 2
        self.browser_max_tabs = 5
        self.browser_navigation_timeout = 30000
        self.browser_action_timeout = 10000
        self.browser_max_page_text_chars = 20000
        self.browser_max_page_text_tokens = 5000
        self.browser_max_elements_returned = 100
        self.browser_max_input_text_length = 2000
        self.browser_interaction_timeout = 10000
        self.browser_max_interactions_per_request = 20
        self.browser_max_wait_timeout = 15000
        
        # Browser screenshot settings
        self.browser_screenshot_enabled = False
        self.browser_max_screenshot_width = 3840
        self.browser_max_screenshot_height = 2160
        self.browser_max_full_page_height = 10000
        self.browser_max_screenshot_size_mb = 20
        self.browser_max_screenshots_per_request = 10
        self.browser_screenshot_timeout = 10000
        
        # Vision settings
        self.vision_enabled = False
        self.vision_provider_type = "local"  # "local" = stub, "real" = RealVisionProvider with preprocessing
        self.vision_model_path = ""
        self.vision_max_image_size_mb = 20
        self.vision_max_image_width = 4096
        self.vision_max_image_height = 4096
        self.vision_max_elements = 100
        self.vision_analysis_timeout = 30000
        
        # Grounding settings
        self.grounding_confidence_threshold = 0.3
        self.grounding_stale_timeout = 30.0
        
        # OCR settings
        self.ocr_enabled = True
        self.ocr_provider_type = "local_tesseract"
        self.ocr_language = "eng"
        self.ocr_config = "--psm 6"
        self.ocr_max_image_size_mb = 20
        self.ocr_max_image_width = 4096
        self.ocr_max_image_height = 4096
        self.ocr_max_text_chars = 100000
        self.ocr_max_blocks = 1000
        self.ocr_timeout_ms = 30000
        
        # Multimodal settings
        self.multimodal_enabled = False
        self.multimodal_max_ocr_chars = 2000
        self.multimodal_max_targets = 15
        self.multimodal_include_coordinates = True
        self.multimodal_include_image_path = True
        self.autonomous_vision_enabled = False
        self.autonomous_max_retries = 3
        self.autonomous_screenshot_delay = 1.0
        
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
        self.mmproj_path = os.getenv("MMPROJ_PATH", self.mmproj_path)
        
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
        self.window_close_enabled = os.getenv("WINDOW_CLOSE_ENABLED", "false").lower() == "true"
        self.window_move_enabled = os.getenv("WINDOW_MOVE_ENABLED", "false").lower() == "true"
        self.window_resize_enabled = os.getenv("WINDOW_RESIZE_ENABLED", "false").lower() == "true"
        self.min_window_width = int(os.getenv("MIN_WINDOW_WIDTH", str(self.min_window_width)))
        self.min_window_height = int(os.getenv("MIN_WINDOW_HEIGHT", str(self.min_window_height)))
        self.max_window_width = int(os.getenv("MAX_WINDOW_WIDTH", str(self.max_window_width)))
        self.max_window_height = int(os.getenv("MAX_WINDOW_HEIGHT", str(self.max_window_height)))
        self.max_mouse_actions_per_request = int(os.getenv("MAX_MOUSE_ACTIONS_PER_REQUEST", str(self.max_mouse_actions_per_request)))
        self.max_keyboard_actions_per_request = int(os.getenv("MAX_KEYBOARD_ACTIONS_PER_REQUEST", str(self.max_keyboard_actions_per_request)))
        self.max_typed_text_length = int(os.getenv("MAX_TYPED_TEXT_LENGTH", str(self.max_typed_text_length)))
        self.mouse_action_timeout = float(os.getenv("MOUSE_ACTION_TIMEOUT", str(self.mouse_action_timeout)))
        self.keyboard_action_timeout = float(os.getenv("KEYBOARD_ACTION_TIMEOUT", str(self.keyboard_action_timeout)))
        self.max_scroll_amount = int(os.getenv("MAX_SCROLL_AMOUNT", str(self.max_scroll_amount)))
        
        # Browser automation settings
        self.browser_automation_enabled = os.getenv("BROWSER_AUTOMATION_ENABLED", "false").lower() == "true"
        self.browser_headless = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
        self.browser_max_sessions = int(os.getenv("BROWSER_MAX_SESSIONS", str(self.browser_max_sessions)))
        self.browser_max_tabs = int(os.getenv("BROWSER_MAX_TABS", str(self.browser_max_tabs)))
        self.browser_navigation_timeout = int(os.getenv("BROWSER_NAVIGATION_TIMEOUT", str(self.browser_navigation_timeout)))
        self.browser_action_timeout = int(os.getenv("BROWSER_ACTION_TIMEOUT", str(self.browser_action_timeout)))
        self.browser_max_page_text_chars = int(os.getenv("BROWSER_MAX_PAGE_TEXT_CHARS", str(self.browser_max_page_text_chars)))
        self.browser_max_page_text_tokens = int(os.getenv("BROWSER_MAX_PAGE_TEXT_TOKENS", str(self.browser_max_page_text_tokens)))
        self.browser_max_elements_returned = int(os.getenv("BROWSER_MAX_ELEMENTS_RETURNED", str(self.browser_max_elements_returned)))
        self.browser_max_input_text_length = int(os.getenv("BROWSER_MAX_INPUT_TEXT_LENGTH", str(self.browser_max_input_text_length)))
        self.browser_interaction_timeout = int(os.getenv("BROWSER_INTERACTION_TIMEOUT", str(self.browser_interaction_timeout)))
        self.browser_max_interactions_per_request = int(os.getenv("BROWSER_MAX_INTERACTIONS_PER_REQUEST", str(self.browser_max_interactions_per_request)))
        self.browser_max_wait_timeout = int(os.getenv("BROWSER_MAX_WAIT_TIMEOUT", str(self.browser_max_wait_timeout)))
        
        # Browser screenshot settings
        self.browser_screenshot_enabled = os.getenv("BROWSER_SCREENSHOT_ENABLED", "false").lower() == "true"
        self.browser_max_screenshot_width = int(os.getenv("BROWSER_MAX_SCREENSHOT_WIDTH", str(self.browser_max_screenshot_width)))
        self.browser_max_screenshot_height = int(os.getenv("BROWSER_MAX_SCREENSHOT_HEIGHT", str(self.browser_max_screenshot_height)))
        self.browser_max_full_page_height = int(os.getenv("BROWSER_MAX_FULL_PAGE_HEIGHT", str(self.browser_max_full_page_height)))
        self.browser_max_screenshot_size_mb = int(os.getenv("BROWSER_MAX_SCREENSHOT_SIZE_MB", str(self.browser_max_screenshot_size_mb)))
        self.browser_max_screenshots_per_request = int(os.getenv("BROWSER_MAX_SCREENSHOTS_PER_REQUEST", str(self.browser_max_screenshots_per_request)))
        self.browser_screenshot_timeout = int(os.getenv("BROWSER_SCREENSHOT_TIMEOUT", str(self.browser_screenshot_timeout)))
        
        # Vision settings
        self.vision_enabled = os.getenv("VISION_ENABLED", "false").lower() == "true"
        self.vision_provider_type = os.getenv("VISION_PROVIDER", self.vision_provider_type)
        self.vision_model_path = os.getenv("VISION_MODEL_PATH", self.vision_model_path)
        self.vision_max_image_size_mb = int(os.getenv("VISION_MAX_IMAGE_SIZE_MB", str(self.vision_max_image_size_mb)))
        self.vision_max_image_width = int(os.getenv("VISION_MAX_IMAGE_WIDTH", str(self.vision_max_image_width)))
        self.vision_max_image_height = int(os.getenv("VISION_MAX_IMAGE_HEIGHT", str(self.vision_max_image_height)))
        self.vision_max_elements = int(os.getenv("VISION_MAX_ELEMENTS", str(self.vision_max_elements)))
        self.vision_analysis_timeout = int(os.getenv("VISION_ANALYSIS_TIMEOUT", str(self.vision_analysis_timeout)))
        
        # Grounding settings
        self.grounding_confidence_threshold = float(os.getenv("GROUNDING_CONFIDENCE_THRESHOLD", str(self.grounding_confidence_threshold)))
        self.grounding_stale_timeout = float(os.getenv("GROUNDING_STALE_TIMEOUT", str(self.grounding_stale_timeout)))
        
        # OCR settings
        self.ocr_enabled = os.getenv("OCR_ENABLED", "true").lower() == "true"
        self.ocr_provider_type = os.getenv("OCR_PROVIDER", self.ocr_provider_type)
        self.ocr_language = os.getenv("OCR_LANGUAGE", self.ocr_language)
        self.ocr_config = os.getenv("OCR_CONFIG", self.ocr_config)
        self.ocr_max_image_size_mb = int(os.getenv("OCR_MAX_IMAGE_SIZE_MB", str(self.ocr_max_image_size_mb)))
        self.ocr_max_image_width = int(os.getenv("OCR_MAX_IMAGE_WIDTH", str(self.ocr_max_image_width)))
        self.ocr_max_image_height = int(os.getenv("OCR_MAX_IMAGE_HEIGHT", str(self.ocr_max_image_height)))
        self.ocr_max_text_chars = int(os.getenv("OCR_MAX_TEXT_CHARS", str(self.ocr_max_text_chars)))
        self.ocr_max_blocks = int(os.getenv("OCR_MAX_BLOCKS", str(self.ocr_max_blocks)))
        self.ocr_timeout_ms = int(os.getenv("OCR_TIMEOUT_MS", str(self.ocr_timeout_ms)))
        
        # Multimodal settings
        self.multimodal_enabled = os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true"
        self.multimodal_max_ocr_chars = int(os.getenv("MULTIMODAL_MAX_OCR_CHARS", str(self.multimodal_max_ocr_chars)))
        self.multimodal_max_targets = int(os.getenv("MULTIMODAL_MAX_TARGETS", str(self.multimodal_max_targets)))
        self.multimodal_include_coordinates = os.getenv("MULTIMODAL_INCLUDE_COORDINATES", "true").lower() == "true"
        self.multimodal_include_image_path = os.getenv("MULTIMODAL_INCLUDE_IMAGE_PATH", "true").lower() == "true"
        self.autonomous_vision_enabled = os.getenv("AUTONOMOUS_VISION_ENABLED", "false").lower() == "true"
        self.autonomous_max_retries = int(os.getenv("AUTONOMOUS_MAX_RETRIES", str(self.autonomous_max_retries)))
        self.autonomous_screenshot_delay = float(os.getenv("AUTONOMOUS_SCREENSHOT_DELAY", str(self.autonomous_screenshot_delay)))
    
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
