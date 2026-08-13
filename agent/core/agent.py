"""Main agent class for the local AI agent."""

import json
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import Config
from ..llm.base import LLMProvider, LLMConfig, LLMResponse, LLMProviderType, VisionCapability, ImageInput
from ..llm.local_provider import LocalLLMProvider
from ..memory import (
    SessionManager,
    ConversationManager,
    LongTermMemory,
    ContextManager,
    ConversationSummarizer,
    MessageRole,
    MemoryType,
    MemoryRecord,
)
from ..tools import (
    ToolRegistry,
    ToolRouter,
    PermissionManager,
    AuditLogger,
    ToolRequest,
    ToolResult,
    FilesystemTool,
    PythonSandboxTool,
    CLITool,
)
from ..orchestration import (
    Task,
    Plan,
    PlanStep,
    TaskStatus,
    Planner,
    PlanValidator,
    TaskExecutor,
    Verifier,
    TaskPersistence,
    OrchestrationLimits,
    EventLogger,
)
from ..media import (
    MediaStorage,
    MediaRouter,
    VisionProvider,
    LocalVisionProvider,
    RealVisionProvider,
    VisionAnalyzer,
    VisionAnalyzeTool,
    VisualGrounder,
    VisualGroundingTool,
    ObserveActVerifyLoop,
    ObserveActVerifyTool,
    register_vision_permissions,
    ImageGenProvider,
    VideoGenProvider,
)
from ..media.tools import (
    ImageAnalyzeTool,
    ImageGenerateTool,
    VideoGenerateTool,
    MediaInfoTool,
)
from ..os_control import (
    ScreenCaptureTool,
    SystemInfoTool,
    MouseTool,
    KeyboardTool,
    WindowTool,
    register_os_permissions,
)
from ..browser import (
    BrowserManager,
    BrowserSessionTool,
    BrowserNavigationTool,
    BrowserPageReadTool,
    BrowserInteractionTool,
    BrowserScreenshotTool,
    register_browser_permissions,
)
from ..media.vision_decision import VisionDecisionSystem, VisionSource, VisionRequirement
from ..media.multimodal_pipeline import MultimodalRequestPipeline, RequestType
from .model_health import ModelHealthChecker, ModelHealthStatus
from ..orchestration.autonomous_loop import AutonomousLoop

logger = logging.getLogger(__name__)


class Agent:
    """Main agent class that orchestrates LLM interactions.
    
    This agent uses a clean architecture with an LLM abstraction layer,
    allowing easy swapping between different providers.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the agent.
        
        Args:
            config: Optional configuration. If not provided, uses default.
        """
        self.config = config or Config()
        self._llm_provider: Optional[LLMProvider] = None
        self._conversation_history: List[Dict[str, str]] = []
        
        # Memory system components
        self._session_manager: Optional[SessionManager] = None
        self._conversation_manager: Optional[ConversationManager] = None
        self._long_term_memory: Optional[LongTermMemory] = None
        self._context_manager: Optional[ContextManager] = None
        self._summarizer: Optional[ConversationSummarizer] = None
        
        # Tool system components
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_router: Optional[ToolRouter] = None
        self._permission_manager: Optional[PermissionManager] = None
        self._audit_logger: Optional[AuditLogger] = None
        self._max_tool_iterations: int = 5
        
        # Orchestration components
        self._planner: Optional[Planner] = None
        self._plan_validator: Optional[PlanValidator] = None
        self._task_executor: Optional[TaskExecutor] = None
        self._verifier: Optional[Verifier] = None
        self._task_persistence: Optional[TaskPersistence] = None
        self._event_logger: Optional[EventLogger] = None
        self._orchestration_limits: Optional[OrchestrationLimits] = None
        self._current_task: Optional[Task] = None
        
        # Media system components
        self._media_storage: Optional[MediaStorage] = None
        self._media_router: Optional[MediaRouter] = None
        self._vision_provider: Optional[VisionProvider] = None
        self._image_gen_provider: Optional[ImageGenProvider] = None
        self._video_gen_provider: Optional[VideoGenProvider] = None
        
        # OS control components
        self._screen_capture_tool: Optional[ScreenCaptureTool] = None
        self._system_info_tool: Optional[SystemInfoTool] = None
        self._mouse_tool: Optional[MouseTool] = None
        self._keyboard_tool: Optional[KeyboardTool] = None
        self._window_tool: Optional[WindowTool] = None
        
        # Browser system components
        self._browser_manager: Optional[BrowserManager] = None
        self._browser_session_tool: Optional[BrowserSessionTool] = None
        self._browser_navigation_tool: Optional[BrowserNavigationTool] = None
        self._browser_page_read_tool: Optional[BrowserPageReadTool] = None
        self._browser_interaction_tool: Optional[BrowserInteractionTool] = None
        self._browser_screenshot_tool: Optional[BrowserScreenshotTool] = None
        
        # New production components
        self._vision_decision = VisionDecisionSystem()
        self._multimodal_pipeline = MultimodalRequestPipeline()
        self._model_health_checker = ModelHealthChecker(config=self.config)
        self._autonomous_loop: Optional[AutonomousLoop] = None
        self._vision_pipeline = None
        
        logger.info(f"Agent initialized: {self.config.project_name} v{self.config.version}")
    
    def _create_llm_provider(self) -> LLMProvider:
        """Create the appropriate LLM provider based on configuration.
        
        Returns:
            LLM provider instance.
            
        Raises:
            ValueError: If unknown provider type.
        """
        provider_type = self.config.llm_provider.lower()
        
        if provider_type == "local":
            # Resolve mmproj path
            mmproj_path = None
            if self.config.mmproj_path:
                mmproj_full = self.config.base_dir / self.config.mmproj_path
                if mmproj_full.exists():
                    mmproj_path = str(mmproj_full)
            
            # Determine vision capability
            from ..llm.base import VisionCapability
            vision_cap = VisionCapability.NONE
            if mmproj_path:
                vision_cap = VisionCapability.MULTIPLE
            
            llm_config = LLMConfig(
                provider_type=LLMProviderType.LOCAL,
                model_path=str(self.config.get_model_full_path()),
                model_name=self.config.model_name,
                context_length=self.config.model_context_length,
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p,
                max_tokens=self.config.llm_max_tokens,
                repeat_penalty=self.config.llm_repeat_penalty,
                n_gpu_layers=self.config.llm_gpu_layers,
                n_batch=self.config.llm_batch_size,
                verbose=self.config.llm_verbose,
                mmproj_path=mmproj_path,
                vision_capability=vision_cap,
                max_images=4 if mmproj_path else 0,
            )
            return LocalLLMProvider(llm_config)
        
        # Future providers can be added here
        # elif provider_type == "cloud":
        #     return CloudLLMProvider(...)
        
        raise ValueError(f"Unknown LLM provider: {provider_type}")
    
    def initialize(self) -> bool:
        """Initialize the agent and load the LLM.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            logger.info("Initializing agent...")
            
            self._llm_provider = self._create_llm_provider()
            
            if not self._llm_provider.initialize():
                logger.error("Failed to initialize LLM provider")
                return False
            
            self._init_memory_system()
            self._init_tool_system()
            self._init_autonomous_loop()
            self._init_vision_pipeline()
            
            # Check model health
            self._model_health = self._model_health_checker.check_health(self._llm_provider)
            logger.info(f"Model health: {self._model_health.status_summary}")
            
            logger.info("Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False
    
    def _init_autonomous_loop(self):
        """Initialize the autonomous execution loop."""
        if not self._tool_router:
            logger.warning("Cannot initialize autonomous loop without tool router")
            return
        
        self._autonomous_loop = AutonomousLoop(
            tool_router=self._tool_router,
            permission_manager=self._permission_manager,
            limits=self._orchestration_limits,
            event_logger=self._event_logger,
        )
        logger.info("Autonomous loop initialized")
    
    def _init_vision_pipeline(self):
        """Initialize the unified vision pipeline."""
        try:
            from ..media.vision_pipeline import VisionPipeline
            self._vision_pipeline = VisionPipeline(
                llm_provider=self._llm_provider,
                vision_provider=self._vision_provider,
                prefer_vl=True,
            )
            logger.info("Vision pipeline initialized")
        except Exception as e:
            logger.warning(f"Vision pipeline initialization failed: {e}")
    
    def _init_tool_system(self):
        """Initialize the tool system."""
        logger.info("Initializing tool system...")
        
        self._permission_manager = PermissionManager()
        self._audit_logger = AuditLogger(log_dir="logs")
        self._tool_registry = ToolRegistry()
        
        fs_tool = FilesystemTool(workspace_dir="workspace")
        py_tool = PythonSandboxTool(workspace_dir="workspace")
        cli_tool = CLITool(workspace_dir="workspace")
        
        self._tool_registry.register(fs_tool)
        self._tool_registry.register(py_tool)
        self._tool_registry.register(cli_tool)
        
        self._init_media_system()
        self._init_os_control_system()
        self._init_browser_system()
        
        self._tool_router = ToolRouter(
            registry=self._tool_registry,
            permission_manager=self._permission_manager,
            audit_logger=self._audit_logger,
            default_timeout=30.0,
            max_output_size=10000,
            max_tool_calls_per_request=self._max_tool_iterations,
        )
        
        logger.info(f"Tool system initialized with {self._tool_registry.count()} tools")
        self._init_orchestration()
    
    def _init_media_system(self):
        """Initialize the multimodal media system."""
        logger.info("Initializing media system...")
        
        self._media_storage = MediaStorage(workspace_dir="workspace")
        self._media_router = MediaRouter(storage=self._media_storage)
        
        # Select vision provider based on config
        if self.config.vision_provider_type == "real":
            logger.info("Using RealVisionProvider with image preprocessing")
            self._vision_provider = RealVisionProvider(
                max_image_size_mb=self.config.vision_max_image_size_mb,
                max_image_width=self.config.vision_max_image_width,
                max_image_height=self.config.vision_max_image_height,
            )
        else:
            self._vision_provider = LocalVisionProvider(
                model_path=self.config.vision_model_path if self.config.vision_model_path else None,
                max_image_size_mb=self.config.vision_max_image_size_mb,
                max_image_width=self.config.vision_max_image_width,
                max_image_height=self.config.vision_max_image_height,
                max_elements=self.config.vision_max_elements,
                analysis_timeout=self.config.vision_analysis_timeout / 1000.0,
            )
        self._vision_provider.initialize()
        
        self._vision_analyzer = VisionAnalyzer(
            vision_provider=self._vision_provider,
            media_storage=self._media_storage,
            max_elements=self.config.vision_max_elements,
            analysis_timeout=self.config.vision_analysis_timeout / 1000.0,
        )
        
        self._image_gen_provider = ImageGenProvider()
        self._video_gen_provider = VideoGenProvider()
        
        self._media_router.register_provider(self._vision_provider)
        self._media_router.register_provider(self._image_gen_provider)
        self._media_router.register_provider(self._video_gen_provider)
        
        if self.config.vision_enabled:
            register_vision_permissions(
                self._permission_manager,
                vision_enabled=self.config.vision_enabled,
            )
            
            self._vision_analyze_tool = VisionAnalyzeTool(
                vision_analyzer=self._vision_analyzer,
                media_storage=self._media_storage,
                vision_enabled=self.config.vision_enabled,
                max_image_size_mb=self.config.vision_max_image_size_mb,
                workspace_dir="workspace",
            )
            self._tool_registry.register(self._vision_analyze_tool)
            
            self._visual_grounder = VisualGrounder(
                screen_width=self.config.max_screenshot_width,
                screen_height=self.config.max_screenshot_height,
            )
            
            self._visual_grounding_tool = VisualGroundingTool(
                vision_analyzer=self._vision_analyzer,
                visual_grounder=self._visual_grounder,
                media_storage=self._media_storage,
                vision_enabled=self.config.vision_enabled,
                grounding_enabled=self.config.vision_enabled,
                screen_width=self.config.max_screenshot_width,
                screen_height=self.config.max_screenshot_height,
            )
            self._tool_registry.register(self._visual_grounding_tool)
            
            self._observe_act_verify_tool = ObserveActVerifyTool(
                vision_analyzer=self._vision_analyzer,
                visual_grounder=self._visual_grounder,
                vision_enabled=self.config.vision_enabled,
                max_iterations=self.config.max_plan_steps,
                max_actions=self.config.max_tool_calls,
                timeout=self.config.max_task_duration,
            )
            self._tool_registry.register(self._observe_act_verify_tool)
        
        image_analyze_tool = ImageAnalyzeTool(self._media_router)
        image_generate_tool = ImageGenerateTool(self._media_router)
        video_generate_tool = VideoGenerateTool(self._media_router)
        media_info_tool = MediaInfoTool(self._media_storage)
        
        self._tool_registry.register(image_analyze_tool)
        self._tool_registry.register(image_generate_tool)
        self._tool_registry.register(video_generate_tool)
        self._tool_registry.register(media_info_tool)
        
        logger.info(f"Media system initialized with {len(self._media_router.list_providers())} providers")
    
    def _init_os_control_system(self):
        """Initialize the OS control and perception system."""
        logger.info("Initializing OS control system...")
        
        if not self.config.os_control_enabled:
            logger.info("OS control system is disabled by configuration")
            return
        
        register_os_permissions(
            self._permission_manager,
            mouse_enabled=self.config.mouse_control_enabled,
            keyboard_enabled=self.config.keyboard_control_enabled,
            window_enabled=self.config.window_control_enabled,
        )
        
        if self.config.screen_capture_enabled:
            self._screen_capture_tool = ScreenCaptureTool(workspace_dir="workspace")
            self._tool_registry.register(self._screen_capture_tool)
        
        if self.config.system_info_enabled:
            self._system_info_tool = SystemInfoTool()
            self._tool_registry.register(self._system_info_tool)
        
        self._mouse_tool = MouseTool(enabled=self.config.mouse_control_enabled)
        self._keyboard_tool = KeyboardTool(enabled=self.config.keyboard_control_enabled)
        self._window_tool = WindowTool(
            enabled=self.config.window_control_enabled,
            control_enabled=self.config.window_control_enabled,
            close_enabled=self.config.window_close_enabled,
            move_enabled=self.config.window_move_enabled,
            resize_enabled=self.config.window_resize_enabled,
            min_width=self.config.min_window_width,
            min_height=self.config.min_window_height,
            max_width=self.config.max_window_width,
            max_height=self.config.max_window_height,
        )
        self._tool_registry.register(self._mouse_tool)
        self._tool_registry.register(self._keyboard_tool)
        self._tool_registry.register(self._window_tool)
        
        logger.info("OS control system initialized")
    
    def _init_browser_system(self):
        """Initialize the browser automation system."""
        logger.info("Initializing browser system...")
        
        if not self.config.browser_automation_enabled:
            logger.info("Browser system is disabled by configuration")
            self._browser_manager = BrowserManager(
                headless=self.config.browser_headless,
                max_sessions=self.config.browser_max_sessions,
                max_tabs=self.config.browser_max_tabs,
            )
            return
        
        register_browser_permissions(
            self._permission_manager,
            browser_enabled=self.config.browser_automation_enabled,
        )
        
        self._browser_manager = BrowserManager(
            headless=self.config.browser_headless,
            max_sessions=self.config.browser_max_sessions,
            max_tabs=self.config.browser_max_tabs,
            navigation_timeout=self.config.browser_navigation_timeout,
            action_timeout=self.config.browser_action_timeout,
        )
        
        if not self._browser_manager.initialize():
            logger.warning("Browser manager failed to initialize")
        
        self._browser_session_tool = BrowserSessionTool(
            browser_manager=self._browser_manager,
            browser_enabled=self.config.browser_automation_enabled,
        )
        self._tool_registry.register(self._browser_session_tool)
        
        self._browser_navigation_tool = BrowserNavigationTool(
            browser_manager=self._browser_manager,
            browser_enabled=self.config.browser_automation_enabled,
        )
        self._tool_registry.register(self._browser_navigation_tool)
        
        self._browser_page_read_tool = BrowserPageReadTool(
            browser_manager=self._browser_manager,
            browser_enabled=self.config.browser_automation_enabled,
            max_page_text_chars=self.config.browser_max_page_text_chars,
            max_page_text_tokens=self.config.browser_max_page_text_tokens,
        )
        self._tool_registry.register(self._browser_page_read_tool)
        
        self._browser_interaction_tool = BrowserInteractionTool(
            browser_manager=self._browser_manager,
            browser_enabled=self.config.browser_automation_enabled,
            max_elements_returned=self.config.browser_max_elements_returned,
            max_input_text_length=self.config.browser_max_input_text_length,
            interaction_timeout=self.config.browser_interaction_timeout,
            max_wait_timeout=self.config.browser_max_wait_timeout,
        )
        self._tool_registry.register(self._browser_interaction_tool)
        
        self._browser_screenshot_tool = BrowserScreenshotTool(
            browser_manager=self._browser_manager,
            media_storage=self._media_storage,
            browser_enabled=self.config.browser_screenshot_enabled and self.config.browser_automation_enabled,
            max_screenshot_width=self.config.browser_max_screenshot_width,
            max_screenshot_height=self.config.browser_max_screenshot_height,
            max_full_page_height=self.config.browser_max_full_page_height,
            max_screenshot_size_mb=self.config.browser_max_screenshot_size_mb,
            max_screenshots_per_request=self.config.browser_max_screenshots_per_request,
            screenshot_timeout=self.config.browser_screenshot_timeout,
        )
        self._tool_registry.register(self._browser_screenshot_tool)
        
        logger.info("Browser system initialized")
    
    def _init_orchestration(self):
        """Initialize the orchestration system."""
        logger.info("Initializing orchestration system...")
        
        self._orchestration_limits = OrchestrationLimits(
            max_plan_steps=12,
            max_tool_calls=20,
            max_replans=3,
            max_step_retries=2,
            max_task_duration=300.0,
        )
        
        self._event_logger = EventLogger(log_dir="logs")
        self._task_persistence = TaskPersistence(data_dir="data/tasks")
        self._planner = Planner(
            llm_provider=self._llm_provider,
            max_plan_steps=self._orchestration_limits.max_plan_steps,
        )
        self._plan_validator = PlanValidator(
            available_tools=["filesystem", "python_sandbox", "generate",
                           "image_analyze", "image_generate", "video_generate", "media_info",
                           "screen_capture", "system_info", "mouse", "keyboard", "window"],
            max_steps=self._orchestration_limits.max_plan_steps,
        )
        self._verifier = Verifier(tool_router=self._tool_router)
        self._task_executor = TaskExecutor(
            tool_router=self._tool_router,
            limits=self._orchestration_limits,
            event_logger=self._event_logger,
        )
        
        logger.info("Orchestration system initialized")
    
    def _init_memory_system(self):
        """Initialize the memory system components."""
        logger.info("Initializing memory system...")
        
        self._session_manager = SessionManager(data_dir="sessions")
        self._conversation_manager = ConversationManager(self._session_manager)
        self._long_term_memory = LongTermMemory(db_path="data/memory.db")
        
        system_prompt = (
            "You are a helpful AI assistant running locally. "
            "You have access to conversation history and stored memories "
            "to provide context-aware responses."
        )
        
        self._context_manager = ContextManager(
            conversation_manager=self._conversation_manager,
            long_term_memory=self._long_term_memory,
            system_prompt=system_prompt,
            max_context_tokens=self.config.model_context_length - 512,
            reserved_output_tokens=512,
            recent_message_count=6,
            memory_retrieval_limit=5,
        )
        
        self._summarizer = ConversationSummarizer(
            conversation_manager=self._conversation_manager,
            long_term_memory=self._long_term_memory,
            max_context_tokens=self.config.model_context_length - 512,
            summary_threshold=int((self.config.model_context_length - 512) * 0.7),
            preserve_recent=6,
        )
        
        # Start a new session
        self._conversation_manager.start_conversation()
        
        logger.info("Memory system initialized")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from a prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            LLMResponse containing the generated text.
            
        Raises:
            RuntimeError: If agent not initialized.
        """
        if self._llm_provider is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        logger.debug(f"Generating response for prompt: {prompt[:50]}...")
        response = self._llm_provider.generate(prompt, **kwargs)
        logger.debug(f"Response generated: {response.text[:50]}...")
        
        return response
    
    def chat(self, message: str, **kwargs) -> LLMResponse:
        """Chat with the agent with memory, vision, and tool context."""
        if self._llm_provider is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        if self._conversation_manager:
            self._conversation_manager.add_user_message(message)
            if self._summarizer and self._summarizer.should_summarize():
                logger.info("Summarizing conversation to free context space")
                self._summarizer.summarize_and_compress()
        
        # Check if vision is needed
        vision_decision = self._vision_decision.decide(message)
        logger.debug(f"Vision decision: {vision_decision.requirement.value} - {vision_decision.reasoning}")
        
        # If vision is required and we have a screenshot, analyze it
        vision_context = ""
        if vision_decision.requirement == VisionRequirement.REQUIRED and vision_decision.needs_screenshot:
            vision_context = self._handle_vision_request(message, vision_decision)
        
        tool_context = []
        if vision_context:
            tool_context.append({
                "role": "system",
                "content": f"Visual context from screen analysis:\n{vision_context}",
            })
        
        iteration = 0
        
        while iteration < self._max_tool_iterations:
            iteration += 1
            
            if self._context_manager:
                context_messages = self._context_manager.build_context(
                    user_query=message if iteration == 1 else "",
                    include_memories=True,
                    session_id=self._conversation_manager.current_session.session_id
                    if self._conversation_manager and self._conversation_manager.current_session
                    else None,
                    tool_results=tool_context if tool_context else None,
                )
            else:
                self._conversation_history.append({"role": "user", "content": message})
                if len(self._conversation_history) > self.config.agent_max_history:
                    self._conversation_history = self._conversation_history[-self.config.agent_max_history:]
                context_messages = self._conversation_history
            
            response = self._llm_provider.chat(context_messages, **kwargs)
            response_text = response.text
            
            tool_request = self._parse_tool_request(response_text)
            if tool_request is None:
                break
            
            logger.info(f"Tool request detected: {tool_request.tool}")
            
            if self._tool_router:
                self._tool_router.reset_call_count()
                result = self._tool_router.route(tool_request)
            else:
                result = ToolResult(
                    success=False, tool_name=tool_request.tool,
                    error="Tool system not initialized",
                )
            
            tool_context.append({
                "role": "system",
                "content": f"Tool '{result.tool_name}' result:\n"
                f"Success: {result.success}\n"
                f"Output: {result.output[:2000] if result.output else 'none'}\n"
                f"Error: {result.error if result.error else 'none'}",
            })
            
            if self._conversation_manager:
                tool_msg = f"[Tool: {result.tool_name}] {'Success' if result.success else 'Failed'}: {result.output[:200] if result.output else result.error}"
                self._conversation_manager.add_system_message(tool_msg)
        
        if self._conversation_manager:
            self._conversation_manager.add_assistant_message(response_text)
        
        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({"role": "assistant", "content": response_text})
        if len(self._conversation_history) > self.config.agent_max_history:
            self._conversation_history = self._conversation_history[-self.config.agent_max_history:]
        
        logger.debug(f"Chat response generated: {response_text[:50]}...")
        
        return response
    
    def _handle_vision_request(self, message: str, vision_decision) -> str:
        """Handle a request that requires visual context."""
        try:
            # Capture screen
            if self._screen_capture_tool:
                capture_result = self._screen_capture_tool.execute({"action": "capture"})
                if capture_result.success:
                    screenshot_path = capture_result.output
                    
                    # Use vision pipeline if available
                    if self._vision_pipeline:
                        from ..media.vision_pipeline import VisionMode
                        mode = VisionMode.VL_NATIVE if self._vision_pipeline.is_vl_available else VisionMode.CLASSICAL
                        result = self._vision_pipeline.analyze_screenshot(
                            screenshot_path=screenshot_path,
                            query=message,
                            mode=mode,
                        )
                        if result.success:
                            return result.description
                    
                    # Fallback to vision analyzer
                    if hasattr(self, '_vision_analyzer') and self._vision_analyzer:
                        from ..media.base import VisionRequest
                        request = VisionRequest(image_path=screenshot_path)
                        result = self._vision_analyzer.analyze(request)
                        if result and hasattr(result, 'description'):
                            return result.description
            
            return "Unable to capture screen for visual analysis"
        except Exception as e:
            logger.error(f"Vision request failed: {e}")
            return f"Vision analysis failed: {str(e)}"
    
    def _parse_tool_request(self, text: str) -> Optional[ToolRequest]:
        """Parse a tool request from LLM output."""
        text = text.strip()
        
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.index("```", start)
                json_str = text[start:end].strip()
                data = json.loads(json_str)
                if "tool" in data:
                    return ToolRequest(
                        tool=data["tool"],
                        arguments=data.get("arguments", {}),
                    )
            except (ValueError, json.JSONDecodeError):
                pass
        
        if "```" in text:
            try:
                start = text.index("```") + 3
                end = text.index("```", start)
                json_str = text[start:end].strip()
                data = json.loads(json_str)
                if "tool" in data:
                    return ToolRequest(
                        tool=data["tool"],
                        arguments=data.get("arguments", {}),
                    )
            except (ValueError, json.JSONDecodeError):
                pass
        
        try:
            data = json.loads(text)
            if "tool" in data:
                return ToolRequest(
                    tool=data["tool"],
                    arguments=data.get("arguments", {}),
                )
        except json.JSONDecodeError:
            pass
        
        return None
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool directly."""
        if not self._tool_router:
            return ToolResult(
                success=False, tool_name=tool_name,
                error="Tool system not initialized",
            )
        self._tool_router.reset_call_count()
        return self._tool_router.execute_tool(tool_name, arguments)
    
    def get_tool_info(self) -> List[Dict[str, Any]]:
        """Get information about available tools."""
        if not self._tool_registry:
            return []
        return self._tool_registry.list_tools()
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool system statistics."""
        stats = {
            "initialized": self._tool_registry is not None,
            "tool_count": self._tool_registry.count() if self._tool_registry else 0,
            "tools": self._tool_registry.list_names() if self._tool_registry else [],
            "max_iterations": self._max_tool_iterations,
        }
        if self._audit_logger:
            stats["audit_records"] = self._audit_logger.get_record_count()
        return stats
    
    def execute_task(self, user_request: str, session_id: Optional[str] = None) -> Task:
        """Execute a multi-step task from user request to completion."""
        if not all([self._planner, self._plan_validator, self._task_executor]):
            raise RuntimeError("Orchestration system not initialized")
        
        task = Task(user_request=user_request, session_id=session_id)
        
        logger.info(f"Creating plan for task: {task.task_id}")
        plan = self._planner.create_plan(user_request, task.task_id)
        task.plan = plan
        
        ok, errors = self._plan_validator.validate(plan)
        if not ok:
            task.status = TaskStatus.FAILED
            task.error = f"Invalid plan: {'; '.join(errors)}"
            task.updated_at = time.time()
            return task
        
        logger.info(f"Executing task {task.task_id} with {len(plan.steps)} steps")
        task = self._task_executor.execute_task(task)
        
        if self._task_persistence:
            self._task_persistence.save_task(task)
        
        self._current_task = task
        return task
    
    def execute_autonomous(self, user_request: str) -> 'AutonomousTaskState':
        """Execute a task using the autonomous loop.
        
        Args:
            user_request: Natural language objective from user.
            
        Returns:
            AutonomousTaskState with full execution results.
        """
        if not self._autonomous_loop:
            raise RuntimeError("Autonomous loop not initialized")
        
        logger.info(f"Autonomous execution: {user_request[:100]}...")
        state = self._autonomous_loop.execute(user_request)
        
        # Store task result in memory if successful
        if state.phase.value == "completed" and self._long_term_memory:
            try:
                from ..memory.base import MemoryType, MemoryRecord
                record = MemoryRecord(
                    content=f"Completed task: {user_request[:200]}. Result: {(state.result or '')[:200]}",
                    memory_type=MemoryType.TASK_INFO,
                    importance=0.7,
                    confidence=0.9,
                )
                self._long_term_memory.store(record)
            except Exception as e:
                logger.warning(f"Failed to store task memory: {e}")
        
        return state
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        if self._task_persistence:
            return self._task_persistence.load_task(task_id)
        return None
    
    def list_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent tasks."""
        if self._task_persistence:
            return self._task_persistence.list_tasks(limit=limit)
        return []
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if self._current_task and self._current_task.task_id == task_id:
            self._task_executor.cancel_task(self._current_task)
            if self._task_persistence:
                self._task_persistence.save_task(self._current_task)
            return True
        return False
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration system statistics."""
        stats = {
            "initialized": self._planner is not None,
            "limits": self._orchestration_limits.to_dict() if self._orchestration_limits else {},
            "task_count": self._task_persistence.get_task_count() if self._task_persistence else 0,
        }
        if self._current_task:
            stats["current_task"] = {
                "task_id": self._current_task.task_id,
                "status": self._current_task.status.value,
                "progress": f"{len(self._current_task.completed_steps)}/{len(self._current_task.plan.steps) if self._current_task.plan else 0}",
            }
        return stats
    
    def store_memory(self, content: str, memory_type: str = "fact",
                     importance: float = 0.7, confidence: float = 0.8) -> bool:
        """Store a memory in long-term storage.
        
        Args:
            content: The memory content.
            memory_type: Type of memory (fact, user_preference, decision, etc.).
            importance: Importance score (0.0 to 1.0).
            confidence: Confidence score (0.0 to 1.0).
            
        Returns:
            True if stored successfully.
        """
        if not self._long_term_memory:
            return False
        
        session_id = None
        if self._conversation_manager and self._conversation_manager.current_session:
            session_id = self._conversation_manager.current_session.session_id
        
        record = MemoryRecord(
            content=content,
            memory_type=MemoryType(memory_type),
            importance=importance,
            confidence=confidence,
            session_id=session_id,
        )
        
        return self._long_term_memory.store(record)
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search long-term memories.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List of memory records as dictionaries.
        """
        if not self._long_term_memory:
            return []
        
        memories = self._long_term_memory.retrieve(query=query, limit=limit)
        return [m.to_dict() for m in memories]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        stats = {
            "session_manager": None,
            "conversation": None,
            "long_term_memory": None,
            "context": None,
            "summarizer": None,
        }
        
        if self._session_manager:
            stats["session_manager"] = {
                "session_count": self._session_manager.get_session_count(),
            }
        
        if self._conversation_manager:
            stats["conversation"] = self._conversation_manager.get_conversation_stats()
        
        if self._long_term_memory:
            stats["long_term_memory"] = self._long_term_memory.health_check()
        
        if self._context_manager:
            stats["context"] = self._context_manager.get_context_stats()
        
        if self._summarizer:
            stats["summarizer"] = self._summarizer.get_summarization_stats()
        
        return stats
    
    def get_media_stats(self) -> Dict[str, Any]:
        """Get media system statistics."""
        stats = {
            "initialized": self._media_router is not None,
            "providers": [],
            "storage": None,
        }
        
        if self._media_router:
            stats["providers"] = self._media_router.list_providers()
            stats["router_stats"] = self._media_router.get_stats()
        
        if self._media_storage:
            stats["storage"] = self._media_storage.get_storage_stats()
        
        return stats
    
    def start_new_session(self, title: Optional[str] = None) -> str:
        """Start a new conversation session.
        
        Args:
            title: Optional session title.
            
        Returns:
            The new session ID.
        """
        if not self._conversation_manager:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        session = self._conversation_manager.start_conversation(title)
        return session.session_id
    
    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions."""
        if not self._session_manager:
            return []
        return self._session_manager.list_sessions(limit=limit)
    
    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()
        if self._conversation_manager:
            self._conversation_manager.clear_conversation()
        logger.debug("Conversation history cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the agent and all components."""
        status = {
            "agent": {
                "initialized": self._llm_provider is not None and self._llm_provider.is_initialized,
                "project": self.config.project_name,
                "version": self.config.version,
                "stage": self.config.stage,
            },
            "llm": {},
            "config": {
                "provider": self.config.llm_provider,
                "model": self.config.model_name,
            },
            "memory": {
                "initialized": self._session_manager is not None,
            },
            "tools": {
                "initialized": self._tool_registry is not None,
            },
            "orchestration": {
                "initialized": self._planner is not None,
                "autonomous_loop": self._autonomous_loop is not None,
            },
            "media": {
                "initialized": self._media_router is not None,
            },
            "vision_pipeline": {
                "initialized": self._vision_pipeline is not None,
                "vl_available": self._vision_pipeline.is_vl_available if self._vision_pipeline else False,
            },
            "vision_decision": {
                "initialized": self._vision_decision is not None,
                "stats": self._vision_decision.get_stats() if self._vision_decision else {},
            },
            "os_control": {
                "initialized": self.config.os_control_enabled,
                "screen_capture": "PASS" if self._screen_capture_tool else "DISABLED",
                "system_info": "PASS" if self._system_info_tool else "DISABLED",
                "mouse": "ENABLED" if self.config.mouse_control_enabled else "DISABLED",
                "keyboard": "ENABLED" if self.config.keyboard_control_enabled else "DISABLED",
                "window": "ENABLED" if self.config.window_control_enabled else "DISABLED",
                "window_control": "ENABLED" if self.config.window_control_enabled else "DISABLED",
            },
            "browser": {
                "enabled": self.config.browser_automation_enabled,
                "headless": self.config.browser_headless,
                "max_sessions": self.config.browser_max_sessions,
                "initialized": self._browser_manager is not None and self._browser_manager.initialized,
                "page_read": "ENABLED" if self.config.browser_automation_enabled else "DISABLED",
                "screenshot": "ENABLED" if (self.config.browser_screenshot_enabled and self.config.browser_automation_enabled) else "DISABLED",
            },
            "vision": {
                "enabled": self.config.vision_enabled,
                "provider": self.config.vision_provider_type,
                "initialized": self._vision_provider is not None,
                "analyzer_initialized": self._vision_analyzer is not None if hasattr(self, '_vision_analyzer') else False,
            },
        }
        
        if self._llm_provider:
            status["llm"] = self._llm_provider.health_check()
        
        if self._long_term_memory:
            status["memory"]["long_term"] = self._long_term_memory.health_check()
        
        if self._conversation_manager:
            status["memory"]["conversation"] = self._conversation_manager.get_conversation_stats()
        
        if self._tool_registry:
            status["tools"]["count"] = self._tool_registry.count()
            status["tools"]["available"] = self._tool_registry.list_names()
        
        if self._orchestration_limits:
            status["orchestration"]["limits"] = self._orchestration_limits.to_dict()
        
        # Add model health status
        if hasattr(self, '_model_health'):
            status["model_health"] = self._model_health.to_dict()
        
        return status
    
    def model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information.
        """
        if self._llm_provider:
            return self._llm_provider.model_info()
        return {"status": "no_provider"}
    
    def shutdown(self):
        """Shutdown the agent and free resources."""
        logger.info("Shutting down agent...")
        
        if self._conversation_manager and self._conversation_manager.current_session:
            self._session_manager.save_session(self._conversation_manager.current_session)
        
        if self._llm_provider:
            self._llm_provider.unload()
            self._llm_provider = None
        
        self._conversation_history.clear()
        self._session_manager = None
        self._conversation_manager = None
        self._long_term_memory = None
        self._context_manager = None
        self._summarizer = None
        self._tool_registry = None
        self._tool_router = None
        self._permission_manager = None
        self._audit_logger = None
        self._planner = None
        self._plan_validator = None
        self._task_executor = None
        self._verifier = None
        self._task_persistence = None
        self._event_logger = None
        self._orchestration_limits = None
        self._current_task = None
        self._media_storage = None
        self._media_router = None
        self._vision_provider = None
        self._vision_analyzer = None
        self._vision_analyze_tool = None
        self._visual_grounder = None
        self._visual_grounding_tool = None
        self._observe_act_verify_tool = None
        self._image_gen_provider = None
        self._video_gen_provider = None
        self._screen_capture_tool = None
        self._system_info_tool = None
        self._mouse_tool = None
        self._keyboard_tool = None
        self._window_tool = None
        self._autonomous_loop = None
        self._vision_pipeline = None
        self._vision_decision = None
        self._multimodal_pipeline = None
        
        if self._browser_manager:
            self._browser_manager.shutdown()
        self._browser_manager = None
        self._browser_session_tool = None
        self._browser_navigation_tool = None
        self._browser_page_read_tool = None
        self._browser_interaction_tool = None
        self._browser_screenshot_tool = None
        
        logger.info("Agent shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
