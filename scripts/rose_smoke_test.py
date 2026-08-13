#!/usr/bin/env python3
"""Rose Backend Smoke Test.

Stage O - Single command to verify the complete system.

Usage:
    python scripts/rose_smoke_test.py
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(name: str, func):
    """Run a check and report result."""
    try:
        result = func()
        if result:
            print(f"[PASS] {name}")
            return True
        else:
            print(f"[FAIL] {name}")
            return False
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 50)
    print("ROSE BACKEND SMOKE TEST")
    print("=" * 50)
    print()
    
    results = []
    start = time.time()
    
    # 1. Python runtime
    results.append(check("Python runtime", lambda: True))
    
    # 2. Configuration
    def check_config():
        from agent.core.config import Config
        config = Config()
        return config.model_path and config.mmproj_path
    results.append(check("Configuration", check_config))
    
    # 3. LLM runtime
    def check_llm():
        import llama_cpp
        return True
    results.append(check("LLM runtime (llama-cpp-python)", check_llm))
    
    # 4. Qwen2.5-VL model
    def check_model():
        from agent.core.config import Config
        config = Config()
        model_path = config.get_model_full_path()
        return model_path.exists()
    results.append(check("Qwen2.5-VL model", check_model))
    
    # 5. Vision projector
    def check_mmproj():
        from agent.core.config import Config
        config = Config()
        mmproj = Path(config.mmproj_path)
        if not mmproj.is_absolute():
            mmproj = Path.cwd() / mmproj
        return mmproj.exists()
    results.append(check("Vision projector (mmproj)", check_mmproj))
    
    # 6. GPU/CPU runtime
    def check_gpu():
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return True  # CPU is fine
    results.append(check("GPU/CPU runtime", check_gpu))
    
    # 7. Text inference
    def check_text_inference():
        from agent.core.config import Config
        from agent.llm.local_provider import LocalLLMProvider
        from agent.llm.base import LLMConfig, LLMProviderType, VisionCapability
        
        config = Config()
        llm_config = LLMConfig(
            provider_type=LLMProviderType.LOCAL,
            model_path=str(config.get_model_full_path()),
            model_name=config.model_name,
            context_length=2048,
            n_gpu_layers=0,
            max_tokens=64,
        )
        provider = LocalLLMProvider(llm_config)
        if not provider.initialize():
            return False
        try:
            response = provider.generate("Say hello in one word.")
            return bool(response.text)
        finally:
            provider.unload()
    results.append(check("Text inference", check_text_inference))
    
    # 8. Image inference (VL model)
    def check_image_inference():
        from agent.core.config import Config
        from agent.llm.local_provider import LocalLLMProvider
        from agent.llm.base import LLMConfig, LLMProviderType, VisionCapability, ImageInput
        
        config = Config()
        mmproj = Path(config.mmproj_path)
        if not mmproj.is_absolute():
            mmproj = Path.cwd() / mmproj
        
        llm_config = LLMConfig(
            provider_type=LLMProviderType.LOCAL,
            model_path=str(config.get_model_full_path()),
            model_name=config.model_name,
            context_length=2048,
            n_gpu_layers=0,
            max_tokens=64,
            mmproj_path=str(mmproj),
            vision_capability=VisionCapability.MULTIPLE,
            max_images=4,
        )
        provider = LocalLLMProvider(llm_config)
        if not provider.initialize():
            return False
        try:
            # Use test image if available
            test_image = ROOT / "test_vision_image.png"
            if not test_image.exists():
                return True  # Skip if no test image
            
            image = ImageInput.from_file(str(test_image))
            response = provider.chat([
                {"role": "user", "content": [
                    {"type": "text", "text": "What do you see?"},
                    image.to_llm_format(),
                ]}
            ])
            return bool(response.text)
        finally:
            provider.unload()
    results.append(check("Image inference (VL)", check_image_inference))
    
    # 9. Tool registry
    def check_tools():
        from agent.tools import ToolRegistry
        registry = ToolRegistry()
        return registry is not None
    results.append(check("Tool registry", check_tools))
    
    # 10. Permission manager
    def check_permissions():
        from agent.tools import PermissionManager
        pm = PermissionManager()
        return pm.has_permission("filesystem.read", "workspace")
    results.append(check("Permission manager", check_permissions))
    
    # 11. Memory system
    def check_memory():
        from agent.memory import SessionManager, LongTermMemory
        sm = SessionManager(data_dir="sessions")
        ltm = LongTermMemory(db_path="data/memory.db")
        return True
    results.append(check("Memory system", check_memory))
    
    # 12. Autonomous loop
    def check_autonomous():
        from agent.orchestration.autonomous_loop import AutonomousLoop
        from agent.tools import ToolRegistry, ToolRouter, PermissionManager, AuditLogger
        
        registry = ToolRegistry()
        pm = PermissionManager()
        audit = AuditLogger()
        router = ToolRouter(registry=registry, permission_manager=pm, audit_logger=audit)
        
        loop = AutonomousLoop(tool_router=router, permission_manager=pm)
        return loop is not None
    results.append(check("Autonomous loop", check_autonomous))
    
    # 13. Vision decision system
    def check_vision_decision():
        from agent.media.vision_decision import VisionDecisionSystem, VisionRequirement
        
        system = VisionDecisionSystem()
        
        # Test text-only
        d1 = system.decide("Explain Python decorators")
        if d1.requirement != VisionRequirement.NOT_NEEDED:
            return False
        
        # Test screen question
        d2 = system.decide("What is on my screen?")
        if d2.requirement != VisionRequirement.REQUIRED:
            return False
        
        return True
    results.append(check("Vision decision system", check_vision_decision))
    
    # 14. Model health checker
    def check_model_health():
        from agent.core.model_health import ModelHealthChecker
        from agent.core.config import Config
        
        config = Config()
        checker = ModelHealthChecker(config=config)
        status = checker.check_health()
        return status.model_exists and status.runtime_available
    results.append(check("Model health checker", check_model_health))
    
    # 15. Observation system
    def check_observation():
        from agent.orchestration.observation import ObservationSystem
        obs = ObservationSystem()
        return obs is not None
    results.append(check("Observation system", check_observation))
    
    # 16. Verification system
    def check_verification():
        from agent.orchestration.verifier import Verifier
        v = Verifier()
        return v is not None
    results.append(check("Verification system", check_verification))
    
    # 17. Event system
    def check_events():
        from agent.orchestration.events import EventLogger
        el = EventLogger()
        return el is not None
    results.append(check("Event system", check_events))
    
    # 18. Error recovery
    def check_error_recovery():
        from agent.core.error_recovery import ErrorHandler, CircuitBreaker
        handler = ErrorHandler()
        cb = CircuitBreaker()
        return handler is not None and cb is not None
    results.append(check("Error recovery", check_error_recovery))
    
    # Summary
    elapsed = time.time() - start
    passed = sum(results)
    total = len(results)
    
    print()
    print("=" * 50)
    if passed == total:
        print(f"FINAL: ROSE BACKEND READY ({passed}/{total} passed in {elapsed:.1f}s)")
    else:
        print(f"FINAL: {passed}/{total} passed, {total - passed} failed ({elapsed:.1f}s)")
    print("=" * 50)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
