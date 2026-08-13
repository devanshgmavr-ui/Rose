#!/usr/bin/env python3
"""Rose Performance Validation.

Stage R - Measures performance characteristics of the backend.

Usage:
    python scripts/rose_perf_test.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def measure(name: str, func, iterations: int = 1):
    """Measure execution time of a function."""
    times = []
    for _ in range(iterations):
        start = time.time()
        try:
            result = func()
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception as e:
            return None, str(e)
    
    avg = sum(times) / len(times)
    return avg, None


def main():
    """Run performance measurements."""
    print("=" * 60)
    print("ROSE PERFORMANCE VALIDATION")
    print("=" * 60)
    print()
    
    results = []
    
    # 1. Config initialization
    avg, err = measure("Config initialization", lambda: __import__('agent.core.config', fromlist=['Config']).Config(), iterations=3)
    results.append(("Config initialization", avg, err))
    print(f"Config initialization:     {avg*1000:.1f}ms" if avg else f"Config initialization:     FAIL ({err})")
    
    # 2. Tool registry creation
    def create_registry():
        from agent.tools import ToolRegistry
        return ToolRegistry()
    avg, err = measure("Tool registry creation", create_registry, iterations=3)
    results.append(("Tool registry creation", avg, err))
    print(f"Tool registry creation:    {avg*1000:.1f}ms" if avg else f"Tool registry creation:    FAIL ({err})")
    
    # 3. Permission check
    def check_permission():
        from agent.tools import PermissionManager
        pm = PermissionManager()
        return pm.has_permission("filesystem.read", "workspace")
    avg, err = measure("Permission check", check_permission, iterations=10)
    results.append(("Permission check", avg, err))
    print(f"Permission check:          {avg*1000:.2f}ms" if avg else f"Permission check:          FAIL ({err})")
    
    # 4. Vision decision
    def vision_decision():
        from agent.media.vision_decision import VisionDecisionSystem
        system = VisionDecisionSystem()
        return system.decide("What is on my screen?")
    avg, err = measure("Vision decision", vision_decision, iterations=10)
    results.append(("Vision decision", avg, err))
    print(f"Vision decision:           {avg*1000:.2f}ms" if avg else f"Vision decision:           FAIL ({err})")
    
    # 5. Model health check
    def model_health():
        from agent.core.model_health import ModelHealthChecker
        from agent.core.config import Config
        config = Config()
        checker = ModelHealthChecker(config=config)
        return checker.check_health()
    avg, err = measure("Model health check", model_health, iterations=3)
    results.append(("Model health check", avg, err))
    print(f"Model health check:        {avg*1000:.1f}ms" if avg else f"Model health check:        FAIL ({err})")
    
    # 6. Memory operation
    def memory_store():
        from agent.memory import LongTermMemory
        from agent.memory.base import MemoryType, MemoryRecord
        ltm = LongTermMemory(db_path="data/perf_test.db")
        record = MemoryRecord(
            content="Performance test memory",
            memory_type=MemoryType.FACT,
            importance=0.5,
            confidence=0.8,
        )
        return ltm.store(record)
    avg, err = measure("Memory store", memory_store, iterations=5)
    results.append(("Memory store", avg, err))
    print(f"Memory store:              {avg*1000:.1f}ms" if avg else f"Memory store:              FAIL ({err})")
    
    # Cleanup
    try:
        Path("data/perf_test.db").unlink()
    except Exception:
        pass
    
    # 7. Audit log write
    def audit_write():
        from agent.tools.audit import AuditLogger, AuditRecord
        audit = AuditLogger(log_dir="logs")
        record = AuditRecord(tool_name="test", arguments={"key": "value"})
        return audit.log_execution(record)
    avg, err = measure("Audit log write", audit_write, iterations=5)
    results.append(("Audit log write", avg, err))
    print(f"Audit log write:           {avg*1000:.1f}ms" if avg else f"Audit log write:           FAIL ({err})")
    
    # 8. Error handler creation
    def error_handler():
        from agent.core.error_recovery import ErrorHandler, CircuitBreaker
        return ErrorHandler(), CircuitBreaker()
    avg, err = measure("Error handler creation", error_handler, iterations=5)
    results.append(("Error handler creation", avg, err))
    print(f"Error handler creation:    {avg*1000:.2f}ms" if avg else f"Error handler creation:    FAIL ({err})")
    
    # Summary
    print()
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, _, err in results if err is None)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print()
    print("Practical expectations:")
    print("- Config init: <50ms")
    print("- Permission check: <1ms")
    print("- Vision decision: <1ms")
    print("- Model health check: <100ms")
    print("- Memory store: <50ms")
    print("- Audit log write: <10ms")
    print()
    print("Note: VL inference latency depends on GPU/CPU.")
    print("Expected: ~69s per image on RTX 4050 (CPU offload).")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
