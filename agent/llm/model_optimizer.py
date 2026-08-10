"""Local model optimization and performance tuning.

Stage 6.1 - Local Model Optimization.

Provides:
- GPU memory optimization
- Context window management
- Batch inference optimization
- Model configuration tuning
- Performance monitoring
- Auto-tuning
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class ModelConfig:
    n_ctx: int = 4096
    n_gpu_layers: int = 28
    n_threads: int = 4
    n_batch: int = 512
    rope_freq_base: float = 1000000.0
    rope_freq_scale: float = 1.0
    use_mmap: bool = True
    use_mlock: bool = False
    vocab_only: bool = False
    flash_attn: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "rope_freq_base": self.rope_freq_base,
            "rope_freq_scale": self.rope_freq_scale,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "vocab_only": self.vocab_only,
            "flash_attn": self.flash_attn,
        }


@dataclass
class PerformanceMetrics:
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time: float = 0.0
    first_token_time: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    ram_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_time": self.total_time,
            "first_token_time": self.first_token_time,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "ram_usage_mb": self.ram_usage_mb,
        }

    def to_text(self) -> str:
        lines = [
            f"Speed: {self.tokens_per_second:.1f} tok/s",
            f"Tokens: {self.prompt_tokens} prompt + {self.completion_tokens} completion",
            f"Time: {self.total_time:.2f}s (first token: {self.first_token_time:.2f}s)",
        ]
        if self.gpu_memory_total_mb > 0:
            pct = (self.gpu_memory_used_mb / self.gpu_memory_total_mb) * 100
            lines.append(f"GPU: {self.gpu_memory_used_mb:.0f}/{self.gpu_memory_total_mb:.0f} MB ({pct:.0f}%)")
        return "\n".join(lines)


@dataclass
class OptimizationResult:
    success: bool
    level: str
    config_before: Dict[str, Any]
    config_after: Dict[str, Any]
    improvements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "level": self.level,
            "config_before": self.config_before,
            "config_after": self.config_after,
            "improvements": self.improvements,
            "warnings": self.warnings,
        }


class ModelOptimizer:
    """Optimizes local model performance."""

    def __init__(
        self,
        gpu_memory_mb: float = 6144,
        cpu_cores: int = 8,
        ram_gb: float = 16,
        current_config: Optional[ModelConfig] = None,
    ):
        self._gpu_mb = gpu_memory_mb
        self._cpu_cores = cpu_cores
        self._ram_gb = ram_gb
        self._config = current_config or ModelConfig()
        self._metrics_history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()

    def optimize(
        self, level: OptimizationLevel = OptimizationLevel.BALANCED
    ) -> OptimizationResult:
        """Optimize model configuration."""
        before = self._config.to_dict()
        improvements = []
        warnings = []

        if level == OptimizationLevel.CONSERVATIVE:
            self._optimize_conservative(improvements, warnings)
        elif level == OptimizationLevel.BALANCED:
            self._optimize_balanced(improvements, warnings)
        else:
            self._optimize_aggressive(improvements, warnings)

        after = self._config.to_dict()
        changed = any(before[k] != after[k] for k in before)

        return OptimizationResult(
            success=changed,
            level=level.value,
            config_before=before,
            config_after=after,
            improvements=improvements,
            warnings=warnings,
        )

    def _optimize_conservative(
        self, improvements: List[str], warnings: List[str]
    ):
        """Conservative optimization: safe defaults."""
        self._config.n_gpu_layers = min(self._config.n_gpu_layers, 28)
        self._config.n_threads = min(self._config.n_threads, self._cpu_cores // 2)
        self._config.n_batch = min(self._config.n_batch, 512)
        self._config.flash_attn = True
        improvements.append("Set conservative GPU layer limit")
        improvements.append("Enabled flash attention")

    def _optimize_balanced(
        self, improvements: List[str], warnings: List[str]
    ):
        """Balanced optimization: good performance with safety."""
        self._config.n_gpu_layers = 28
        self._config.n_threads = min(self._cpu_cores, 8)
        self._config.n_batch = 1024
        self._config.flash_attn = True
        self._config.use_mmap = True
        self._config.use_mlock = False
        improvements.append("Maximized GPU layers for speed")
        improvements.append("Optimized batch size for throughput")
        improvements.append("Enabled memory-mapped files")
        improvements.append("Enabled flash attention")

    def _optimize_aggressive(
        self, improvements: List[str], warnings: List[str]
    ):
        """Aggressive optimization: maximum performance."""
        self._config.n_gpu_layers = 99
        self._config.n_threads = self._cpu_cores
        self._config.n_batch = 2048
        self._config.flash_attn = True
        self._config.use_mmap = True
        self._config.use_mlock = True
        self._config.rope_freq_base = 1000000.0
        improvements.append("Set maximum GPU layers (all layers on GPU)")
        improvements.append("Maximized CPU threads")
        improvements.append("Doubled batch size")
        improvements.append("Enabled memory lock")
        warnings.append("Aggressive settings may cause OOM on limited VRAM")

    def adjust_for_task(
        self, task_type: str, config: Optional[ModelConfig] = None
    ) -> ModelConfig:
        """Adjust config for specific task type."""
        cfg = config or ModelConfig()

        if task_type == "code":
            cfg.n_ctx = 4096
            cfg.n_batch = 1024
        elif task_type == "chat":
            cfg.n_ctx = 2048
            cfg.n_batch = 512
        elif task_type == "analysis":
            cfg.n_ctx = 8192
            cfg.n_batch = 2048
        elif task_type == "vision":
            cfg.n_ctx = 4096
            cfg.n_batch = 512
        elif task_type == "long_context":
            cfg.n_ctx = 16384
            cfg.n_batch = 2048
            warnings = ["Long context may exceed VRAM capacity"]
            logger.warning(warnings[0])

        return cfg

    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]

    def get_average_metrics(self, last_n: int = 10) -> PerformanceMetrics:
        """Get average metrics over last N measurements."""
        with self._lock:
            recent = self._metrics_history[-last_n:]
            if not recent:
                return PerformanceMetrics()

            avg = PerformanceMetrics(
                tokens_per_second=sum(m.tokens_per_second for m in recent) / len(recent),
                prompt_tokens=sum(m.prompt_tokens for m in recent) // len(recent),
                completion_tokens=sum(m.completion_tokens for m in recent) // len(recent),
                total_time=sum(m.total_time for m in recent) / len(recent),
                first_token_time=sum(m.first_token_time for m in recent) / len(recent),
                gpu_memory_used_mb=sum(m.gpu_memory_used_mb for m in recent) / len(recent),
                gpu_memory_total_mb=recent[-1].gpu_memory_total_mb,
                cpu_usage_percent=sum(m.cpu_usage_percent for m in recent) / len(recent),
                ram_usage_mb=sum(m.ram_usage_mb for m in recent) / len(recent),
            )
            return avg

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on history."""
        recs = []
        avg = self.get_average_metrics()

        if avg.tokens_per_second > 0 and avg.tokens_per_second < 10:
            recs.append("Low token/s: consider reducing context or batch size")

        if avg.gpu_memory_total_mb > 0:
            usage_pct = (avg.gpu_memory_used_mb / avg.gpu_memory_total_mb) * 100
            if usage_pct > 90:
                recs.append("High GPU memory usage: consider fewer GPU layers")
            elif usage_pct < 50:
                recs.append("Low GPU memory usage: can increase GPU layers")

        if avg.first_token_time > 2.0:
            recs.append("Slow first token: consider enabling flash attention")

        if not recs:
            recs.append("Performance looks good!")

        return recs

    def get_config(self) -> ModelConfig:
        """Get current configuration."""
        return self._config

    def set_config(self, config: ModelConfig):
        """Set configuration."""
        self._config = config

    def benchmark_config(
        self, config: ModelConfig, iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark a configuration (simulated)."""
        base_speed = 25.0
        speed = base_speed

        speed *= (config.n_gpu_layers / 28)
        speed *= (config.n_batch / 512)
        if config.flash_attn:
            speed *= 1.2

        return {
            "config": config.to_dict(),
            "estimated_tokens_per_second": round(speed, 1),
            "estimated_vram_mb": round(config.n_gpu_layers * 220, 0),
            "iterations": iterations,
        }
