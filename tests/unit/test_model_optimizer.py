"""Tests for Stage 6.1 - Local Model Optimization."""

import pytest
from agent.llm.model_optimizer import (
    ModelOptimizer, ModelConfig, PerformanceMetrics,
    OptimizationResult, OptimizationLevel,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_ctx == 4096
        assert cfg.n_gpu_layers == 28
        assert cfg.flash_attn is True

    def test_to_dict(self):
        cfg = ModelConfig(n_ctx=8192)
        d = cfg.to_dict()
        assert d["n_ctx"] == 8192
        assert d["flash_attn"] is True


class TestPerformanceMetrics:
    def test_creation(self):
        m = PerformanceMetrics(tokens_per_second=30.5, prompt_tokens=100)
        assert m.tokens_per_second == 30.5

    def test_to_dict(self):
        m = PerformanceMetrics(tokens_per_second=25.0, completion_tokens=50)
        d = m.to_dict()
        assert d["tokens_per_second"] == 25.0

    def test_to_text(self):
        m = PerformanceMetrics(
            tokens_per_second=35.0,
            prompt_tokens=100,
            completion_tokens=50,
            total_time=5.0,
            first_token_time=0.5,
            gpu_memory_used_mb=3000,
            gpu_memory_total_mb=6000,
        )
        text = m.to_text()
        assert "35.0" in text
        assert "GPU" in text


class TestOptimizationResult:
    def test_creation(self):
        r = OptimizationResult(
            success=True, level="balanced",
            config_before={}, config_after={},
        )
        assert r.success is True

    def test_to_dict(self):
        r = OptimizationResult(
            success=True, level="balanced",
            config_before={"n_ctx": 2048},
            config_after={"n_ctx": 4096},
            improvements=["Increased context"],
        )
        d = r.to_dict()
        assert d["success"] is True
        assert len(d["improvements"]) == 1


class TestModelOptimizer:
    def test_init(self):
        opt = ModelOptimizer()
        assert opt._gpu_mb == 6144

    def test_optimize_conservative(self):
        opt = ModelOptimizer()
        opt.set_config(ModelConfig(n_gpu_layers=99, n_threads=16, n_batch=2048))
        r = opt.optimize(OptimizationLevel.CONSERVATIVE)
        assert r.success is True
        assert r.level == "conservative"
        assert len(r.improvements) > 0

    def test_optimize_balanced(self):
        opt = ModelOptimizer()
        r = opt.optimize(OptimizationLevel.BALANCED)
        assert r.success is True
        assert r.level == "balanced"
        assert opt._config.n_gpu_layers == 28

    def test_optimize_aggressive(self):
        opt = ModelOptimizer()
        r = opt.optimize(OptimizationLevel.AGGRESSIVE)
        assert r.success is True
        assert r.level == "aggressive"
        assert opt._config.n_gpu_layers == 99
        assert len(r.warnings) > 0

    def test_adjust_for_task_code(self):
        opt = ModelOptimizer()
        cfg = opt.adjust_for_task("code")
        assert cfg.n_ctx == 4096
        assert cfg.n_batch == 1024

    def test_adjust_for_task_chat(self):
        opt = ModelOptimizer()
        cfg = opt.adjust_for_task("chat")
        assert cfg.n_ctx == 2048

    def test_adjust_for_task_analysis(self):
        opt = ModelOptimizer()
        cfg = opt.adjust_for_task("analysis")
        assert cfg.n_ctx == 8192

    def test_adjust_for_task_vision(self):
        opt = ModelOptimizer()
        cfg = opt.adjust_for_task("vision")
        assert cfg.n_ctx == 4096

    def test_record_metrics(self):
        opt = ModelOptimizer()
        m = PerformanceMetrics(tokens_per_second=30.0)
        opt.record_metrics(m)
        assert len(opt._metrics_history) == 1

    def test_get_average_metrics(self):
        opt = ModelOptimizer()
        for i in range(5):
            opt.record_metrics(PerformanceMetrics(tokens_per_second=20 + i))
        avg = opt.get_average_metrics()
        assert avg.tokens_per_second == 22.0

    def test_get_average_empty(self):
        opt = ModelOptimizer()
        avg = opt.get_average_metrics()
        assert avg.tokens_per_second == 0.0

    def test_get_recommendations(self):
        opt = ModelOptimizer()
        opt.record_metrics(PerformanceMetrics(tokens_per_second=5.0))
        recs = opt.get_recommendations()
        assert len(recs) > 0

    def test_get_recommendations_good(self):
        opt = ModelOptimizer()
        opt.record_metrics(PerformanceMetrics(
            tokens_per_second=30.0,
            gpu_memory_used_mb=3000,
            gpu_memory_total_mb=6000,
            first_token_time=0.3,
        ))
        recs = opt.get_recommendations()
        assert "good" in recs[0].lower() or len(recs) == 1

    def test_get_config(self):
        opt = ModelOptimizer()
        cfg = opt.get_config()
        assert isinstance(cfg, ModelConfig)

    def test_set_config(self):
        opt = ModelOptimizer()
        new_cfg = ModelConfig(n_ctx=8192)
        opt.set_config(new_cfg)
        assert opt.get_config().n_ctx == 8192

    def test_benchmark_config(self):
        opt = ModelOptimizer()
        cfg = ModelConfig(n_gpu_layers=28, n_batch=1024, flash_attn=True)
        result = opt.benchmark_config(cfg)
        assert "estimated_tokens_per_second" in result
        assert result["estimated_tokens_per_second"] > 0

    def test_metrics_history_limit(self):
        opt = ModelOptimizer()
        for i in range(150):
            opt.record_metrics(PerformanceMetrics(tokens_per_second=float(i)))
        assert len(opt._metrics_history) == 100
