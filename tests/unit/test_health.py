"""Tests for Stage 7.3 - System Health Check."""

import pytest
import os
from agent.core.health import (
    HealthChecker, ComponentHealth, HealthReport, HealthStatus,
)


class TestComponentHealth:
    def test_creation(self):
        h = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        assert h.name == "test"
        assert h.status == HealthStatus.HEALTHY

    def test_to_dict(self):
        h = ComponentHealth(
            name="disk", status=HealthStatus.DEGRADED,
            message="Low space", latency_ms=50.0,
        )
        d = h.to_dict()
        assert d["name"] == "disk"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 50.0


class TestHealthReport:
    def test_creation(self):
        r = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            components=[],
        )
        assert r.overall_status == HealthStatus.HEALTHY

    def test_to_dict(self):
        r = HealthReport(
            overall_status=HealthStatus.DEGRADED,
            components=[ComponentHealth("c1", HealthStatus.HEALTHY)],
        )
        d = r.to_dict()
        assert d["overall_status"] == "degraded"
        assert len(d["components"]) == 1

    def test_to_text(self):
        r = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            components=[ComponentHealth("c1", HealthStatus.HEALTHY, "OK")],
            recommendations=["Do this"],
        )
        text = r.to_text()
        assert "HEALTHY" in text
        assert "Do this" in text


class TestHealthChecker:
    def test_init(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        assert hc._workspace == str(tmp_path)

    def test_register_check(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        hc.register_check(lambda: ComponentHealth("custom", HealthStatus.HEALTHY))
        assert len(hc._checks) == 1

    def test_check_workspace(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        h = hc.check_workspace()
        assert h.status == HealthStatus.HEALTHY

    def test_check_workspace_nonexistent(self):
        hc = HealthChecker(workspace_dir="C:\\nonexistent_path_xyz")
        h = hc.check_workspace()
        assert h.status == HealthStatus.UNHEALTHY

    def test_check_disk_space(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        h = hc.check_disk_space()
        assert h.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def test_check_model_not_found(self):
        hc = HealthChecker()
        h = hc.check_model("C:\\nonexistent\\model.gguf")
        assert h.status == HealthStatus.UNHEALTHY

    def test_check_model_not_configured(self):
        hc = HealthChecker()
        h = hc.check_model()
        assert h.status == HealthStatus.UNKNOWN

    def test_check_all(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        hc.register_check(lambda: ComponentHealth("test", HealthStatus.HEALTHY))
        report = hc.check_all()
        assert report.overall_status == HealthStatus.HEALTHY
        assert len(report.components) == 1

    def test_check_all_with_failure(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        def bad_check():
            raise RuntimeError("check failed")
        hc.register_check(bad_check)
        report = hc.check_all()
        assert report.overall_status == HealthStatus.UNHEALTHY

    def test_check_all_mixed(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        hc.register_check(lambda: ComponentHealth("a", HealthStatus.HEALTHY))
        hc.register_check(lambda: ComponentHealth("b", HealthStatus.DEGRADED))
        report = hc.check_all()
        assert report.overall_status == HealthStatus.DEGRADED

    def test_get_history(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        hc.check_all()
        hc.check_all()
        history = hc.get_history()
        assert len(history) == 2

    def test_system_info(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        report = hc.check_all()
        assert "platform" in report.system_info
        assert "python" in report.system_info

    def test_recommendations(self, tmp_path):
        hc = HealthChecker(workspace_dir=str(tmp_path))
        hc.register_check(lambda: ComponentHealth("c1", HealthStatus.DEGRADED, "watch out"))
        report = hc.check_all()
        assert len(report.recommendations) > 0
