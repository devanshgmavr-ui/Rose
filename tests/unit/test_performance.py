"""Tests for Stage 6.2 - Performance Monitoring."""

import time
import pytest
from agent.core.performance import (
    PerformanceMonitor, Metric, MetricType, PerformanceSnapshot,
    Alert, TimerContext,
)


class TestMetric:
    def test_creation(self):
        m = Metric(name="cpu", value=50.0, metric_type=MetricType.GAUGE)
        assert m.name == "cpu"
        assert m.value == 50.0

    def test_to_dict(self):
        m = Metric(name="gpu", value=80.0, metric_type=MetricType.COUNTER, labels={"device": "cuda"})
        d = m.to_dict()
        assert d["name"] == "gpu"
        assert d["labels"]["device"] == "cuda"


class TestPerformanceSnapshot:
    def test_creation(self):
        s = PerformanceSnapshot(timestamp=1.0, cpu_percent=45.0, ram_used_mb=8000)
        assert s.cpu_percent == 45.0
        assert s.ram_used_mb == 8000

    def test_to_dict(self):
        s = PerformanceSnapshot(timestamp=1.0, cpu_percent=50.0)
        d = s.to_dict()
        assert d["cpu_percent"] == 50.0

    def test_to_text(self):
        s = PerformanceSnapshot(
            timestamp=1.0, cpu_percent=50.0,
            ram_used_mb=8000, ram_total_mb=16000,
            active_threads=10,
        )
        text = s.to_text()
        assert "8000" in text
        assert "CPU" in text


class TestAlert:
    def test_creation(self):
        a = Alert(level="warning", message="High CPU", timestamp=1.0)
        assert a.level == "warning"

    def test_to_dict(self):
        a = Alert(level="critical", message="OOM", timestamp=2.0, metric_name="ram", value=95)
        d = a.to_dict()
        assert d["level"] == "critical"
        assert d["value"] == 95


class TestPerformanceMonitor:
    def test_init(self):
        mon = PerformanceMonitor()
        assert mon._gpu_total == 6144

    def test_record_metric(self):
        mon = PerformanceMonitor()
        mon.record_metric("cpu", 50.0)
        history = mon.get_metric_history("cpu")
        assert len(history) == 1
        assert history[0].value == 50.0

    def test_record_snapshot(self):
        mon = PerformanceMonitor()
        s = PerformanceSnapshot(timestamp=1.0, cpu_percent=40.0)
        mon.record_snapshot(s)
        latest = mon.get_latest_snapshot()
        assert latest.cpu_percent == 40.0

    def test_get_metric_history(self):
        mon = PerformanceMonitor()
        for i in range(10):
            mon.record_metric("cpu", float(i))
        history = mon.get_metric_history("cpu", last_n=5)
        assert len(history) == 5

    def test_get_average(self):
        mon = PerformanceMonitor()
        for i in range(5):
            mon.record_metric("cpu", float(i * 10))
        avg = mon.get_average("cpu")
        assert avg == 20.0

    def test_get_max(self):
        mon = PerformanceMonitor()
        for v in [10, 50, 30]:
            mon.record_metric("cpu", float(v))
        assert mon.get_max("cpu") == 50.0

    def test_get_min(self):
        mon = PerformanceMonitor()
        for v in [10, 50, 30]:
            mon.record_metric("cpu", float(v))
        assert mon.get_min("cpu") == 10.0

    def test_get_percentile(self):
        mon = PerformanceMonitor()
        for i in range(100):
            mon.record_metric("cpu", float(i))
        p95 = mon.get_percentile("cpu", 95.0)
        assert p95 >= 94.0

    def test_alert_trigger(self):
        mon = PerformanceMonitor(alert_threshold_cpu=80.0)
        alerts = []
        mon.register_alert_callback(lambda a: alerts.append(a))
        mon.record_metric("cpu_usage", 90.0)
        assert len(alerts) == 1
        assert alerts[0].level == "warning"

    def test_alert_no_trigger(self):
        mon = PerformanceMonitor(alert_threshold_cpu=90.0)
        alerts = []
        mon.register_alert_callback(lambda a: alerts.append(a))
        mon.record_metric("cpu_usage", 50.0)
        assert len(alerts) == 0

    def test_get_alerts(self):
        mon = PerformanceMonitor(alert_threshold_cpu=50.0)
        mon.record_metric("cpu_usage", 80.0)
        mon.record_metric("cpu_usage", 90.0)
        alerts = mon.get_alerts()
        assert len(alerts) == 2

    def test_clear_alerts(self):
        mon = PerformanceMonitor(alert_threshold_cpu=50.0)
        mon.record_metric("cpu_usage", 90.0)
        mon.clear_alerts()
        assert len(mon.get_alerts()) == 0

    def test_get_summary(self):
        mon = PerformanceMonitor()
        mon.record_metric("cpu", 50.0)
        mon.record_snapshot(PerformanceSnapshot(timestamp=1.0))
        summary = mon.get_summary()
        assert summary["total_metrics"] == 1
        assert summary["total_snapshots"] == 1

    def test_timer(self):
        mon = PerformanceMonitor()
        with mon.timer("operation"):
            time.sleep(0.05)
        history = mon.get_metric_history("operation")
        assert len(history) == 1
        assert history[0].value >= 0.04

    def test_get_snapshots(self):
        mon = PerformanceMonitor()
        for i in range(5):
            mon.record_snapshot(PerformanceSnapshot(timestamp=float(i)))
        snaps = mon.get_snapshots(last_n=3)
        assert len(snaps) == 3

    def test_get_average_empty(self):
        mon = PerformanceMonitor()
        assert mon.get_average("nonexistent") == 0.0

    def test_get_max_empty(self):
        mon = PerformanceMonitor()
        assert mon.get_max("nonexistent") == 0.0
