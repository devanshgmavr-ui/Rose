"""Performance monitoring and metrics collection.

Stage 6.2 - Performance Monitoring.

Provides:
- Real-time performance tracking
- Memory usage monitoring
- GPU utilization tracking
- Latency measurement
- Throughput monitoring
- Alerting on anomalies
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


@dataclass
class PerformanceSnapshot:
    timestamp: float
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_threads: int = 0
    open_files: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "gpu_used_mb": self.gpu_used_mb,
            "gpu_total_mb": self.gpu_total_mb,
            "gpu_utilization": self.gpu_utilization,
            "active_threads": self.active_threads,
        }

    def to_text(self) -> str:
        lines = [f"Timestamp: {self.timestamp:.1f}"]
        if self.ram_total_mb > 0:
            pct = (self.ram_used_mb / self.ram_total_mb) * 100
            lines.append(f"RAM: {self.ram_used_mb:.0f}/{self.ram_total_mb:.0f} MB ({pct:.0f}%)")
        if self.gpu_total_mb > 0:
            pct = (self.gpu_used_mb / self.gpu_total_mb) * 100
            lines.append(f"GPU: {self.gpu_used_mb:.0f}/{self.gpu_total_mb:.0f} MB ({pct:.0f}%)")
        lines.append(f"CPU: {self.cpu_percent:.1f}%")
        lines.append(f"Threads: {self.active_threads}")
        return "\n".join(lines)


@dataclass
class Alert:
    level: str
    message: str
    timestamp: float
    metric_name: str = ""
    value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "value": self.value,
        }


class PerformanceMonitor:
    """Monitors system and application performance."""

    def __init__(
        self,
        gpu_total_mb: float = 6144,
        ram_total_mb: float = 16384,
        history_size: int = 1000,
        alert_threshold_cpu: float = 90.0,
        alert_threshold_ram: float = 85.0,
        alert_threshold_gpu: float = 90.0,
    ):
        self._gpu_total = gpu_total_mb
        self._ram_total = ram_total_mb
        self._history_size = history_size
        self._thresholds = {
            "cpu": alert_threshold_cpu,
            "ram": alert_threshold_ram,
            "gpu": alert_threshold_gpu,
        }
        self._metrics: Dict[str, deque] = {}
        self._snapshots: deque = deque(maxlen=history_size)
        self._alerts: List[Alert] = []
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self._running = False

    def record_metric(
        self, name: str, value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self._history_size)

            metric = Metric(
                name=name, value=value, metric_type=metric_type,
                timestamp=time.time(), labels=labels or {},
            )
            self._metrics[name].append(metric)

            self._check_alerts(name, value)

    def record_snapshot(self, snapshot: PerformanceSnapshot):
        """Record a performance snapshot."""
        with self._lock:
            self._snapshots.append(snapshot)

    def get_metric_history(
        self, name: str, last_n: int = 100
    ) -> List[Metric]:
        """Get metric history."""
        with self._lock:
            if name not in self._metrics:
                return []
            return list(self._metrics[name])[-last_n:]

    def get_latest_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Get latest snapshot."""
        with self._lock:
            if not self._snapshots:
                return None
            return self._snapshots[-1]

    def get_snapshots(
        self, last_n: int = 100
    ) -> List[PerformanceSnapshot]:
        """Get recent snapshots."""
        with self._lock:
            return list(self._snapshots)[-last_n:]

    def get_average(
        self, name: str, last_n: int = 100
    ) -> float:
        """Get average value for a metric."""
        history = self.get_metric_history(name, last_n)
        if not history:
            return 0.0
        return sum(m.value for m in history) / len(history)

    def get_max(self, name: str, last_n: int = 100) -> float:
        """Get max value for a metric."""
        history = self.get_metric_history(name, last_n)
        if not history:
            return 0.0
        return max(m.value for m in history)

    def get_min(self, name: str, last_n: int = 100) -> float:
        """Get min value for a metric."""
        history = self.get_metric_history(name, last_n)
        if not history:
            return 0.0
        return min(m.value for m in history)

    def get_percentile(
        self, name: str, percentile: float = 95.0, last_n: int = 100
    ) -> float:
        """Get percentile value for a metric."""
        history = self.get_metric_history(name, last_n)
        if not history:
            return 0.0
        values = sorted(m.value for m in history)
        idx = int(len(values) * percentile / 100)
        return values[min(idx, len(values) - 1)]

    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register alert callback."""
        self._callbacks.append(callback)

    def get_alerts(self, last_n: int = 50) -> List[Alert]:
        """Get recent alerts."""
        with self._lock:
            return self._alerts[-last_n:]

    def clear_alerts(self):
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            summary = {
                "total_metrics": len(self._metrics),
                "total_snapshots": len(self._snapshots),
                "total_alerts": len(self._alerts),
                "metric_names": list(self._metrics.keys()),
            }

            if self._snapshots:
                latest = self._snapshots[-1]
                summary["latest"] = latest.to_dict()

            return summary

    def _check_alerts(self, name: str, value: float):
        """Check if metric triggers an alert."""
        threshold_key = None
        if "cpu" in name.lower():
            threshold_key = "cpu"
        elif "ram" in name.lower() or "memory" in name.lower():
            threshold_key = "ram"
        elif "gpu" in name.lower():
            threshold_key = "gpu"

        if threshold_key:
            threshold = self._thresholds.get(threshold_key, 100)
            if value > threshold:
                alert = Alert(
                    level="warning",
                    message=f"{name} exceeded threshold: {value:.1f} > {threshold}",
                    timestamp=time.time(),
                    metric_name=name,
                    value=value,
                )
                self._alerts.append(alert)
                for callback in self._callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.warning(f"Alert callback failed: {e}")

    def timer(self, name: str) -> "TimerContext":
        """Create a timer context manager."""
        return TimerContext(self, name)


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, name: str):
        self._monitor = monitor
        self._name = name
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        self._monitor.record_metric(
            self._name, elapsed, MetricType.TIMER
        )
        return False
