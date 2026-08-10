"""System health check and diagnostics.

Stage 7.3 - System Health Check.

Provides:
- Component health monitoring
- Dependency checking
- Resource health assessment
- Diagnostic reporting
- Health alerts
"""

import os
import time
import logging
import platform
import shutil
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "checked_at": self.checked_at,
        }


@dataclass
class HealthReport:
    overall_status: HealthStatus
    components: List[ComponentHealth]
    timestamp: float = 0.0
    system_info: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "components": [c.to_dict() for c in self.components],
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"Health Report: {self.overall_status.value.upper()}",
            f"Timestamp: {self.timestamp}",
            f"Components: {len(self.components)}",
        ]
        for c in self.components:
            icon = "+" if c.status == HealthStatus.HEALTHY else "!"
            lines.append(f"  [{icon}] {c.name}: {c.status.value} ({c.latency_ms:.0f}ms)")
            if c.message:
                lines.append(f"      {c.message}")
        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)


class HealthChecker:
    """Checks system and component health."""

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        checks: Optional[List[Callable[[], ComponentHealth]]] = None,
    ):
        self._workspace = workspace_dir or os.getcwd()
        self._checks = checks or []
        self._history: List[HealthReport] = []

    def register_check(self, check: Callable[[], ComponentHealth]):
        """Register a health check function."""
        self._checks.append(check)

    def check_all(self) -> HealthReport:
        """Run all health checks."""
        components = []
        for check in self._checks:
            try:
                start = time.time()
                health = check()
                health.latency_ms = (time.time() - start) * 1000
                health.checked_at = time.time()
                components.append(health)
            except Exception as e:
                components.append(ComponentHealth(
                    name=check.__name__,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    checked_at=time.time(),
                ))

        overall = self._determine_overall(components)
        system_info = self._get_system_info()
        recommendations = self._generate_recommendations(components)

        report = HealthReport(
            overall_status=overall,
            components=components,
            timestamp=time.time(),
            system_info=system_info,
            recommendations=recommendations,
        )

        self._history.append(report)
        if len(self._history) > 50:
            self._history = self._history[-50:]

        return report

    def check_workspace(self) -> ComponentHealth:
        """Check workspace health."""
        try:
            exists = os.path.exists(self._workspace)
            writable = os.access(self._workspace, os.W_OK) if exists else False

            if not exists:
                return ComponentHealth(
                    name="workspace",
                    status=HealthStatus.UNHEALTHY,
                    message="Workspace directory does not exist",
                )
            if not writable:
                return ComponentHealth(
                    name="workspace",
                    status=HealthStatus.DEGRADED,
                    message="Workspace directory is not writable",
                )

            files = len(os.listdir(self._workspace))
            return ComponentHealth(
                name="workspace",
                status=HealthStatus.HEALTHY,
                message=f"Workspace OK ({files} items)",
                details={"path": self._workspace, "file_count": files},
            )
        except Exception as e:
            return ComponentHealth(
                name="workspace",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    def check_disk_space(self) -> ComponentHealth:
        """Check disk space."""
        try:
            total, used, free = shutil.disk_usage(self._workspace)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            used_pct = (used / total) * 100

            if used_pct > 95:
                status = HealthStatus.UNHEALTHY
                msg = f"Critical: {free_gb:.1f}GB free ({used_pct:.0f}% used)"
            elif used_pct > 85:
                status = HealthStatus.DEGRADED
                msg = f"Low: {free_gb:.1f}GB free ({used_pct:.0f}% used)"
            else:
                status = HealthStatus.HEALTHY
                msg = f"{free_gb:.1f}GB free ({used_pct:.0f}% used)"

            return ComponentHealth(
                name="disk",
                status=status,
                message=msg,
                details={"total_gb": total_gb, "free_gb": free_gb, "used_pct": used_pct},
            )
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=str(e),
            )

    def check_model(self, model_path: Optional[str] = None) -> ComponentHealth:
        """Check model availability."""
        try:
            if model_path and os.path.exists(model_path):
                size_gb = os.path.getsize(model_path) / (1024**3)
                return ComponentHealth(
                    name="model",
                    status=HealthStatus.HEALTHY,
                    message=f"Model found ({size_gb:.1f}GB)",
                    details={"path": model_path, "size_gb": size_gb},
                )
            elif model_path:
                return ComponentHealth(
                    name="model",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Model not found at {model_path}",
                )
            return ComponentHealth(
                name="model",
                status=HealthStatus.UNKNOWN,
                message="Model path not configured",
            )
        except Exception as e:
            return ComponentHealth(
                name="model",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    def get_history(self, last_n: int = 10) -> List[HealthReport]:
        """Get health check history."""
        return self._history[-last_n:]

    def _determine_overall(self, components: List[ComponentHealth]) -> HealthStatus:
        """Determine overall health from components."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "architecture": platform.machine(),
        }

    def _generate_recommendations(
        self, components: List[ComponentHealth]
    ) -> List[str]:
        """Generate health recommendations."""
        recs = []
        for c in components:
            if c.status == HealthStatus.UNHEALTHY:
                recs.append(f"Fix {c.name}: {c.message}")
            elif c.status == HealthStatus.DEGRADED:
                recs.append(f"Monitor {c.name}: {c.message}")
        return recs


def default_checks(workspace_dir: str) -> List[Callable[[], ComponentHealth]]:
    """Create default health checks."""
    checker = HealthChecker(workspace_dir=workspace_dir)
    return [checker.check_workspace, checker.check_disk_space]
