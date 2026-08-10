"""Resource management for memory, CPU, and GPU.

Stage 6.3 - Resource Management.

Provides:
- Memory pool management
- GPU memory allocation tracking
- CPU throttling controls
- Resource cleanup
- Leak detection
- Resource limits enforcement
"""

import gc
import os
import time
import threading
import logging
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    CPU = "cpu"
    RAM = "ram"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceAllocation:
    resource_type: ResourceType
    allocated_mb: float
    allocated_at: float
    owner: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.resource_type.value,
            "allocated_mb": self.allocated_mb,
            "allocated_at": self.allocated_at,
            "owner": self.owner,
            "description": self.description,
        }


@dataclass
class ResourceLimits:
    max_ram_mb: float = 12288
    max_gpu_mb: float = 6144
    max_cpu_percent: float = 80.0
    max_threads: int = 16
    max_open_files: int = 512
    warning_threshold: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_ram_mb": self.max_ram_mb,
            "max_gpu_mb": self.max_gpu_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_threads": self.max_threads,
            "max_open_files": self.max_open_files,
            "warning_threshold": self.warning_threshold,
        }


@dataclass
class ResourceSnapshot:
    timestamp: float
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_available_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    open_files: int = 0
    allocations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "ram_used_mb": self.ram_used_mb,
            "ram_available_mb": self.ram_available_mb,
            "gpu_used_mb": self.gpu_used_mb,
            "gpu_available_mb": self.gpu_available_mb,
            "cpu_percent": self.cpu_percent,
            "thread_count": self.thread_count,
            "open_files": self.open_files,
            "allocations": self.allocations,
        }


class ResourceManager:
    """Manages system resources with limits and tracking."""

    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        gpu_total_mb: float = 6144,
        ram_total_mb: float = 16384,
    ):
        self._limits = limits or ResourceLimits()
        self._gpu_total = gpu_total_mb
        self._ram_total = ram_total_mb
        self._allocations: List[ResourceAllocation] = []
        self._snapshots: List[ResourceSnapshot] = []
        self._lock = threading.Lock()
        self._cleanup_callbacks: List[callable] = []
        self._leak_detector_enabled = True

    def allocate(
        self,
        resource_type: ResourceType,
        amount_mb: float,
        owner: str = "",
        description: str = "",
    ) -> bool:
        """Allocate resources."""
        with self._lock:
            if not self._check_limits(resource_type, amount_mb):
                return False

            allocation = ResourceAllocation(
                resource_type=resource_type,
                allocated_mb=amount_mb,
                allocated_at=time.time(),
                owner=owner,
                description=description,
            )
            self._allocations.append(allocation)
            return True

    def release(self, owner: str, resource_type: Optional[ResourceType] = None) -> int:
        """Release resources by owner."""
        with self._lock:
            released = 0
            remaining = []
            for alloc in self._allocations:
                if alloc.owner == owner and (
                    resource_type is None or alloc.resource_type == resource_type
                ):
                    released += 1
                else:
                    remaining.append(alloc)
            self._allocations = remaining
            return released

    def release_all(self):
        """Release all allocations."""
        with self._lock:
            self._allocations.clear()

    def get_usage(self, resource_type: ResourceType) -> float:
        """Get total usage for a resource type."""
        with self._lock:
            return self._get_usage_unlocked(resource_type)

    def _get_usage_unlocked(self, resource_type: ResourceType) -> float:
        """Get total usage without locking (internal)."""
        return sum(
            a.allocated_mb for a in self._allocations
            if a.resource_type == resource_type
        )

    def get_usage_summary(self) -> Dict[str, float]:
        """Get usage summary for all resource types."""
        with self._lock:
            summary = defaultdict(float)
            for alloc in self._allocations:
                summary[alloc.resource_type.value] += alloc.allocated_mb
            return dict(summary)

    def get_allocations(
        self, resource_type: Optional[ResourceType] = None
    ) -> List[ResourceAllocation]:
        """Get all allocations."""
        with self._lock:
            if resource_type is None:
                return list(self._allocations)
            return [a for a in self._allocations if a.resource_type == resource_type]

    def check_limits(self) -> Dict[str, Any]:
        """Check current resource limits."""
        usage = self.get_usage_summary()
        return {
            "ram": {
                "used": usage.get("ram", 0),
                "limit": self._limits.max_ram_mb,
                "percent": (usage.get("ram", 0) / self._limits.max_ram_mb) * 100,
            },
            "gpu": {
                "used": usage.get("gpu", 0),
                "limit": self._limits.max_gpu_mb,
                "percent": (usage.get("gpu", 0) / self._limits.max_gpu_mb) * 100,
            },
        }

    def snapshot(self) -> ResourceSnapshot:
        """Take a resource snapshot."""
        usage = self.get_usage_summary()
        snap = ResourceSnapshot(
            timestamp=time.time(),
            ram_used_mb=usage.get("ram", 0),
            ram_available_mb=self._ram_total - usage.get("ram", 0),
            gpu_used_mb=usage.get("gpu", 0),
            gpu_available_mb=self._gpu_total - usage.get("gpu", 0),
            thread_count=threading.active_count(),
            allocations=[a.to_dict() for a in self._allocations],
        )
        with self._lock:
            self._snapshots.append(snap)
            if len(self._snapshots) > 100:
                self._snapshots = self._snapshots[-100:]
        return snap

    def get_snapshots(self, last_n: int = 10) -> List[ResourceSnapshot]:
        """Get recent snapshots."""
        with self._lock:
            return self._snapshots[-last_n:]

    def register_cleanup(self, callback: callable):
        """Register cleanup callback."""
        self._cleanup_callbacks.append(callback)

    def cleanup(self) -> int:
        """Run cleanup callbacks and force GC."""
        cleaned = 0
        for callback in self._cleanup_callbacks:
            try:
                callback()
                cleaned += 1
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")

        gc.collect()
        return cleaned

    def detect_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        if not self._leak_detector_enabled or len(self._snapshots) < 3:
            return []

        leaks = []
        recent = self._snapshots[-3:]

        for i in range(1, len(recent)):
            prev, curr = recent[i - 1], recent[i]
            if curr.ram_used_mb > prev.ram_used_mb * 1.1:
                growth = curr.ram_used_mb - prev.ram_used_mb
                leaks.append({
                    "type": "ram_growth",
                    "growth_mb": growth,
                    "timestamp": curr.timestamp,
                })

        return leaks

    def get_limits(self) -> ResourceLimits:
        """Get current limits."""
        return self._limits

    def set_limits(self, limits: ResourceLimits):
        """Set resource limits."""
        self._limits = limits

    def _check_limits(self, resource_type: ResourceType, amount_mb: float) -> bool:
        """Check if allocation would exceed limits."""
        current = self._get_usage_unlocked(resource_type)

        if resource_type == ResourceType.RAM:
            limit = self._limits.max_ram_mb
        elif resource_type == ResourceType.GPU:
            limit = self._limits.max_gpu_mb
        else:
            return True

        if current + amount_mb > limit:
            logger.warning(
                f"Resource limit exceeded: {resource_type.value} "
                f"{current + amount_mb:.0f} > {limit:.0f}"
            )
            return False

        usage_pct = (current + amount_mb) / limit
        if usage_pct > self._limits.warning_threshold:
            logger.warning(
                f"Resource usage high: {resource_type.value} "
                f"at {usage_pct * 100:.0f}%"
            )

        return True

    def get_status_text(self) -> str:
        """Get human-readable status."""
        usage = self.get_usage_summary()
        lines = ["Resource Status:"]
        lines.append(f"  RAM: {usage.get('ram', 0):.0f}/{self._limits.max_ram_mb:.0f} MB")
        lines.append(f"  GPU: {usage.get('gpu', 0):.0f}/{self._limits.max_gpu_mb:.0f} MB")
        lines.append(f"  Allocations: {len(self._allocations)}")
        return "\n".join(lines)
