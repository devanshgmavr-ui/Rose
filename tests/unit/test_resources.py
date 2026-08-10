"""Tests for Stage 6.3 - Resource Management."""

import pytest
import time
from agent.core.resources import (
    ResourceManager, ResourceAllocation, ResourceLimits,
    ResourceSnapshot, ResourceType,
)


class TestResourceAllocation:
    def test_creation(self):
        a = ResourceAllocation(
            resource_type=ResourceType.RAM, allocated_mb=1024,
            allocated_at=time.time(), owner="test",
        )
        assert a.resource_type == ResourceType.RAM
        assert a.allocated_mb == 1024

    def test_to_dict(self):
        a = ResourceAllocation(
            resource_type=ResourceType.GPU, allocated_mb=512,
            allocated_at=1.0, owner="model",
        )
        d = a.to_dict()
        assert d["type"] == "gpu"
        assert d["owner"] == "model"


class TestResourceLimits:
    def test_defaults(self):
        l = ResourceLimits()
        assert l.max_ram_mb == 12288
        assert l.max_gpu_mb == 6144

    def test_to_dict(self):
        l = ResourceLimits(max_ram_mb=8192)
        d = l.to_dict()
        assert d["max_ram_mb"] == 8192


class TestResourceSnapshot:
    def test_creation(self):
        s = ResourceSnapshot(timestamp=1.0, ram_used_mb=4000, gpu_used_mb=2000)
        assert s.ram_used_mb == 4000

    def test_to_dict(self):
        s = ResourceSnapshot(timestamp=1.0, cpu_percent=50.0)
        d = s.to_dict()
        assert d["cpu_percent"] == 50.0


class TestResourceManager:
    def test_init(self):
        rm = ResourceManager()
        assert rm._limits.max_ram_mb == 12288

    def test_allocate(self):
        rm = ResourceManager()
        result = rm.allocate(ResourceType.RAM, 1024, owner="test")
        assert result is True
        assert rm.get_usage(ResourceType.RAM) == 1024

    def test_allocate_exceeds_limit(self):
        rm = ResourceManager(limits=ResourceLimits(max_ram_mb=1000))
        rm.allocate(ResourceType.RAM, 800, owner="a")
        result = rm.allocate(ResourceType.RAM, 300, owner="b")
        assert result is False

    def test_release(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 1024, owner="test")
        released = rm.release("test")
        assert released == 1
        assert rm.get_usage(ResourceType.RAM) == 0

    def test_release_by_type(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 1024, owner="test")
        rm.allocate(ResourceType.GPU, 512, owner="test")
        released = rm.release("test", ResourceType.RAM)
        assert released == 1
        assert rm.get_usage(ResourceType.GPU) == 512

    def test_release_all(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 1024, owner="a")
        rm.allocate(ResourceType.GPU, 512, owner="b")
        rm.release_all()
        assert rm.get_usage(ResourceType.RAM) == 0

    def test_get_usage_summary(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 1024, owner="a")
        rm.allocate(ResourceType.GPU, 512, owner="b")
        summary = rm.get_usage_summary()
        assert summary["ram"] == 1024
        assert summary["gpu"] == 512

    def test_get_allocations(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 1024, owner="a")
        rm.allocate(ResourceType.GPU, 512, owner="b")
        all_allocs = rm.get_allocations()
        assert len(all_allocs) == 2
        gpu_allocs = rm.get_allocations(ResourceType.GPU)
        assert len(gpu_allocs) == 1

    def test_check_limits(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 4096, owner="a")
        limits = rm.check_limits()
        assert limits["ram"]["used"] == 4096

    def test_snapshot(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 1024, owner="a")
        snap = rm.snapshot()
        assert snap.ram_used_mb == 1024
        assert snap.thread_count > 0

    def test_get_snapshots(self):
        rm = ResourceManager()
        rm.snapshot()
        rm.snapshot()
        snaps = rm.get_snapshots()
        assert len(snaps) == 2

    def test_cleanup(self):
        rm = ResourceManager()
        cleaned = []
        rm.register_cleanup(lambda: cleaned.append(1))
        rm.cleanup()
        assert len(cleaned) == 1

    def test_detect_leaks(self):
        rm = ResourceManager()
        rm._snapshots.append(ResourceSnapshot(timestamp=1.0, ram_used_mb=1000))
        rm._snapshots.append(ResourceSnapshot(timestamp=2.0, ram_used_mb=1500))
        rm._snapshots.append(ResourceSnapshot(timestamp=3.0, ram_used_mb=2000))
        leaks = rm.detect_leaks()
        assert len(leaks) > 0

    def test_detect_leaks_no_data(self):
        rm = ResourceManager()
        leaks = rm.detect_leaks()
        assert len(leaks) == 0

    def test_get_set_limits(self):
        rm = ResourceManager()
        new_limits = ResourceLimits(max_ram_mb=8192)
        rm.set_limits(new_limits)
        assert rm.get_limits().max_ram_mb == 8192

    def test_get_status_text(self):
        rm = ResourceManager()
        rm.allocate(ResourceType.RAM, 1024, owner="a")
        text = rm.get_status_text()
        assert "RAM" in text
        assert "1024" in text
