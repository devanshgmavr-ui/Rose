"""Unit tests for Stage 3.3 Observe/Act/Verify."""

import os
import tempfile
import time
import pytest
from pathlib import Path

from agent.media.vision import VisionResult, DetectedElement, VisionConfidence
from agent.media.analyzer import VisionAnalyzer
from agent.media.vision import VisionProvider
from agent.media.grounding import VisualGrounder, GroundedTarget, TargetType, GroundingConfidence, Point
from agent.media.observe_act_verify import (
    ObserveActVerifyLoop, LoopConfig, LoopResult,
    LoopState, LoopExitReason, LoopStep,
)
from agent.media.oav_tool import ObserveActVerifyTool


class TestLoopState:
    def test_values(self):
        assert LoopState.IDLE.value == "idle"
        assert LoopState.OBSERVING.value == "observing"
        assert LoopState.COMPLETE.value == "complete"
        assert LoopState.FAILED.value == "failed"


class TestLoopExitReason:
    def test_values(self):
        assert LoopExitReason.GOAL_ACHIEVED.value == "goal_achieved"
        assert LoopExitReason.MAX_ITERATIONS.value == "max_iterations"
        assert LoopExitReason.TIMEOUT.value == "timeout"
        assert LoopExitReason.USER_CANCELLED.value == "user_cancelled"


class TestLoopConfig:
    def test_creation(self):
        config = LoopConfig(max_iterations=5, max_actions=10, timeout=60.0)
        assert config.max_iterations == 5
        assert config.max_actions == 10
        assert config.timeout == 60.0

    def test_to_dict(self):
        config = LoopConfig()
        d = config.to_dict()
        assert "max_iterations" in d
        assert "max_actions" in d
        assert "timeout" in d


class TestLoopStep:
    def test_creation(self):
        step = LoopStep(step_number=1, state=LoopState.OBSERVING)
        assert step.step_number == 1
        assert step.state == LoopState.OBSERVING

    def test_to_dict(self):
        step = LoopStep(step_number=2, state=LoopState.ACTING, success=True)
        d = step.to_dict()
        assert d["step_number"] == 2
        assert d["state"] == "acting"
        assert d["success"] is True


class TestLoopResult:
    def test_creation(self):
        result = LoopResult(
            success=True,
            exit_reason=LoopExitReason.GOAL_ACHIEVED,
        )
        assert result.success is True
        assert result.exit_reason == LoopExitReason.GOAL_ACHIEVED

    def test_to_dict(self):
        result = LoopResult(
            success=True,
            exit_reason=LoopExitReason.GOAL_ACHIEVED,
            total_iterations=3,
            total_actions=2,
            total_time=5.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["exit_reason"] == "goal_achieved"
        assert d["total_iterations"] == 3

    def test_to_text(self):
        result = LoopResult(
            success=True,
            exit_reason=LoopExitReason.GOAL_ACHIEVED,
            steps=[
                LoopStep(step_number=1, state=LoopState.COMPLETE, success=True),
            ],
            total_iterations=1,
            total_actions=0,
            total_time=1.0,
        )
        text = result.to_text()
        assert "[BEGIN UNTRUSTED LOOP RESULT]" in text
        assert "goal_achieved" in text


class TestObserveActVerifyLoop:
    def test_init(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        config = LoopConfig(max_iterations=5)
        loop = ObserveActVerifyLoop(analyzer, grounder, config)
        assert loop.state == LoopState.IDLE

    def test_cancel(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        loop = ObserveActVerifyLoop(analyzer, grounder)
        loop.cancel()
        assert loop._cancelled is True

    def test_execute_no_callbacks(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        loop = ObserveActVerifyLoop(analyzer, grounder)
        result = loop.execute(goal="Test goal")
        assert result.success is False
        assert "callback" in result.error.lower()

    def test_execute_goal_achieved(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        config = LoopConfig(max_iterations=3, timeout=5.0)
        loop = ObserveActVerifyLoop(analyzer, grounder, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='green')
            img.save(img_path)

            def observe():
                return img_path

            def plan(vr):
                return {"type": "goal_achieved"}

            def act(action):
                return True

            def verify(path):
                return img_path

            result = loop.execute(
                goal="Open calculator",
                observe_fn=observe,
                plan_fn=plan,
                act_fn=act,
                verify_fn=verify,
            )
            assert result.success is True
            assert result.exit_reason == LoopExitReason.GOAL_ACHIEVED

    def test_execute_max_iterations(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        config = LoopConfig(max_iterations=2, timeout=10.0)
        loop = ObserveActVerifyLoop(analyzer, grounder, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(img_path)

            call_count = [0]

            def observe():
                return img_path

            def plan(vr):
                call_count[0] += 1
                return {"type": "action", "tool": "click", "x": 50, "y": 50}

            def act(action):
                return True

            def verify(path):
                return img_path

            result = loop.execute(
                goal="Keep clicking",
                observe_fn=observe,
                plan_fn=plan,
                act_fn=act,
                verify_fn=verify,
            )
            assert result.success is False
            assert result.exit_reason == LoopExitReason.MAX_ITERATIONS

    def test_execute_user_cancel(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        config = LoopConfig(max_iterations=10, timeout=10.0)
        loop = ObserveActVerifyLoop(analyzer, grounder, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(img_path)

            def observe():
                loop.cancel()
                return img_path

            def plan(vr):
                return {"type": "action", "tool": "click", "x": 50, "y": 50}

            def act(action):
                return True

            def verify(path):
                return img_path

            result = loop.execute(
                goal="Cancel test",
                observe_fn=observe,
                plan_fn=plan,
                act_fn=act,
                verify_fn=verify,
            )
            assert result.exit_reason == LoopExitReason.USER_CANCELLED

    def test_is_running(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        loop = ObserveActVerifyLoop(analyzer, grounder)
        assert loop.is_running is False


class TestObserveActVerifyTool:
    def test_init(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=True)
        assert tool.name == "observe_act_verify"

    def test_validate_disabled(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=False)
        ok, errors = tool.validate({})
        assert ok is False
        assert "disabled" in errors[0].lower()

    def test_validate_no_goal(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=True)
        ok, errors = tool.validate({})
        assert ok is False
        assert "goal" in errors[0].lower()

    def test_validate_empty_goal(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=True)
        ok, errors = tool.validate({"goal": "  "})
        assert ok is False

    def test_validate_invalid_iterations(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=True)
        ok, errors = tool.validate({"goal": "test", "max_iterations": -1})
        assert ok is False

    def test_validate_too_large_iterations(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=True)
        ok, errors = tool.validate({"goal": "test", "max_iterations": 100})
        assert ok is False

    def test_execute(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=True)
        result = tool.execute({"goal": "Open calculator"})
        assert result.success is True
        assert "loop" in result.output.lower()

    def test_permissions(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=True)
        assert "vision.analyze" in tool.required_permissions

    def test_disabled_deny(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        tool = ObserveActVerifyTool(analyzer, grounder, vision_enabled=False)
        assert tool.confirmation_level.value == "deny"


class TestOAVIntegration:
    def test_full_loop_with_retries(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        config = LoopConfig(max_iterations=5, max_actions=3, timeout=10.0)
        loop = ObserveActVerifyLoop(analyzer, grounder, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='yellow')
            img.save(img_path)

            action_count = [0]

            def observe():
                return img_path

            def plan(vr):
                action_count[0] += 1
                if action_count[0] >= 3:
                    return {"type": "goal_achieved"}
                return {"type": "action", "tool": "click", "x": 50, "y": 50}

            def act(action):
                return True

            def verify(path):
                return img_path

            result = loop.execute(
                goal="Click until done",
                observe_fn=observe,
                plan_fn=plan,
                act_fn=act,
                verify_fn=verify,
            )
            assert result.success is True
            assert result.total_actions == 2

    def test_loop_step_counting(self):
        vp = VisionProvider()
        analyzer = VisionAnalyzer(vision_provider=vp)
        grounder = VisualGrounder()
        config = LoopConfig(max_iterations=3, timeout=10.0)
        loop = ObserveActVerifyLoop(analyzer, grounder, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='cyan')
            img.save(img_path)

            def observe():
                return img_path

            def plan(vr):
                return {"type": "goal_achieved"}

            def act(action):
                return True

            def verify(path):
                return img_path

            result = loop.execute(
                goal="Quick goal",
                observe_fn=observe,
                plan_fn=plan,
                act_fn=act,
                verify_fn=verify,
            )
            assert result.total_iterations == 1
