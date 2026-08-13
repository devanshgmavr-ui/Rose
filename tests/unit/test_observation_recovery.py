"""Phase 13 - Observation, Verification & Failure Recovery Tests.

Tests the observation system, failure recovery, error classification,
recovery decisions, and tool fallback logic.
"""

import time
import pytest
from unittest.mock import MagicMock

from agent.orchestration.observation import (
    ObservationSystem, Observation, ObservationResult, ObservationRule,
    ObservationType, ObservationStatus,
)
from agent.orchestration.recovery import (
    FailureRecovery, RecoveryDecision, FailureContext,
    ErrorCategory, RecoveryStrategy, TOOL_FALLBACKS,
)
from agent.orchestration.verifier import Verifier
from agent.orchestration.executor import TaskExecutor
from agent.orchestration.models import Task, Plan, PlanStep, TaskStatus, StepStatus


# === Observation Tests ===

class TestObservationType:
    def test_values(self):
        assert ObservationType.FILE_EXISTS.value == "file_exists"
        assert ObservationType.WINDOW_STATE.value == "window_state"
        assert ObservationType.URL_STATE.value == "url_state"
        assert ObservationType.CUSTOM.value == "custom"


class TestObservationStatus:
    def test_values(self):
        assert ObservationStatus.PASSED.value == "passed"
        assert ObservationStatus.FAILED.value == "failed"
        assert ObservationStatus.SKIPPED.value == "skipped"
        assert ObservationStatus.ERROR.value == "error"


class TestObservation:
    def test_creation(self):
        obs = Observation(
            observation_type=ObservationType.FILE_EXISTS,
            description="Check if file exists",
            expected="file exists",
            actual="file exists",
            status=ObservationStatus.PASSED,
        )
        assert obs.passed is True
        assert obs.observation_type == ObservationType.FILE_EXISTS

    def test_to_dict(self):
        obs = Observation(
            observation_type=ObservationType.FILE_EXISTS,
            description="Check file",
            status=ObservationStatus.PASSED,
        )
        d = obs.to_dict()
        assert d["observation_type"] == "file_exists"
        assert d["status"] == "passed"
        assert "timestamp" in d

    def test_failed_observation(self):
        obs = Observation(
            observation_type=ObservationType.FILE_EXISTS,
            description="Check file",
            status=ObservationStatus.FAILED,
        )
        assert obs.passed is False


class TestObservationResult:
    def test_creation(self):
        result = ObservationResult(
            action_description="test:action",
            overall_status=ObservationStatus.PASSED,
        )
        assert result.passed is True

    def test_with_observations(self):
        obs1 = Observation(
            observation_type=ObservationType.FILE_EXISTS,
            description="obs1",
            status=ObservationStatus.PASSED,
        )
        obs2 = Observation(
            observation_type=ObservationType.FILE_EXISTS,
            description="obs2",
            status=ObservationStatus.FAILED,
        )
        result = ObservationResult(
            action_description="test",
            observations=[obs1, obs2],
            overall_status=ObservationStatus.FAILED,
        )
        assert result.passed_count == 1
        assert result.failed_count == 1
        assert result.passed is False

    def test_to_dict(self):
        result = ObservationResult(
            action_description="test",
            overall_status=ObservationStatus.PASSED,
        )
        d = result.to_dict()
        assert d["overall_status"] == "passed"
        assert d["passed_count"] == 0
        assert d["total_count"] == 0


class TestObservationRule:
    def test_creation(self):
        rule = ObservationRule(
            action_pattern="filesystem:write",
            observation_type=ObservationType.FILE_EXISTS,
            description="Verify file created",
        )
        assert rule.action_pattern == "filesystem:write"

    def test_to_dict(self):
        rule = ObservationRule(
            action_pattern="test",
            observation_type=ObservationType.CUSTOM,
            description="test rule",
        )
        d = rule.to_dict()
        assert d["action_pattern"] == "test"
        assert d["observation_type"] == "custom"


class TestObservationSystem:
    def test_init(self):
        system = ObservationSystem()
        assert len(system._rules) > 0

    def test_get_rules_for_action(self):
        system = ObservationSystem()
        rules = system.get_rules_for_action("filesystem", "write")
        assert len(rules) > 0
        assert any(r.observation_type == ObservationType.FILE_EXISTS for r in rules)

    def test_observe_without_router(self):
        system = ObservationSystem()
        result = system.observe(
            tool_name="filesystem",
            action="write",
            tool_result=MagicMock(success=True),
            arguments={"path": "test.txt"},
        )
        assert result.action_description == "filesystem:write"

    def test_observe_with_router(self):
        system = ObservationSystem()
        router = MagicMock()
        router.execute_tool.return_value = MagicMock(success=True)
        result = system.observe(
            tool_name="filesystem",
            action="write",
            tool_result=MagicMock(success=True),
            arguments={"path": "test.txt"},
            tool_router=router,
        )
        assert result.overall_status == ObservationStatus.PASSED

    def test_observe_file_not_found(self):
        system = ObservationSystem()
        router = MagicMock()
        router.execute_tool.return_value = MagicMock(success=False)
        result = system.observe(
            tool_name="filesystem",
            action="write",
            tool_result=MagicMock(success=True),
            arguments={"path": "test.txt"},
            tool_router=router,
        )
        assert result.overall_status == ObservationStatus.FAILED

    def test_observe_no_matching_rules(self):
        system = ObservationSystem()
        result = system.observe(
            tool_name="unknown_tool",
            action="unknown_action",
            tool_result=MagicMock(success=True),
            arguments={},
        )
        assert result.overall_status == ObservationStatus.PASSED

    def test_register_rule(self):
        system = ObservationSystem()
        initial_count = len(system._rules)
        system.register_rule(ObservationRule(
            action_pattern="custom:action",
            observation_type=ObservationType.CUSTOM,
            description="custom rule",
        ))
        assert len(system._rules) == initial_count + 1

    def test_history(self):
        system = ObservationSystem()
        system.observe("filesystem", "write", MagicMock(success=True), {"path": "a.txt"})
        system.observe("filesystem", "write", MagicMock(success=True), {"path": "b.txt"})
        history = system.get_history()
        assert len(history) == 2

    def test_stats(self):
        system = ObservationSystem()
        system.observe("unknown_tool", "unknown_action", MagicMock(success=True), {})
        stats = system.get_stats()
        assert stats["total_observations"] == 1
        assert stats["passed"] == 1


# === Failure Recovery Tests ===

class TestErrorCategory:
    def test_values(self):
        assert ErrorCategory.TOOL_FAILURE.value == "tool_failure"
        assert ErrorCategory.PERMISSION_DENIED.value == "permission_denied"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.VERIFICATION_FAILURE.value == "verification_failure"


class TestRecoveryStrategy:
    def test_values(self):
        assert RecoveryStrategy.RETRY_SAME.value == "retry_same"
        assert RecoveryStrategy.USE_FALLBACK_TOOL.value == "use_fallback_tool"
        assert RecoveryStrategy.REPLAN.value == "replan"
        assert RecoveryStrategy.STOP_SAFE.value == "stop_safe"


class TestToolFallbacks:
    def test_vision_analyze_fallbacks(self):
        assert "screen_capture" in TOOL_FALLBACKS["vision_analyze"]

    def test_mouse_fallbacks(self):
        assert "keyboard" in TOOL_FALLBACKS["mouse"]

    def test_all_fallbacks_are_lists(self):
        for key, val in TOOL_FALLBACKS.items():
            assert isinstance(val, list)


class TestFailureRecovery:
    def test_classify_permission_denied(self):
        recovery = FailureRecovery()
        category = recovery.classify_error("Permission denied for mouse control")
        assert category == ErrorCategory.PERMISSION_DENIED

    def test_classify_timeout(self):
        recovery = FailureRecovery()
        category = recovery.classify_error("Operation timed out after 30s")
        assert category == ErrorCategory.TIMEOUT

    def test_classify_not_found(self):
        recovery = FailureRecovery()
        category = recovery.classify_error("File not found: /path/to/file")
        assert category in (ErrorCategory.TOOL_FAILURE, ErrorCategory.INVALID_INPUT)

    def test_classify_unavailable(self):
        recovery = FailureRecovery()
        category = recovery.classify_error("Vision analysis is not available")
        assert category == ErrorCategory.UNAVAILABLE_CAPABILITY

    def test_classify_verification(self):
        recovery = FailureRecovery()
        category = recovery.classify_error("Verification failed: expected window to be active")
        assert category == ErrorCategory.VERIFICATION_FAILURE

    def test_classify_environment(self):
        recovery = FailureRecovery()
        category = recovery.classify_error("Connection refused: network unreachable")
        assert category == ErrorCategory.ENVIRONMENT_FAILURE

    def test_classify_unknown(self):
        recovery = FailureRecovery()
        category = recovery.classify_error("Something weird happened")
        assert category == ErrorCategory.UNKNOWN

    def test_decide_tool_failure_retry(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="filesystem",
            action="read",
            error_message="IO error",
            attempt_number=1,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.RETRY_SAME

    def test_decide_tool_failure_fallback(self):
        recovery = FailureRecovery()
        # Simulate multiple failures
        for _ in range(3):
            recovery._tool_failures["vision_analyze"] = recovery._tool_failures.get("vision_analyze", 0) + 1
        context = FailureContext(
            tool_name="vision_analyze",
            action="analyze",
            error_message="Provider error",
            attempt_number=3,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.USE_FALLBACK_TOOL
        assert decision.alternative_tool is not None

    def test_decide_permission_denied_with_fallback(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="mouse",
            action="click",
            error_message="Permission denied",
            attempt_number=1,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy in (RecoveryStrategy.USE_FALLBACK_TOOL, RecoveryStrategy.ASK_CLARIFICATION)

    def test_decide_permission_denied_ask_user(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="mouse",
            action="click",
            error_message="Permission denied",
            attempt_number=3,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.ASK_CLARIFICATION

    def test_decide_timeout_retry(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="browser",
            action="navigate",
            error_message="Timeout",
            attempt_number=1,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.RETRY_SAME

    def test_decide_timeout_fallback(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="browser",
            action="navigate",
            error_message="Timeout",
            attempt_number=3,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.USE_FALLBACK_TOOL

    def test_decide_unavailable_skip(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="unknown_tool",
            action="do_something",
            error_message="Not available",
            attempt_number=1,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.SKIP_STEP

    def test_decide_verification_retry(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="window",
            action="activate",
            error_message="Verification failed",
            attempt_number=1,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.RETRY_SAME

    def test_decide_verification_replan(self):
        recovery = FailureRecovery()
        context = FailureContext(
            tool_name="window",
            action="activate",
            error_message="Verification failed",
            attempt_number=3,
            max_attempts=3,
        )
        decision = recovery.decide_recovery(context)
        assert decision.strategy == RecoveryStrategy.REPLAN

    def test_record_recovery(self):
        recovery = FailureRecovery()
        decision = RecoveryDecision(
            error_category=ErrorCategory.TOOL_FAILURE,
            strategy=RecoveryStrategy.RETRY_SAME,
            reasoning="test",
        )
        recovery.record_recovery(decision)
        assert len(recovery._history) == 1

    def test_stats(self):
        recovery = FailureRecovery()
        decision = RecoveryDecision(
            error_category=ErrorCategory.TOOL_FAILURE,
            strategy=RecoveryStrategy.RETRY_SAME,
            reasoning="test",
        )
        recovery.record_recovery(decision)
        stats = recovery.get_stats()
        assert stats["total_recoveries"] == 1

    def test_reset_tool_failures(self):
        recovery = FailureRecovery()
        recovery._tool_failures["test"] = 5
        recovery.reset_tool_failures("test")
        assert "test" not in recovery._tool_failures

    def test_find_alternative_prohibited(self):
        recovery = FailureRecovery()
        result = recovery._find_alternative_tool("mouse", prohibited={"keyboard"})
        # mouse fallback is keyboard, but it's prohibited
        assert result is None

    def test_find_alternative_available(self):
        recovery = FailureRecovery()
        result = recovery._find_alternative_tool("mouse", prohibited=set())
        assert result == "keyboard"


class TestRecoveryDecision:
    def test_to_dict(self):
        decision = RecoveryDecision(
            error_category=ErrorCategory.TOOL_FAILURE,
            strategy=RecoveryStrategy.RETRY_SAME,
            reasoning="test reasoning",
            alternative_tool="fallback_tool",
        )
        d = decision.to_dict()
        assert d["error_category"] == "tool_failure"
        assert d["strategy"] == "retry_same"
        assert d["alternative_tool"] == "fallback_tool"


class TestFailureContext:
    def test_to_dict(self):
        context = FailureContext(
            tool_name="filesystem",
            action="read",
            error_message="File not found",
            attempt_number=2,
            max_attempts=3,
        )
        d = context.to_dict()
        assert d["tool_name"] == "filesystem"
        assert d["attempt_number"] == 2


# === Verifier Enhancement Tests ===

class TestVerifierBasic:
    def test_verify_no_plan(self):
        verifier = Verifier()
        task = Task(user_request="test")
        result = verifier.verify(task)
        assert result is False

    def test_verify_no_criteria(self):
        verifier = Verifier()
        task = Task(user_request="test")
        task.plan = Plan(objective="test", steps=[])
        task.completed_steps = ["step1"]
        result = verifier.verify(task)
        assert result is True

    def test_verify_with_file_criterion(self):
        verifier = Verifier()
        router = MagicMock()
        router.execute_tool.return_value = MagicMock(success=True)
        verifier._tool_router = router
        task = Task(user_request="test")
        task.plan = Plan(
            objective="test",
            steps=[],
            completion_criteria=['File "test.txt" exists'],
        )
        result = verifier.verify(task)
        assert result is True

    def test_verify_step(self):
        verifier = Verifier()
        assert verifier.verify_step("success", "success") is True
        assert verifier.verify_step("", "expected") is False
        assert verifier.verify_step("result", "") is True


class TestTaskExecutorDecisionMaking:
    def test_fail_task(self):
        executor = TaskExecutor(tool_router=MagicMock())
        task = Task(user_request="test")
        from agent.orchestration.state import StateMachine
        state = StateMachine()
        result = executor._fail_task(task, state, "test error")
        assert result.status == TaskStatus.FAILED
        assert result.error == "test error"

    def test_advance_step(self):
        executor = TaskExecutor(tool_router=MagicMock())
        task = Task(user_request="test")
        task.plan = Plan(
            objective="test",
            steps=[
                PlanStep(step_id="s1", description="step1", tool_name="test", arguments={}),
                PlanStep(step_id="s2", description="step2", tool_name="test", arguments={}),
            ],
        )
        task.current_step_index = 0
        executor._advance_step(task)
        assert task.current_step_index == 1

    def test_dependencies_met(self):
        executor = TaskExecutor(tool_router=MagicMock())
        task = Task(user_request="test")
        step = PlanStep(
            step_id="s2",
            description="step2",
            tool_name="test",
            arguments={},
            dependencies=["s1"],
        )
        task.completed_steps = ["s1"]
        assert executor._dependencies_met(task, step) is True

    def test_dependencies_not_met(self):
        executor = TaskExecutor(tool_router=MagicMock())
        task = Task(user_request="test")
        step = PlanStep(
            step_id="s2",
            description="step2",
            tool_name="test",
            arguments={},
            dependencies=["s1"],
        )
        task.completed_steps = []
        assert executor._dependencies_met(task, step) is False
