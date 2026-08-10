"""Tests for Stage 7.2 - Error Recovery."""

import pytest
import time
from agent.core.error_recovery import (
    ErrorHandler, CircuitBreaker, RetryConfig,
    ErrorInfo, ErrorSeverity, RecoveryAction, retry,
)


class TestErrorInfo:
    def test_creation(self):
        e = ErrorInfo(
            error_type="ValueError",
            message="bad value",
            severity=ErrorSeverity.MEDIUM,
        )
        assert e.error_type == "ValueError"

    def test_to_dict(self):
        e = ErrorInfo(
            error_type="IOError",
            message="file not found",
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.RETRY,
        )
        d = e.to_dict()
        assert d["severity"] == "high"
        assert d["recovery_action"] == "retry"


class TestRetryConfig:
    def test_defaults(self):
        c = RetryConfig()
        assert c.max_attempts == 3

    def test_get_delay(self):
        c = RetryConfig(base_delay=1.0, exponential_base=2.0)
        assert c.get_delay(0) == 1.0
        assert c.get_delay(1) == 2.0
        assert c.get_delay(2) == 4.0

    def test_get_delay_max(self):
        c = RetryConfig(base_delay=1.0, max_delay=5.0)
        assert c.get_delay(10) == 5.0


class TestCircuitBreaker:
    def test_init(self):
        cb = CircuitBreaker()
        assert cb.can_execute() is True

    def test_record_failure_opens(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.can_execute() is False

    def test_record_success_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            cb.record_failure()
        cb.record_success()
        state = cb.get_state()
        assert state["failure_count"] == 0

    def test_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is False
        time.sleep(0.2)
        assert cb.can_execute() is True

    def test_half_open_recovery(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_max=2)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.2)
        cb.can_execute()
        cb.record_success()
        cb.record_success()
        state = cb.get_state()
        assert state["closed"] is True


class TestErrorHandler:
    def test_init(self):
        h = ErrorHandler()
        assert h._retry_config.max_attempts == 3

    def test_execute_success(self):
        h = ErrorHandler()
        result = h.execute_with_recovery(lambda: 42)
        assert result == 42

    def test_execute_with_retry(self):
        counter = {"n": 0}
        def flaky():
            counter["n"] += 1
            if counter["n"] < 3:
                raise ValueError("not yet")
            return "ok"

        h = ErrorHandler(retry_config=RetryConfig(max_attempts=3, base_delay=0.01))
        result = h.execute_with_recovery(flaky)
        assert result == "ok"
        assert counter["n"] == 3

    def test_execute_all_attempts_fail(self):
        h = ErrorHandler(retry_config=RetryConfig(max_attempts=2, base_delay=0.01))
        with pytest.raises(ValueError):
            h.execute_with_recovery(lambda: (_ for _ in ()).throw(ValueError("always fail")))

    def test_execute_with_fallback(self):
        h = ErrorHandler(retry_config=RetryConfig(max_attempts=1, base_delay=0.01))
        result = h.execute_with_recovery(
            lambda: (_ for _ in ()).throw(ValueError("fail")),
            fallback=lambda: "fallback result",
        )
        assert result == "fallback result"

    def test_register_fallback(self):
        h = ErrorHandler()
        h.register_fallback("ValueError", lambda: "fallback")
        assert "ValueError" in h._fallback_handlers

    def test_get_error_history(self):
        h = ErrorHandler(retry_config=RetryConfig(max_attempts=1, base_delay=0.01))
        for _ in range(3):
            try:
                h.execute_with_recovery(lambda: (_ for _ in ()).throw(ValueError("err")))
            except (ValueError, RuntimeError):
                pass
        history = h.get_error_history()
        assert len(history) >= 2

    def test_get_error_summary(self):
        h = ErrorHandler(retry_config=RetryConfig(max_attempts=1, base_delay=0.01))
        try:
            h.execute_with_recovery(lambda: (_ for _ in ()).throw(ValueError("err")))
        except (ValueError, RuntimeError):
            pass
        summary = h.get_error_summary()
        assert summary["total_errors"] >= 1

    def test_circuit_breaker_integration(self):
        cb = CircuitBreaker(failure_threshold=2)
        h = ErrorHandler(
            retry_config=RetryConfig(max_attempts=0),
            circuit_breaker=cb,
        )
        for _ in range(2):
            try:
                h.execute_with_recovery(lambda: (_ for _ in ()).throw(IOError("fail")))
            except (IOError, RuntimeError):
                pass
        assert cb.get_state()["closed"] is False


class TestRetryDecorator:
    def test_retry_success(self):
        counter = {"n": 0}

        @retry(max_attempts=3, base_delay=0.01)
        def flaky():
            counter["n"] += 1
            if counter["n"] < 2:
                raise ValueError("not yet")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert counter["n"] == 2

    def test_retry_all_fail(self):
        @retry(max_attempts=2, base_delay=0.01)
        def always_fail():
            raise ValueError("always")

        with pytest.raises(ValueError):
            always_fail()
