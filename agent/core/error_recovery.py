"""Error recovery and resilience mechanisms.

Stage 7.2 - Error Recovery.

Provides:
- Automatic error detection and classification
- Retry strategies with exponential backoff
- Graceful degradation
- State rollback
- Circuit breaker pattern
- Error reporting and logging
"""

import time
import logging
import functools
from typing import Optional, Dict, Any, List, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    ROLLBACK = "rollback"


@dataclass
class ErrorInfo:
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_action: RecoveryAction = RecoveryAction.ABORT
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "context": self.context,
            "recovery_action": self.recovery_action.value,
            "attempts": self.attempts,
        }


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (Exception,)

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


@dataclass
class CircuitState:
    closed: bool = True
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0
    state_changed_at: float = 0.0


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 3,
    ):
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max
        self._state = CircuitState(state_changed_at=time.time())
        self._lock = __import__("threading").Lock()

    def can_execute(self) -> bool:
        with self._lock:
            if self._state.closed:
                return True

            if time.time() - self._state.last_failure_time > self._recovery_timeout:
                self._state.closed = True
                self._state.success_count = 0
                self._state.state_changed_at = time.time()
                logger.info("Circuit breaker: half-open state")
                return True

            return False

    def record_success(self):
        with self._lock:
            if not self._state.closed:
                self._state.success_count += 1
                if self._state.success_count >= self._half_open_max:
                    self._state.closed = True
                    self._state.failure_count = 0
                    self._state.state_changed_at = time.time()
                    logger.info("Circuit breaker: closed (recovered)")
            else:
                self._state.failure_count = 0

    def record_failure(self):
        with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()

            if self._state.failure_count >= self._threshold:
                self._state.closed = False
                self._state.state_changed_at = time.time()
                logger.warning(
                    f"Circuit breaker: open after {self._state.failure_count} failures"
                )

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "closed": self._state.closed,
                "failure_count": self._state.failure_count,
                "success_count": self._state.success_count,
                "last_failure": self._state.last_failure_time,
            }


class ErrorHandler:
    """Handles errors with recovery strategies."""

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        max_error_history: int = 100,
    ):
        self._retry_config = retry_config or RetryConfig()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._error_history: deque = deque(maxlen=max_error_history)
        self._fallback_handlers: Dict[str, Callable] = {}

    def execute_with_recovery(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Execute function with error recovery."""
        last_error = None

        for attempt in range(self._retry_config.max_attempts + 1):
            if not self._circuit_breaker.can_execute():
                logger.warning("Circuit breaker open, using fallback")
                if fallback:
                    return fallback(*args, **kwargs)
                raise RuntimeError("Circuit breaker is open")

            try:
                result = func(*args, **kwargs)
                self._circuit_breaker.record_success()
                return result

            except self._retry_config.retryable_exceptions as e:
                last_error = e
                self._circuit_breaker.record_failure()

                error_info = ErrorInfo(
                    error_type=type(e).__name__,
                    message=str(e),
                    severity=self._classify_error(e),
                    timestamp=time.time(),
                    attempts=attempt + 1,
                )
                self._error_history.append(error_info)

                if attempt < self._retry_config.max_attempts:
                    delay = self._retry_config.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self._retry_config.max_attempts} attempts failed")

        if fallback:
            try:
                return fallback(*args, **kwargs)
            except Exception as fb_error:
                logger.error(f"Fallback also failed: {fb_error}")

        raise last_error

    def register_fallback(self, error_type: str, handler: Callable):
        """Register fallback handler for error type."""
        self._fallback_handlers[error_type] = handler

    def get_error_history(self, last_n: int = 10) -> List[ErrorInfo]:
        """Get recent errors."""
        return list(self._error_history)[-last_n:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        errors = list(self._error_history)
        by_type = {}
        for e in errors:
            by_type[e.error_type] = by_type.get(e.error_type, 0) + 1

        return {
            "total_errors": len(errors),
            "by_type": by_type,
            "circuit_breaker": self._circuit_breaker.get_state(),
        }

    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        error_name = type(error).__name__

        critical_errors = {"MemoryError", "SystemExit", "KeyboardInterrupt"}
        high_errors = {"ConnectionError", "TimeoutError", "IOError"}
        medium_errors = {"ValueError", "TypeError", "RuntimeError"}

        if error_name in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_name in high_errors:
            return ErrorSeverity.HIGH
        elif error_name in medium_errors:
            return ErrorSeverity.MEDIUM
        return ErrorSeverity.LOW


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator for retrying functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                retryable_exceptions=retryable_exceptions,
            )
            handler = ErrorHandler(retry_config=config)
            return handler.execute_with_recovery(func, *args, **kwargs)
        return wrapper
    return decorator
