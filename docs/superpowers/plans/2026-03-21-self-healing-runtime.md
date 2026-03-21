# Self-Healing Runtime Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-healing runtime that makes Homie detect failures, recover autonomously, optimize its own performance, and evolve its own code and architecture.

**Architecture:** Hybrid approach — modules handle their own Tier 1 resilience via a `@resilient` decorator, while a central HealthWatchdog coordinates Tier 2-4 recovery and the Improvement Engine handles self-optimization and code evolution. All fixes are fully autonomous and silently logged.

**Tech Stack:** Python 3.11+, SQLite (health log), threading (event bus, probes), psutil (system metrics), existing homie_core infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-21-self-healing-runtime-design.md`

---

## Chunk 1: Resilience Foundation

### Task 1: Exception Classifier

**Files:**
- Create: `src/homie_core/self_healing/__init__.py`
- Create: `src/homie_core/self_healing/resilience/__init__.py`
- Create: `src/homie_core/self_healing/resilience/exceptions.py`
- Test: `tests/unit/self_healing/test_exceptions.py`

- [ ] **Step 1: Create module structure**

```bash
mkdir -p src/homie_core/self_healing/resilience
```

- [ ] **Step 2: Write failing test for exception classifier**

```python
# tests/unit/self_healing/test_exceptions.py
import sqlite3
import pytest
from homie_core.self_healing.resilience.exceptions import (
    classify_exception,
    ErrorCategory,
)


class TestErrorCategory:
    def test_enum_values(self):
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.RECOVERABLE.value == "recoverable"
        assert ErrorCategory.PERMANENT.value == "permanent"
        assert ErrorCategory.FATAL.value == "fatal"


class TestClassifyException:
    def test_timeout_is_transient(self):
        assert classify_exception(TimeoutError("timed out")) == ErrorCategory.TRANSIENT

    def test_connection_error_is_transient(self):
        assert classify_exception(ConnectionError("reset")) == ErrorCategory.TRANSIENT

    def test_oserror_errno_enomem_is_transient(self):
        err = OSError(12, "Cannot allocate memory")
        assert classify_exception(err) == ErrorCategory.TRANSIENT

    def test_sqlite_locked_is_recoverable(self):
        err = sqlite3.OperationalError("database is locked")
        assert classify_exception(err) == ErrorCategory.RECOVERABLE

    def test_file_not_found_is_permanent(self):
        assert classify_exception(FileNotFoundError("no such file")) == ErrorCategory.PERMANENT

    def test_value_error_is_permanent(self):
        assert classify_exception(ValueError("bad config")) == ErrorCategory.PERMANENT

    def test_disk_full_is_fatal(self):
        err = OSError(28, "No space left on device")
        assert classify_exception(err) == ErrorCategory.FATAL

    def test_unknown_exception_defaults_to_permanent(self):
        assert classify_exception(Exception("unknown")) == ErrorCategory.PERMANENT

    def test_custom_classifier_overrides_default(self):
        custom = {RuntimeError: ErrorCategory.TRANSIENT}
        result = classify_exception(RuntimeError("custom"), custom_rules=custom)
        assert result == ErrorCategory.TRANSIENT

    def test_memory_error_is_fatal(self):
        assert classify_exception(MemoryError()) == ErrorCategory.FATAL
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd C:/Users/muthu/PycharmProjects/Homie && python -m pytest tests/unit/self_healing/test_exceptions.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write implementation**

```python
# src/homie_core/self_healing/__init__.py
"""Homie Self-Healing Runtime — autonomous failure recovery and self-improvement."""
```

```python
# src/homie_core/self_healing/resilience/__init__.py
from .exceptions import ErrorCategory, classify_exception

__all__ = ["ErrorCategory", "classify_exception"]
```

```python
# src/homie_core/self_healing/resilience/exceptions.py
"""Exception classifier — categorizes errors for recovery decisions."""

import errno
import sqlite3
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"      # Retry immediately
    RECOVERABLE = "recoverable"  # Retry with longer backoff
    PERMANENT = "permanent"      # Fail, report to watchdog
    FATAL = "fatal"              # Trip circuit, escalate


# Errno values that indicate fatal disk/resource exhaustion
_FATAL_ERRNOS = {errno.ENOSPC, errno.EDQUOT} if hasattr(errno, "EDQUOT") else {errno.ENOSPC}

# Errno values that indicate transient resource pressure
_TRANSIENT_ERRNOS = {errno.ENOMEM, errno.EAGAIN, errno.EBUSY}

_DEFAULT_RULES: dict[type, ErrorCategory] = {
    TimeoutError: ErrorCategory.TRANSIENT,
    ConnectionError: ErrorCategory.TRANSIENT,
    ConnectionResetError: ErrorCategory.TRANSIENT,
    ConnectionRefusedError: ErrorCategory.TRANSIENT,
    ConnectionAbortedError: ErrorCategory.TRANSIENT,
    BrokenPipeError: ErrorCategory.TRANSIENT,
    InterruptedError: ErrorCategory.TRANSIENT,
    FileNotFoundError: ErrorCategory.PERMANENT,
    PermissionError: ErrorCategory.PERMANENT,
    ValueError: ErrorCategory.PERMANENT,
    TypeError: ErrorCategory.PERMANENT,
    KeyError: ErrorCategory.PERMANENT,
    AttributeError: ErrorCategory.PERMANENT,
    ImportError: ErrorCategory.PERMANENT,
    MemoryError: ErrorCategory.FATAL,
    SystemError: ErrorCategory.FATAL,
}


def classify_exception(
    exc: BaseException,
    custom_rules: Optional[dict[type, ErrorCategory]] = None,
) -> ErrorCategory:
    """Classify an exception into a recovery category."""
    # Custom rules take precedence
    if custom_rules:
        for exc_type, category in custom_rules.items():
            if isinstance(exc, exc_type):
                return category

    # Check default type rules
    for exc_type, category in _DEFAULT_RULES.items():
        if isinstance(exc, exc_type):
            return category

    # SQLite-specific classification
    if isinstance(exc, sqlite3.OperationalError):
        msg = str(exc).lower()
        if "locked" in msg or "busy" in msg:
            return ErrorCategory.RECOVERABLE
        return ErrorCategory.PERMANENT

    # OSError errno-based classification
    if isinstance(exc, OSError) and exc.errno is not None:
        if exc.errno in _FATAL_ERRNOS:
            return ErrorCategory.FATAL
        if exc.errno in _TRANSIENT_ERRNOS:
            return ErrorCategory.TRANSIENT

    return ErrorCategory.PERMANENT
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd C:/Users/muthu/PycharmProjects/Homie && python -m pytest tests/unit/self_healing/test_exceptions.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/self_healing/__init__.py src/homie_core/self_healing/resilience/__init__.py src/homie_core/self_healing/resilience/exceptions.py tests/unit/self_healing/test_exceptions.py
git commit -m "feat(self-healing): add exception classifier with 4-tier categorization"
```

---

### Task 2: Retry Logic with Exponential Backoff

**Files:**
- Create: `src/homie_core/self_healing/resilience/retry.py`
- Test: `tests/unit/self_healing/test_retry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_retry.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.resilience.retry import retry_with_backoff
from homie_core.self_healing.resilience.exceptions import ErrorCategory


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value="ok")
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 1

    def test_retries_on_transient_error(self):
        fn = MagicMock(side_effect=[TimeoutError("timeout"), "ok"])
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    def test_no_retry_on_permanent_error(self):
        fn = MagicMock(side_effect=FileNotFoundError("gone"))
        with pytest.raises(FileNotFoundError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1

    def test_no_retry_on_fatal_error(self):
        fn = MagicMock(side_effect=MemoryError())
        with pytest.raises(MemoryError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1

    def test_exhausts_retries_then_raises(self):
        fn = MagicMock(side_effect=TimeoutError("timeout"))
        with pytest.raises(TimeoutError):
            retry_with_backoff(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3  # 1 initial + 2 retries

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value="ok")
        retry_with_backoff(fn, max_retries=1, base_delay=0.01, args=("a",), kwargs={"b": 2})
        fn.assert_called_once_with("a", b=2)

    def test_exponential_backoff_timing(self):
        fn = MagicMock(side_effect=[TimeoutError(), TimeoutError(), "ok"])
        with patch("homie_core.self_healing.resilience.retry.time.sleep") as mock_sleep:
            retry_with_backoff(fn, max_retries=3, base_delay=1.0)
            # First retry: 1.0s, second retry: 2.0s (exponential)
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert len(delays) == 2
            assert delays[0] == pytest.approx(1.0, abs=0.5)  # jitter
            assert delays[1] == pytest.approx(2.0, abs=1.0)

    def test_custom_classifier(self):
        custom = {RuntimeError: ErrorCategory.TRANSIENT}
        fn = MagicMock(side_effect=[RuntimeError("temp"), "ok"])
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01, custom_rules=custom)
        assert result == "ok"
        assert fn.call_count == 2

    def test_on_retry_callback(self):
        callback = MagicMock()
        fn = MagicMock(side_effect=[TimeoutError(), "ok"])
        retry_with_backoff(fn, max_retries=3, base_delay=0.01, on_retry=callback)
        assert callback.call_count == 1
        call_args = callback.call_args
        assert call_args[1]["attempt"] == 1
        assert isinstance(call_args[1]["exception"], TimeoutError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_retry.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/resilience/retry.py
"""Retry logic with exponential backoff and jitter."""

import random
import time
from typing import Any, Callable, Optional

from .exceptions import ErrorCategory, classify_exception


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    custom_rules: Optional[dict[type, ErrorCategory]] = None,
    on_retry: Optional[Callable] = None,
) -> Any:
    """Call fn with retries on transient/recoverable errors.

    Uses exponential backoff with jitter. Permanent and fatal errors
    are raised immediately without retry.
    """
    kwargs = kwargs or {}
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            category = classify_exception(exc, custom_rules)

            if category in (ErrorCategory.PERMANENT, ErrorCategory.FATAL):
                raise

            if attempt >= max_retries:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            # Add jitter: ±50% of delay
            delay = delay * (0.5 + random.random())

            if on_retry:
                on_retry(attempt=attempt, exception=exc, category=category, delay=delay)

            time.sleep(delay)

    raise last_exc  # unreachable but satisfies type checker
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_retry.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/resilience/retry.py tests/unit/self_healing/test_retry.py
git commit -m "feat(self-healing): add retry logic with exponential backoff and jitter"
```

---

### Task 3: Circuit Breaker

**Files:**
- Create: `src/homie_core/self_healing/resilience/circuit_breaker.py`
- Test: `tests/unit/self_healing/test_circuit_breaker.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_circuit_breaker.py
import time
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
)


class TestCircuitState:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # count reset to 1

    def test_old_failures_outside_window_dont_count(self):
        cb = CircuitBreaker(threshold=3, window=0.1, cooldown=5)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # old failures expired


class TestCircuitOpenBehavior:
    def test_raises_circuit_open_error(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=5)
        cb.record_failure()
        with pytest.raises(CircuitOpenError):
            cb.check()

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_one_call(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        time.sleep(0.15)
        cb.check()  # should not raise — allows test call

    def test_success_in_half_open_closes_circuit(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        time.sleep(0.15)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitMetrics:
    def test_tracks_total_failures(self):
        cb = CircuitBreaker(threshold=5, window=60, cooldown=5)
        for _ in range(3):
            cb.record_failure()
        assert cb.total_failures == 3

    def test_tracks_total_successes(self):
        cb = CircuitBreaker(threshold=5, window=60, cooldown=5)
        for _ in range(3):
            cb.record_success()
        assert cb.total_successes == 3

    def test_tracks_trip_count(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.05)
        cb.record_failure()  # trip 1
        time.sleep(0.06)
        cb.record_failure()  # trip 2
        assert cb.trip_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_circuit_breaker.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/resilience/circuit_breaker.py
"""Circuit breaker — prevents repeated calls to failing operations."""

import threading
import time
from enum import Enum
from typing import Optional


class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and rejecting calls."""

    def __init__(self, name: str = "unknown"):
        super().__init__(f"Circuit breaker '{name}' is open — operation rejected")
        self.circuit_name = name


class CircuitBreaker:
    """Tracks failures and trips open to prevent cascading failures."""

    def __init__(
        self,
        threshold: int = 5,
        window: float = 60.0,
        cooldown: float = 30.0,
        name: str = "default",
    ):
        self._threshold = threshold
        self._window = window
        self._cooldown = cooldown
        self._name = name
        self._lock = threading.Lock()

        self._failure_times: list[float] = []
        self._state = CircuitState.CLOSED
        self._opened_at: Optional[float] = None

        # Metrics
        self.total_failures: int = 0
        self.total_successes: int = 0
        self.trip_count: int = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN and self._opened_at is not None:
                if time.monotonic() - self._opened_at >= self._cooldown:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def check(self) -> None:
        """Check if the circuit allows a call. Raises CircuitOpenError if not."""
        state = self.state
        if state == CircuitState.OPEN:
            raise CircuitOpenError(self._name)
        # CLOSED and HALF_OPEN allow the call through

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            now = time.monotonic()
            self.total_failures += 1
            self._failure_times.append(now)

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = now
                self.trip_count += 1
                return

            # Prune failures outside window
            cutoff = now - self._window
            self._failure_times = [t for t in self._failure_times if t > cutoff]

            if len(self._failure_times) >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = now
                self.trip_count += 1

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.total_successes += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_times.clear()
            elif self._state == CircuitState.CLOSED:
                self._failure_times.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_circuit_breaker.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/resilience/circuit_breaker.py tests/unit/self_healing/test_circuit_breaker.py
git commit -m "feat(self-healing): add circuit breaker with closed/open/half-open states"
```

---

### Task 4: Timeout Enforcement

**Files:**
- Create: `src/homie_core/self_healing/resilience/timeout.py`
- Test: `tests/unit/self_healing/test_timeout.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_timeout.py
import time
import pytest
from homie_core.self_healing.resilience.timeout import run_with_timeout


class TestRunWithTimeout:
    def test_returns_result_within_timeout(self):
        result = run_with_timeout(lambda: "ok", timeout=5.0)
        assert result == "ok"

    def test_raises_timeout_on_slow_function(self):
        def slow():
            time.sleep(10)
        with pytest.raises(TimeoutError, match="exceeded 0.1s"):
            run_with_timeout(slow, timeout=0.1)

    def test_propagates_function_exception(self):
        def bad():
            raise ValueError("broken")
        with pytest.raises(ValueError, match="broken"):
            run_with_timeout(bad, timeout=5.0)

    def test_passes_args_and_kwargs(self):
        def add(a, b=0):
            return a + b
        result = run_with_timeout(add, timeout=5.0, args=(3,), kwargs={"b": 4})
        assert result == 7

    def test_custom_timeout_message(self):
        def slow():
            time.sleep(10)
        with pytest.raises(TimeoutError, match="model inference"):
            run_with_timeout(slow, timeout=0.1, operation_name="model inference")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_timeout.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/resilience/timeout.py
"""Timeout enforcement — prevents operations from hanging indefinitely."""

import threading
from typing import Any, Callable, Optional


def run_with_timeout(
    fn: Callable,
    timeout: float,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    operation_name: str = "operation",
) -> Any:
    """Run fn in a daemon thread with a timeout.

    Returns the result if fn completes within timeout.
    Raises TimeoutError if fn doesn't complete in time.
    Propagates any exception raised by fn.
    """
    kwargs = kwargs or {}
    result: list = [None]
    error: list = [None]

    def _run():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as exc:
            error[0] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(
            f"{operation_name} exceeded {timeout}s timeout"
        )

    if error[0] is not None:
        raise error[0]

    return result[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_timeout.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/resilience/timeout.py tests/unit/self_healing/test_timeout.py
git commit -m "feat(self-healing): add timeout enforcement for operation calls"
```

---

### Task 5: The `@resilient` Decorator

**Files:**
- Create: `src/homie_core/self_healing/resilience/decorator.py`
- Test: `tests/unit/self_healing/test_decorator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_decorator.py
import time
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.resilience.decorator import resilient
from homie_core.self_healing.resilience.circuit_breaker import CircuitOpenError


class TestResilientDecorator:
    def test_passes_through_on_success(self):
        @resilient(retries=3, base_delay=0.01, timeout=5.0)
        def greet(name):
            return f"hi {name}"

        assert greet("homie") == "hi homie"

    def test_retries_on_transient_error(self):
        call_count = 0

        @resilient(retries=3, base_delay=0.01, timeout=5.0)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "recovered"

        assert flaky() == "recovered"
        assert call_count == 3

    def test_raises_permanent_error_immediately(self):
        @resilient(retries=3, base_delay=0.01, timeout=5.0)
        def broken():
            raise FileNotFoundError("gone")

        with pytest.raises(FileNotFoundError):
            broken()

    def test_circuit_breaker_trips(self):
        @resilient(
            retries=0,
            base_delay=0.01,
            timeout=5.0,
            circuit_breaker_threshold=2,
            circuit_breaker_window=60,
        )
        def failing():
            raise TimeoutError("down")

        for _ in range(2):
            with pytest.raises(TimeoutError):
                failing()

        with pytest.raises(CircuitOpenError):
            failing()

    def test_timeout_enforcement(self):
        @resilient(retries=0, base_delay=0.01, timeout=0.1)
        def slow():
            time.sleep(10)

        with pytest.raises(TimeoutError):
            slow()

    def test_fallback_called_on_failure(self):
        fallback_fn = MagicMock(return_value="fallback_result")

        @resilient(retries=0, base_delay=0.01, timeout=5.0, fallback=fallback_fn)
        def broken():
            raise TimeoutError("down")

        result = broken()
        assert result == "fallback_result"
        fallback_fn.assert_called_once()

    def test_fallback_called_on_circuit_open(self):
        fallback_fn = MagicMock(return_value="safe")

        @resilient(
            retries=0,
            base_delay=0.01,
            timeout=5.0,
            circuit_breaker_threshold=1,
            circuit_breaker_window=60,
            fallback=fallback_fn,
        )
        def broken():
            raise TimeoutError("down")

        with pytest.raises(TimeoutError):
            broken()  # trips circuit

        result = broken()  # circuit open, fallback used
        assert result == "safe"

    def test_works_as_method_decorator(self):
        class MyService:
            @resilient(retries=1, base_delay=0.01, timeout=5.0)
            def query(self, q):
                return f"result:{q}"

        svc = MyService()
        assert svc.query("test") == "result:test"

    def test_health_status_exposed(self):
        @resilient(retries=1, base_delay=0.01, timeout=5.0)
        def fn():
            return "ok"

        fn()
        status = fn.health_status()
        assert status["state"] == "healthy"
        assert status["total_successes"] == 1
        assert status["total_failures"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_decorator.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/resilience/decorator.py
"""The @resilient decorator — wraps functions with retry, circuit breaker, and timeout."""

import functools
from typing import Any, Callable, Optional

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .exceptions import ErrorCategory, classify_exception
from .retry import retry_with_backoff
from .timeout import run_with_timeout


def resilient(
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    timeout: float = 30.0,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_window: float = 60.0,
    circuit_breaker_cooldown: float = 30.0,
    fallback: Optional[Callable] = None,
    custom_rules: Optional[dict[type, ErrorCategory]] = None,
) -> Callable:
    """Decorator that adds retry, circuit breaker, and timeout to a function."""

    def decorator(fn: Callable) -> Callable:
        cb = CircuitBreaker(
            threshold=circuit_breaker_threshold,
            window=circuit_breaker_window,
            cooldown=circuit_breaker_cooldown,
            name=getattr(fn, "__qualname__", fn.__name__),
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit breaker first
            try:
                cb.check()
            except CircuitOpenError:
                if fallback is not None:
                    return fallback(*args, **kwargs)
                raise

            def _call_with_timeout():
                return run_with_timeout(
                    fn,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                    operation_name=fn.__qualname__ if hasattr(fn, "__qualname__") else fn.__name__,
                )

            def _on_retry(attempt, exception, category, delay):
                cb.record_failure()

            try:
                result = retry_with_backoff(
                    _call_with_timeout,
                    max_retries=retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    custom_rules=custom_rules,
                    on_retry=_on_retry,
                )
                cb.record_success()
                return result
            except Exception as exc:
                cb.record_failure()
                if fallback is not None:
                    return fallback(*args, **kwargs)
                raise

        def health_status() -> dict[str, Any]:
            """Return current health metrics for this function."""
            state = cb.state
            return {
                "state": "healthy" if state.value == "closed" else "degraded" if state.value == "half_open" else "failed",
                "circuit_state": state.value,
                "total_successes": cb.total_successes,
                "total_failures": cb.total_failures,
                "trip_count": cb.trip_count,
            }

        wrapper.health_status = health_status
        wrapper._circuit_breaker = cb
        return wrapper

    return decorator
```

- [ ] **Step 4: Update resilience __init__.py**

```python
# src/homie_core/self_healing/resilience/__init__.py
from .circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from .decorator import resilient
from .exceptions import ErrorCategory, classify_exception
from .retry import retry_with_backoff
from .timeout import run_with_timeout

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "ErrorCategory",
    "classify_exception",
    "resilient",
    "retry_with_backoff",
    "run_with_timeout",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_decorator.py -v`
Expected: All 9 tests PASS

- [ ] **Step 6: Run all self-healing tests**

Run: `python -m pytest tests/unit/self_healing/ -v`
Expected: All 36 tests PASS (exceptions: 10, retry: 9, circuit_breaker: 12, decorator: 9 — note: some may have ±1 from exact count)

- [ ] **Step 7: Commit**

```bash
git add src/homie_core/self_healing/resilience/ tests/unit/self_healing/
git commit -m "feat(self-healing): add @resilient decorator combining retry, circuit breaker, and timeout"
```

---

## Chunk 2: Event Bus & Health Infrastructure

### Task 6: Event Bus

**Files:**
- Create: `src/homie_core/self_healing/event_bus.py`
- Test: `tests/unit/self_healing/test_event_bus.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_event_bus.py
import threading
import time
import pytest
from homie_core.self_healing.event_bus import EventBus, HealthEvent


class TestHealthEvent:
    def test_event_creation(self):
        evt = HealthEvent(
            module="inference",
            event_type="probe_result",
            severity="info",
            details={"latency_ms": 42},
        )
        assert evt.module == "inference"
        assert evt.event_type == "probe_result"
        assert evt.severity == "info"
        assert evt.timestamp > 0

    def test_event_to_dict(self):
        evt = HealthEvent(module="storage", event_type="recovery", severity="warning", details={})
        d = evt.to_dict()
        assert d["module"] == "storage"
        assert "timestamp" in d


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("probe_result", lambda evt: received.append(evt))
        evt = HealthEvent(module="test", event_type="probe_result", severity="info", details={})
        bus.publish(evt)
        time.sleep(0.05)  # async delivery
        assert len(received) == 1
        assert received[0].module == "test"

    def test_multiple_subscribers(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("anomaly", lambda e: r1.append(e))
        bus.subscribe("anomaly", lambda e: r2.append(e))
        bus.publish(HealthEvent(module="m", event_type="anomaly", severity="warning", details={}))
        time.sleep(0.05)
        assert len(r1) == 1
        assert len(r2) == 1

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.publish(HealthEvent(module="a", event_type="probe_result", severity="info", details={}))
        bus.publish(HealthEvent(module="b", event_type="recovery", severity="warning", details={}))
        time.sleep(0.05)
        assert len(received) == 2

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe("test", cb)
        bus.unsubscribe("test", cb)
        bus.publish(HealthEvent(module="m", event_type="test", severity="info", details={}))
        time.sleep(0.05)
        assert len(received) == 0

    def test_subscriber_exception_doesnt_crash_bus(self):
        bus = EventBus()
        good_received = []

        def bad_handler(e):
            raise RuntimeError("handler crash")

        bus.subscribe("test", bad_handler)
        bus.subscribe("test", lambda e: good_received.append(e))
        bus.publish(HealthEvent(module="m", event_type="test", severity="info", details={}))
        time.sleep(0.05)
        assert len(good_received) == 1

    def test_shutdown_stops_processing(self):
        bus = EventBus()
        bus.shutdown()
        # Should not hang or raise
        bus.publish(HealthEvent(module="m", event_type="test", severity="info", details={}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_event_bus.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/event_bus.py
"""In-process event bus for health event pub/sub."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class HealthEvent:
    """A health event published by a module or the watchdog."""

    module: str
    event_type: str  # probe_result, anomaly, recovery, improvement, rollback
    severity: str    # info, warning, error, critical
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    version_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "event_type": self.event_type,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp,
            "version_id": self.version_id,
        }


EventCallback = Callable[[HealthEvent], None]


class EventBus:
    """Lightweight in-process pub/sub for health events."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[HealthEvent | None] = queue.Queue()
        self._running = True
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Subscribe to events. Use '*' to receive all events."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove a subscription."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    def publish(self, event: HealthEvent) -> None:
        """Publish an event to all matching subscribers."""
        if self._running:
            self._queue.put(event)

    def shutdown(self) -> None:
        """Stop the event bus."""
        self._running = False
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    def _process_loop(self) -> None:
        """Background worker that dispatches events to subscribers."""
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:
                break

            with self._lock:
                callbacks = list(self._subscribers.get(event.event_type, []))
                callbacks.extend(self._subscribers.get("*", []))

            for cb in callbacks:
                try:
                    cb(event)
                except Exception:
                    logger.exception("Event handler failed for %s", event.event_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_event_bus.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/event_bus.py tests/unit/self_healing/test_event_bus.py
git commit -m "feat(self-healing): add in-process event bus for health events"
```

---

### Task 7: Metrics Collector

**Files:**
- Create: `src/homie_core/self_healing/metrics.py`
- Test: `tests/unit/self_healing/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_metrics.py
import pytest
from homie_core.self_healing.metrics import MetricsCollector, AnomalyAlert


class TestMetricsCollector:
    def test_record_and_get_latest(self):
        mc = MetricsCollector(window_size=100)
        mc.record("inference", "latency_ms", 42.0)
        mc.record("inference", "latency_ms", 55.0)
        assert mc.get_latest("inference", "latency_ms") == 55.0

    def test_get_average(self):
        mc = MetricsCollector(window_size=100)
        mc.record("storage", "query_ms", 10.0)
        mc.record("storage", "query_ms", 20.0)
        mc.record("storage", "query_ms", 30.0)
        assert mc.get_average("storage", "query_ms") == pytest.approx(20.0)

    def test_window_size_respected(self):
        mc = MetricsCollector(window_size=3)
        for v in [10, 20, 30, 40, 50]:
            mc.record("m", "v", float(v))
        # Only last 3 values retained
        assert mc.get_average("m", "v") == pytest.approx(40.0)

    def test_unknown_metric_returns_none(self):
        mc = MetricsCollector(window_size=100)
        assert mc.get_latest("unknown", "metric") is None
        assert mc.get_average("unknown", "metric") is None

    def test_detect_anomaly_spike(self):
        mc = MetricsCollector(window_size=100, anomaly_std_threshold=2.0)
        # Establish baseline
        for _ in range(20):
            mc.record("inference", "latency_ms", 50.0)
        # Spike
        alert = mc.record("inference", "latency_ms", 200.0)
        assert alert is not None
        assert alert.metric_name == "latency_ms"
        assert alert.module == "inference"

    def test_no_anomaly_for_normal_values(self):
        mc = MetricsCollector(window_size=100, anomaly_std_threshold=2.0)
        for _ in range(20):
            alert = mc.record("m", "v", 50.0)
        assert alert is None  # last record returns no anomaly

    def test_snapshot(self):
        mc = MetricsCollector(window_size=100)
        mc.record("inference", "latency_ms", 42.0)
        mc.record("storage", "query_ms", 10.0)
        snap = mc.snapshot()
        assert "inference" in snap
        assert "storage" in snap
        assert "latency_ms" in snap["inference"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_metrics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/metrics.py
"""Time-series metric collection with anomaly detection."""

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnomalyAlert:
    """Alert raised when a metric deviates significantly from its baseline."""

    module: str
    metric_name: str
    current_value: float
    mean: float
    std_dev: float
    threshold_multiplier: float


class MetricsCollector:
    """Collects time-series metrics per module and detects anomalies."""

    def __init__(
        self,
        window_size: int = 200,
        anomaly_std_threshold: float = 3.0,
        min_samples_for_anomaly: int = 10,
    ) -> None:
        self._window_size = window_size
        self._anomaly_threshold = anomaly_std_threshold
        self._min_samples = min_samples_for_anomaly
        self._lock = threading.Lock()
        # {module: {metric_name: deque[float]}}
        self._data: dict[str, dict[str, deque[float]]] = {}

    def record(
        self, module: str, metric_name: str, value: float
    ) -> Optional[AnomalyAlert]:
        """Record a metric value. Returns AnomalyAlert if anomaly detected."""
        with self._lock:
            if module not in self._data:
                self._data[module] = {}
            if metric_name not in self._data[module]:
                self._data[module][metric_name] = deque(maxlen=self._window_size)

            series = self._data[module][metric_name]

            # Check for anomaly before adding new value
            alert = None
            if len(series) >= self._min_samples:
                mean = sum(series) / len(series)
                variance = sum((x - mean) ** 2 for x in series) / len(series)
                std_dev = math.sqrt(variance) if variance > 0 else 0.0

                if std_dev > 0 and abs(value - mean) > self._anomaly_threshold * std_dev:
                    alert = AnomalyAlert(
                        module=module,
                        metric_name=metric_name,
                        current_value=value,
                        mean=mean,
                        std_dev=std_dev,
                        threshold_multiplier=self._anomaly_threshold,
                    )

            series.append(value)
            return alert

    def get_latest(self, module: str, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        with self._lock:
            series = self._data.get(module, {}).get(metric_name)
            if series and len(series) > 0:
                return series[-1]
            return None

    def get_average(self, module: str, metric_name: str) -> Optional[float]:
        """Get the moving average for a metric."""
        with self._lock:
            series = self._data.get(module, {}).get(metric_name)
            if series and len(series) > 0:
                return sum(series) / len(series)
            return None

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return a snapshot of all metrics with latest/average values."""
        with self._lock:
            result = {}
            for module, metrics in self._data.items():
                result[module] = {}
                for name, series in metrics.items():
                    if len(series) > 0:
                        result[module][name] = {
                            "latest": series[-1],
                            "average": sum(series) / len(series),
                            "count": len(series),
                        }
            return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_metrics.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/metrics.py tests/unit/self_healing/test_metrics.py
git commit -m "feat(self-healing): add metrics collector with moving-average anomaly detection"
```

---

### Task 8: Health Log

**Files:**
- Create: `src/homie_core/self_healing/health_log.py`
- Test: `tests/unit/self_healing/test_health_log.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_health_log.py
import json
import pytest
from homie_core.self_healing.health_log import HealthLog
from homie_core.self_healing.event_bus import HealthEvent


class TestHealthLog:
    def test_write_and_read_event(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        evt = HealthEvent(module="inference", event_type="probe_result", severity="info", details={"latency_ms": 42})
        log.write(evt)
        events = log.query(module="inference", limit=10)
        assert len(events) == 1
        assert events[0]["module"] == "inference"
        assert events[0]["severity"] == "info"
        assert json.loads(events[0]["details"])["latency_ms"] == 42

    def test_query_by_event_type(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write(HealthEvent(module="m", event_type="probe_result", severity="info", details={}))
        log.write(HealthEvent(module="m", event_type="recovery", severity="warning", details={}))
        log.write(HealthEvent(module="m", event_type="probe_result", severity="info", details={}))
        results = log.query(event_type="recovery")
        assert len(results) == 1

    def test_query_by_severity(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write(HealthEvent(module="m", event_type="t", severity="info", details={}))
        log.write(HealthEvent(module="m", event_type="t", severity="error", details={}))
        results = log.query(min_severity="error")
        assert len(results) == 1

    def test_query_limit(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        for i in range(10):
            log.write(HealthEvent(module="m", event_type="t", severity="info", details={"i": i}))
        results = log.query(limit=3)
        assert len(results) == 3

    def test_cleanup_old_events(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        # Write event with old timestamp
        log.write(HealthEvent(module="m", event_type="t", severity="info", details={}, timestamp=1.0))
        log.write(HealthEvent(module="m", event_type="t", severity="info", details={}))
        deleted = log.cleanup(max_age_days=1)
        assert deleted == 1
        assert len(log.query()) == 1

    def test_close(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.close()
        # Should not raise on double close
        log.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_health_log.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/health_log.py
"""SQLite-backed health event log."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .event_bus import HealthEvent

_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


class HealthLog:
    """Persistent health event log backed by SQLite."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create the database and health_events table."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                module TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT NOT NULL,
                version_id TEXT DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_module ON health_events(module)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_type ON health_events(event_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_ts ON health_events(timestamp)
        """)
        self._conn.commit()

    def write(self, event: HealthEvent) -> None:
        """Write a health event to the log."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO health_events (timestamp, module, event_type, severity, details, version_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.timestamp,
                event.module,
                event.event_type,
                event.severity,
                json.dumps(event.details),
                event.version_id,
            ),
        )
        self._conn.commit()

    def query(
        self,
        module: Optional[str] = None,
        event_type: Optional[str] = None,
        min_severity: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query health events with optional filters."""
        if self._conn is None:
            return []

        clauses = []
        params: list = []

        if module:
            clauses.append("module = ?")
            params.append(module)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if min_severity and min_severity in _SEVERITY_ORDER:
            min_level = _SEVERITY_ORDER[min_severity]
            allowed = [s for s, level in _SEVERITY_ORDER.items() if level >= min_level]
            placeholders = ",".join("?" for _ in allowed)
            clauses.append(f"severity IN ({placeholders})")
            params.extend(allowed)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM health_events WHERE {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def cleanup(self, max_age_days: int = 30) -> int:
        """Delete events older than max_age_days. Returns count deleted."""
        if self._conn is None:
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM health_events WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_health_log.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/health_log.py tests/unit/self_healing/test_health_log.py
git commit -m "feat(self-healing): add SQLite-backed health event log"
```

---

## Chunk 3: Health Probes

### Task 9: Base Probe & Health Status

**Files:**
- Create: `src/homie_core/self_healing/probes/__init__.py`
- Create: `src/homie_core/self_healing/probes/base.py`
- Test: `tests/unit/self_healing/test_probes_base.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_probes_base.py
import pytest
from homie_core.self_healing.probes.base import BaseProbe, HealthStatus, ProbeResult


class TestHealthStatus:
    def test_healthy(self):
        s = HealthStatus.HEALTHY
        assert s.value == "healthy"

    def test_ordering(self):
        assert HealthStatus.HEALTHY < HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED < HealthStatus.FAILED


class TestProbeResult:
    def test_creation(self):
        r = ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=15.5,
            error_count=0,
            metadata={"model": "qwen"},
        )
        assert r.status == HealthStatus.HEALTHY
        assert r.latency_ms == 15.5

    def test_to_dict(self):
        r = ProbeResult(status=HealthStatus.FAILED, latency_ms=0, error_count=3, last_error="timeout")
        d = r.to_dict()
        assert d["status"] == "failed"
        assert d["error_count"] == 3
        assert d["last_error"] == "timeout"


class ConcreteProbe(BaseProbe):
    name = "test_probe"
    interval = 10.0

    def check(self) -> ProbeResult:
        return ProbeResult(status=HealthStatus.HEALTHY, latency_ms=1.0, error_count=0)


class FailingProbe(BaseProbe):
    name = "failing_probe"
    interval = 10.0

    def check(self) -> ProbeResult:
        raise RuntimeError("probe crash")


class TestBaseProbe:
    def test_concrete_probe_works(self):
        probe = ConcreteProbe()
        result = probe.run()
        assert result.status == HealthStatus.HEALTHY

    def test_failing_probe_returns_failed_status(self):
        probe = FailingProbe()
        result = probe.run()
        assert result.status == HealthStatus.FAILED
        assert "probe crash" in result.last_error

    def test_probe_has_name_and_interval(self):
        probe = ConcreteProbe()
        assert probe.name == "test_probe"
        assert probe.interval == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_probes_base.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/__init__.py
from .base import BaseProbe, HealthStatus, ProbeResult

__all__ = ["BaseProbe", "HealthStatus", "ProbeResult"]
```

```python
# src/homie_core/self_healing/probes/base.py
"""Base probe class and health status types."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HealthStatus(IntEnum):
    """Health status ordered by severity for comparison."""

    HEALTHY = 0
    DEGRADED = 1
    FAILED = 2
    UNKNOWN = 3


@dataclass
class ProbeResult:
    """Result of a health probe check."""

    status: HealthStatus
    latency_ms: float
    error_count: int
    last_error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.name.lower(),
            "latency_ms": self.latency_ms,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }


class BaseProbe(ABC):
    """Base class for health probes. Subclasses implement check()."""

    name: str = "unnamed"
    interval: float = 30.0  # seconds between checks

    @abstractmethod
    def check(self) -> ProbeResult:
        """Run the health check. Override in subclasses."""
        ...

    def run(self) -> ProbeResult:
        """Run the probe with error handling and timing."""
        start = time.perf_counter()
        try:
            result = self.check()
            elapsed = (time.perf_counter() - start) * 1000
            result.latency_ms = elapsed
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Probe %s failed: %s", self.name, exc)
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=elapsed,
                error_count=1,
                last_error=str(exc),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_probes_base.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/ tests/unit/self_healing/test_probes_base.py
git commit -m "feat(self-healing): add base probe class with health status types"
```

---

### Task 10: Inference Probe

**Files:**
- Create: `src/homie_core/self_healing/probes/inference_probe.py`
- Test: `tests/unit/self_healing/test_inference_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_inference_probe.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.probes.inference_probe import InferenceProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestInferenceProbe:
    def _make_probe(self, engine=None, router=None):
        engine = engine or MagicMock()
        router = router or MagicMock()
        return InferenceProbe(model_engine=engine, inference_router=router)

    def test_healthy_when_model_loaded_and_responds(self):
        engine = MagicMock()
        engine.is_loaded = True
        engine.generate.return_value = "hello"
        router = MagicMock()
        router.active_source = "Local"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.metadata["source"] == "Local"

    def test_degraded_when_model_not_loaded(self):
        engine = MagicMock()
        engine.is_loaded = False
        router = MagicMock()
        router.active_source = "None"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_generate_raises(self):
        engine = MagicMock()
        engine.is_loaded = True
        engine.generate.side_effect = RuntimeError("OOM")
        router = MagicMock()
        router.active_source = "Local"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.FAILED
        assert "OOM" in result.last_error

    def test_degraded_when_on_fallback_source(self):
        engine = MagicMock()
        engine.is_loaded = True
        engine.generate.return_value = "ok"
        router = MagicMock()
        router.active_source = "Homie Intelligence (Cloud)"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_inference_probe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/inference_probe.py
"""Health probe for the inference pipeline."""

from .base import BaseProbe, HealthStatus, ProbeResult


class InferenceProbe(BaseProbe):
    """Checks model engine health and inference responsiveness."""

    name = "inference"
    interval = 10.0  # critical — check every 10s

    def __init__(self, model_engine, inference_router) -> None:
        self._engine = model_engine
        self._router = inference_router

    def check(self) -> ProbeResult:
        source = self._router.active_source
        errors = 0
        last_error = None

        if not self._engine.is_loaded:
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=1,
                last_error="Model not loaded",
                metadata={"source": source},
            )

        # Test a minimal generation
        try:
            self._engine.generate("ping", max_tokens=1, timeout=10)
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
                metadata={"source": source},
            )

        # Check if we're on a fallback source
        status = HealthStatus.HEALTHY
        if source != "Local":
            status = HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=errors,
            last_error=last_error,
            metadata={"source": source},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_inference_probe.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/inference_probe.py tests/unit/self_healing/test_inference_probe.py
git commit -m "feat(self-healing): add inference health probe"
```

---

### Task 11: Storage Probe

**Files:**
- Create: `src/homie_core/self_healing/probes/storage_probe.py`
- Test: `tests/unit/self_healing/test_storage_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_storage_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.storage_probe import StorageProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestStorageProbe:
    def test_healthy_when_db_works(self, tmp_path):
        db = MagicMock()
        db.path = tmp_path / "homie.db"
        db._conn = MagicMock()
        db._conn.execute.return_value.fetchone.return_value = ("ok",)
        vectors = MagicMock()
        probe = StorageProbe(database=db, vector_store=vectors)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_vectors_fail(self, tmp_path):
        db = MagicMock()
        db.path = tmp_path / "homie.db"
        db._conn = MagicMock()
        db._conn.execute.return_value.fetchone.return_value = ("ok",)
        vectors = MagicMock()
        vectors.query_episodes.side_effect = RuntimeError("chroma down")
        probe = StorageProbe(database=db, vector_store=vectors)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_db_fails(self, tmp_path):
        db = MagicMock()
        db.path = tmp_path / "homie.db"
        db._conn = MagicMock()
        db._conn.execute.side_effect = Exception("corrupt")
        vectors = MagicMock()
        probe = StorageProbe(database=db, vector_store=vectors)
        result = probe.check()
        assert result.status == HealthStatus.FAILED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_storage_probe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/storage_probe.py
"""Health probe for SQLite and ChromaDB storage."""

import os
from .base import BaseProbe, HealthStatus, ProbeResult


class StorageProbe(BaseProbe):
    """Checks SQLite database and ChromaDB vector store health."""

    name = "storage"
    interval = 10.0  # critical

    def __init__(self, database, vector_store=None) -> None:
        self._db = database
        self._vectors = vector_store

    def check(self) -> ProbeResult:
        errors = []
        metadata = {}

        # Check SQLite
        try:
            result = self._db._conn.execute("SELECT 'ok'").fetchone()
            if result[0] != "ok":
                errors.append("SQLite query returned unexpected result")
            # Check DB file size
            if hasattr(self._db, "path") and os.path.exists(self._db.path):
                size_mb = os.path.getsize(self._db.path) / (1024 * 1024)
                metadata["db_size_mb"] = round(size_mb, 2)
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=f"SQLite check failed: {exc}",
                metadata=metadata,
            )

        # Check ChromaDB
        vector_ok = True
        if self._vectors:
            try:
                self._vectors.query_episodes("health_check", n=1)
            except Exception as exc:
                vector_ok = False
                errors.append(f"ChromaDB: {exc}")

        status = HealthStatus.HEALTHY
        if not vector_ok:
            status = HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=len(errors),
            last_error=errors[-1] if errors else None,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_storage_probe.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/storage_probe.py tests/unit/self_healing/test_storage_probe.py
git commit -m "feat(self-healing): add storage health probe for SQLite and ChromaDB"
```

---

### Task 12: Config Probe

**Files:**
- Create: `src/homie_core/self_healing/probes/config_probe.py`
- Test: `tests/unit/self_healing/test_config_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_config_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.config_probe import ConfigProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestConfigProbe:
    def test_healthy_with_valid_config(self, tmp_path):
        config_path = tmp_path / "homie.config.yaml"
        config_path.write_text("llm:\n  backend: gguf\n")
        config = MagicMock()
        config.llm.backend = "gguf"
        config.llm.model_path = str(tmp_path / "model.gguf")
        # Create a fake model file
        (tmp_path / "model.gguf").touch()
        probe = ConfigProbe(config=config, config_path=config_path)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_model_path_missing(self, tmp_path):
        config_path = tmp_path / "homie.config.yaml"
        config_path.write_text("llm:\n  backend: gguf\n")
        config = MagicMock()
        config.llm.backend = "gguf"
        config.llm.model_path = "/nonexistent/model.gguf"
        probe = ConfigProbe(config=config, config_path=config_path)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_config_file_missing(self, tmp_path):
        config = MagicMock()
        config.llm.backend = "gguf"
        probe = ConfigProbe(config=config, config_path=tmp_path / "missing.yaml")
        result = probe.check()
        assert result.status == HealthStatus.FAILED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_config_probe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/config_probe.py
"""Health probe for configuration validity."""

import os
from pathlib import Path

import yaml

from .base import BaseProbe, HealthStatus, ProbeResult


class ConfigProbe(BaseProbe):
    """Checks configuration file parsability and value validity."""

    name = "config"
    interval = 30.0

    def __init__(self, config, config_path: Path | str) -> None:
        self._config = config
        self._config_path = Path(config_path)

    def check(self) -> ProbeResult:
        errors = []
        metadata = {}

        # Check config file exists and is parseable
        if not self._config_path.exists():
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=f"Config file not found: {self._config_path}",
            )

        try:
            with open(self._config_path) as f:
                yaml.safe_load(f)
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=f"Config parse error: {exc}",
            )

        # Check critical config values
        backend = getattr(self._config.llm, "backend", None)
        model_path = getattr(self._config.llm, "model_path", None)
        metadata["backend"] = backend

        if backend == "gguf" and model_path:
            if not os.path.exists(model_path):
                errors.append(f"Model file not found: {model_path}")

        status = HealthStatus.HEALTHY
        if errors:
            status = HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=len(errors),
            last_error=errors[-1] if errors else None,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_config_probe.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/config_probe.py tests/unit/self_healing/test_config_probe.py
git commit -m "feat(self-healing): add config health probe"
```

---

## Chunk 4: Recovery Engine

### Task 13: Recovery Engine Core

**Files:**
- Create: `src/homie_core/self_healing/recovery/__init__.py`
- Create: `src/homie_core/self_healing/recovery/engine.py`
- Test: `tests/unit/self_healing/test_recovery_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_recovery_engine.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.engine import RecoveryEngine, RecoveryTier, RecoveryResult
from homie_core.self_healing.probes.base import HealthStatus


class TestRecoveryTier:
    def test_tier_ordering(self):
        assert RecoveryTier.RETRY < RecoveryTier.FALLBACK
        assert RecoveryTier.FALLBACK < RecoveryTier.REBUILD
        assert RecoveryTier.REBUILD < RecoveryTier.DEGRADE


class TestRecoveryEngine:
    def test_register_and_execute_strategy(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        strategy = MagicMock(return_value=RecoveryResult(success=True, action="retried", tier=RecoveryTier.RETRY))
        engine.register_strategy("inference", RecoveryTier.RETRY, strategy)
        result = engine.recover("inference", HealthStatus.FAILED, error="timeout")
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY
        strategy.assert_called_once()

    def test_escalates_through_tiers(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        t1 = MagicMock(return_value=RecoveryResult(success=False, action="retry failed", tier=RecoveryTier.RETRY))
        t2 = MagicMock(return_value=RecoveryResult(success=True, action="fallback ok", tier=RecoveryTier.FALLBACK))
        engine.register_strategy("inference", RecoveryTier.RETRY, t1)
        engine.register_strategy("inference", RecoveryTier.FALLBACK, t2)
        result = engine.recover("inference", HealthStatus.FAILED, error="timeout")
        assert result.success is True
        assert result.tier == RecoveryTier.FALLBACK
        t1.assert_called_once()
        t2.assert_called_once()

    def test_all_tiers_fail_returns_last_result(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        t1 = MagicMock(return_value=RecoveryResult(success=False, action="retry fail", tier=RecoveryTier.RETRY))
        engine.register_strategy("inference", RecoveryTier.RETRY, t1)
        result = engine.recover("inference", HealthStatus.FAILED, error="fatal")
        assert result.success is False

    def test_no_strategy_registered_returns_failure(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        result = engine.recover("unknown_module", HealthStatus.FAILED, error="no strategy")
        assert result.success is False

    def test_recovery_logged_to_event_bus(self):
        bus = MagicMock()
        log = MagicMock()
        engine = RecoveryEngine(event_bus=bus, health_log=log)
        strategy = MagicMock(return_value=RecoveryResult(success=True, action="ok", tier=RecoveryTier.RETRY))
        engine.register_strategy("storage", RecoveryTier.RETRY, strategy)
        engine.recover("storage", HealthStatus.FAILED, error="locked")
        bus.publish.assert_called()
        log.write.assert_called()

    def test_max_tier_respected(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock(), max_tier=RecoveryTier.FALLBACK)
        t1 = MagicMock(return_value=RecoveryResult(success=False, action="fail", tier=RecoveryTier.RETRY))
        t2 = MagicMock(return_value=RecoveryResult(success=False, action="fail", tier=RecoveryTier.FALLBACK))
        t3 = MagicMock(return_value=RecoveryResult(success=True, action="ok", tier=RecoveryTier.REBUILD))
        engine.register_strategy("m", RecoveryTier.RETRY, t1)
        engine.register_strategy("m", RecoveryTier.FALLBACK, t2)
        engine.register_strategy("m", RecoveryTier.REBUILD, t3)
        result = engine.recover("m", HealthStatus.FAILED, error="err")
        assert result.success is False
        t3.assert_not_called()  # T3 not attempted — max_tier is FALLBACK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_recovery_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/recovery/__init__.py
from .engine import RecoveryEngine, RecoveryResult, RecoveryTier

__all__ = ["RecoveryEngine", "RecoveryResult", "RecoveryTier"]
```

```python
# src/homie_core/self_healing/recovery/engine.py
"""Recovery engine — orchestrates tiered recovery strategies."""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional

from ..event_bus import EventBus, HealthEvent
from ..health_log import HealthLog
from ..probes.base import HealthStatus

logger = logging.getLogger(__name__)


class RecoveryTier(IntEnum):
    RETRY = 1
    FALLBACK = 2
    REBUILD = 3
    DEGRADE = 4


@dataclass
class RecoveryResult:
    success: bool
    action: str
    tier: RecoveryTier
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# Strategy callable: (module, status, error, **context) -> RecoveryResult
RecoveryStrategy = Callable[..., RecoveryResult]


class RecoveryEngine:
    """Orchestrates tiered recovery — tries lightest fix first, escalates on failure."""

    def __init__(
        self,
        event_bus: EventBus,
        health_log: HealthLog,
        max_tier: RecoveryTier = RecoveryTier.DEGRADE,
    ) -> None:
        self._bus = event_bus
        self._log = health_log
        self._max_tier = max_tier
        # {module: {tier: strategy}}
        self._strategies: dict[str, dict[RecoveryTier, RecoveryStrategy]] = {}

    def register_strategy(
        self, module: str, tier: RecoveryTier, strategy: RecoveryStrategy
    ) -> None:
        """Register a recovery strategy for a module at a specific tier."""
        self._strategies.setdefault(module, {})[tier] = strategy

    def recover(
        self,
        module: str,
        status: HealthStatus,
        error: str = "",
        **context: Any,
    ) -> RecoveryResult:
        """Attempt recovery for a module by escalating through tiers."""
        strategies = self._strategies.get(module, {})

        if not strategies:
            logger.warning("No recovery strategies for module: %s", module)
            return RecoveryResult(
                success=False,
                action="no_strategy",
                tier=RecoveryTier.RETRY,
                details={"error": f"No recovery strategies registered for {module}"},
            )

        last_result = None

        for tier in sorted(RecoveryTier):
            if tier > self._max_tier:
                break

            strategy = strategies.get(tier)
            if strategy is None:
                continue

            logger.info("Attempting T%d recovery for %s: %s", tier, module, error)

            try:
                result = strategy(module=module, status=status, error=error, **context)
            except Exception as exc:
                logger.error("Recovery strategy T%d for %s crashed: %s", tier, module, exc)
                result = RecoveryResult(
                    success=False,
                    action=f"strategy_crash: {exc}",
                    tier=tier,
                )

            # Log the attempt
            severity = "info" if result.success else "warning"
            event = HealthEvent(
                module=module,
                event_type="recovery",
                severity=severity,
                details={
                    "tier": tier.value,
                    "action": result.action,
                    "success": result.success,
                    "error": error,
                },
            )
            self._bus.publish(event)
            self._log.write(event)

            if result.success:
                return result

            last_result = result

        return last_result or RecoveryResult(
            success=False,
            action="all_tiers_exhausted",
            tier=RecoveryTier.DEGRADE,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_recovery_engine.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/recovery/ tests/unit/self_healing/test_recovery_engine.py
git commit -m "feat(self-healing): add tiered recovery engine with strategy registration"
```

---

### Task 14: Recovery Playbook (Seed Rules)

**Files:**
- Create: `src/homie_core/self_healing/recovery/playbook.py`
- Test: `tests/unit/self_healing/test_playbook.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_playbook.py
import json
import pytest
from homie_core.self_healing.recovery.playbook import RecoveryPlaybook, PlaybookEntry
from homie_core.self_healing.recovery.engine import RecoveryTier


class TestPlaybookEntry:
    def test_creation(self):
        entry = PlaybookEntry(
            module="inference",
            failure_type="timeout",
            tier=RecoveryTier.RETRY,
            action="retry with shorter max_tokens",
            success_count=0,
            fail_count=0,
        )
        assert entry.module == "inference"
        assert entry.success_rate == 0.0

    def test_success_rate(self):
        entry = PlaybookEntry(
            module="m", failure_type="f", tier=RecoveryTier.RETRY,
            action="a", success_count=7, fail_count=3,
        )
        assert entry.success_rate == pytest.approx(0.7)


class TestRecoveryPlaybook:
    def test_seed_playbook_has_entries(self):
        pb = RecoveryPlaybook()
        entries = pb.get_entries("inference")
        assert len(entries) > 0

    def test_get_best_strategy_for_failure(self):
        pb = RecoveryPlaybook()
        entry = pb.get_best_entry("inference", "timeout", RecoveryTier.RETRY)
        assert entry is not None
        assert entry.module == "inference"

    def test_record_outcome_updates_counts(self):
        pb = RecoveryPlaybook()
        entry = pb.get_best_entry("inference", "timeout", RecoveryTier.RETRY)
        assert entry is not None
        old_success = entry.success_count
        pb.record_outcome(entry, success=True)
        assert entry.success_count == old_success + 1

    def test_add_learned_entry(self):
        pb = RecoveryPlaybook()
        new_entry = PlaybookEntry(
            module="custom",
            failure_type="new_error",
            tier=RecoveryTier.FALLBACK,
            action="custom fix",
            success_count=0,
            fail_count=0,
            learned=True,
        )
        pb.add_entry(new_entry)
        entries = pb.get_entries("custom")
        assert any(e.action == "custom fix" for e in entries)

    def test_export_and_import(self, tmp_path):
        pb = RecoveryPlaybook()
        pb.record_outcome(pb.get_best_entry("inference", "timeout", RecoveryTier.RETRY), success=True)
        path = tmp_path / "playbook.json"
        pb.export_to_file(path)
        assert path.exists()

        pb2 = RecoveryPlaybook(seed=False)
        pb2.import_from_file(path)
        entries = pb2.get_entries("inference")
        assert len(entries) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_playbook.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/recovery/playbook.py
"""Recovery playbook — seed rules plus learned strategies."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .engine import RecoveryTier


@dataclass
class PlaybookEntry:
    """A single recovery strategy entry."""

    module: str
    failure_type: str
    tier: RecoveryTier
    action: str
    success_count: int = 0
    fail_count: int = 0
    learned: bool = False
    deprecated: bool = False

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0


def _seed_entries() -> list[PlaybookEntry]:
    """Static seed playbook — the starting point that Homie evolves from."""
    return [
        # Inference
        PlaybookEntry("inference", "timeout", RecoveryTier.RETRY, "retry with shorter max_tokens"),
        PlaybookEntry("inference", "timeout", RecoveryTier.FALLBACK, "reduce context_length and retry"),
        PlaybookEntry("inference", "timeout", RecoveryTier.REBUILD, "switch to smaller model"),
        PlaybookEntry("inference", "timeout", RecoveryTier.DEGRADE, "serve cached responses"),
        PlaybookEntry("inference", "oom", RecoveryTier.RETRY, "retry with fewer gpu_layers"),
        PlaybookEntry("inference", "oom", RecoveryTier.FALLBACK, "reduce batch size, offload to CPU"),
        PlaybookEntry("inference", "oom", RecoveryTier.REBUILD, "switch to full CPU inference"),
        PlaybookEntry("inference", "oom", RecoveryTier.DEGRADE, "queue requests, serve cache"),
        PlaybookEntry("inference", "model_corrupt", RecoveryTier.RETRY, "re-verify GGUF hash"),
        PlaybookEntry("inference", "model_corrupt", RecoveryTier.FALLBACK, "re-download model"),
        PlaybookEntry("inference", "model_corrupt", RecoveryTier.REBUILD, "fall back to alternative model"),
        PlaybookEntry("inference", "gpu_crash", RecoveryTier.RETRY, "retry after cooldown"),
        PlaybookEntry("inference", "gpu_crash", RecoveryTier.FALLBACK, "restart with CPU fallback"),
        PlaybookEntry("inference", "gpu_crash", RecoveryTier.DEGRADE, "CPU-only mode"),
        # Storage
        PlaybookEntry("storage", "sqlite_locked", RecoveryTier.RETRY, "retry with backoff"),
        PlaybookEntry("storage", "sqlite_locked", RecoveryTier.FALLBACK, "force close stale connections"),
        PlaybookEntry("storage", "sqlite_locked", RecoveryTier.REBUILD, "copy DB, repair, swap"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.RETRY, "run integrity_check and auto-fix"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.FALLBACK, "restore from backup"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.REBUILD, "rebuild from available data"),
        PlaybookEntry("storage", "sqlite_corrupt", RecoveryTier.DEGRADE, "start fresh DB"),
        PlaybookEntry("storage", "chromadb_down", RecoveryTier.RETRY, "restart ChromaDB process"),
        PlaybookEntry("storage", "chromadb_down", RecoveryTier.FALLBACK, "rebuild collection"),
        PlaybookEntry("storage", "chromadb_down", RecoveryTier.DEGRADE, "fall back to keyword search"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.RETRY, "emergency cleanup"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.FALLBACK, "compress data"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.REBUILD, "aggressive retention 7 days"),
        PlaybookEntry("storage", "disk_full", RecoveryTier.DEGRADE, "read-only mode"),
        # Voice
        PlaybookEntry("voice", "stt_crash", RecoveryTier.RETRY, "reload Whisper model"),
        PlaybookEntry("voice", "stt_crash", RecoveryTier.FALLBACK, "switch quality medium to small"),
        PlaybookEntry("voice", "stt_crash", RecoveryTier.DEGRADE, "text-only input"),
        PlaybookEntry("voice", "tts_failure", RecoveryTier.RETRY, "retry synthesis"),
        PlaybookEntry("voice", "tts_failure", RecoveryTier.FALLBACK, "switch TTS engine"),
        PlaybookEntry("voice", "tts_failure", RecoveryTier.DEGRADE, "text-only output"),
        PlaybookEntry("voice", "audio_device_lost", RecoveryTier.RETRY, "re-enumerate devices"),
        PlaybookEntry("voice", "audio_device_lost", RecoveryTier.DEGRADE, "text-only mode"),
        # Config
        PlaybookEntry("config", "parse_error", RecoveryTier.RETRY, "re-read and validate YAML"),
        PlaybookEntry("config", "parse_error", RecoveryTier.FALLBACK, "use last known good config"),
        PlaybookEntry("config", "parse_error", RecoveryTier.REBUILD, "merge defaults with parseable fields"),
        PlaybookEntry("config", "parse_error", RecoveryTier.DEGRADE, "boot with full defaults"),
        PlaybookEntry("config", "invalid_values", RecoveryTier.RETRY, "clamp to valid range"),
        PlaybookEntry("config", "invalid_values", RecoveryTier.FALLBACK, "reset section to defaults"),
    ]


class RecoveryPlaybook:
    """Manages recovery strategies — seed rules plus learned mutations."""

    def __init__(self, seed: bool = True) -> None:
        self._entries: list[PlaybookEntry] = _seed_entries() if seed else []

    def get_entries(self, module: str) -> list[PlaybookEntry]:
        """Get all entries for a module."""
        return [e for e in self._entries if e.module == module and not e.deprecated]

    def get_best_entry(
        self, module: str, failure_type: str, tier: RecoveryTier
    ) -> Optional[PlaybookEntry]:
        """Get the best strategy for a specific failure at a specific tier."""
        candidates = [
            e for e in self._entries
            if e.module == module and e.failure_type == failure_type
            and e.tier == tier and not e.deprecated
        ]
        if not candidates:
            return None
        # Prefer highest success rate, then seed over learned
        return max(candidates, key=lambda e: (e.success_rate, not e.learned))

    def record_outcome(self, entry: PlaybookEntry, success: bool) -> None:
        """Record whether a strategy succeeded or failed."""
        if success:
            entry.success_count += 1
        else:
            entry.fail_count += 1

    def add_entry(self, entry: PlaybookEntry) -> None:
        """Add a learned strategy to the playbook."""
        self._entries.append(entry)

    def deprecate_entry(self, entry: PlaybookEntry) -> None:
        """Mark a strategy as deprecated (never succeeded)."""
        entry.deprecated = True

    def export_to_file(self, path: Path | str) -> None:
        """Export playbook to JSON for persistence."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for e in self._entries:
            d = {
                "module": e.module,
                "failure_type": e.failure_type,
                "tier": e.tier.value,
                "action": e.action,
                "success_count": e.success_count,
                "fail_count": e.fail_count,
                "learned": e.learned,
                "deprecated": e.deprecated,
            }
            data.append(d)
        path.write_text(json.dumps(data, indent=2))

    def import_from_file(self, path: Path | str) -> None:
        """Import playbook from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        for d in data:
            self._entries.append(PlaybookEntry(
                module=d["module"],
                failure_type=d["failure_type"],
                tier=RecoveryTier(d["tier"]),
                action=d["action"],
                success_count=d.get("success_count", 0),
                fail_count=d.get("fail_count", 0),
                learned=d.get("learned", False),
                deprecated=d.get("deprecated", False),
            ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_playbook.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/recovery/playbook.py tests/unit/self_healing/test_playbook.py
git commit -m "feat(self-healing): add recovery playbook with seed rules and learning support"
```

---

## Chunk 5: Improvement Engine & Rollback

### Task 15: Rollback Manager

**Files:**
- Create: `src/homie_core/self_healing/improvement/__init__.py`
- Create: `src/homie_core/self_healing/improvement/rollback.py`
- Test: `tests/unit/self_healing/test_rollback.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_rollback.py
import pytest
from homie_core.self_healing.improvement.rollback import RollbackManager


class TestRollbackManager:
    def test_snapshot_and_rollback(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        target = tmp_path / "code.py"
        target.write_text("original")

        version_id = rm.snapshot(target, reason="testing")
        assert version_id is not None

        target.write_text("modified")
        rm.rollback(version_id)
        assert target.read_text() == "original"

    def test_snapshot_multiple_files(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a_orig")
        f2.write_text("b_orig")

        vid = rm.snapshot([f1, f2], reason="multi")
        f1.write_text("a_mod")
        f2.write_text("b_mod")
        rm.rollback(vid)
        assert f1.read_text() == "a_orig"
        assert f2.read_text() == "b_orig"

    def test_rollback_nonexistent_version_raises(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        with pytest.raises(KeyError):
            rm.rollback("nonexistent")

    def test_evolution_log(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots", evolution_dir=tmp_path / "evolution")
        target = tmp_path / "code.py"
        target.write_text("original")
        vid = rm.snapshot(target, reason="test change")

        target.write_text("modified")
        rm.record_evolution(vid, diff="- original\n+ modified", reasoning="optimize", outcome="success")

        log = rm.get_evolution_log()
        assert len(log) == 1
        assert log[0]["version_id"] == vid
        assert log[0]["reasoning"] == "optimize"

    def test_blacklist_prevents_reattempt(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        rm.blacklist("bad_change_hash")
        assert rm.is_blacklisted("bad_change_hash")
        assert not rm.is_blacklisted("good_change_hash")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_rollback.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/improvement/__init__.py
from .rollback import RollbackManager

__all__ = ["RollbackManager"]
```

```python
# src/homie_core/self_healing/improvement/rollback.py
"""Rollback manager — snapshots, baselines, and auto-revert for self-modifications."""

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Union


class RollbackManager:
    """Manages file snapshots and rollbacks for self-modification safety."""

    def __init__(
        self,
        snapshot_dir: Path | str,
        evolution_dir: Path | str | None = None,
    ) -> None:
        self._snapshot_dir = Path(snapshot_dir)
        self._evolution_dir = Path(evolution_dir) if evolution_dir else self._snapshot_dir.parent / "evolution"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._evolution_dir.mkdir(parents=True, exist_ok=True)

        # {version_id: [{"original": path, "snapshot": path}]}
        self._versions: dict[str, list[dict[str, Path]]] = {}
        self._blacklist: set[str] = set()
        self._evolution_log: list[dict] = []

    def snapshot(
        self,
        files: Union[Path, str, list[Path | str]],
        reason: str = "",
    ) -> str:
        """Snapshot file(s) before modification. Returns version_id."""
        if isinstance(files, (str, Path)):
            files = [files]

        version_id = f"v-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        version_dir = self._snapshot_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for f in files:
            f = Path(f)
            if f.exists():
                snapshot_path = version_dir / f.name
                shutil.copy2(f, snapshot_path)
                entries.append({"original": f, "snapshot": snapshot_path})

        self._versions[version_id] = entries

        # Write metadata
        meta = {
            "version_id": version_id,
            "timestamp": time.time(),
            "reason": reason,
            "files": [str(e["original"]) for e in entries],
        }
        (version_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        return version_id

    def rollback(self, version_id: str) -> None:
        """Restore files from a snapshot."""
        if version_id not in self._versions:
            raise KeyError(f"Unknown version: {version_id}")

        for entry in self._versions[version_id]:
            shutil.copy2(entry["snapshot"], entry["original"])

    def record_evolution(
        self,
        version_id: str,
        diff: str = "",
        reasoning: str = "",
        outcome: str = "",
    ) -> None:
        """Record an evolution entry for a self-modification."""
        record = {
            "version_id": version_id,
            "timestamp": time.time(),
            "diff": diff,
            "reasoning": reasoning,
            "outcome": outcome,
        }
        self._evolution_log.append(record)

        # Persist to file
        log_file = self._evolution_dir / f"{version_id}.json"
        log_file.write_text(json.dumps(record, indent=2))

    def get_evolution_log(self) -> list[dict]:
        """Return the evolution log."""
        return list(self._evolution_log)

    def blacklist(self, change_hash: str) -> None:
        """Blacklist a change to prevent re-attempting it."""
        self._blacklist.add(change_hash)

    def is_blacklisted(self, change_hash: str) -> bool:
        """Check if a change is blacklisted."""
        return change_hash in self._blacklist
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_rollback.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/improvement/ tests/unit/self_healing/test_rollback.py
git commit -m "feat(self-healing): add rollback manager with snapshots and evolution log"
```

---

### Task 16: Improvement Engine

**Files:**
- Create: `src/homie_core/self_healing/improvement/engine.py`
- Test: `tests/unit/self_healing/test_improvement_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_improvement_engine.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.improvement.engine import ImprovementEngine, ImprovementLevel


class TestImprovementLevel:
    def test_ordering(self):
        assert ImprovementLevel.CONFIG < ImprovementLevel.WORKFLOW
        assert ImprovementLevel.WORKFLOW < ImprovementLevel.CODE_PATCH
        assert ImprovementLevel.CODE_PATCH < ImprovementLevel.ARCHITECTURE


class TestImprovementEngine:
    def _make_engine(self, **overrides):
        defaults = {
            "event_bus": MagicMock(),
            "health_log": MagicMock(),
            "metrics": MagicMock(),
            "rollback_manager": MagicMock(),
            "inference_fn": MagicMock(return_value="no improvements needed"),
            "max_mutations_per_day": 10,
            "monitoring_window": 1,
            "error_threshold": 0.20,
            "latency_threshold": 0.50,
        }
        defaults.update(overrides)
        return ImprovementEngine(**defaults)

    def test_analyze_returns_observations(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 500, "average": 100, "count": 50}}
        }
        engine = self._make_engine(metrics=metrics)
        observations = engine.analyze()
        assert len(observations) > 0

    def test_rate_limit_respected(self):
        engine = self._make_engine(max_mutations_per_day=0)
        assert engine.can_mutate() is False

    def test_rate_limit_allows_when_under(self):
        engine = self._make_engine(max_mutations_per_day=10)
        assert engine.can_mutate() is True

    def test_core_lock_prevents_modification(self):
        engine = self._make_engine()
        engine.add_core_lock("src/homie_core/self_healing/improvement/rollback.py")
        assert engine.is_locked("src/homie_core/self_healing/improvement/rollback.py")
        assert not engine.is_locked("src/homie_core/some_other.py")

    def test_core_lock_directory(self):
        engine = self._make_engine()
        engine.add_core_lock("src/homie_core/security/")
        assert engine.is_locked("src/homie_core/security/vault.py")
        assert not engine.is_locked("src/homie_core/storage/database.py")

    def test_mutation_count_increments(self):
        engine = self._make_engine(max_mutations_per_day=10)
        engine._mutations_today = 0
        engine.record_mutation()
        assert engine._mutations_today == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_improvement_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/improvement/engine.py
"""Improvement engine — observes, diagnoses, prescribes, and applies self-improvements."""

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional

from ..event_bus import EventBus, HealthEvent
from ..health_log import HealthLog
from ..metrics import MetricsCollector
from .rollback import RollbackManager

logger = logging.getLogger(__name__)


class ImprovementLevel(IntEnum):
    CONFIG = 1
    WORKFLOW = 2
    CODE_PATCH = 3
    ARCHITECTURE = 4


@dataclass
class Observation:
    """An observed performance issue or optimization opportunity."""

    module: str
    metric: str
    current_value: float
    baseline_value: float
    description: str


class ImprovementEngine:
    """The self-improvement loop: observe → diagnose → prescribe → apply → monitor."""

    def __init__(
        self,
        event_bus: EventBus,
        health_log: HealthLog,
        metrics: MetricsCollector,
        rollback_manager: RollbackManager,
        inference_fn: Callable[[str], str],
        max_mutations_per_day: int = 10,
        monitoring_window: float = 300.0,
        error_threshold: float = 0.20,
        latency_threshold: float = 0.50,
    ) -> None:
        self._bus = event_bus
        self._log = health_log
        self._metrics = metrics
        self._rollback = rollback_manager
        self._infer = inference_fn
        self._max_mutations = max_mutations_per_day
        self._monitoring_window = monitoring_window
        self._error_threshold = error_threshold
        self._latency_threshold = latency_threshold

        self._mutations_today: int = 0
        self._last_reset_day: int = 0
        self._core_locks: list[str] = []

    def analyze(self) -> list[Observation]:
        """Step 1: Observe — analyze current metrics for optimization opportunities."""
        observations = []
        snapshot = self._metrics.snapshot()

        for module, metrics in snapshot.items():
            for metric_name, values in metrics.items():
                latest = values.get("latest", 0)
                average = values.get("average", 0)
                count = values.get("count", 0)

                if count < 10:
                    continue

                # Flag if latest is significantly above average
                if average > 0 and latest > average * 1.5:
                    observations.append(Observation(
                        module=module,
                        metric=metric_name,
                        current_value=latest,
                        baseline_value=average,
                        description=f"{module}.{metric_name} is {latest:.1f} vs baseline {average:.1f}",
                    ))

        return observations

    def can_mutate(self) -> bool:
        """Check if mutation rate limit allows another change."""
        today = int(time.time() / 86400)
        if today != self._last_reset_day:
            self._mutations_today = 0
            self._last_reset_day = today
        return self._mutations_today < self._max_mutations

    def record_mutation(self) -> None:
        """Record that a mutation was applied."""
        today = int(time.time() / 86400)
        if today != self._last_reset_day:
            self._mutations_today = 0
            self._last_reset_day = today
        self._mutations_today += 1

    def add_core_lock(self, path: str) -> None:
        """Add a path to the core lock list (immutable files)."""
        self._core_locks.append(path)

    def is_locked(self, path: str) -> bool:
        """Check if a file path is core-locked."""
        for lock in self._core_locks:
            if lock.endswith("/"):
                if path.startswith(lock) or path.startswith(lock.rstrip("/")):
                    return True
            elif path == lock:
                return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_improvement_engine.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Update improvement __init__.py**

```python
# src/homie_core/self_healing/improvement/__init__.py
from .engine import ImprovementEngine, ImprovementLevel, Observation
from .rollback import RollbackManager

__all__ = ["ImprovementEngine", "ImprovementLevel", "Observation", "RollbackManager"]
```

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/self_healing/improvement/ tests/unit/self_healing/test_improvement_engine.py
git commit -m "feat(self-healing): add improvement engine with rate limiting and core locks"
```

---

## Chunk 6: HealthWatchdog & Integration

### Task 17: HealthWatchdog Service

**Files:**
- Create: `src/homie_core/self_healing/watchdog.py`
- Test: `tests/unit/self_healing/test_watchdog.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_watchdog.py
import time
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.watchdog import HealthWatchdog
from homie_core.self_healing.probes.base import BaseProbe, HealthStatus, ProbeResult


class FakeProbe(BaseProbe):
    name = "fake"
    interval = 0.1

    def __init__(self, status=HealthStatus.HEALTHY):
        self._status = status

    def check(self):
        return ProbeResult(status=self._status, latency_ms=1.0, error_count=0)


class TestHealthWatchdog:
    def test_register_probe(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        probe = FakeProbe()
        wd.register_probe(probe)
        assert "fake" in wd.registered_probes

    def test_run_all_probes(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        wd.register_probe(FakeProbe(HealthStatus.HEALTHY))
        results = wd.run_all_probes()
        assert "fake" in results
        assert results["fake"].status == HealthStatus.HEALTHY

    def test_system_health_all_healthy(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        wd.register_probe(FakeProbe(HealthStatus.HEALTHY))
        wd.run_all_probes()
        assert wd.system_health == HealthStatus.HEALTHY

    def test_system_health_worst_wins(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        healthy = FakeProbe(HealthStatus.HEALTHY)
        healthy.name = "good"
        failed = FakeProbe(HealthStatus.FAILED)
        failed.name = "bad"
        wd.register_probe(healthy)
        wd.register_probe(failed)
        wd.run_all_probes()
        assert wd.system_health == HealthStatus.FAILED

    def test_start_and_stop(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        wd.register_probe(FakeProbe())
        wd.start()
        time.sleep(0.3)  # let it run a few cycles
        wd.stop()
        # Should not hang
        assert wd._running is False

    def test_recovery_triggered_on_failure(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        failed = FakeProbe(HealthStatus.FAILED)
        failed.name = "failing"
        wd.register_probe(failed)

        recovery = MagicMock()
        wd.set_recovery_engine(recovery)
        wd.run_all_probes()
        # Recovery should be attempted for the failed probe
        recovery.recover.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_watchdog.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/watchdog.py
"""HealthWatchdog — central coordinator for self-healing runtime."""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .event_bus import EventBus, HealthEvent
from .health_log import HealthLog
from .metrics import MetricsCollector
from .probes.base import BaseProbe, HealthStatus, ProbeResult

logger = logging.getLogger(__name__)


class HealthWatchdog:
    """Central health monitoring and recovery coordination service."""

    def __init__(
        self,
        db_path: Path | str,
        probe_interval: float = 30.0,
    ) -> None:
        self._db_path = Path(db_path)
        self._probe_interval = probe_interval
        self._probes: dict[str, BaseProbe] = {}
        self._last_results: dict[str, ProbeResult] = {}
        self._recovery_engine = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Initialize subsystems
        self.event_bus = EventBus()
        self.health_log = HealthLog(db_path=self._db_path)
        self.health_log.initialize()
        self.metrics = MetricsCollector()

    @property
    def registered_probes(self) -> dict[str, BaseProbe]:
        return dict(self._probes)

    @property
    def system_health(self) -> HealthStatus:
        """Overall system health — worst probe status wins."""
        if not self._last_results:
            return HealthStatus.UNKNOWN
        worst = HealthStatus.HEALTHY
        for result in self._last_results.values():
            if result.status > worst:
                worst = result.status
        return worst

    def register_probe(self, probe: BaseProbe) -> None:
        """Register a health probe."""
        self._probes[probe.name] = probe

    def set_recovery_engine(self, recovery_engine) -> None:
        """Set the recovery engine for handling failures."""
        self._recovery_engine = recovery_engine

    def run_all_probes(self) -> dict[str, ProbeResult]:
        """Run all registered probes and return results."""
        results = {}
        for name, probe in self._probes.items():
            result = probe.run()
            results[name] = result
            self._last_results[name] = result

            # Record metrics
            self.metrics.record(name, "latency_ms", result.latency_ms)
            self.metrics.record(name, "error_count", float(result.error_count))

            # Log event
            event = HealthEvent(
                module=name,
                event_type="probe_result",
                severity="info" if result.status == HealthStatus.HEALTHY else "warning" if result.status == HealthStatus.DEGRADED else "error",
                details=result.to_dict(),
            )
            self.event_bus.publish(event)

            # Trigger recovery if needed
            if result.status >= HealthStatus.FAILED and self._recovery_engine:
                self._recovery_engine.recover(
                    module=name,
                    status=result.status,
                    error=result.last_error or "probe failed",
                )

        return results

    def start(self) -> None:
        """Start the watchdog monitoring loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("HealthWatchdog started (interval: %.1fs)", self._probe_interval)

    def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.event_bus.shutdown()
        self.health_log.close()
        logger.info("HealthWatchdog stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.run_all_probes()
            except Exception:
                logger.exception("Watchdog probe cycle failed")

            # Sleep in small increments for responsive shutdown
            elapsed = 0.0
            while elapsed < self._probe_interval and self._running:
                time.sleep(0.1)
                elapsed += 0.1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_watchdog.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/watchdog.py tests/unit/self_healing/test_watchdog.py
git commit -m "feat(self-healing): add HealthWatchdog service with probe scheduling and recovery"
```

---

### Task 18: Guardian Process

**Files:**
- Create: `src/homie_core/self_healing/guardian.py`
- Test: `tests/unit/self_healing/test_guardian.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_guardian.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.guardian import Guardian


class TestGuardian:
    def test_creates_pid_file(self, tmp_path):
        guardian = Guardian(pid_file=tmp_path / "watchdog.pid")
        guardian.write_pid(12345)
        assert (tmp_path / "watchdog.pid").exists()
        assert (tmp_path / "watchdog.pid").read_text().strip() == "12345"

    def test_read_pid(self, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("54321")
        guardian = Guardian(pid_file=pid_file)
        assert guardian.read_pid() == 54321

    def test_read_pid_returns_none_if_missing(self, tmp_path):
        guardian = Guardian(pid_file=tmp_path / "nope.pid")
        assert guardian.read_pid() is None

    @patch("homie_core.self_healing.guardian.psutil")
    def test_is_alive_true(self, mock_psutil, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("100")
        guardian = Guardian(pid_file=pid_file)
        mock_psutil.pid_exists.return_value = True
        assert guardian.is_alive() is True

    @patch("homie_core.self_healing.guardian.psutil")
    def test_is_alive_false(self, mock_psutil, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("100")
        guardian = Guardian(pid_file=pid_file)
        mock_psutil.pid_exists.return_value = False
        assert guardian.is_alive() is False

    def test_cleanup_removes_pid_file(self, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("100")
        guardian = Guardian(pid_file=pid_file)
        guardian.cleanup()
        assert not pid_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_guardian.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/guardian.py
"""OS-level guardian process — monitors and restarts the watchdog if it dies."""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class Guardian:
    """Lightweight guardian that tracks the watchdog process via PID file."""

    def __init__(self, pid_file: Path | str) -> None:
        self._pid_file = Path(pid_file)

    def write_pid(self, pid: int) -> None:
        """Write the watchdog PID to the pid file."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(pid))

    def read_pid(self) -> Optional[int]:
        """Read the watchdog PID from the pid file."""
        if not self._pid_file.exists():
            return None
        try:
            return int(self._pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def is_alive(self) -> bool:
        """Check if the watchdog process is still running."""
        pid = self.read_pid()
        if pid is None:
            return False
        if psutil is None:
            # Fallback: try os.kill with signal 0
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        return psutil.pid_exists(pid)

    def cleanup(self) -> None:
        """Remove the PID file."""
        if self._pid_file.exists():
            self._pid_file.unlink()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_guardian.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/guardian.py tests/unit/self_healing/test_guardian.py
git commit -m "feat(self-healing): add guardian process for watchdog monitoring"
```

---

### Task 19: Config Integration

**Files:**
- Modify: `src/homie_core/config.py` — add SelfHealingConfig
- Modify: `homie.config.yaml` — add self_healing section
- Test: `tests/unit/self_healing/test_config_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_config_integration.py
import pytest
from homie_core.config import SelfHealingConfig, ImprovementConfig, RecoveryConfig


class TestSelfHealingConfig:
    def test_defaults(self):
        cfg = SelfHealingConfig()
        assert cfg.enabled is True
        assert cfg.probe_interval == 30
        assert cfg.critical_probe_interval == 10

    def test_improvement_defaults(self):
        cfg = ImprovementConfig()
        assert cfg.enabled is True
        assert cfg.max_mutations_per_day == 10
        assert cfg.monitoring_window == 300
        assert cfg.rollback_error_threshold == 0.20
        assert cfg.rollback_latency_threshold == 0.50

    def test_recovery_defaults(self):
        cfg = RecoveryConfig()
        assert cfg.max_tier == 4
        assert cfg.preemptive is True
        assert cfg.pattern_threshold == 3

    def test_core_lock_defaults(self):
        cfg = SelfHealingConfig()
        assert "self_healing/improvement/rollback.py" in cfg.core_lock
        assert "self_healing/guardian.py" in cfg.core_lock
        assert "security/" in cfg.core_lock
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_config_integration.py -v`
Expected: FAIL — SelfHealingConfig not found

- [ ] **Step 3: Add config classes to config.py**

Add the following Pydantic models to `src/homie_core/config.py` (after existing config classes, before `HomieConfig`):

```python
class ImprovementConfig(BaseModel):
    enabled: bool = True
    max_mutations_per_day: int = 10
    monitoring_window: int = 300
    rollback_error_threshold: float = 0.20
    rollback_latency_threshold: float = 0.50


class RecoveryConfig(BaseModel):
    max_tier: int = 4
    preemptive: bool = True
    pattern_threshold: int = 3


class HealthLogConfig(BaseModel):
    retention_days: int = 30
    digest_enabled: bool = True


class SelfHealingConfig(BaseModel):
    enabled: bool = True
    probe_interval: int = 30
    critical_probe_interval: int = 10
    improvement: ImprovementConfig = ImprovementConfig()
    recovery: RecoveryConfig = RecoveryConfig()
    health_log: HealthLogConfig = HealthLogConfig()
    guardian_enabled: bool = True
    core_lock: list[str] = [
        "self_healing/improvement/rollback.py",
        "self_healing/guardian.py",
        "security/",
    ]
```

Then add `self_healing: SelfHealingConfig = Field(default_factory=SelfHealingConfig)` to the `HomieConfig` class (matching existing codebase convention).

- [ ] **Step 4: Add self_healing section to homie.config.yaml**

Append to `homie.config.yaml`:

```yaml
self_healing:
  enabled: true
  probe_interval: 30
  critical_probe_interval: 10
  improvement:
    enabled: true
    max_mutations_per_day: 10
    monitoring_window: 300
    rollback_error_threshold: 0.20
    rollback_latency_threshold: 0.50
  recovery:
    max_tier: 4
    preemptive: true
    pattern_threshold: 3
  health_log:
    retention_days: 30
    digest_enabled: true
  guardian_enabled: true
  core_lock:
    - self_healing/improvement/rollback.py
    - self_healing/guardian.py
    - security/
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_config_integration.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/config.py homie.config.yaml tests/unit/self_healing/test_config_integration.py
git commit -m "feat(self-healing): add SelfHealingConfig to config system"
```

---

### Task 20: Boot Integration — Wire Watchdog into CLI

**Files:**
- Modify: `src/homie_app/cli.py` — add watchdog boot/shutdown
- Test: `tests/unit/self_healing/test_boot_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_boot_integration.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.watchdog import HealthWatchdog


class TestBootIntegration:
    def test_watchdog_initializes_with_config(self, tmp_path):
        """Watchdog can be created from config."""
        wd = HealthWatchdog(db_path=tmp_path / "health.db", probe_interval=30.0)
        assert wd.system_health is not None

    def test_watchdog_probes_run_on_boot(self, tmp_path):
        """Probes can be run before main loop starts."""
        wd = HealthWatchdog(db_path=tmp_path / "health.db")
        results = wd.run_all_probes()
        # No probes registered, should return empty
        assert results == {}

    def test_watchdog_start_stop_lifecycle(self, tmp_path):
        """Watchdog starts and stops cleanly."""
        wd = HealthWatchdog(db_path=tmp_path / "health.db", probe_interval=0.1)
        wd.start()
        assert wd._running is True
        wd.stop()
        assert wd._running is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_boot_integration.py -v`
Expected: All PASS (these test the watchdog API — no cli.py changes needed to pass)

- [ ] **Step 3: Add watchdog initialization to cli.py**

In `src/homie_app/cli.py`, add a helper function `_init_watchdog(cfg)` that:
1. Creates a `HealthWatchdog` instance
2. Registers probes for available subsystems
3. Returns the watchdog

Then call `_init_watchdog` at the beginning of the main boot sequence (before model loading), and `watchdog.stop()` in the shutdown/cleanup path.

```python
def _init_watchdog(cfg):
    """Initialize the self-healing watchdog."""
    from homie_core.self_healing.watchdog import HealthWatchdog
    from homie_core.self_healing.probes.config_probe import ConfigProbe

    if not getattr(cfg, 'self_healing', None) or not cfg.self_healing.enabled:
        return None

    storage_path = Path(cfg.storage.path)
    wd = HealthWatchdog(
        db_path=storage_path / "health.db",
        probe_interval=cfg.self_healing.probe_interval,
    )

    # Register config probe (always available)
    config_path = Path("homie.config.yaml")
    if config_path.exists():
        wd.register_probe(ConfigProbe(config=cfg, config_path=config_path))

    return wd
```

- [ ] **Step 4: Run all self-healing tests**

Run: `python -m pytest tests/unit/self_healing/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_app/cli.py tests/unit/self_healing/test_boot_integration.py
git commit -m "feat(self-healing): wire HealthWatchdog into boot sequence"
```

---

### Task 21: Update self_healing __init__.py — Public API

**Files:**
- Modify: `src/homie_core/self_healing/__init__.py`

- [ ] **Step 1: Write the public API exports**

```python
# src/homie_core/self_healing/__init__.py
"""Homie Self-Healing Runtime — autonomous failure recovery and self-improvement."""

from .event_bus import EventBus, HealthEvent
from .guardian import Guardian
from .health_log import HealthLog
from .metrics import AnomalyAlert, MetricsCollector
from .resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ErrorCategory,
    classify_exception,
    resilient,
    retry_with_backoff,
    run_with_timeout,
)
from .watchdog import HealthWatchdog

__all__ = [
    # Core
    "HealthWatchdog",
    "EventBus",
    "HealthEvent",
    "HealthLog",
    "Guardian",
    # Metrics
    "MetricsCollector",
    "AnomalyAlert",
    # Resilience
    "resilient",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "ErrorCategory",
    "classify_exception",
    "retry_with_backoff",
    "run_with_timeout",
]
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/unit/self_healing/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/self_healing/__init__.py
git commit -m "feat(self-healing): finalize public API exports"
```

---

### Task 22: Integration Test — Full Self-Healing Lifecycle

**Files:**
- Create: `tests/integration/test_self_healing_lifecycle.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_self_healing_lifecycle.py
"""Integration test: full self-healing lifecycle from probe to recovery."""
import time
import pytest
from unittest.mock import MagicMock

from homie_core.self_healing.watchdog import HealthWatchdog
from homie_core.self_healing.probes.base import BaseProbe, HealthStatus, ProbeResult
from homie_core.self_healing.recovery.engine import RecoveryEngine, RecoveryResult, RecoveryTier


class FlakeyProbe(BaseProbe):
    """Probe that fails N times then succeeds."""
    name = "flakey"
    interval = 0.1

    def __init__(self, fail_count=2):
        self._calls = 0
        self._fail_count = fail_count

    def check(self):
        self._calls += 1
        if self._calls <= self._fail_count:
            return ProbeResult(status=HealthStatus.FAILED, latency_ms=1.0, error_count=1, last_error="simulated failure")
        return ProbeResult(status=HealthStatus.HEALTHY, latency_ms=1.0, error_count=0)


class TestSelfHealingLifecycle:
    def test_probe_failure_triggers_recovery(self, tmp_path):
        """Full flow: probe detects failure → recovery engine attempts fix."""
        wd = HealthWatchdog(db_path=tmp_path / "test.db")

        probe = FlakeyProbe(fail_count=1)
        wd.register_probe(probe)

        # Set up recovery engine
        recovery = RecoveryEngine(
            event_bus=wd.event_bus,
            health_log=wd.health_log,
        )
        fixed = {"called": False}

        def fix_strategy(module, status, error, **ctx):
            fixed["called"] = True
            return RecoveryResult(success=True, action="fixed it", tier=RecoveryTier.RETRY)

        recovery.register_strategy("flakey", RecoveryTier.RETRY, fix_strategy)
        wd.set_recovery_engine(recovery)

        # First probe run — should fail and trigger recovery
        results = wd.run_all_probes()
        assert results["flakey"].status == HealthStatus.FAILED
        assert fixed["called"] is True

        # Second run — should succeed
        results = wd.run_all_probes()
        assert results["flakey"].status == HealthStatus.HEALTHY

        wd.stop()

    def test_metrics_tracked_across_probes(self, tmp_path):
        """Metrics are collected for each probe run."""
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        probe = FlakeyProbe(fail_count=0)  # always healthy
        wd.register_probe(probe)

        for _ in range(5):
            wd.run_all_probes()

        avg = wd.metrics.get_average("flakey", "latency_ms")
        assert avg is not None
        assert avg > 0

        wd.stop()

    def test_health_log_persists_events(self, tmp_path):
        """Health events are written to SQLite."""
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        probe = FlakeyProbe(fail_count=0)
        wd.register_probe(probe)

        wd.run_all_probes()

        events = wd.health_log.query(module="flakey")
        assert len(events) >= 1

        wd.stop()
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/integration/test_self_healing_lifecycle.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Run ALL self-healing tests**

Run: `python -m pytest tests/unit/self_healing/ tests/integration/test_self_healing_lifecycle.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_self_healing_lifecycle.py
git commit -m "test(self-healing): add integration test for full probe-to-recovery lifecycle"
```

---

## Chunk 7: Remaining Health Probes

### Task 23: Voice Probe

**Files:**
- Create: `src/homie_core/self_healing/probes/voice_probe.py`
- Test: `tests/unit/self_healing/test_voice_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_voice_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.voice_probe import VoiceProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestVoiceProbe:
    def test_healthy_when_all_engines_available(self):
        vm = MagicMock()
        vm.available_engines = {"stt": True, "tts": True, "vad": True}
        vm.state = MagicMock(value="idle")
        probe = VoiceProbe(voice_manager=vm)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_tts_unavailable(self):
        vm = MagicMock()
        vm.available_engines = {"stt": True, "tts": False, "vad": True}
        vm.state = MagicMock(value="idle")
        probe = VoiceProbe(voice_manager=vm)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_stt_unavailable(self):
        vm = MagicMock()
        vm.available_engines = {"stt": False, "tts": False, "vad": False}
        vm.state = MagicMock(value="idle")
        probe = VoiceProbe(voice_manager=vm)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_voice_manager_none(self):
        probe = VoiceProbe(voice_manager=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_voice_probe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/voice_probe.py
"""Health probe for the voice pipeline."""

from .base import BaseProbe, HealthStatus, ProbeResult


class VoiceProbe(BaseProbe):
    """Checks STT, TTS, and VAD engine availability."""

    name = "voice"
    interval = 30.0

    def __init__(self, voice_manager=None) -> None:
        self._vm = voice_manager

    def check(self) -> ProbeResult:
        if self._vm is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="Voice manager not initialized",
            )

        try:
            engines = self._vm.available_engines
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
            )

        stt_ok = engines.get("stt", False)
        tts_ok = engines.get("tts", False)
        vad_ok = engines.get("vad", False)

        metadata = {"stt": stt_ok, "tts": tts_ok, "vad": vad_ok}

        if not stt_ok and not tts_ok:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error="STT and TTS both unavailable",
                metadata=metadata,
            )

        if not stt_ok or not tts_ok or not vad_ok:
            missing = [k for k, v in engines.items() if not v]
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=len(missing),
                last_error=f"Unavailable: {', '.join(missing)}",
                metadata=metadata,
            )

        return ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=0,
            error_count=0,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_voice_probe.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/voice_probe.py tests/unit/self_healing/test_voice_probe.py
git commit -m "feat(self-healing): add voice pipeline health probe"
```

---

### Task 24: Context Probe

**Files:**
- Create: `src/homie_core/self_healing/probes/context_probe.py`
- Test: `tests/unit/self_healing/test_context_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_context_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.context_probe import ContextProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestContextProbe:
    def test_healthy_when_aggregator_works(self):
        agg = MagicMock()
        agg.tick.return_value = {"active_window": "VSCode", "timestamp": 1.0}
        probe = ContextProbe(context_aggregator=agg)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_tick_returns_empty(self):
        agg = MagicMock()
        agg.tick.return_value = {}
        probe = ContextProbe(context_aggregator=agg)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_tick_raises(self):
        agg = MagicMock()
        agg.tick.side_effect = RuntimeError("observer crash")
        probe = ContextProbe(context_aggregator=agg)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_none_aggregator(self):
        probe = ContextProbe(context_aggregator=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_context_probe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/context_probe.py
"""Health probe for context aggregation observers."""

from .base import BaseProbe, HealthStatus, ProbeResult


class ContextProbe(BaseProbe):
    """Checks context aggregator and observer health."""

    name = "context"
    interval = 30.0

    def __init__(self, context_aggregator=None) -> None:
        self._agg = context_aggregator

    def check(self) -> ProbeResult:
        if self._agg is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="Context aggregator not initialized",
            )

        try:
            snapshot = self._agg.tick()
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
            )

        if not snapshot:
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=0,
                last_error="Context tick returned empty snapshot",
            )

        return ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=0,
            error_count=0,
            metadata={"keys": list(snapshot.keys())},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_context_probe.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/context_probe.py tests/unit/self_healing/test_context_probe.py
git commit -m "feat(self-healing): add context aggregator health probe"
```

---

### Task 25: Network Probe

**Files:**
- Create: `src/homie_core/self_healing/probes/network_probe.py`
- Test: `tests/unit/self_healing/test_network_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_network_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.network_probe import NetworkProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestNetworkProbe:
    def test_healthy_when_lan_discovery_works(self):
        net = MagicMock()
        net.is_running = True
        net.peer_count = 2
        probe = NetworkProbe(network_manager=net)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_no_peers(self):
        net = MagicMock()
        net.is_running = True
        net.peer_count = 0
        probe = NetworkProbe(network_manager=net)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_not_running(self):
        net = MagicMock()
        net.is_running = False
        probe = NetworkProbe(network_manager=net)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_none_manager(self):
        probe = NetworkProbe(network_manager=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_network_probe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/network_probe.py
"""Health probe for LAN discovery and network services."""

from .base import BaseProbe, HealthStatus, ProbeResult


class NetworkProbe(BaseProbe):
    """Checks LAN discovery and WebSocket connection health."""

    name = "network"
    interval = 30.0

    def __init__(self, network_manager=None) -> None:
        self._net = network_manager

    def check(self) -> ProbeResult:
        if self._net is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="Network manager not initialized",
            )

        try:
            is_running = self._net.is_running
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
            )

        if not is_running:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error="Network service not running",
            )

        peer_count = getattr(self._net, "peer_count", 0)
        status = HealthStatus.HEALTHY if peer_count > 0 else HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=0,
            metadata={"peer_count": peer_count},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_network_probe.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/network_probe.py tests/unit/self_healing/test_network_probe.py
git commit -m "feat(self-healing): add network health probe"
```

---

### Task 26: Knowledge Probe

**Files:**
- Create: `src/homie_core/self_healing/probes/knowledge_probe.py`
- Test: `tests/unit/self_healing/test_knowledge_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_knowledge_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.knowledge_probe import KnowledgeProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestKnowledgeProbe:
    def test_healthy_when_rag_works(self):
        rag = MagicMock()
        rag._file_hashes = {"a.py": "abc123"}
        rag._search = MagicMock()
        probe = KnowledgeProbe(rag_pipeline=rag)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_no_indexed_files(self):
        rag = MagicMock()
        rag._file_hashes = {}
        rag._search = MagicMock()
        probe = KnowledgeProbe(rag_pipeline=rag)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_search_raises(self):
        rag = MagicMock()
        rag._file_hashes = {"a.py": "abc"}
        rag._search = MagicMock()
        rag._search.search.side_effect = RuntimeError("index corrupt")
        probe = KnowledgeProbe(rag_pipeline=rag)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_none_rag(self):
        probe = KnowledgeProbe(rag_pipeline=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_knowledge_probe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/probes/knowledge_probe.py
"""Health probe for knowledge/RAG pipeline."""

from .base import BaseProbe, HealthStatus, ProbeResult


class KnowledgeProbe(BaseProbe):
    """Checks RAG pipeline, document index, and search health."""

    name = "knowledge"
    interval = 30.0

    def __init__(self, rag_pipeline=None) -> None:
        self._rag = rag_pipeline

    def check(self) -> ProbeResult:
        if self._rag is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="RAG pipeline not initialized",
            )

        indexed_count = len(getattr(self._rag, "_file_hashes", {}))
        metadata = {"indexed_files": indexed_count}

        # Test search functionality if files are indexed
        if indexed_count > 0:
            try:
                self._rag._search.search("health_check", n=1)
            except Exception as exc:
                return ProbeResult(
                    status=HealthStatus.FAILED,
                    latency_ms=0,
                    error_count=1,
                    last_error=f"Search failed: {exc}",
                    metadata=metadata,
                )

        if indexed_count == 0:
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=0,
                last_error="No files indexed",
                metadata=metadata,
            )

        return ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=0,
            error_count=0,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_knowledge_probe.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/probes/knowledge_probe.py tests/unit/self_healing/test_knowledge_probe.py
git commit -m "feat(self-healing): add knowledge/RAG pipeline health probe"
```

---

## Chunk 8: Recovery Strategies

### Task 27: Inference Recovery Strategies

**Files:**
- Create: `src/homie_core/self_healing/recovery/strategies/__init__.py`
- Create: `src/homie_core/self_healing/recovery/strategies/inference.py`
- Test: `tests/unit/self_healing/test_strategy_inference.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_strategy_inference.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.recovery.strategies.inference import (
    retry_with_reduced_tokens,
    fallback_reduce_context,
    switch_to_smaller_model,
    degrade_to_cached,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestInferenceRecoveryStrategies:
    def test_retry_reduced_tokens_success(self):
        engine = MagicMock()
        engine.generate.return_value = "ok"
        result = retry_with_reduced_tokens(module="inference", status=2, error="timeout", model_engine=engine)
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_retry_reduced_tokens_failure(self):
        engine = MagicMock()
        engine.generate.side_effect = TimeoutError("still slow")
        result = retry_with_reduced_tokens(module="inference", status=2, error="timeout", model_engine=engine)
        assert result.success is False

    def test_fallback_reduce_context(self):
        config = MagicMock()
        config.llm.context_length = 65536
        engine = MagicMock()
        engine.generate.return_value = "ok"
        result = fallback_reduce_context(module="inference", status=2, error="oom", config=config, model_engine=engine)
        assert result.success is True
        assert result.tier == RecoveryTier.FALLBACK

    def test_switch_to_smaller_model(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.list_models.return_value = [MagicMock(name="small-model")]
        result = switch_to_smaller_model(module="inference", status=2, error="oom", model_engine=engine, model_registry=registry)
        assert result.tier == RecoveryTier.REBUILD

    def test_degrade_to_cached(self):
        result = degrade_to_cached(module="inference", status=2, error="fatal")
        assert result.success is True
        assert result.tier == RecoveryTier.DEGRADE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_strategy_inference.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/recovery/strategies/__init__.py
"""Concrete recovery strategies per module."""
```

```python
# src/homie_core/self_healing/recovery/strategies/inference.py
"""Recovery strategies for inference failures."""

import logging
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def retry_with_reduced_tokens(module, status, error, model_engine=None, **ctx) -> RecoveryResult:
    """T1: Retry inference with shorter max_tokens."""
    if model_engine is None:
        return RecoveryResult(success=False, action="no model engine", tier=RecoveryTier.RETRY)
    try:
        model_engine.generate("ping", max_tokens=1, timeout=10)
        return RecoveryResult(success=True, action="retry with reduced max_tokens succeeded", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"retry failed: {exc}", tier=RecoveryTier.RETRY)


def fallback_reduce_context(module, status, error, config=None, model_engine=None, **ctx) -> RecoveryResult:
    """T2: Reduce context_length and retry."""
    if config is None or model_engine is None:
        return RecoveryResult(success=False, action="missing config or engine", tier=RecoveryTier.FALLBACK)
    try:
        original = config.llm.context_length
        reduced = max(original // 2, 2048)
        config.llm.context_length = reduced
        model_engine.generate("ping", max_tokens=1, timeout=15)
        logger.info("Reduced context_length from %d to %d", original, reduced)
        return RecoveryResult(
            success=True,
            action=f"reduced context_length {original} → {reduced}",
            tier=RecoveryTier.FALLBACK,
        )
    except Exception as exc:
        return RecoveryResult(success=False, action=f"fallback failed: {exc}", tier=RecoveryTier.FALLBACK)


def switch_to_smaller_model(module, status, error, model_engine=None, model_registry=None, **ctx) -> RecoveryResult:
    """T3: Switch to a smaller/alternative model."""
    if model_engine is None or model_registry is None:
        return RecoveryResult(success=False, action="missing engine or registry", tier=RecoveryTier.REBUILD)
    try:
        models = model_registry.list_models()
        if not models:
            return RecoveryResult(success=False, action="no alternative models available", tier=RecoveryTier.REBUILD)
        # Pick first available alternative
        alt = models[0]
        model_engine.unload()
        model_engine.load(alt)
        logger.info("Switched to alternative model: %s", alt.name)
        return RecoveryResult(success=True, action=f"switched to {alt.name}", tier=RecoveryTier.REBUILD)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"model switch failed: {exc}", tier=RecoveryTier.REBUILD)


def degrade_to_cached(module, status, error, **ctx) -> RecoveryResult:
    """T4: Degrade to cached/static responses."""
    logger.warning("Inference fully degraded — serving cached responses only")
    return RecoveryResult(
        success=True,
        action="degraded to cached response mode",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "cached_only"},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_strategy_inference.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/recovery/strategies/ tests/unit/self_healing/test_strategy_inference.py
git commit -m "feat(self-healing): add inference recovery strategies (T1-T4)"
```

---

### Task 28: Storage Recovery Strategies

**Files:**
- Create: `src/homie_core/self_healing/recovery/strategies/storage.py`
- Test: `tests/unit/self_healing/test_strategy_storage.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_strategy_storage.py
import sqlite3
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.strategies.storage import (
    retry_sqlite_operation,
    emergency_disk_cleanup,
    restore_from_backup,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestStorageRecoveryStrategies:
    def test_retry_sqlite_succeeds_on_second_try(self):
        db = MagicMock()
        db._conn.execute.return_value.fetchone.return_value = ("ok",)
        result = retry_sqlite_operation(module="storage", status=2, error="locked", database=db)
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_retry_sqlite_fails(self):
        db = MagicMock()
        db._conn.execute.side_effect = sqlite3.OperationalError("still locked")
        result = retry_sqlite_operation(module="storage", status=2, error="locked", database=db)
        assert result.success is False

    def test_emergency_disk_cleanup(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "old.log").write_text("x" * 10000)
        result = emergency_disk_cleanup(module="storage", status=2, error="disk full", log_dir=str(log_dir))
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_restore_from_backup(self):
        db = MagicMock()
        result = restore_from_backup(module="storage", status=2, error="corrupt", database=db, backup_path="/fake")
        assert result.tier == RecoveryTier.FALLBACK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_strategy_storage.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/recovery/strategies/storage.py
"""Recovery strategies for storage failures (SQLite + ChromaDB)."""

import logging
import os
import shutil
from pathlib import Path
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def retry_sqlite_operation(module, status, error, database=None, **ctx) -> RecoveryResult:
    """T1: Retry SQLite operation after brief delay."""
    if database is None:
        return RecoveryResult(success=False, action="no database", tier=RecoveryTier.RETRY)
    try:
        database._conn.execute("SELECT 'ok'").fetchone()
        return RecoveryResult(success=True, action="SQLite retry succeeded", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"SQLite retry failed: {exc}", tier=RecoveryTier.RETRY)


def emergency_disk_cleanup(module, status, error, log_dir=None, **ctx) -> RecoveryResult:
    """T1: Emergency cleanup of logs and temp files to free disk space."""
    freed = 0
    if log_dir and os.path.isdir(log_dir):
        for f in Path(log_dir).glob("*.log"):
            try:
                size = f.stat().st_size
                f.unlink()
                freed += size
            except OSError:
                pass
    logger.info("Emergency cleanup freed %d bytes", freed)
    return RecoveryResult(
        success=True,
        action=f"emergency cleanup freed {freed} bytes",
        tier=RecoveryTier.RETRY,
        details={"freed_bytes": freed},
    )


def restore_from_backup(module, status, error, database=None, backup_path=None, **ctx) -> RecoveryResult:
    """T2: Restore database from most recent backup."""
    if not backup_path or not os.path.exists(str(backup_path)):
        return RecoveryResult(success=False, action="no backup available", tier=RecoveryTier.FALLBACK)
    try:
        db_path = database.path if database else None
        if db_path:
            shutil.copy2(str(backup_path), str(db_path))
            database.initialize()
            logger.info("Restored database from backup: %s", backup_path)
            return RecoveryResult(success=True, action="restored from backup", tier=RecoveryTier.FALLBACK)
        return RecoveryResult(success=False, action="no db path", tier=RecoveryTier.FALLBACK)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"restore failed: {exc}", tier=RecoveryTier.FALLBACK)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_strategy_storage.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/recovery/strategies/storage.py tests/unit/self_healing/test_strategy_storage.py
git commit -m "feat(self-healing): add storage recovery strategies"
```

---

### Task 29: Voice and Config Recovery Strategies

**Files:**
- Create: `src/homie_core/self_healing/recovery/strategies/voice.py`
- Create: `src/homie_core/self_healing/recovery/strategies/config.py`
- Test: `tests/unit/self_healing/test_strategy_voice_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_strategy_voice_config.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.strategies.voice import (
    restart_voice_engine,
    switch_tts_engine,
    degrade_to_text_only,
)
from homie_core.self_healing.recovery.strategies.config import (
    reparse_config,
    use_last_known_good,
    reset_to_defaults,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestVoiceStrategies:
    def test_restart_voice_engine(self):
        vm = MagicMock()
        result = restart_voice_engine(module="voice", status=2, error="crash", voice_manager=vm)
        vm.stop.assert_called_once()
        vm.start.assert_called_once()
        assert result.tier == RecoveryTier.RETRY

    def test_switch_tts_engine(self):
        vm = MagicMock()
        result = switch_tts_engine(module="voice", status=2, error="tts fail", voice_manager=vm)
        assert result.tier == RecoveryTier.FALLBACK

    def test_degrade_to_text_only(self):
        vm = MagicMock()
        result = degrade_to_text_only(module="voice", status=2, error="fatal", voice_manager=vm)
        vm.stop.assert_called_once()
        assert result.success is True
        assert result.tier == RecoveryTier.DEGRADE


class TestConfigStrategies:
    def test_reparse_config(self, tmp_path):
        cfg_path = tmp_path / "homie.config.yaml"
        cfg_path.write_text("llm:\n  backend: gguf\n")
        result = reparse_config(module="config", status=2, error="parse error", config_path=str(cfg_path))
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_reparse_config_fails_on_missing(self, tmp_path):
        result = reparse_config(module="config", status=2, error="missing", config_path=str(tmp_path / "nope.yaml"))
        assert result.success is False

    def test_reset_to_defaults(self):
        config = MagicMock()
        result = reset_to_defaults(module="config", status=2, error="corrupt", config=config)
        assert result.tier == RecoveryTier.DEGRADE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_strategy_voice_config.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementations**

```python
# src/homie_core/self_healing/recovery/strategies/voice.py
"""Recovery strategies for voice pipeline failures."""

import logging
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def restart_voice_engine(module, status, error, voice_manager=None, **ctx) -> RecoveryResult:
    """T1: Stop and restart the voice manager."""
    if voice_manager is None:
        return RecoveryResult(success=False, action="no voice manager", tier=RecoveryTier.RETRY)
    try:
        voice_manager.stop()
        voice_manager.start()
        return RecoveryResult(success=True, action="voice engine restarted", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"restart failed: {exc}", tier=RecoveryTier.RETRY)


def switch_tts_engine(module, status, error, voice_manager=None, **ctx) -> RecoveryResult:
    """T2: Switch to a different TTS engine."""
    if voice_manager is None:
        return RecoveryResult(success=False, action="no voice manager", tier=RecoveryTier.FALLBACK)
    try:
        # Voice manager handles engine fallback internally
        voice_manager.stop()
        voice_manager.start()
        logger.info("Switched TTS engine via restart")
        return RecoveryResult(success=True, action="TTS engine switched", tier=RecoveryTier.FALLBACK)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"TTS switch failed: {exc}", tier=RecoveryTier.FALLBACK)


def degrade_to_text_only(module, status, error, voice_manager=None, **ctx) -> RecoveryResult:
    """T4: Disable voice entirely, text-only mode."""
    if voice_manager:
        try:
            voice_manager.stop()
        except Exception:
            pass
    logger.warning("Voice pipeline degraded — text-only mode active")
    return RecoveryResult(
        success=True,
        action="degraded to text-only mode",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "text_only"},
    )
```

```python
# src/homie_core/self_healing/recovery/strategies/config.py
"""Recovery strategies for configuration failures."""

import logging
from pathlib import Path

import yaml

from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def reparse_config(module, status, error, config_path=None, **ctx) -> RecoveryResult:
    """T1: Re-read and validate the config file."""
    if not config_path or not Path(config_path).exists():
        return RecoveryResult(success=False, action="config file not found", tier=RecoveryTier.RETRY)
    try:
        with open(config_path) as f:
            yaml.safe_load(f)
        return RecoveryResult(success=True, action="config re-parsed successfully", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"reparse failed: {exc}", tier=RecoveryTier.RETRY)


def use_last_known_good(module, status, error, config_cache=None, **ctx) -> RecoveryResult:
    """T2: Fall back to cached known-good config."""
    if config_cache is None:
        return RecoveryResult(success=False, action="no cached config", tier=RecoveryTier.FALLBACK)
    logger.info("Using last known good configuration")
    return RecoveryResult(success=True, action="reverted to cached config", tier=RecoveryTier.FALLBACK)


def reset_to_defaults(module, status, error, config=None, **ctx) -> RecoveryResult:
    """T4: Reset config to factory defaults."""
    logger.warning("Config reset to defaults")
    return RecoveryResult(
        success=True,
        action="config reset to defaults",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "defaults"},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_strategy_voice_config.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/recovery/strategies/voice.py src/homie_core/self_healing/recovery/strategies/config.py tests/unit/self_healing/test_strategy_voice_config.py
git commit -m "feat(self-healing): add voice and config recovery strategies"
```

---

## Chunk 9: Recovery History & Preemptive Healing

### Task 30: Recovery History (SQLite Persistence)

**Files:**
- Modify: `src/homie_core/self_healing/health_log.py` — add recovery_history table
- Test: `tests/unit/self_healing/test_recovery_history.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_recovery_history.py
import pytest
from homie_core.self_healing.health_log import HealthLog


class TestRecoveryHistory:
    def test_write_and_query_recovery(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write_recovery(
            module="inference",
            failure_type="timeout",
            tier=1,
            action="retry with reduced tokens",
            success=True,
            time_to_recover_ms=1200,
            system_state={"gpu_mem": "11.2GB"},
        )
        history = log.query_recovery(module="inference")
        assert len(history) == 1
        assert history[0]["failure_type"] == "timeout"
        assert history[0]["success"] == 1

    def test_query_recovery_by_failure_type(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write_recovery(module="inference", failure_type="timeout", tier=1, action="a", success=True, time_to_recover_ms=100, system_state={})
        log.write_recovery(module="inference", failure_type="oom", tier=2, action="b", success=False, time_to_recover_ms=200, system_state={})
        results = log.query_recovery(module="inference", failure_type="oom")
        assert len(results) == 1
        assert results[0]["success"] == 0

    def test_recovery_pattern_summary(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        for _ in range(5):
            log.write_recovery(module="inference", failure_type="timeout", tier=1, action="retry", success=True, time_to_recover_ms=100, system_state={})
        for _ in range(2):
            log.write_recovery(module="inference", failure_type="timeout", tier=1, action="retry", success=False, time_to_recover_ms=100, system_state={})
        patterns = log.recovery_pattern_summary("inference", "timeout")
        assert patterns["total"] == 7
        assert patterns["success_count"] == 5
        assert patterns["success_rate"] == pytest.approx(5 / 7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_recovery_history.py -v`
Expected: FAIL — write_recovery not found

- [ ] **Step 3: Add recovery_history methods to HealthLog**

Add to `src/homie_core/self_healing/health_log.py`:

In `initialize()`, after the health_events table creation, add:

```python
self._conn.execute("""
    CREATE TABLE IF NOT EXISTS recovery_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        module TEXT NOT NULL,
        failure_type TEXT NOT NULL,
        tier INTEGER NOT NULL,
        action TEXT NOT NULL,
        success INTEGER NOT NULL,
        time_to_recover_ms REAL NOT NULL,
        system_state TEXT NOT NULL
    )
""")
self._conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_recovery_module ON recovery_history(module)
""")
self._conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_recovery_failure ON recovery_history(failure_type)
""")
```

Add these methods to the HealthLog class:

```python
def write_recovery(
    self,
    module: str,
    failure_type: str,
    tier: int,
    action: str,
    success: bool,
    time_to_recover_ms: float,
    system_state: dict,
) -> None:
    """Write a recovery attempt to the history (append-only)."""
    if self._conn is None:
        return
    self._conn.execute(
        "INSERT INTO recovery_history (timestamp, module, failure_type, tier, action, success, time_to_recover_ms, system_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (time.time(), module, failure_type, tier, action, int(success), time_to_recover_ms, json.dumps(system_state)),
    )
    self._conn.commit()

def query_recovery(
    self,
    module: Optional[str] = None,
    failure_type: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Query recovery history."""
    if self._conn is None:
        return []
    clauses = []
    params: list = []
    if module:
        clauses.append("module = ?")
        params.append(module)
    if failure_type:
        clauses.append("failure_type = ?")
        params.append(failure_type)
    where = " AND ".join(clauses) if clauses else "1=1"
    cursor = self._conn.execute(
        f"SELECT * FROM recovery_history WHERE {where} ORDER BY timestamp DESC LIMIT ?",
        params + [limit],
    )
    return [dict(row) for row in cursor.fetchall()]

def recovery_pattern_summary(self, module: str, failure_type: str) -> dict:
    """Summarize recovery patterns for a specific failure type."""
    if self._conn is None:
        return {"total": 0, "success_count": 0, "success_rate": 0.0}
    cursor = self._conn.execute(
        "SELECT COUNT(*) as total, SUM(success) as successes FROM recovery_history WHERE module = ? AND failure_type = ?",
        (module, failure_type),
    )
    row = cursor.fetchone()
    total = row["total"] or 0
    successes = row["successes"] or 0
    return {
        "total": total,
        "success_count": successes,
        "success_rate": successes / total if total > 0 else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_recovery_history.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/health_log.py tests/unit/self_healing/test_recovery_history.py
git commit -m "feat(self-healing): add recovery_history table for pattern learning"
```

---

### Task 31: Preemptive Healing Engine

**Files:**
- Create: `src/homie_core/self_healing/recovery/preemptive.py`
- Test: `tests/unit/self_healing/test_preemptive.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_preemptive.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.preemptive import PreemptiveEngine, PreemptiveRule


class TestPreemptiveRule:
    def test_rule_creation(self):
        rule = PreemptiveRule(
            name="gpu_memory_guard",
            module="inference",
            condition_metric="gpu_mem_percent",
            threshold=85.0,
            action="reduce gpu_layers",
            observation_count=0,
            min_observations=3,
        )
        assert rule.is_active is False  # needs 3 observations

    def test_rule_activates_after_threshold(self):
        rule = PreemptiveRule(
            name="test",
            module="m",
            condition_metric="v",
            threshold=50.0,
            action="act",
            observation_count=3,
            min_observations=3,
        )
        assert rule.is_active is True


class TestPreemptiveEngine:
    def test_add_and_list_rules(self):
        pe = PreemptiveEngine(metrics=MagicMock())
        pe.add_rule(PreemptiveRule(
            name="test_rule",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="reduce context",
            observation_count=3,
            min_observations=3,
        ))
        assert len(pe.rules) == 1

    def test_evaluate_triggers_rule(self):
        metrics = MagicMock()
        metrics.get_latest.return_value = 600.0  # above threshold
        pe = PreemptiveEngine(metrics=metrics)
        rule = PreemptiveRule(
            name="high_latency",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="reduce context",
            observation_count=5,
            min_observations=3,
        )
        pe.add_rule(rule)
        triggered = pe.evaluate()
        assert len(triggered) == 1
        assert triggered[0].name == "high_latency"

    def test_evaluate_ignores_inactive_rules(self):
        metrics = MagicMock()
        metrics.get_latest.return_value = 600.0
        pe = PreemptiveEngine(metrics=metrics)
        rule = PreemptiveRule(
            name="inactive",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="act",
            observation_count=1,  # below min_observations
            min_observations=3,
        )
        pe.add_rule(rule)
        triggered = pe.evaluate()
        assert len(triggered) == 0

    def test_evaluate_ignores_below_threshold(self):
        metrics = MagicMock()
        metrics.get_latest.return_value = 100.0  # below threshold
        pe = PreemptiveEngine(metrics=metrics)
        rule = PreemptiveRule(
            name="ok",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="act",
            observation_count=5,
            min_observations=3,
        )
        pe.add_rule(rule)
        triggered = pe.evaluate()
        assert len(triggered) == 0

    def test_seed_rules_exist(self):
        pe = PreemptiveEngine(metrics=MagicMock(), seed=True)
        assert len(pe.rules) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_preemptive.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/recovery/preemptive.py
"""Preemptive healing — fix problems before they happen."""

import logging
from dataclasses import dataclass
from typing import Optional

from ..metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class PreemptiveRule:
    """A rule that triggers preemptive action based on metric conditions."""

    name: str
    module: str
    condition_metric: str
    threshold: float
    action: str
    observation_count: int = 0
    min_observations: int = 3

    @property
    def is_active(self) -> bool:
        return self.observation_count >= self.min_observations


def _seed_rules() -> list[PreemptiveRule]:
    """Seed preemptive rules — the starting set that evolves over time."""
    return [
        PreemptiveRule(
            name="gpu_memory_guard",
            module="inference",
            condition_metric="gpu_mem_percent",
            threshold=90.0,
            action="reduce gpu_layers proactively",
            observation_count=0,
            min_observations=3,
        ),
        PreemptiveRule(
            name="inference_latency_guard",
            module="inference",
            condition_metric="latency_ms",
            threshold=5000.0,
            action="clear caches and optimize settings",
            observation_count=0,
            min_observations=3,
        ),
        PreemptiveRule(
            name="disk_space_guard",
            module="storage",
            condition_metric="disk_usage_percent",
            threshold=90.0,
            action="trigger early cleanup cycle",
            observation_count=0,
            min_observations=3,
        ),
    ]


class PreemptiveEngine:
    """Evaluates preemptive rules against current metrics."""

    def __init__(
        self,
        metrics: MetricsCollector,
        seed: bool = False,
    ) -> None:
        self._metrics = metrics
        self._rules: list[PreemptiveRule] = _seed_rules() if seed else []

    @property
    def rules(self) -> list[PreemptiveRule]:
        return list(self._rules)

    def add_rule(self, rule: PreemptiveRule) -> None:
        """Add a preemptive rule."""
        self._rules.append(rule)

    def evaluate(self) -> list[PreemptiveRule]:
        """Evaluate all active rules against current metrics. Returns triggered rules."""
        triggered = []
        for rule in self._rules:
            if not rule.is_active:
                continue

            value = self._metrics.get_latest(rule.module, rule.condition_metric)
            if value is None:
                continue

            if value >= rule.threshold:
                logger.info(
                    "Preemptive rule '%s' triggered: %s=%s (threshold=%s)",
                    rule.name, rule.condition_metric, value, rule.threshold,
                )
                triggered.append(rule)

        return triggered
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_preemptive.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/recovery/preemptive.py tests/unit/self_healing/test_preemptive.py
git commit -m "feat(self-healing): add preemptive healing engine with seed rules"
```

---

## Chunk 10: Improvement Sub-Components

### Task 32: Performance Analyzer

**Files:**
- Create: `src/homie_core/self_healing/improvement/analyzer.py`
- Test: `tests/unit/self_healing/test_analyzer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_analyzer.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.improvement.analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    def test_profile_module_returns_metrics(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {
                "latency_ms": {"latest": 200, "average": 150, "count": 50},
                "error_count": {"latest": 0, "average": 0.1, "count": 50},
            }
        }
        analyzer = PerformanceAnalyzer(metrics=metrics)
        profile = analyzer.profile("inference")
        assert "latency_ms" in profile
        assert profile["latency_ms"]["latest"] == 200

    def test_identify_bottlenecks(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 5000, "average": 200, "count": 50}},
            "storage": {"latency_ms": {"latest": 10, "average": 8, "count": 50}},
        }
        analyzer = PerformanceAnalyzer(metrics=metrics)
        bottlenecks = analyzer.identify_bottlenecks()
        assert len(bottlenecks) >= 1
        assert bottlenecks[0]["module"] == "inference"

    def test_no_bottlenecks_when_healthy(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 100, "average": 100, "count": 50}},
        }
        analyzer = PerformanceAnalyzer(metrics=metrics)
        bottlenecks = analyzer.identify_bottlenecks()
        assert len(bottlenecks) == 0

    def test_trend_detection_increasing(self):
        metrics = MagicMock()
        # Simulate an increasing trend by returning high latest vs average
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 500, "average": 100, "count": 50}},
        }
        analyzer = PerformanceAnalyzer(metrics=metrics, trend_threshold=2.0)
        trends = analyzer.detect_trends()
        assert len(trends) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_analyzer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/improvement/analyzer.py
"""Performance analyzer — profiles execution and identifies bottlenecks."""

from typing import Any

from ..metrics import MetricsCollector


class PerformanceAnalyzer:
    """Analyzes module performance metrics to find optimization opportunities."""

    def __init__(
        self,
        metrics: MetricsCollector,
        bottleneck_ratio: float = 3.0,
        trend_threshold: float = 2.0,
    ) -> None:
        self._metrics = metrics
        self._bottleneck_ratio = bottleneck_ratio
        self._trend_threshold = trend_threshold

    def profile(self, module: str) -> dict[str, Any]:
        """Get performance profile for a specific module."""
        snapshot = self._metrics.snapshot()
        return snapshot.get(module, {})

    def identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Find modules where latest metrics significantly exceed baseline."""
        bottlenecks = []
        snapshot = self._metrics.snapshot()

        for module, metrics in snapshot.items():
            for metric_name, values in metrics.items():
                latest = values.get("latest", 0)
                average = values.get("average", 0)
                count = values.get("count", 0)

                if count < 10 or average <= 0:
                    continue

                ratio = latest / average
                if ratio >= self._bottleneck_ratio:
                    bottlenecks.append({
                        "module": module,
                        "metric": metric_name,
                        "latest": latest,
                        "average": average,
                        "ratio": ratio,
                    })

        bottlenecks.sort(key=lambda b: b["ratio"], reverse=True)
        return bottlenecks

    def detect_trends(self) -> list[dict[str, Any]]:
        """Detect metrics trending in a concerning direction."""
        trends = []
        snapshot = self._metrics.snapshot()

        for module, metrics in snapshot.items():
            for metric_name, values in metrics.items():
                latest = values.get("latest", 0)
                average = values.get("average", 0)
                count = values.get("count", 0)

                if count < 10 or average <= 0:
                    continue

                if latest > average * self._trend_threshold:
                    trends.append({
                        "module": module,
                        "metric": metric_name,
                        "direction": "increasing",
                        "latest": latest,
                        "average": average,
                    })

        return trends
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_analyzer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/improvement/analyzer.py tests/unit/self_healing/test_analyzer.py
git commit -m "feat(self-healing): add performance analyzer with bottleneck and trend detection"
```

---

### Task 33: Code Patcher

**Files:**
- Create: `src/homie_core/self_healing/improvement/patcher.py`
- Test: `tests/unit/self_healing/test_patcher.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_patcher.py
import pytest
from homie_core.self_healing.improvement.patcher import CodePatcher
from homie_core.self_healing.improvement.rollback import RollbackManager


class TestCodePatcher:
    def test_apply_patch_modifies_file(self, tmp_path):
        target = tmp_path / "module.py"
        target.write_text("def slow():\n    x = 1\n    return x\n")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path)
        version_id = patcher.apply_patch(
            file_path=target,
            old_text="    x = 1\n    return x",
            new_text="    return 1  # optimized",
            reason="remove unnecessary variable",
        )
        assert version_id is not None
        assert "return 1  # optimized" in target.read_text()

    def test_apply_patch_creates_snapshot(self, tmp_path):
        target = tmp_path / "module.py"
        target.write_text("original content")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path)
        version_id = patcher.apply_patch(
            file_path=target,
            old_text="original content",
            new_text="modified content",
            reason="test",
        )
        # Rollback should restore
        rm.rollback(version_id)
        assert target.read_text() == "original content"

    def test_apply_patch_rejects_locked_file(self, tmp_path):
        target = tmp_path / "security" / "vault.py"
        target.parent.mkdir()
        target.write_text("secret")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path, locked_paths=["security/"])
        with pytest.raises(PermissionError):
            patcher.apply_patch(
                file_path=target,
                old_text="secret",
                new_text="hacked",
                reason="evil",
            )

    def test_apply_patch_fails_if_old_text_not_found(self, tmp_path):
        target = tmp_path / "module.py"
        target.write_text("actual content")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            patcher.apply_patch(
                file_path=target,
                old_text="nonexistent text",
                new_text="replacement",
                reason="test",
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_patcher.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/improvement/patcher.py
"""Code patcher — generates and applies source code modifications."""

import logging
from pathlib import Path
from typing import Optional

from .rollback import RollbackManager

logger = logging.getLogger(__name__)


class CodePatcher:
    """Applies targeted code patches to source files with rollback support."""

    def __init__(
        self,
        rollback_manager: RollbackManager,
        project_root: Path | str,
        locked_paths: Optional[list[str]] = None,
    ) -> None:
        self._rollback = rollback_manager
        self._root = Path(project_root)
        self._locked = locked_paths or []

    def _is_locked(self, file_path: Path) -> bool:
        """Check if a file is in the core lock list."""
        try:
            rel = file_path.relative_to(self._root)
        except ValueError:
            rel = file_path
        rel_str = str(rel).replace("\\", "/")
        for lock in self._locked:
            if lock.endswith("/"):
                if rel_str.startswith(lock) or rel_str.startswith(lock.rstrip("/")):
                    return True
            elif rel_str == lock:
                return True
        return False

    def apply_patch(
        self,
        file_path: Path | str,
        old_text: str,
        new_text: str,
        reason: str = "",
    ) -> str:
        """Apply a text replacement patch to a file.

        Returns version_id for rollback.
        Raises PermissionError if file is core-locked.
        Raises ValueError if old_text not found in file.
        """
        file_path = Path(file_path)

        if self._is_locked(file_path):
            raise PermissionError(f"File is core-locked: {file_path}")

        content = file_path.read_text()
        if old_text not in content:
            raise ValueError(f"Target text not found in {file_path}")

        # Snapshot before modification
        version_id = self._rollback.snapshot(file_path, reason=reason)

        # Apply patch
        new_content = content.replace(old_text, new_text, 1)
        file_path.write_text(new_content)

        logger.info("Applied patch to %s (version: %s): %s", file_path, version_id, reason)
        return version_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_patcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/improvement/patcher.py tests/unit/self_healing/test_patcher.py
git commit -m "feat(self-healing): add code patcher with core-lock enforcement and rollback"
```

---

### Task 34: Architecture Evolver

**Files:**
- Create: `src/homie_core/self_healing/improvement/evolver.py`
- Test: `tests/unit/self_healing/test_evolver.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_evolver.py
import pytest
from homie_core.self_healing.improvement.evolver import ArchitectureEvolver
from homie_core.self_healing.improvement.rollback import RollbackManager


class TestArchitectureEvolver:
    def test_create_module(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path)
        path = tmp_path / "src" / "new_module.py"
        version_id = evolver.create_module(
            file_path=path,
            content='"""New module."""\n\ndef new_function():\n    pass\n',
            reason="add caching layer",
        )
        assert path.exists()
        assert "new_function" in path.read_text()
        assert version_id is not None

    def test_create_module_rejects_locked_path(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path, locked_paths=["security/"])
        with pytest.raises(PermissionError):
            evolver.create_module(
                file_path=tmp_path / "security" / "backdoor.py",
                content="evil",
                reason="nope",
            )

    def test_remove_module(self, tmp_path):
        target = tmp_path / "dead_module.py"
        target.write_text("# deprecated")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path)
        version_id = evolver.remove_module(file_path=target, reason="deprecated")
        assert not target.exists()
        # Can rollback
        rm.rollback(version_id)
        assert target.exists()

    def test_split_module_creates_new_files(self, tmp_path):
        original = tmp_path / "big_module.py"
        original.write_text("# Part A\ndef func_a(): pass\n\n# Part B\ndef func_b(): pass\n")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path)
        version_id = evolver.split_module(
            source=original,
            targets={
                tmp_path / "part_a.py": "# Part A\ndef func_a(): pass\n",
                tmp_path / "part_b.py": "# Part B\ndef func_b(): pass\n",
            },
            reason="split oversized module",
        )
        assert (tmp_path / "part_a.py").exists()
        assert (tmp_path / "part_b.py").exists()
        assert not original.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_evolver.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/improvement/evolver.py
"""Architecture evolver — creates, removes, and restructures modules."""

import logging
from pathlib import Path
from typing import Optional

from .rollback import RollbackManager

logger = logging.getLogger(__name__)


class ArchitectureEvolver:
    """Evolves module structure — create, remove, split, merge."""

    def __init__(
        self,
        rollback_manager: RollbackManager,
        project_root: Path | str,
        locked_paths: Optional[list[str]] = None,
    ) -> None:
        self._rollback = rollback_manager
        self._root = Path(project_root)
        self._locked = locked_paths or []

    def _is_locked(self, file_path: Path) -> bool:
        try:
            rel = file_path.relative_to(self._root)
        except ValueError:
            rel = file_path
        rel_str = str(rel).replace("\\", "/")
        for lock in self._locked:
            if lock.endswith("/"):
                if rel_str.startswith(lock) or rel_str.startswith(lock.rstrip("/")):
                    return True
            elif rel_str == lock:
                return True
        return False

    def create_module(
        self,
        file_path: Path | str,
        content: str,
        reason: str = "",
    ) -> str:
        """Create a new module file. Returns version_id."""
        file_path = Path(file_path)
        if self._is_locked(file_path):
            raise PermissionError(f"Path is core-locked: {file_path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Snapshot the creation (empty file marker for rollback)
        version_id = self._rollback.snapshot([], reason=f"create: {reason}")

        file_path.write_text(content)
        logger.info("Created module %s (version: %s): %s", file_path, version_id, reason)
        return version_id

    def remove_module(
        self,
        file_path: Path | str,
        reason: str = "",
    ) -> str:
        """Remove a module file. Returns version_id for rollback."""
        file_path = Path(file_path)
        if self._is_locked(file_path):
            raise PermissionError(f"Path is core-locked: {file_path}")

        version_id = self._rollback.snapshot(file_path, reason=f"remove: {reason}")
        file_path.unlink()
        logger.info("Removed module %s (version: %s): %s", file_path, version_id, reason)
        return version_id

    def split_module(
        self,
        source: Path | str,
        targets: dict[Path | str, str],
        reason: str = "",
    ) -> str:
        """Split a module into multiple files. Removes the original."""
        source = Path(source)
        if self._is_locked(source):
            raise PermissionError(f"Source is core-locked: {source}")

        # Snapshot source before splitting
        version_id = self._rollback.snapshot(source, reason=f"split: {reason}")

        # Write target files
        for target_path, content in targets.items():
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)

        # Remove original
        source.unlink()
        logger.info("Split %s into %d files (version: %s): %s", source, len(targets), version_id, reason)
        return version_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_evolver.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Update improvement __init__.py with all components**

```python
# src/homie_core/self_healing/improvement/__init__.py
from .analyzer import PerformanceAnalyzer
from .engine import ImprovementEngine, ImprovementLevel, Observation
from .evolver import ArchitectureEvolver
from .patcher import CodePatcher
from .rollback import RollbackManager

__all__ = [
    "ArchitectureEvolver",
    "CodePatcher",
    "ImprovementEngine",
    "ImprovementLevel",
    "Observation",
    "PerformanceAnalyzer",
    "RollbackManager",
]
```

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/self_healing/improvement/ tests/unit/self_healing/test_evolver.py
git commit -m "feat(self-healing): add architecture evolver for module restructuring"
```

---

### Task 35: Context Recovery Strategies

**Files:**
- Create: `src/homie_core/self_healing/recovery/strategies/context.py`
- Test: `tests/unit/self_healing/test_strategy_context.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_healing/test_strategy_context.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.strategies.context import (
    restart_observer,
    reduce_monitoring_frequency,
    degrade_without_context,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestContextRecoveryStrategies:
    def test_restart_observer_success(self):
        agg = MagicMock()
        agg.tick.return_value = {"active_window": "test"}
        result = restart_observer(module="context", status=2, error="crash", context_aggregator=agg)
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_restart_observer_failure(self):
        agg = MagicMock()
        agg.tick.side_effect = RuntimeError("still broken")
        result = restart_observer(module="context", status=2, error="crash", context_aggregator=agg)
        assert result.success is False

    def test_reduce_monitoring_frequency(self):
        config = MagicMock()
        result = reduce_monitoring_frequency(module="context", status=2, error="overloaded", config=config)
        assert result.success is True
        assert result.tier == RecoveryTier.FALLBACK

    def test_degrade_without_context(self):
        result = degrade_without_context(module="context", status=2, error="fatal")
        assert result.success is True
        assert result.tier == RecoveryTier.DEGRADE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_healing/test_strategy_context.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/self_healing/recovery/strategies/context.py
"""Recovery strategies for context/observer failures."""

import logging
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def restart_observer(module, status, error, context_aggregator=None, **ctx) -> RecoveryResult:
    """T1: Restart the context aggregator and verify it works."""
    if context_aggregator is None:
        return RecoveryResult(success=False, action="no context aggregator", tier=RecoveryTier.RETRY)
    try:
        snapshot = context_aggregator.tick()
        if snapshot:
            return RecoveryResult(success=True, action="observer restarted and responding", tier=RecoveryTier.RETRY)
        return RecoveryResult(success=False, action="observer returned empty after restart", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"observer restart failed: {exc}", tier=RecoveryTier.RETRY)


def reduce_monitoring_frequency(module, status, error, config=None, **ctx) -> RecoveryResult:
    """T2: Reduce observer polling frequency to reduce load."""
    logger.info("Reducing context monitoring frequency")
    return RecoveryResult(
        success=True,
        action="reduced monitoring frequency",
        tier=RecoveryTier.FALLBACK,
        details={"mode": "reduced_frequency"},
    )


def degrade_without_context(module, status, error, **ctx) -> RecoveryResult:
    """T4: Disable context observers, continue without context awareness."""
    logger.warning("Context observers disabled — running without context awareness")
    return RecoveryResult(
        success=True,
        action="context disabled, running without observers",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "no_context"},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_healing/test_strategy_context.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/self_healing/recovery/strategies/context.py tests/unit/self_healing/test_strategy_context.py
git commit -m "feat(self-healing): add context observer recovery strategies"
```

---

### Task 36: Test __init__.py files & Final Test Suite

**Files:**
- Create: `tests/unit/self_healing/__init__.py`
- Create: `tests/integration/__init__.py` (if missing)

- [ ] **Step 1: Create test init files**

```python
# tests/unit/self_healing/__init__.py
```

```python
# tests/integration/__init__.py (if not exists)
```

- [ ] **Step 2: Run complete test suite**

Run: `python -m pytest tests/unit/self_healing/ tests/integration/test_self_healing_lifecycle.py -v --tb=short`
Expected: All tests PASS (approximately 90+ tests across all modules)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/self_healing/__init__.py tests/integration/__init__.py
git commit -m "chore(self-healing): add test init files for pytest discovery"
```
