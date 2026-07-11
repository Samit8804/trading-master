"""Tests for the retry decorator with exponential back-off."""

from unittest.mock import Mock, patch

import pytest

from crowdwisdom_quant.utils.retry import retry


class TestRetryDecorator:
    """Verify that @retry re-invokes on failure, respects max_attempts,
    and propagates the exception when all attempts are exhausted."""

    def test_success_first_attempt(self) -> None:
        call_count = 0

        def succeed() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        decorated = retry(max_attempts=3)(succeed)
        result = decorated()
        assert result == 42
        assert call_count == 1

    def test_success_on_second_attempt(self) -> None:
        call_count = 0

        def fail_once() -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first")
            return 99

        decorated = retry(max_attempts=3, base_delay=0.01, jitter=0)(fail_once)
        result = decorated()
        assert result == 99
        assert call_count == 2

    def test_all_attempts_fail(self) -> None:
        call_count = 0

        def always_fail() -> int:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        decorated = retry(max_attempts=3, base_delay=0.01, jitter=0)(always_fail)
        with pytest.raises(ValueError, match="always fails"):
            decorated()
        assert call_count == 3

    def test_only_catches_specified_exceptions(self) -> None:
        call_count = 0

        def type_error_func() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("not caught")

        decorated = retry(
            max_attempts=3,
            base_delay=0.01,
            exceptions=(ValueError,),
        )(type_error_func)
        with pytest.raises(TypeError):
            decorated()
        assert call_count == 1

    def test_on_retry_callback(self) -> None:
        callback = Mock()
        call_count = 0

        def fail_twice() -> int:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError(f"fail{call_count}")
            return 42

        decorated = retry(
            max_attempts=3,
            base_delay=0.01,
            jitter=0,
            on_retry=callback,
        )(fail_twice)
        result = decorated()
        assert result == 42
        assert callback.call_count == 2
        args1 = callback.call_args_list[0]
        assert isinstance(args1[0][0], ValueError)
        assert args1[0][1] == 1
        args2 = callback.call_args_list[1]
        assert args2[0][1] == 2

    def test_max_delay_cap(self) -> None:
        call_count = 0

        def fail_always() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with patch("time.sleep") as mock_sleep:
            decorated = retry(
                max_attempts=3,
                base_delay=1000.0,
                max_delay=5.0,
                jitter=0,
            )(fail_always)
            with pytest.raises(RuntimeError):
                decorated()
            for call_arg in mock_sleep.call_args_list:
                delay = call_arg[0][0]
                assert delay <= 5.0, f"Delay {delay} exceeded max_delay"
