"""Retry decorator with exponential back-off and jitter.

Provides a ``@retry`` decorator that re-invokes a callable up to
``max_attempts`` times when it raises a specified exception.  Between
attempts the thread sleeps for ``base_delay * 2^attempt`` seconds plus
uniform random jitter.
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[F], F]:
    """Decorator: re-invoke the wrapped function on failure.

    Parameters
    ----------
    max_attempts : int
        Maximum number of calls (including the first).  Default 3.
    base_delay : float
        Base sleep time in seconds (doubles after each failure).  Default 1.0.
    max_delay : float
        Upper bound on sleep time.  Default 30.0.
    jitter : float
        Uniform random jitter range ``[0, jitter]`` added to each sleep.
        Default 0.5.
    exceptions : tuple[Type[Exception]]
        Exception types that trigger a retry.  Default ``(Exception,)``.
    on_retry : callable, optional
        Callback invoked before each retry with the exception and attempt
        number (1-indexed).  By default logs a warning.

    Usage::

        @retry(exceptions=(requests.RequestException,), max_attempts=3)
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts,
                            func.__name__,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay += random.uniform(0, jitter)

                    if on_retry is not None:
                        on_retry(exc, attempt)
                    else:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. "
                            "Retrying in %.1fs ...",
                            attempt,
                            max_attempts,
                            func.__name__,
                            exc,
                            delay,
                        )
                    time.sleep(delay)

            # Unreachable — the loop always returns or raises.
            raise RuntimeError("Unexpected: retry loop exited without returning.")  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    return decorator
