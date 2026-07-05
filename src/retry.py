"""
Retry with exponential backoff for transient API failures.

Handles rate limits (429), server errors (500/503), and network timeouts.
Applied to OpenAI embedding and chat API calls.

Citations:
  - _instructions.md L626 (retry/backoff for rate limits, timeouts)
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """Call func with exponential backoff on failure.

    Args:
        func: The function to call.
        max_retries: Maximum number of retries after the initial attempt.
        base_delay: Initial delay in seconds (doubles each retry).
        retryable_exceptions: Exception types that trigger a retry.
            Non-matching exceptions raise immediately.
        args: Positional arguments to pass to func.
        kwargs: Keyword arguments to pass to func.

    Returns:
        The return value of func on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    if kwargs is None:
        kwargs = {}

    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt)
                jitter = random.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)
            else:
                raise
        except Exception:
            # Non-retryable exception — raise immediately
            raise
