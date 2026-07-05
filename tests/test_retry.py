"""Tests for retry with exponential backoff.

TDD: verifies retry behavior, backoff timing, and exception handling
without hitting real APIs.

Citations:
  - _instructions.md L626 (retry/backoff for rate limits, timeouts)
"""

from unittest.mock import patch, MagicMock

import pytest

from src.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Verify retry wrapper handles transient failures correctly."""

    def test_returns_result_on_first_success(self):
        """No retries needed when the function succeeds immediately."""
        func = MagicMock(return_value="ok")
        result = retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert func.call_count == 1

    def test_retries_on_transient_error(self):
        """Should retry and eventually succeed."""
        func = MagicMock(side_effect=[Exception("rate limit"), "ok"])
        with patch("src.retry.time.sleep"):  # don't actually sleep
            result = retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert func.call_count == 2

    def test_retries_up_to_max(self):
        """Should stop after max_retries and raise the last exception."""
        func = MagicMock(side_effect=Exception("always fails"))
        with patch("src.retry.time.sleep"):
            with pytest.raises(Exception, match="always fails"):
                retry_with_backoff(func, max_retries=3)
        assert func.call_count == 4  # initial + 3 retries

    def test_backoff_increases_exponentially(self):
        """Sleep durations should double each retry."""
        func = MagicMock(side_effect=[
            Exception("fail"),
            Exception("fail"),
            Exception("fail"),
            "ok",
        ])
        with patch("src.retry.time.sleep") as mock_sleep:
            retry_with_backoff(func, max_retries=3, base_delay=1.0)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # Should be approximately 1, 2, 4 (with jitter, so check order)
        assert len(delays) == 3
        assert delays[0] < delays[1] < delays[2]

    def test_passes_args_and_kwargs(self):
        """Should forward arguments to the wrapped function."""
        func = MagicMock(return_value="ok")
        retry_with_backoff(func, max_retries=1, args=("a", "b"), kwargs={"x": 1})
        func.assert_called_once_with("a", "b", x=1)

    def test_retries_only_on_specified_exceptions(self):
        """Non-retryable exceptions should raise immediately."""
        func = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            retry_with_backoff(
                func, max_retries=3,
                retryable_exceptions=(ConnectionError,),
            )
        assert func.call_count == 1  # no retries

    def test_default_retryable_exceptions(self):
        """By default, should retry on Exception (catch-all)."""
        func = MagicMock(side_effect=[RuntimeError("oops"), "ok"])
        with patch("src.retry.time.sleep"):
            result = retry_with_backoff(func, max_retries=3)
        assert result == "ok"
