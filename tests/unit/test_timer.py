"""
Unit tests for timer.py
Tests the Timer class and utility functions.
"""
import pytest
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from timer import Timer, timed, fmt_seconds


class TestTimer:
    """Tests for the Timer class."""

    def test_timer_init(self):
        """Test Timer initialization."""
        t = Timer()
        assert t.total == 0.0
        assert t._start is None
        assert not t.running()

    def test_timer_start_stop(self):
        """Test basic start/stop functionality."""
        t = Timer()
        t.start()
        assert t.running()
        time.sleep(0.1)
        t.stop()
        assert not t.running()
        assert t.elapsed() >= 0.1

    def test_timer_multiple_start_stop(self):
        """Test accumulating time across multiple start/stop cycles."""
        t = Timer()
        t.start()
        time.sleep(0.05)
        t.stop()
        first_elapsed = t.elapsed()

        t.start()
        time.sleep(0.05)
        t.stop()
        second_elapsed = t.elapsed()

        assert second_elapsed > first_elapsed
        assert second_elapsed >= 0.1

    def test_timer_reset(self):
        """Test timer reset."""
        t = Timer()
        t.start()
        time.sleep(0.05)
        t.stop()
        assert t.elapsed() > 0

        t.reset()
        assert t.total == 0.0
        assert t._start is None
        assert t.elapsed() == 0.0

    def test_timer_elapsed_while_running(self):
        """Test elapsed() returns current time while running."""
        t = Timer()
        t.start()
        time.sleep(0.05)
        elapsed1 = t.elapsed()
        time.sleep(0.05)
        elapsed2 = t.elapsed()
        t.stop()

        assert elapsed2 > elapsed1
        assert t.running() is False

    def test_timer_context_manager(self):
        """Test Timer as context manager."""
        with Timer() as t:
            time.sleep(0.1)

        assert not t.running()
        assert t.elapsed() >= 0.1

    def test_timer_double_start(self):
        """Test that double start doesn't reset timer."""
        t = Timer()
        t.start()
        time.sleep(0.05)
        start_time = t._start
        t.start()  # Should be ignored
        assert t._start == start_time

    def test_timer_double_stop(self):
        """Test that double stop is safe."""
        t = Timer()
        t.start()
        time.sleep(0.05)
        t.stop()
        elapsed1 = t.elapsed()
        t.stop()  # Should be ignored
        elapsed2 = t.elapsed()
        assert elapsed1 == elapsed2


class TestFmtSeconds:
    """Tests for the fmt_seconds function."""

    def test_fmt_seconds_zero(self):
        """Test formatting zero seconds."""
        assert fmt_seconds(0) == "0:00:00.00"

    def test_fmt_seconds_seconds_only(self):
        """Test formatting seconds only."""
        assert fmt_seconds(30.5) == "0:00:30.50"

    def test_fmt_seconds_minutes(self):
        """Test formatting with minutes."""
        assert fmt_seconds(90.25) == "0:01:30.25"

    def test_fmt_seconds_hours(self):
        """Test formatting with hours."""
        assert fmt_seconds(3661.5) == "1:01:01.50"

    def test_fmt_seconds_large(self):
        """Test formatting large values."""
        assert fmt_seconds(7200) == "2:00:00.00"


class TestTimedDecorator:
    """Tests for the @timed decorator."""

    def test_timed_decorator_returns_value(self):
        """Test that timed decorator returns function result."""
        @timed
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_timed_decorator_preserves_name(self):
        """Test that timed decorator preserves function name."""
        @timed
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_timed_decorator_with_exception(self, capsys):
        """Test that timed decorator still prints time on exception."""
        @timed
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        captured = capsys.readouterr()
        assert "[timed] failing_function took" in captured.out
