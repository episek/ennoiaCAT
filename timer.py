# timer.py
from time import perf_counter
from functools import wraps

class Timer:
    def __init__(self):
        self._start = None
        self.total = 0.0  # accumulated seconds

    def start(self):
        if self._start is None:
            self._start = perf_counter()

    def stop(self):
        if self._start is not None:
            self.total += perf_counter() - self._start
            self._start = None

    def reset(self):
        self._start = None
        self.total = 0.0

    def running(self) -> bool:
        return self._start is not None

    def elapsed(self) -> float:
        if self._start is None:
            return self.total
        return self.total + (perf_counter() - self._start)

    # Context manager: with Timer() as t: ...  print(t.elapsed())
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

def timed(fn):
    """Decorator to print runtime of a function."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = perf_counter() - t0
            print(f"[timed] {fn.__name__} took {dt:.3f}s")
    return wrapper

def fmt_seconds(s: float) -> str:
    m, sec = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{sec:05.2f}"
