"""Cooperative run control, independent of any sampler backend."""
from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable


class SamplingStoppedError(RuntimeError):
    """Sampling stopped before a complete posterior draw could be recovered."""


class RunControl:
    """Thread-safe stop flag with a throttled external predicate.

    Predicate failures warn once and are retried; they never abort sampling or
    silently disable future stop checks. A local stop always takes precedence.
    """

    def __init__(self, stop_requested: Callable[[], bool] | None = None, *,
                 poll: Callable[[], bool] | None = None):
        if stop_requested is not None and poll is not None:
            raise TypeError("specify only one external stop predicate")
        predicate = stop_requested if stop_requested is not None else poll
        if predicate is not None and not callable(predicate):
            raise TypeError("stop_requested must be callable")
        self._flag = threading.Event()
        self._external_poll = predicate
        self._lock = threading.Lock()
        self._next_poll = 0.0
        self._warned = False

    def is_stop_requested(self) -> bool:
        if self._flag.is_set() or self._external_poll is None:
            return self._flag.is_set()
        with self._lock:
            if self._flag.is_set():
                return True
            now = time.monotonic()
            if now < self._next_poll:
                return False
            self._next_poll = now + 0.5
            try:
                if self._external_poll():
                    self._flag.set()
            except Exception as exc:
                if not self._warned:
                    warnings.warn(f"external stop polling failed: {exc}", RuntimeWarning,
                                  stacklevel=2)
                    self._warned = True
        return self._flag.is_set()

    def request_stop(self) -> None:
        self._flag.set()

    # Backward-compatible local and polling names.
    stop = request_stop
    poll = is_stop_requested

    @property
    def stopped(self) -> bool:
        return self._flag.is_set()
