"""Cooperative run control, independent of any sampler backend."""
from __future__ import annotations

import threading
from collections.abc import Callable


class SamplingStoppedError(RuntimeError):
    """Sampling stopped before a complete posterior draw could be recovered."""


class RunControl:
    """Cooperative stop flag with an optional external stop predicate.

    ``poll`` is called by the event loop, never by sampler callbacks. A true
    return value latches the stop flag. Poll failures stop the run and propagate;
    losing the control channel must not silently disable cancellation.
    """

    def __init__(self, poll: Callable[[], bool] | None = None):
        if poll is not None and not callable(poll):
            raise TypeError("poll must be callable")
        self._flag = threading.Event()
        self._external_poll = poll

    def poll(self) -> bool:
        if not self.stopped and self._external_poll is not None:
            try:
                if self._external_poll():
                    self.stop()
            except BaseException:
                self.stop()
                raise
        return self.stopped

    def stop(self) -> None:
        self._flag.set()

    @property
    def stopped(self) -> bool:
        return self._flag.is_set()
