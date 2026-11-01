"""Synchronous and asynchronous public entry points."""
from __future__ import annotations

import threading
from importlib.metadata import PackageNotFoundError, version

from .control import RunControl
from .dashboard import DashboardState
from .engine import iter_sample
from .events import dispatch_event
from .request import resolve


class SamplingHandle:
    """Owns a background sampling run and its optional loopback dashboard."""

    def __init__(self, model, kwargs, *, until, initial_strategy, progress, dashboard, control,
                 on_event, nuts_sampler):
        self._control = control or RunControl()
        self._state = DashboardState()
        self._done = threading.Event()
        self._result = None
        self._error = None
        self._latest = None
        self._sinks = [self._state]
        self._dashboard = None
        self._dashboard_run_id = None
        self._owns_dashboard = False
        if progress:
            from .display import NotebookDisplay, TerminalDisplay, _is_marimo, _is_notebook
            if _is_notebook() or _is_marimo():
                try:
                    sink = NotebookDisplay(control=self._control)
                except ImportError:
                    sink = TerminalDisplay()
            else:
                sink = TerminalDisplay()
            self._sinks.append(sink)
            if isinstance(sink, TerminalDisplay):
                sink.start()
        if dashboard:
            from .dashboard import Dashboard
            if dashboard is True:
                self._dashboard = Dashboard()
                self._owns_dashboard = True
            elif isinstance(dashboard, Dashboard):
                self._dashboard = dashboard
            else:
                raise TypeError("dashboard must be a Dashboard instance or bool")
            self._dashboard_run_id = self._dashboard.add_run(self._state, self._control)

        def run():
            stream = iter_sample(model, kwargs, until=until, initial_strategy=initial_strategy,
                                 control=self._control, nuts_sampler=nuts_sampler)
            try:
                while True:
                    try:
                        event = next(stream)
                    except StopIteration as end:
                        self._result = end.value
                        break
                    self._latest = event
                    dispatch_event(event, self._sinks)
                    if on_event is not None:
                        on_event(event)
            except BaseException as exc:
                self._error = exc
            finally:
                try:
                    stream.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
                for sink in self._sinks:
                    try:
                        close = getattr(sink, "stop", None)
                        if callable(close):
                            close()
                    except Exception as exc:
                        if self._error is None:
                            self._error = exc
                self._state.finish("error" if self._error else
                                   self._result.report.get("status", "complete") if self._result else "error")
                self._done.set()

        self._thread = threading.Thread(target=run, daemon=True, name="pymc-monitor")
        self._thread.start()

    @property
    def done(self):
        return self._done.is_set()

    @property
    def dashboard_url(self):
        return self._dashboard.url(self._dashboard_run_id) if self._dashboard else None

    def stop(self):
        self._control.stop()
        if not self.done:
            self._state.request_stop()

    def wait(self, timeout=None):
        if threading.current_thread() is self._thread:
            raise RuntimeError("wait cannot be called from an event callback")
        if not self._done.wait(timeout):
            raise TimeoutError("sampling is still running")
        if self._error:
            raise self._error
        return self._result.idata

    def snapshot(self):
        return {"done": self.done, "event": self._latest,
                "report": self._result.report if self._result else None,
                "state": self._state.snapshot(),
                "dashboard": self._state.snapshot() if self._dashboard else None}

    def close(self):
        self.stop()
        if threading.current_thread() is not self._thread:
            self._thread.join()
        if self._dashboard:
            self._dashboard.remove_run(self._dashboard_run_id)
            if self._owns_dashboard:
                self._dashboard.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def start(*, model=None, until=None, initial_strategy=None, progress=True, dashboard=False,
          control=None, on_event=None, nuts_sampler=None, **sample_kwargs):
    """Begin sampling in a background thread and return a control handle."""
    if on_event is not None and not callable(on_event):
        raise TypeError("on_event must be callable")
    request = resolve(model, sample_kwargs, nuts_sampler=nuts_sampler, until=until,
                      initial_strategy=initial_strategy, control=control)
    kwargs = {**request.settings["extra"], "draws": request.settings["draws"],
              "chains": request.chains, "random_seed": request.settings["seed"]}
    if not request.tune_default:
        kwargs["tune"] = request.settings["tune"]
    return SamplingHandle(request.model, kwargs, until=request.target,
                          initial_strategy=request.initial_strategy,
                          progress=progress, dashboard=dashboard, control=request.control,
                          on_event=on_event, nuts_sampler=nuts_sampler)


def sample(*, model=None, until=None, initial_strategy=None, progress=True, dashboard=False,
           control=None, on_event=None, nuts_sampler=None, **sample_kwargs):
    """Sample locally, display diagnostics, and return PyMC inference data."""
    handle = start(model=model, until=until, initial_strategy=initial_strategy, progress=progress,
                   dashboard=dashboard, control=control, on_event=on_event,
                   nuts_sampler=nuts_sampler, **sample_kwargs)
    try:
        return handle.wait()
    finally:
        handle.close()


def _normalize_cache(model, kwargs):
    """Normalize a cache call made as ``hook(model, keyword_arguments)``."""
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary")
    return _normalize_cache_kwargs(model, **kwargs)


def _normalize_cache_kwargs(model, *, until=None, initial_strategy=None, progress=True,
                            dashboard=False, control=None, on_event=None, nuts_sampler=None,
                            **sample_kwargs):
    """Pure cache integration protocol: no dependency on or import of pymc-cache."""
    if control is not None or on_event is not None or sample_kwargs.get("callback") is not None:
        raise ValueError("live control and event callbacks are not cacheable")
    request = resolve(model, sample_kwargs, nuts_sampler=nuts_sampler, until=until,
                      initial_strategy=initial_strategy)
    sampler = request.sampler
    key_kwargs = {**request.settings["extra"], "draws": request.settings["draws"],
                  "tune": request.settings["tune"], "chains": request.chains,
                  "random_seed": request.settings["seed"], "until": request.target,
                  "initial_strategy": initial_strategy, "nuts_sampler": sampler}
    call_kwargs = dict(progress=progress, dashboard=dashboard, **key_kwargs)
    if nuts_sampler is None:
        call_kwargs["nuts_sampler"] = None
    if request.tune_default:
        call_kwargs.pop("tune")
    versions = {}
    packages = ["pymc-monitor", "pymc", "arviz", "pytensor", "numpy"]
    if sampler == "nutpie":
        packages.append("nutpie")
    elif sampler in ("numpyro", "blackjax"):
        packages.extend([sampler, "jax", "jaxlib"])
    for package in packages:
        if package is not None:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
    return {"call_kwargs": call_kwargs, "key_kwargs": key_kwargs,
            "sampler_identity": sampler, "versions": versions}


sample.__pymc_cache_normalize__ = _normalize_cache
