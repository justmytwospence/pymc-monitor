"""Public handle ownership, timeouts, display fallback and observer failures."""
import threading
import time

import pymc as pm
import pytest

import pymc_monitor as monitor
from pymc_monitor import api, display
from pymc_monitor.events import JobPhase, PhaseUpdate, SamplingResult


@pytest.fixture
def fake_run(monkeypatch):
    release = threading.Event()
    closed = threading.Event()
    idata = object()
    def stream(model, kwargs, *, control, **options):
        try:
            yield PhaseUpdate(JobPhase.SAMPLING, "in_progress", "running", 0)
            while not release.is_set() and not control.stopped:
                time.sleep(.01)
            yield PhaseUpdate(JobPhase.SAMPLING, "done", "complete", 1)
            return SamplingResult(idata, {"status": "complete"})
        finally:
            closed.set()
    monkeypatch.setattr(api, "iter_sample", stream)
    with pm.Model() as model:
        pm.Normal("x")
    return model, release, closed, idata


def test_wait_timeout_stop_dashboard_and_repeated_close(fake_run):
    model, release, closed, idata = fake_run
    handle = monitor.start(model=model, progress=False, dashboard=True)
    assert handle.dashboard_url.startswith("http://127.0.0.1:")
    with pytest.raises(TimeoutError):
        handle.wait(.01)
    handle.stop()
    assert handle.wait(5) is idata
    assert handle.snapshot()["dashboard"]["complete"]
    assert handle.snapshot()["state"]["stop_requested"]
    assert closed.is_set()
    handle.close()
    handle.close()
    assert handle.dashboard_url is None


@pytest.mark.parametrize("notebook,missing", [(False, False), (True, False), (True, True)])
def test_display_selection_and_fallback(fake_run, monkeypatch, notebook, missing):
    model, release, closed, idata = fake_run
    monkeypatch.setattr(display, "_is_notebook", lambda: notebook)
    monkeypatch.setattr(display, "_is_marimo", lambda: False)
    if missing:
        def unavailable(*args, **kwargs):
            raise ImportError("anywidget not installed")
        monkeypatch.setattr(display, "NotebookDisplay", unavailable)
    else:
        monkeypatch.setattr(display.NotebookDisplay, "_mount", lambda self: None)
    release.set()
    with monitor.start(model=model, progress=True) as handle:
        assert handle.wait(5) is idata
    assert closed.is_set()


def test_display_cleanup_error_does_not_hang_handle(fake_run, monkeypatch):
    model, release, closed, _ = fake_run
    monkeypatch.setattr(display, "_is_notebook", lambda: False)
    monkeypatch.setattr(display, "_is_marimo", lambda: False)
    def failed_stop(self):
        self._live.stop()
        raise RuntimeError("display cleanup failed")
    monkeypatch.setattr(display.TerminalDisplay, "stop", failed_stop)
    release.set()
    handle = monitor.start(model=model, progress=True)
    try:
        with pytest.raises(RuntimeError, match="display cleanup"):
            handle.wait(5)
        assert handle.done and closed.is_set()
    finally:
        handle.close()


def test_callback_cannot_wait_on_own_thread(fake_run):
    model, release, closed, _ = fake_run
    ready = threading.Event()
    def callback(event):
        ready.wait(5)
        handle.wait()
    handle = monitor.start(model=model, progress=False, on_event=callback)
    ready.set()
    try:
        with pytest.raises(RuntimeError, match="event callback"):
            handle.wait(5)
        assert closed.is_set()
    finally:
        handle.close()


def test_normalizer_sampler_versions_and_absent_packages(fake_run, monkeypatch):
    model, *_ = fake_run
    def version(package):
        if package == "blackjax":
            raise api.PackageNotFoundError(package)
        return "test"
    monkeypatch.setattr(api, "version", version)
    normalized = monitor.sample.__pymc_cache_normalize__(model=model, nuts_sampler="blackjax")
    assert normalized["versions"]["blackjax"] is None
    assert normalized["versions"]["jax"] == "test"
    nutpie = monitor.sample.__pymc_cache_normalize__(model=model, nuts_sampler="nutpie")
    assert nutpie["versions"]["nutpie"] == "test"
