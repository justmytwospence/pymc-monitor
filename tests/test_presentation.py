"""The three presentation surfaces share honest diagnostics and safe text."""
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

from pymc_monitor import RunControl, display
from pymc_monitor.dashboard import DashboardService, DashboardState, dashboard_assets
from pymc_monitor.events import ConvergenceUpdate, JobPhase, ParamConvergence, PhaseUpdate
from pymc_monitor.rendering import convergence_html, number


@pytest.fixture
def convergence():
    return ConvergenceUpdate({"<img src=x onerror=alert(1)>": ParamConvergence(None, 10, 5)},
        draws=100, verdict={"decision": "cap", "hint": "<script>bad()</script>",
        "checks": {"bfmi": {"value": None, "threshold": .3, "ok": False}}}, final=True)


def test_convergence_html_escapes_all_fields(convergence):
    html = convergence_html(convergence)
    assert "<script>" not in html and "<img " not in html
    assert "&lt;script&gt;" in html and "unavailable" in html and "(fail)" in html
    assert number(float("inf")) == "unavailable"
    assert number("unusual") == "unusual"


def test_notebook_convergence_and_local_stop(monkeypatch, convergence):
    monkeypatch.setattr(display.NotebookDisplay, "_mount", lambda self: None)
    control = RunControl()
    widget = display.NotebookDisplay(control=control)
    widget.show_convergence(convergence)
    assert "Bulk ESS" in widget._widget.html and "&lt;img" in widget._widget.html
    assert widget._widget.stop_enabled
    widget._on_message(widget._widget, {"type": "other"}, [])
    assert not control.stopped
    widget._on_message(widget._widget, {"type": "stop"}, [])
    assert control.stopped and not widget._widget.stop_enabled
    widget.stop()


def test_terminal_convergence_treats_markup_as_text(convergence):
    terminal = display.TerminalDisplay()
    terminal.start()
    terminal.show_phase(PhaseUpdate(JobPhase.SAMPLING, "in_progress", "[bad]literal", 0))
    terminal.show_phase(PhaseUpdate(JobPhase.SAMPLING, "done", "[bad]literal", 1))
    terminal.show_phase(PhaseUpdate(JobPhase.SAMPLING, "error", "[bad]literal", 1))
    terminal.show_convergence(convergence)
    with terminal._console.capture() as capture:
        terminal._console.print(terminal._live.renderable)
    output = capture.get()
    assert "[bad]literal" in output
    assert "unavailable" in output and "cap" in output and "bfmi" in output
    terminal.stop()


def test_reducer_copies_and_sanitizes(convergence):
    reducer = DashboardState()
    convergence.params["bad"] = ParamConvergence(float("inf"), 0, 0)
    convergence.traces = {"x": [[float("nan"), 2]]}
    reducer.show_convergence(convergence)
    reducer.request_stop()
    snap = reducer.snapshot()
    assert snap["convergence"]["params"]["bad"]["rhat"] is None
    assert snap["traces"]["x"][0][0] is None
    snap["convergence"]["draws"] = 999
    assert reducer.snapshot()["convergence"]["draws"] == 100
    json.dumps(reducer.snapshot(), allow_nan=False)
    reducer.show_phase(PhaseUpdate(JobPhase.COMPILING, "done", "compiled", 1))
    assert not reducer.snapshot()["complete"]
    reducer.show_phase(PhaseUpdate(JobPhase.SAMPLING, "error", "failed", 1))
    assert reducer.snapshot()["complete"]


def test_http_assets_auth_origin_and_close():
    service = DashboardService(RunControl())
    service.start()
    url = urlsplit(service.url)
    base = f"http://{url.netloc}"
    headers = {"Authorization": f"Bearer {url.fragment}"}
    try:
        with pytest.raises(RuntimeError, match="already"):
            service.start()
        for path in ["/", "/dashboard.js", "/dashboard.css", "/renderer.js"]:
            with urllib.request.urlopen(base + path) as response:
                assert response.status == 200
                assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
                assert response.read()
        for method, path, extra, status in [
            ("GET", "/unknown", {}, 404), ("POST", "/unknown", {}, 404),
            ("GET", "/stop", {}, 404),
            ("GET", "/state", {"Host": "evil.test"}, 403),
            ("GET", "/state", {"Origin": "http://evil.test"}, 403),
            ("POST", "/stop", {"Authorization": "Bearer wrong"}, 403),
        ]:
            req = urllib.request.Request(base + path, method=method, headers={**headers, **extra})
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req)
            assert exc.value.code == status
        req = urllib.request.Request(base + "/stop", method="POST",
                                      headers={**headers, "Origin": base})
        assert urllib.request.urlopen(req).status == 204
        assert service.snapshot()["stop_requested"]
        assert "renderer.js" in dashboard_assets()
    finally:
        service.close()
    service.close()
    assert service.url is None


def test_oneshot_terminal_fallback(monkeypatch):
    monkeypatch.setattr(display, "_is_marimo", lambda: False)
    monkeypatch.setattr(display, "_is_notebook", lambda: False)
    calls = []
    display._emit_oneshot_html(["safe"], terminal_fallback=lambda: calls.append(True))
    assert calls == [True]
