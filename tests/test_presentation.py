"""The three presentation surfaces share honest diagnostics and safe text."""
import http.cookiejar
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

from pymc_monitor import RunControl, display
from pymc_monitor.dashboard import (
    Dashboard,
    DashboardState,
    dashboard_assets,
    render_dashboard_html,
)
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
    state, control = DashboardState(), RunControl()
    service = Dashboard()
    run_id = service.add_run(state, control)
    url = urlsplit(service.url(run_id))
    base = f"http://{url.netloc}"
    token = url.path.split("/")[2]
    headers = {"Authorization": f"Bearer {token}"}
    try:
        assert service.start() is service
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        with opener.open(service.url(run_id)) as response:
            assert response.status == 200
            assert response.url == base + f"/r/{run_id}/"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            assert b"dashboard-config" in response.read()
        assert len(jar) == 1 and jar._cookies["127.0.0.1"]["/"]["pymc_monitor_token"].has_nonstandard_attr("HttpOnly")
        assert "uPlot" in dashboard_assets()["uPlot.iife.min.js"].decode()
        for method, path, extra, status in [
            ("GET", "/unknown", {}, 404), ("POST", "/unknown", {}, 404),
            ("GET", f"/runs/{run_id}/stop", {}, 405),
            ("GET", f"/runs/{run_id}/state", {"Host": "evil.test"}, 403),
            ("GET", f"/runs/{run_id}/state", {"Origin": "http://evil.test"}, 403),
            ("POST", f"/runs/{run_id}/stop", {"Authorization": "Bearer wrong"}, 403),
            ("GET", "/assets/dashboard.js", {"Authorization": "Bearer wrong"}, 403),
        ]:
            req = urllib.request.Request(base + path, method=method, headers={**headers, **extra})
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req)
            assert exc.value.code == status
        req = urllib.request.Request(base + f"/runs/{run_id}/stop", method="POST",
                                     headers={**headers, "Origin": base})
        assert urllib.request.urlopen(req).status == 204
        assert state.snapshot()["stop_requested"]
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 409
        assert "renderer.js" not in dashboard_assets()
    finally:
        service.close()
    service.close()
    assert service.url(run_id) is None


def test_shared_html_escapes_config_and_title():
    html = render_dashboard_html(progress_url='/progress?name=</script><img>',
                                 stop_url='/stop', token='"</script>',
                                 title='<script>bad</script>')
    assert '<img src=x>' not in html and '<script>bad</script>' not in html
    assert '&lt;/script&gt;' in html
    assert '&lt;script&gt;bad&lt;/script&gt;' in html


def test_oneshot_terminal_fallback(monkeypatch):
    monkeypatch.setattr(display, "_is_marimo", lambda: False)
    monkeypatch.setattr(display, "_is_notebook", lambda: False)
    calls = []
    display._emit_oneshot_html(["safe"], terminal_fallback=lambda: calls.append(True))
    assert calls == [True]
