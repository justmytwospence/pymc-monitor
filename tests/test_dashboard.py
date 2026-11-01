"""Dashboard multi-run lifecycle, cloud completion and real HTTP security boundaries."""
import base64
import hashlib
import http.cookiejar
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pytest

from pymc_monitor import RunControl
from pymc_monitor.dashboard import DashboardState, dashboard_csp, render_dashboard_html
from pymc_monitor.events import JobPhase, PhaseUpdate, RunFinished
from pymc_monitor.server import Dashboard


def get(url, **headers):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=4)


def test_cloud_completion_and_concurrent_snapshots():
    state = DashboardState(completion_phase="downloading")
    def update():
        for i in range(100):
            state.update(PhaseUpdate(JobPhase.SAMPLING, "done", str(i), i))
            state.update(PhaseUpdate(JobPhase.DOWNLOADING, "in_progress", str(i), i))
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(update)
        for _ in range(100):
            assert not state.snapshot()["complete"]
        future.result()
    assert not state.snapshot()["complete"]
    state.update(PhaseUpdate(JobPhase.DOWNLOADING, "done", "finished", 1))
    assert state.snapshot()["complete"]
    state.update(RunFinished("met"))
    assert state.snapshot()["status"] == "met"
    pending = DashboardState(completion_phase="downloading")
    pending.update(RunFinished("met"))
    assert pending.snapshot()["status"] == "met" and not pending.snapshot()["complete"]
    pending.update(PhaseUpdate(JobPhase.DOWNLOADING, "done", "ready", 1))
    assert pending.snapshot()["complete"] and pending.snapshot()["status"] == "met"
    failed = DashboardState(completion_phase="downloading")
    failed.update(RunFinished("error"))
    assert failed.snapshot()["complete"] and failed.snapshot()["status"] == "error"


def test_embedded_page_csp_and_url_validation():
    hostile = '\"><script>window.pwned=1</script><img src=x>'
    html = render_dashboard_html(progress_url="https://api.example/progress?x=1",
                                 stop_url="https://stop.example/stop", token=hostile,
                                 title=hostile)
    assert '<script>window.pwned=1</script>' not in html
    assert html.count("<script>") == 1 and "<script src=" not in html
    assert "<link" not in html and "import " not in html.split("<script>")[1][:300]
    match = re.search(r'data-config="([^"]*)"', html)
    assert match and "&quot;" in match.group(1) and "&lt;script&gt;" in match.group(1)
    csp = dashboard_csp(progress_url="https://api.example/progress?x=1",
                        stop_url="https://stop.example/stop")
    assert "connect-src 'self' https://api.example https://stop.example" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    for kind in ("script", "style"):
        source = re.search(rf"<{kind}>(.*?)</{kind}>", html, re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
        assert f"{kind}-src 'sha256-{digest}'" in csp
    for bad in ("https://user:pass@api.example/", "https://api.example:bad/",
                "javascript:alert(1)", "//evil.test/", "/ok\nInjected: yes",
                "https://api.example/\x00", "https://api.example/#fragment", "relative"):
        with pytest.raises(ValueError):
            dashboard_csp(progress_url=bad)
    assert "connect-src 'self'" in dashboard_csp(progress_url="/progress")


def test_overview_targeted_stop_only_affects_selected_run():
    server = Dashboard()
    first, second = DashboardState(), DashboardState()
    a, b = RunControl(), RunControl()
    aid = server.add_run(first, a, run_id="alpha")
    server.add_run(second, b, run_id="beta")
    url = server.overview_url()
    base = f"http://127.0.0.1:{urlsplit(url).port}"
    token = urlsplit(url).path.split("/")[2]
    try:
        req = urllib.request.Request(base + f"/stop?model={aid}", method="POST",
                                     headers={"Authorization": f"Bearer {token}"})
        assert urllib.request.urlopen(req).status == 204
        assert a.stopped and not b.stopped
        assert urllib.request.urlopen(urllib.request.Request(
            base + "/stop", method="POST",
            headers={"Authorization": f"Bearer {token}"})).status == 204
        assert b.stopped
    finally:
        server.close()


def test_multi_run_security_stop_status_and_retention():
    server = Dashboard(idle_timeout=0.2, title="Multi-run")
    a, b = DashboardState(), DashboardState()
    ca, cb = RunControl(), RunControl()
    aid, bid = server.add_run(a, ca), server.add_run(b, cb)
    url, other = server.url(aid), server.url(bid)
    assert urlsplit(url).port == urlsplit(other).port
    token = urlsplit(url).path.split("/")[2]
    base = f"http://127.0.0.1:{urlsplit(url).port}"
    auth = {"Authorization": f"Bearer {token}"}
    cookie_client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(base + "/assets/dashboard.js")
        assert exc.value.code == 403
        page = cookie_client.open(url)
        assert page.url == base + f"/r/{aid}/"
        assert b"Multi-run" in page.read()
        assert "script-src 'sha256-" in page.headers["Content-Security-Policy"]
        overview = cookie_client.open(server.overview_url())
        assert overview.url == base + "/overview/"
        assert b"Multi-run" in overview.read()
        manifest = json.load(cookie_client.open(base + "/progress"))
        assert [model["label"] for model in manifest["models"]] == [aid, bid]
        assert set(manifest["runs"]) == {aid, bid}
        assert cookie_client.open(base + f"/r/{aid}/").status == 200
        assert cookie_client.open(base + f"/runs/{aid}/state").status == 200
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(base + f"/r/{aid}/")
        assert exc.value.code == 403
        for path in (f"/runs/{aid}/state", "/assets/dashboard.js"):
            for headers in ({"Host": "evil.test", **auth},
                            {"Origin": "https://evil.test", **auth},
                            {"Authorization": "Bearer wrong"}):
                with pytest.raises(urllib.error.HTTPError) as exc:
                    get(base + path, **headers)
                assert exc.value.code == 403
        for path in ("/runs/missing/state", "/assets/missing.js"):
            with pytest.raises(urllib.error.HTTPError) as exc:
                get(base + path, **auth)
            assert exc.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as exc:
            cookie_client.open(urllib.request.Request(base + "/stop?model=missing", method="POST"))
        assert exc.value.code == 404
        stop = urllib.request.Request(base + f"/runs/{aid}/stop", headers=auth, method="POST")
        assert urllib.request.urlopen(stop).status == 204 and ca.stopped and not cb.stopped
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(stop)
        assert exc.value.code == 409
        assert cookie_client.open(urllib.request.Request(base + "/stop", method="POST")).status == 204
        assert cb.stopped
        with pytest.raises(urllib.error.HTTPError) as exc:
            cookie_client.open(urllib.request.Request(base + "/stop", method="POST"))
        assert exc.value.code == 409
        b.finish()
        stop_b = urllib.request.Request(base + f"/runs/{bid}/stop", headers=auth, method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(stop_b)
        assert exc.value.code == 409
        server.remove_run(aid)
        assert server.url(aid) is None and server.url(bid) == other
        server.remove_run(bid)
        deadline = time.monotonic() + 4
        while server._server is not None and time.monotonic() < deadline:
            time.sleep(.01)
        assert server._server is None
    finally:
        server.close()
