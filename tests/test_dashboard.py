"""Dashboard multi-run lifecycle, cloud completion and real HTTP security boundaries."""
import http.cookiejar
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pytest

from pymc_monitor import RunControl
from pymc_monitor.dashboard import DashboardState
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
        assert b"Multi-run" in cookie_client.open(url).read()
        assert cookie_client.open(base + f"/runs/{aid}/state").status == 200
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
        stop = urllib.request.Request(base + f"/runs/{aid}/stop", headers=auth, method="POST")
        assert urllib.request.urlopen(stop).status == 204 and ca.stopped and not cb.stopped
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(stop)
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
