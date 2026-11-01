"""Smoke test the optional dashboard in an isolated installed wheel or sdist."""
import http.cookiejar
import urllib.request
from urllib.parse import urlsplit

from pymc_monitor import Dashboard, DashboardState, RunControl, render_dashboard_html

state = DashboardState()
control = RunControl()
with Dashboard(idle_timeout=0) as dashboard:
    run_id = dashboard.add_run(state, control)
    url = dashboard.url(run_id)
    assert urlsplit(url).hostname == "127.0.0.1"
    browser = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with browser.open(url) as response:
        assert response.status == 200 and b"dashboard-config" in response.read()
    assert browser.open(f"http://{urlsplit(url).netloc}/r/{run_id}/").status == 200
    token = urlsplit(url).path.split("/")[2]
    base = f"http://{urlsplit(url).netloc}"
    req = urllib.request.Request(base + f"/runs/{run_id}/state",
                                 headers={"Authorization": f"Bearer {token}"})
    assert urllib.request.urlopen(req).status == 200
    assert "&lt;" in render_dashboard_html(progress_url="/state", stop_url="/stop",
                                               token="", title="<title>")
print("optional dashboard smoke OK")
