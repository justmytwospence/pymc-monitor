"""Transport-neutral dashboard reducer and optional authenticated loopback server."""
from __future__ import annotations

import hmac
import json
import secrets
import socket
import threading
import time
from dataclasses import asdict
from html import escape
from importlib.resources import files

from .convergence import _jsonable
from .events import ConvergenceUpdate, PhaseUpdate, RunFinished, SamplingProgress


def dashboard_assets():
    """Return packaged assets; a cloud transport can serve these unchanged."""
    names = ("dashboard.html", "dashboard.js", "dashboard.css", "renderer.js")
    root = files("pymc_monitor").joinpath("static")
    return {name: root.joinpath(name).read_bytes() for name in names}


def render_dashboard_html(*, progress_url, stop_url, token, title):
    """Render shared UI with escaped, non-executable configuration for either transport."""
    config = json.dumps({"progress_url": progress_url, "stop_url": stop_url,
                         "token": token}, ensure_ascii=True).replace("<", "\\u003c")
    template = dashboard_assets()["dashboard.html"].decode("utf-8")
    return (template.replace("pymc-monitor</title>", f"{escape(title)}</title>")
            .replace("pymc-monitor</h1>", f"{escape(title)}</h1>")
            .replace("<!--CONFIG-->", f'<script id="dashboard-config" type="application/json">{config}</script>'))


class DashboardState:
    """Lock-protected reducer of local or cloud events; snapshots are JSON-safe copies."""

    def __init__(self, *, completion_phase="sampling"):
        self._lock = threading.Lock()
        self.completion_phase = completion_phase
        self._data = {"phases": [], "sampling": None, "complete": False,
                      "stop_requested": False, "status": "running"}

    def update(self, event):
        if isinstance(event, PhaseUpdate):
            self.show_phase(event)
        elif isinstance(event, SamplingProgress):
            self.show_sampling(event)
        elif isinstance(event, ConvergenceUpdate):
            self.show_convergence(event)
        elif isinstance(event, RunFinished):
            self.finish(event.status)
        else:
            raise TypeError(f"unsupported dashboard event: {type(event).__name__}")

    def show_phase(self, update: PhaseUpdate):
        with self._lock:
            phase = {"label": update.phase.value, "status": update.status,
                     "detail": update.message}
            self._data["phases"] = [p for p in self._data["phases"] if p["label"] != phase["label"]] + [phase]
            if update.status == "error":
                self._data.update(complete=True, status="error")
            elif update.phase.value == self.completion_phase and update.status == "done":
                self._data.update(complete=True, status="complete")

    def show_sampling(self, update: SamplingProgress):
        with self._lock:
            self._data["sampling"] = {"chains": {str(k): asdict(v) for k, v in update.chains.items()},
                                      "total_divergences": update.total_divergences,
                                      "elapsed": update.elapsed, "total_draws": update.total_draws}

    def show_convergence(self, update: ConvergenceUpdate):
        with self._lock:
            self._data["convergence"] = {"params": {k: asdict(v) for k, v in update.params.items()},
                                         "draws": update.draws,
                                         "verdict": {**(update.verdict or {}), "final": update.final}}
            if update.traces:
                self._data["traces"] = update.traces

    def request_stop(self):
        with self._lock:
            self._data["stop_requested"] = True

    def finish(self, status="complete"):
        with self._lock:
            self._data.update(complete=True, status=status)

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(_jsonable(self._data), allow_nan=False))


class Dashboard:
    """Reusable FastAPI/uvicorn loopback service; dependencies load only on start.

    Each dashboard has its own secret. A one-time token-bearing navigation sets
    an HttpOnly SameSite cookie for browser asset/API requests; non-browser
    clients may use a Bearer header. Every route requires authentication.
    """

    def __init__(self, *, port=0, title="pymc-monitor", idle_timeout=300):
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if idle_timeout < 0:
            raise ValueError("idle_timeout must be nonnegative")
        self.title = title
        self.idle_timeout = idle_timeout
        self._port = port
        self._token = secrets.token_urlsafe(32)
        self._runs = {}
        self._lock = threading.RLock()
        self._server = self._thread = self._socket = self._timer = None

    def start(self):
        with self._lock:
            if self._server is not None:
                return self
            # Imports are deliberately deferred so local sampling has no web dependency.
            import uvicorn
            from fastapi import FastAPI, Request
            from fastapi.responses import HTMLResponse, JSONResponse, Response

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", self._port))
                sock.listen(128)
                self._port = sock.getsockname()[1]
                app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
                assets = dashboard_assets()
                dashboard = self

                @app.middleware("http")
                async def secure(request: Request, call_next):
                    origin = f"http://127.0.0.1:{dashboard._port}"
                    host = f"127.0.0.1:{dashboard._port}"
                    if request.headers.get("host") != host or request.headers.get("origin", origin) != origin:
                        return Response(status_code=403)
                    path = request.url.path
                    initial = path.startswith("/r/") and path.endswith("/") and request.method == "GET"
                    token = (path.split("/")[2] if initial and len(path.split("/")) == 5 else None)
                    bearer = request.headers.get("authorization", "")
                    cookie = request.cookies.get("pymc_monitor_token", "")
                    if not any(hmac.compare_digest(candidate, dashboard._token) for candidate in
                               (token or "", bearer.removeprefix("Bearer ") if bearer.startswith("Bearer ") else "",
                                cookie)):
                        return Response(status_code=403)
                    response = await call_next(request)
                    response.headers["Cache-Control"] = "no-store"
                    response.headers["Referrer-Policy"] = "no-referrer"
                    response.headers["X-Content-Type-Options"] = "nosniff"
                    response.headers["Content-Security-Policy"] = (
                        "default-src 'none'; script-src 'self'; style-src 'self'; "
                        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                    if token is not None and hmac.compare_digest(token, dashboard._token):
                        response.set_cookie("pymc_monitor_token", dashboard._token,
                                            httponly=True, samesite="strict", path="/")
                    return response

                @app.get("/r/{token}/{run_id}/")
                async def index(token: str, run_id: str):
                    with dashboard._lock:
                        if run_id not in dashboard._runs:
                            return Response(status_code=404)
                    html = render_dashboard_html(progress_url=f"/runs/{run_id}/state",
                                                 stop_url=f"/runs/{run_id}/stop",
                                                 token="", title=dashboard.title)
                    return HTMLResponse(html)

                @app.get("/assets/{name}")
                async def asset(name: str):
                    if name not in assets or name == "dashboard.html":
                        return Response(status_code=404)
                    media = "text/css" if name.endswith(".css") else "text/javascript"
                    return Response(assets[name], media_type=media)

                @app.get("/runs/{run_id}/state")
                async def state(run_id: str):
                    with dashboard._lock:
                        run = dashboard._runs.get(run_id)
                    return JSONResponse(run[0].snapshot()) if run else Response(status_code=404)

                @app.post("/runs/{run_id}/stop")
                async def stop(run_id: str):
                    with dashboard._lock:
                        run = dashboard._runs.get(run_id)
                        if run is None:
                            return Response(status_code=404)
                        state, control = run
                        if state.snapshot()["complete"] or state.snapshot()["stop_requested"]:
                            return Response(status_code=409)
                        control.request_stop()
                        state.request_stop()
                    return Response(status_code=204)

                server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self._port,
                                                       log_level="error", lifespan="off"))
                thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]},
                                          daemon=True, name="pymc-monitor-dashboard")
                self._socket, self._server, self._thread = sock, server, thread
                thread.start()
                deadline = time.monotonic() + 5
                while not server.started and thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not server.started:
                    self._server = self._thread = self._socket = None
                    server.should_exit = True
                    thread.join(timeout=5)
                    raise RuntimeError("dashboard server failed to start")
            except BaseException:
                if self._server is None:
                    sock.close()
                raise
            return self

    def add_run(self, state: DashboardState, control, *, run_id=None):
        self.start()
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            run_id = run_id or secrets.token_urlsafe(12)
            if run_id in self._runs or not run_id.isascii() or not run_id.replace("-", "").replace("_", "").isalnum():
                raise ValueError("invalid or duplicate run id")
            self._runs[run_id] = (state, control)
            return run_id

    def remove_run(self, run_id):
        with self._lock:
            self._runs.pop(run_id, None)
            if not self._runs and self._server is not None:
                self._timer = threading.Timer(self.idle_timeout, self._close_if_idle)
                self._timer.daemon = True
                self._timer.start()

    def _close_if_idle(self):
        with self._lock:
            if not self._runs:
                self.close()

    def url(self, run_id):
        with self._lock:
            return (f"http://127.0.0.1:{self._port}/r/{self._token}/{run_id}/"
                    if self._server and run_id in self._runs else None)

    def close(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            server, thread, sock = self._server, self._thread, self._socket
            self._server = self._thread = self._socket = None
            self._runs.clear()
            if server:
                server.should_exit = True
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)
        if sock:
            sock.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
