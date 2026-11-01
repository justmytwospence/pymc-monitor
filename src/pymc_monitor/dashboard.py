"""Transport-neutral dashboard reducer and optional authenticated loopback server."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import socket
import threading
import time
from dataclasses import asdict
from functools import lru_cache
from html import escape
from importlib.resources import files
from urllib.parse import urlsplit

from .convergence import _jsonable
from .events import ConvergenceUpdate, PhaseUpdate, RunFinished, SamplingProgress


def dashboard_assets():
    """Return the packaged sources used to build the self-contained page."""
    names = ("dashboard.html", "dashboard.js", "dashboard.css", "uPlot.min.css",
             "uPlot.iife.min.js", "uPlot.LICENSE")
    root = files("pymc_monitor").joinpath("static")
    return {name: root.joinpath(name).read_bytes() for name in names}


def _origin(url):
    """Validate an explicit HTTP endpoint; relative paths are local to the page."""
    if not isinstance(url, str) or not url or any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("dashboard URL must be a nonempty HTTP URL without control characters")
    if (any(c.isspace() for c in url) or "\\" in url
            or re.search(r"%(?![0-9a-fA-F]{2})", url)):
        raise ValueError("malformed dashboard URL")
    try:
        parts = urlsplit(url)
        if parts.fragment or parts.username is not None or parts.password is not None:
            raise ValueError("dashboard URL must not contain userinfo or a fragment")
        if parts.scheme:
            if parts.scheme not in {"http", "https"} or not parts.hostname or not parts.netloc:
                raise ValueError("dashboard URL must have an HTTP origin")
            port = parts.port  # validates malformed port numbers
            if any(c.isspace() for c in parts.netloc) or "@" in parts.netloc:
                raise ValueError("malformed dashboard origin")
            host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
            return f"{parts.scheme}://{host}{':' + str(port) if port is not None else ''}"
        if parts.netloc or not url.startswith("/") or url.startswith("//"):
            raise ValueError("dashboard URL must be absolute or root-relative")
        return None
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"invalid dashboard URL: {url!r}") from exc


@lru_cache(maxsize=1)
def _inline_assets():
    assets = dashboard_assets()
    style = (assets["uPlot.min.css"] + b"\n" + assets["dashboard.css"]).decode("utf-8")
    script = (assets["uPlot.iife.min.js"] + b"\n" + assets["dashboard.js"]).decode("utf-8")
    return style, script


def render_dashboard_html(*, progress_url, stop_url, token, title="pymc-monitor"):
    """Render a fully self-contained, transport-independent dashboard page."""
    _origin(progress_url)
    if stop_url is not None:
        _origin(stop_url)
    config = json.dumps({"progress_url": progress_url, "stop_url": stop_url,
                         "token": token}, ensure_ascii=True)
    template = dashboard_assets()["dashboard.html"].decode("utf-8")
    style, script = _inline_assets()
    return (template.replace("<!--TITLE-->", escape(title))
            .replace("<!--HEADING-->", escape(title))
            .replace("<!--CONFIG-->", escape(config, quote=True))
            .replace("<!--STYLE-->", style).replace("<!--SCRIPT-->", script))


def dashboard_csp(*, progress_url, stop_url=None):
    """Hash the exact bundled inline assets and allow only explicit API origins."""
    origins = {_origin(progress_url), _origin(stop_url) if stop_url is not None else None}
    style, script = _inline_assets()

    def digest(value):
        return base64.b64encode(hashlib.sha256(value.encode("utf-8")).digest()).decode("ascii")

    connect = " ".join(["'self'", *sorted(origin for origin in origins if origin)])
    return ("default-src 'none'; "
            f"script-src 'sha256-{digest(script)}'; style-src 'sha256-{digest(style)}'; "
            f"connect-src {connect}; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'")


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
            if self.completion_phase == "downloading" and event.status != "error":
                with self._lock:
                    self._data["status"] = event.status
            else:
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
                self._data["complete"] = True
                if self._data["status"] == "running":
                    self._data["status"] = "complete"

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
            from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", self._port))
                sock.listen(128)
                self._port = sock.getsockname()[1]
                app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
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
                    response.headers["Content-Security-Policy"] = dashboard_csp(
                        progress_url="/progress", stop_url="/stop")
                    if token is not None and hmac.compare_digest(token, dashboard._token):
                        response.set_cookie("pymc_monitor_token", dashboard._token,
                                            httponly=True, samesite="strict", path="/")
                    return response

                @app.get("/r/{run_id}/")
                async def index_from_cookie(run_id: str):
                    with dashboard._lock:
                        if run_id not in dashboard._runs:
                            return Response(status_code=404)
                    return HTMLResponse(render_dashboard_html(
                        progress_url=f"/runs/{run_id}/state",
                        stop_url=f"/runs/{run_id}/stop", token="", title=dashboard.title))

                @app.get("/r/{token}/overview/")
                async def overview_bootstrap(token: str):
                    return RedirectResponse("/overview/", status_code=303)

                @app.get("/r/{token}/{run_id}/")
                async def index(token: str, run_id: str):
                    with dashboard._lock:
                        if run_id not in dashboard._runs:
                            return Response(status_code=404)
                    return RedirectResponse(f"/r/{run_id}/", status_code=303)

                @app.get("/overview/")
                async def overview():
                    return HTMLResponse(render_dashboard_html(
                        progress_url="/progress", stop_url="/stop", token="",
                        title=dashboard.title))

                @app.get("/progress")
                async def progress():
                    with dashboard._lock:
                        return JSONResponse({"models": [{"label": label, "name": label}
                                                        for label in dashboard._runs],
                                             "runs": {label: state.snapshot() for label, (state, _)
                                                      in dashboard._runs.items()}})

                @app.post("/stop")
                async def stop_models(model: str | None = None):
                    with dashboard._lock:
                        if model is not None and model not in dashboard._runs:
                            return Response(status_code=404)
                        labels = [model] if model is not None else list(dashboard._runs)
                        targets = [(state, control) for label in labels
                                   for state, control in [dashboard._runs[label]]
                                   if not state.snapshot()["complete"]
                                   and not state.snapshot()["stop_requested"]
                                   and state.snapshot().get("can_stop", True)]
                        if not targets:
                            return Response(status_code=409)
                        for state, control in targets:
                            control.request_stop()
                            state.request_stop()
                    return Response(status_code=204)

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
                        if (state.snapshot()["complete"] or state.snapshot()["stop_requested"]
                                or not state.snapshot().get("can_stop", True)):
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
            if (run_id in self._runs or run_id == "overview" or not run_id.isascii()
                    or not run_id.replace("-", "").replace("_", "").isalnum()):
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

    def overview_url(self):
        with self._lock:
            return (f"http://127.0.0.1:{self._port}/r/{self._token}/overview/"
                    if self._server else None)

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
