"""Reusable dashboard state reducer and authenticated loopback HTTP service."""
from __future__ import annotations

import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from .convergence import _jsonable
from .events import ConvergenceUpdate, PhaseUpdate, SamplingProgress


def dashboard_assets():
    """Return bundled assets for consumers serving the same safe dashboard UI."""
    names = ("dashboard.html", "dashboard.js", "dashboard.css", "renderer.js")
    root = files("pymc_monitor").joinpath("static")
    return {name: root.joinpath(name).read_bytes() for name in names}


class DashboardState:
    """Transport-independent reducer; cloud consumers may publish the same schema."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {"phases": [], "sampling": None, "complete": False}

    def show_phase(self, update: PhaseUpdate):
        with self._lock:
            phase = {"label": update.phase.value, "status": update.status,
                     "detail": update.message}
            phases = [p for p in self._data["phases"] if p["label"] != phase["label"]]
            phases.append(phase)
            self._data["phases"] = phases
            if update.status == "error" or (update.phase.value == "sampling" and update.status == "done"):
                self._data["complete"] = True

    def show_sampling(self, update: SamplingProgress):
        with self._lock:
            self._data["sampling"] = {
                "chains": {str(k): vars(v).copy() for k, v in update.chains.items()},
                "total_divergences": update.total_divergences,
                "elapsed": update.elapsed, "total_draws": update.total_draws,
            }

    def show_convergence(self, update: ConvergenceUpdate):
        with self._lock:
            self._data["convergence"] = {
                "params": {k: vars(v).copy() for k, v in update.params.items()},
                "draws": update.draws,
                "verdict": {**(update.verdict or {}), "final": update.final},
            }
            if update.traces:
                self._data["traces"] = update.traces

    def request_stop(self):
        with self._lock:
            self._data["stop_requested"] = True

    def finish(self):
        with self._lock:
            self._data["complete"] = True

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(_jsonable(self._data), allow_nan=False))


class DashboardService(DashboardState):
    """Runs only on loopback, using a per-run bearer token and no third-party scripts."""

    def __init__(self, control):
        super().__init__()
        self._control = control
        self._token = secrets.token_urlsafe(32)
        self._server = None
        self._thread = None

    def start(self):
        if self._server is not None:
            raise RuntimeError("dashboard already started")
        state = self
        assets = dashboard_assets()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _headers(self, status, mime="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self.end_headers()

            def _authorized(self):
                token = self.headers.get("Authorization", "")
                host = f"127.0.0.1:{self.server.server_port}"
                return (self.headers.get("Host") == host and self._local_origin()
                        and hmac.compare_digest(token.encode(), ("Bearer " + state._token).encode()))

            def _local_origin(self):
                origin = self.headers.get("Origin")
                return origin is None or origin == f"http://127.0.0.1:{self.server.server_port}"

            def do_GET(self):
                path = urlsplit(self.path).path
                if path in ("/state", "/stop") and not self._authorized():
                    self._headers(403)
                    return
                if path == "/state":
                    payload = json.dumps(state.snapshot()).encode()
                    self._headers(200)
                elif path == "/":
                    payload = assets["dashboard.html"]
                    self._headers(200, "text/html; charset=utf-8")
                elif path in ("/dashboard.js", "/dashboard.css", "/renderer.js"):
                    payload = assets[path[1:]]
                    self._headers(200, "text/javascript" if path.endswith("js") else "text/css")
                else:
                    self._headers(404)
                    return
                self.wfile.write(payload)

            def do_POST(self):
                if urlsplit(self.path).path != "/stop":
                    self._headers(404)
                elif not self._authorized() or not self._local_origin():
                    self._headers(403)
                else:
                    state._control.stop()
                    state.request_stop()
                    self._headers(204)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self._server.server_port}/#{self._token}" if self._server else None

    def close(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._server = None
