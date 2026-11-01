"""Browser-only two-origin fixture for the self-contained dashboard contract."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pymc_monitor.dashboard import dashboard_csp, render_dashboard_html

LABEL = 'model / <img src=x onerror=window.pwned=1> &?'
posts = []
reads = 0
options = 0


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in {"/", "/flat", "/readonly", "/nostop"}:
            self.send_error(404)
            return
        mode = self.path[1:]
        suffix = f"&mode={mode}" if mode else ""
        progress_url = f"{api_origin}/progress?existing=1{suffix}"
        stop_url = None if mode == "readonly" else f"{api_origin}/stop?existing=1{suffix}"
        html = render_dashboard_html(
            progress_url=progress_url, stop_url=stop_url, token="secret?&",
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Security-Policy", dashboard_csp(
            progress_url=progress_url, stop_url=stop_url))
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *args):
        pass


class APIHandler(BaseHTTPRequestHandler):
    def respond(self, status, data):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", dashboard_origin)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        global options
        options += 1
        self.respond(204, {})

    def do_GET(self):
        global reads
        if self.path == "/test/log":
            self.respond(200, {"posts": posts, "options": options, "reads": reads})
            return
        from urllib.parse import parse_qs, urlsplit

        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query)
        if (parsed.path != "/progress" or params.get("existing") != ["1"]
                or params.get("token") != ["secret?&"]):
            self.send_error(404)
            return
        if params.get("mode") in (["flat"], ["nostop"]):
            self.respond(200, {"phases": [{"label": "sampling", "detail": "Flat run",
                                           "status": "in_progress"}], "complete": False,
                               "can_stop": params.get("mode") != ["nostop"]})
            return
        reads += 1
        first = {"phases": [{"label": "sampling", "detail": "Working", "status": "in_progress"}],
                 "sampling": {"chains": {"0": {"phase": "sampling", "draw": 5,
                                               "total": 10}}, "total_draws": 5},
                 "traces": {"x<script>window.pwned=1</script>": [[1, 2, 3]]},
                 "complete": len(posts) >= 3, "status": "stopped" if len(posts) >= 3 else "running"}
        second = {"complete": True, "status": "cached", "can_stop": False} if reads >= 8 else None
        self.respond(200, {"models": [{"label": LABEL, "name": LABEL},
                                      {"label": "cached", "name": "Cached model"}],
                           "runs": {LABEL: first, **({"cached": second} if second else {})}})

    def do_POST(self):
        from urllib.parse import parse_qs, urlsplit

        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query)
        if parsed.path != "/stop" or params.get("token") != ["secret?&"]:
            self.send_error(403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        posts.append({"url": self.path, "authorization": self.headers.get("Authorization"),
                      "contentType": self.headers.get("Content-Type"),
                      "body": self.rfile.read(length).decode()})
        if params.get("mode") == ["flat"]:
            self.respond(403 if sum("mode=flat" in post["url"] for post in posts) == 1 else 200,
                         {"stopped": True})
        elif len(posts) == 1:
            self.respond(200, {"stopped": False})
        elif len(posts) == 2:
            self.respond(500, {"stopped": False})
        else:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", dashboard_origin)
            self.end_headers()

    def log_message(self, *args):
        pass


dashboard = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
api = ThreadingHTTPServer(("127.0.0.1", 0), APIHandler)
dashboard_origin = f"http://127.0.0.1:{dashboard.server_port}"
api_origin = f"http://127.0.0.1:{api.server_port}"
threading.Thread(target=api.serve_forever, daemon=True).start()
print(json.dumps([dashboard_origin, api_origin]), flush=True)
try:
    dashboard.serve_forever()
finally:
    dashboard.server_close()
    api.shutdown()
    api.server_close()
