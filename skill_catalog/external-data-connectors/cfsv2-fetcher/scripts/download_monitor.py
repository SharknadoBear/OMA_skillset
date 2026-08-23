#!/usr/bin/env python3
"""Persistent JSON progress state and a localhost HTML waitbar for long downloads."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
from pathlib import Path
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import webbrowser


TERMINAL_STATES = {"complete", "failed", "cancelled"}
STATUS_NAME = "download_status.json"
HTML_NAME = "download_monitor.html"
SERVER_NAME = "monitor_server.json"


def utc_now() -> str:
    """Return an RFC-3339 UTC timestamp without importing third-party packages."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Publish JSON through an atomic replace in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        # OneDrive and antivirus indexers can briefly hold the destination open
        # on Windows. Retry publication without weakening atomicity.
        for attempt in range(20):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 19:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def safe_message(value: object) -> str:
    """Remove URL queries and obvious local paths from monitor-visible messages."""
    text = str(value)
    for raw in re.findall(r"https?://[^\s\"']+", text):
        parsed = urlsplit(raw)
        clean = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        text = text.replace(raw, clean)
    text = re.sub(r"(?i)[A-Z]:\\[^\s,;]+", "[local-path]", text)
    text = re.sub(r"(?<!:)\/(?:home|Users|scratch)\/[^\s,;]+", "[local-path]", text)
    return text[:1000]


class DownloadStatus:
    """Thread-safe atomic status writer with a periodic heartbeat."""

    def __init__(
        self,
        path: str | Path,
        *,
        connector: str = "cfs-fetcher",
        request_hash: str = "",
        total_chunks: int = 0,
        expected_bytes: int = 0,
        estimate_seconds: float = 0.0,
        artifacts: dict[str, str] | None = None,
        heartbeat_seconds: float = 15.0,
        **initial: Any,
    ) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._started_monotonic = time.monotonic()
        self._heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self.data: dict[str, Any] = {
            "schema_version": "external_download_status_v1",
            "connector": connector,
            "request_hash": request_hash,
            "state": "planned",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "started_utc": None,
            "finished_utc": None,
            "elapsed_seconds": 0.0,
            "estimate_seconds": round(float(estimate_seconds), 3),
            "eta_seconds": round(float(estimate_seconds), 3),
            "total_chunks": int(total_chunks),
            "completed_chunks": 0,
            "failed_chunks": 0,
            "active_chunk": None,
            "expected_bytes": int(expected_bytes),
            "completed_bytes": 0,
            "measured_bytes_per_second": None,
            "attempts": 0,
            "retries": 0,
            "recent_messages": [],
            "artifacts": dict(artifacts or {}),
            **initial,
        }
        self._write()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _write(self) -> None:
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        self.data["elapsed_seconds"] = round(elapsed, 3)
        self.data["updated_utc"] = utc_now()
        completed = int(self.data.get("completed_bytes", 0))
        expected = int(self.data.get("expected_bytes", 0))
        if completed > 0 and elapsed > 0:
            rate = completed / elapsed
            self.data["measured_bytes_per_second"] = round(rate, 3)
            self.data["eta_seconds"] = round(max(0, expected - completed) / rate, 3)
        elif self.data.get("state") in TERMINAL_STATES:
            self.data["eta_seconds"] = 0.0
        atomic_write_json(self.path, self.data)

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._heartbeat_seconds):
            with self._lock:
                if self.data.get("state") in TERMINAL_STATES:
                    return
                self._write()

    def start(self, message: str | None = None) -> None:
        with self._lock:
            self.data.update({"state": "running", "started_utc": utc_now()})
            if message:
                recent = list(self.data.get("recent_messages", []))
                recent.append({"utc": utc_now(), "message": safe_message(message)})
                self.data["recent_messages"] = recent[-12:]
            self._write()

    def update(self, **values: Any) -> None:
        with self._lock:
            if "message" in values:
                message = safe_message(values.pop("message"))
                recent = list(self.data.get("recent_messages", []))
                recent.append({"utc": utc_now(), "message": message})
                self.data["recent_messages"] = recent[-12:]
            self.data.update(values)
            self._write()

    def finish(self, state: str = "complete", message: str | None = None) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"terminal state required, got {state!r}")
        with self._lock:
            self.data.update(
                {"state": state, "active_chunk": None, "finished_utc": utc_now()}
            )
            if message:
                recent = list(self.data.get("recent_messages", []))
                recent.append({"utc": utc_now(), "message": safe_message(message)})
                self.data["recent_messages"] = recent[-12:]
            self._write()
        self._stop.set()
        self._thread.join(timeout=2.0)


def monitor_html(status_name: str = STATUS_NAME) -> str:
    """Return a dependency-free waitbar page using only relative JSON access."""
    safe_name = json.dumps(Path(status_name).name)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>External Data Download Monitor</title>
<style>
:root{{--bg:#07111f;--panel:#10233d;--line:#2c4565;--text:#eaf2ff;--muted:#9fb3cc;--accent:#4ac7ff;--ok:#4ade80;--bad:#fb7185}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#06101d,#0b1d34);color:var(--text);font:15px system-ui,sans-serif}}
main{{max-width:920px;margin:40px auto;padding:0 20px}} .card{{background:rgba(16,35,61,.96);border:1px solid var(--line);border-radius:16px;padding:24px;box-shadow:0 18px 50px #0006}}
h1{{font-size:24px;margin:0 0 6px}} .muted{{color:var(--muted)}} .bar{{height:24px;background:#06101d;border:1px solid var(--line);border-radius:999px;overflow:hidden;margin:22px 0 8px}}
#fill{{height:100%;width:0;background:linear-gradient(90deg,#38bdf8,#4ade80);transition:width .5s}} #pct{{font-size:32px;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}} .metric{{background:#0a1a30;border:1px solid var(--line);border-radius:10px;padding:12px}} .metric b{{display:block;font-size:18px;margin-top:4px}}
pre{{white-space:pre-wrap;background:#07111f;border:1px solid var(--line);border-radius:10px;padding:12px;min-height:88px}} input{{margin-top:12px}} .complete{{color:var(--ok)}} .failed{{color:var(--bad)}}
</style></head><body><main><section class="card">
<h1>External Data Download</h1><div id="subtitle" class="muted">Waiting for status JSON...</div>
<div class="bar"><div id="fill"></div></div><div id="pct">0%</div>
<div class="grid">
 <div class="metric">State<b id="state">planned</b></div>
 <div class="metric">Chunks<b id="chunks">0 / 0</b></div>
 <div class="metric">Transferred<b id="bytes">0 B</b></div>
 <div class="metric">Throughput<b id="rate">--</b></div>
 <div class="metric">Elapsed<b id="elapsed">--</b></div>
 <div class="metric">ETA<b id="eta">--</b></div>
 <div class="metric">Retries / failures<b id="retries">0 / 0</b></div>
 <div class="metric">Last heartbeat<b id="updated">--</b></div>
</div>
<div class="muted">Active chunk</div><pre id="active">--</pre>
<div class="muted">Recent messages</div><pre id="messages">--</pre>
<div id="fallback" class="muted">If automatic refresh is unavailable, select {Path(status_name).name}:</div>
<input id="file" type="file" accept="application/json,.json">
</section></main>
<script>
const STATUS={safe_name};
const el=id=>document.getElementById(id);
const size=n=>{{if(n==null)return'--';let u=['B','KiB','MiB','GiB'];let i=0;while(n>=1024&&i<u.length-1){{n/=1024;i++}}return n.toFixed(i?1:0)+' '+u[i]}};
const span=s=>{{if(s==null||!isFinite(s))return'--';s=Math.max(0,Math.round(s));let d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),q=s%60;return(d?d+'d ':'')+String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(q).padStart(2,'0')}};
function render(d){{let tc=Number(d.total_chunks||0),cc=Number(d.completed_chunks||0);let bp=Number(d.expected_bytes||0)>0?100*Number(d.completed_bytes||0)/Number(d.expected_bytes):0;let cp=tc?100*cc/tc:0;let p=Math.max(0,Math.min(100,Math.max(bp,cp)));el('fill').style.width=p.toFixed(1)+'%';el('pct').textContent=p.toFixed(1)+'%';el('state').textContent=d.state||'unknown';el('state').className=d.state==='complete'?'complete':d.state==='failed'?'failed':'';el('subtitle').textContent=(d.connector||'connector')+' · '+String(d.request_hash||'').slice(0,12);el('chunks').textContent=cc+' / '+tc;el('bytes').textContent=size(d.completed_bytes)+' / '+size(d.expected_bytes);el('rate').textContent=d.measured_bytes_per_second?size(d.measured_bytes_per_second)+'/s':'--';el('elapsed').textContent=span(d.elapsed_seconds);el('eta').textContent=span(d.eta_seconds);el('retries').textContent=(d.retries||0)+' / '+(d.failed_chunks||0);el('updated').textContent=d.updated_utc||'--';el('active').textContent=d.active_chunk||'--';el('messages').textContent=(d.recent_messages||[]).map(x=>x.utc+'  '+x.message).join('\\n')||'--'}}
async function poll(){{try{{let r=await fetch(STATUS+'?t='+Date.now(),{{cache:'no-store'}});if(!r.ok)throw Error(r.status);render(await r.json())}}catch(e){{el('fallback').textContent='Automatic refresh unavailable ('+e+'). Select the status JSON below.'}}}}
el('file').addEventListener('change',async e=>{{let f=e.target.files[0];if(f)render(JSON.parse(await f.text()))}});
poll();setInterval(poll,5000);
</script></body></html>"""


def write_monitor_html(run_dir: str | Path) -> Path:
    path = Path(run_dir) / HTML_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(monitor_html(), encoding="utf-8", newline="\n")
    return path


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return


class _LoopbackServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve_monitor(
    run_dir: str | Path,
    *,
    port: int = 0,
    open_browser: bool = False,
    terminal_grace_seconds: float = 300.0,
    max_hours: float = 72.0,
) -> int:
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_monitor_html(directory)
    handler = functools.partial(_QuietHandler, directory=str(directory))
    started = time.monotonic()
    terminal_seen: float | None = None
    with _LoopbackServer(("127.0.0.1", int(port)), handler) as server:
        server.timeout = 1.0
        actual_port = int(server.server_address[1])
        url = f"http://127.0.0.1:{actual_port}/{HTML_NAME}"
        atomic_write_json(
            directory / SERVER_NAME,
            {
                "schema_version": "external_download_monitor_server_v1",
                "host": "127.0.0.1",
                "port": actual_port,
                "url": url,
                "started_utc": utc_now(),
            },
        )
        if open_browser:
            try:
                webbrowser.open(url, new=2)
            except Exception:
                pass
        while True:
            server.handle_request()
            now = time.monotonic()
            try:
                state = json.loads((directory / STATUS_NAME).read_text(encoding="utf-8")).get("state")
            except (OSError, json.JSONDecodeError):
                state = None
            if state in TERMINAL_STATES:
                terminal_seen = terminal_seen or now
                if now - terminal_seen >= max(0.0, terminal_grace_seconds):
                    return 0
            if now - started >= max_hours * 3600.0:
                return 0


def launch_monitor(run_dir: str | Path, *, open_browser: bool = True) -> dict[str, Any]:
    """Launch the localhost server in a detached helper process."""
    directory = Path(run_dir).resolve()
    html = write_monitor_html(directory)
    server_state = directory / SERVER_NAME
    try:
        server_state.unlink()
    except FileNotFoundError:
        pass
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "serve",
        "--run-dir",
        str(directory),
        "--terminal-grace-seconds",
        "300",
    ]
    if open_browser:
        command.append("--open-browser")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            start_new_session=(os.name != "nt"),
        )
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if server_state.exists():
                state = json.loads(server_state.read_text(encoding="utf-8"))
                return {"launched": True, "html": str(html), **state}
            time.sleep(0.1)
    except Exception as exc:
        return {"launched": False, "html": str(html), "reason": safe_message(exc)}
    return {
        "launched": False,
        "html": str(html),
        "reason": "monitor server did not publish a URL; open the HTML manually",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="Write the static monitor HTML")
    render.add_argument("--run-dir", required=True)
    serve = subparsers.add_parser("serve", help="Serve a run directory on loopback")
    serve.add_argument("--run-dir", required=True)
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--open-browser", action="store_true")
    serve.add_argument("--terminal-grace-seconds", type=float, default=300.0)
    serve.add_argument("--max-hours", type=float, default=72.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "render":
        print(write_monitor_html(args.run_dir))
        return 0
    return serve_monitor(
        args.run_dir,
        port=args.port,
        open_browser=args.open_browser,
        terminal_grace_seconds=args.terminal_grace_seconds,
        max_hours=args.max_hours,
    )


if __name__ == "__main__":
    raise SystemExit(main())
