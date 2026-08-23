#!/usr/bin/env python3
"""Atomic progress JSON and concise loopback waitbar for CFSR downloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import webbrowser


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        for attempt in range(10):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class DownloadStatus:
    def __init__(self, path: str | Path, **initial: Any) -> None:
        self.path = Path(path)
        self.started = time.monotonic()
        self.data: dict[str, Any] = {
            "schema_version": "external_download_status_v1",
            "state": "planned",
            "updated_utc": utc_now(),
            "recent_messages": [],
            **initial,
        }
        atomic_write_json(self.path, self.data)

    def update(self, message: str | None = None, **values: Any) -> None:
        self.data.update(values)
        self.data["updated_utc"] = utc_now()
        self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        if message:
            messages = list(self.data.get("recent_messages", []))
            messages.append({"utc": self.data["updated_utc"], "message": str(message)[:1000]})
            self.data["recent_messages"] = messages[-20:]
        expected = int(self.data.get("expected_bytes", 0) or 0)
        completed = int(self.data.get("completed_bytes", 0) or 0)
        elapsed = max(float(self.data.get("elapsed_seconds", 0) or 0), 1e-6)
        rate = completed / elapsed
        self.data["measured_bytes_per_second"] = round(rate, 3) if completed else None
        self.data["eta_seconds"] = round((expected - completed) / rate, 3) if rate > 0 and completed < expected else 0
        atomic_write_json(self.path, self.data)

    def start(self, message: str = "Download started") -> None:
        self.update(message, state="running", started_utc=utc_now())

    def finish(self, state: str, message: str) -> None:
        self.update(message, state=state, active_source=None, finished_utc=utc_now())


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CFSR download</title><style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#17212b;background:#f4f7f9}body{margin:0;padding:24px}.card{max-width:760px;margin:auto;background:white;border:1px solid #d8e0e6;border-radius:12px;padding:22px;box-shadow:0 4px 18px #0000000d}h1{font-size:20px;margin:0 0 4px}.sub{color:#667680;margin-bottom:18px}.bar{height:18px;background:#e7edf1;border-radius:10px;overflow:hidden}.fill{height:100%;width:0;background:#287e8e;transition:width .25s}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}.metric{padding:10px;background:#f7f9fa;border-radius:8px;color:#687781}.metric b{display:block;color:#17212b;margin-top:3px}.line{display:flex;justify-content:space-between;margin-top:8px}button{border:1px solid #9caab3;background:#fff;border-radius:7px;padding:7px 12px;cursor:pointer}pre{font-size:12px;white-space:pre-wrap;background:#f7f9fa;padding:10px;border-radius:8px;max-height:170px;overflow:auto}@media(max-width:620px){.row{grid-template-columns:1fr 1fr}}</style></head>
<body><div class="card"><div class="line"><div><h1>NCEP CFSR acquisition</h1><div class="sub" id="subtitle">Waiting for status</div></div><button onclick="refreshNow()">Refresh</button></div><div class="bar"><div class="fill" id="fill"></div></div><div class="line"><b id="pct">0.0%</b><span id="state">planned</span></div><div class="row"><div class="metric">Provider<b id="provider">--</b></div><div class="metric">Source units<b id="chunks">0 / 0</b></div><div class="metric">Bytes<b id="bytes">0 B / 0 B</b></div><div class="metric">Rate<b id="rate">--</b></div><div class="metric">ETA<b id="eta">--</b></div><div class="metric">Decoded hours<b id="decoded">0 / 0</b></div></div><div class="metric">Active source<b id="active">--</b></div><pre id="messages">--</pre></div>
<script>const e=x=>document.getElementById(x);const size=n=>{n=Number(n||0);for(const u of ['B','KiB','MiB','GiB']){if(n<1024||u==='GiB')return n.toFixed(u==='B'?0:1)+' '+u;n/=1024}};const span=s=>{s=Number(s);if(!isFinite(s)||s<0)return'--';let h=Math.floor(s/3600),m=Math.floor((s%3600)/60);return(h?h+'h ':'')+m+'m'};function render(d){let t=Number(d.expected_bytes||0),c=Number(d.completed_bytes||0),p=t?100*c/t:0;e('fill').style.width=Math.max(0,Math.min(100,p))+'%';e('pct').textContent=p.toFixed(1)+'%';e('state').textContent=d.state||'unknown';e('provider').textContent=d.provider||'--';e('chunks').textContent=(d.completed_chunks||0)+' / '+(d.total_chunks||0);e('bytes').textContent=size(c)+' / '+size(t);e('rate').textContent=d.measured_bytes_per_second?size(d.measured_bytes_per_second)+'/s':'--';e('eta').textContent=span(d.eta_seconds);e('decoded').textContent=(d.decoded_hours||0)+' / '+(d.expected_hours||0);e('active').textContent=d.active_source||'--';e('subtitle').textContent=(d.updated_utc||'--')+' · '+String(d.request_hash||'').slice(0,12);e('messages').textContent=(d.recent_messages||[]).map(x=>x.utc+'  '+x.message).join('\n')||'--';document.title=p.toFixed(1)+'% - CFSR'}async function refreshNow(){try{let r=await fetch('download_status.json?v='+Date.now(),{cache:'no-store'});render(await r.json())}catch(x){e('subtitle').textContent='Status unavailable: '+x}}refreshNow();setInterval(refreshNow,30000);</script></body></html>'''


def write_monitor_html(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "download_monitor.html"
    path.write_text(HTML, encoding="utf-8")
    return path


def _free_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def launch_monitor(run_dir: str | Path, *, open_browser: bool = True) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    html = write_monitor_html(directory)
    port = _free_port()
    command = [sys.executable, str(Path(__file__).resolve()), "serve", "--run-dir", str(directory), "--port", str(port)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    process = subprocess.Popen(command, cwd=directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    payload = {"schema_version":"external_download_monitor_server_v1","pid":process.pid,"port":port,"url":f"http://127.0.0.1:{port}/{html.name}","started_utc":utc_now()}
    atomic_write_json(directory / "monitor_server.json", payload)
    if open_browser:
        webbrowser.open(payload["url"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--run-dir", type=Path, required=True)
    serve.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
        os.chdir(args.run_dir.resolve())
        ThreadingHTTPServer(("127.0.0.1", args.port), SimpleHTTPRequestHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
