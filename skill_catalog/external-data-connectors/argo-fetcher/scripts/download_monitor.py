#!/usr/bin/env python3
"""Atomic status and loopback-only HTML monitoring for Argo transfers."""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_NAME = "download_status.json"
HTML_NAME = "download_monitor.html"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def safe_message(value: object) -> str:
    """Return status text without URLs, query strings, or local absolute paths."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    if "?" in text:
        text = text.split("?", 1)[0]
    if "://" in text or Path(text).is_absolute():
        text = Path(text.replace("\\", "/")).name
    return text[:240]


def write_status(run_dir: str | Path, **updates: Any) -> Path:
    target = Path(run_dir) / STATUS_NAME
    current: dict[str, Any] = {}
    if target.exists():
        with contextlib.suppress(Exception):
            current = json.loads(target.read_text(encoding="utf-8"))
    clean = {
        key: safe_message(value) if key in {"current_file", "message", "error"} else value
        for key, value in updates.items()
    }
    current.update(clean)
    current["updated_utc"] = utc_now()
    atomic_write_json(target, current)
    return target


def monitor_html() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Argo download</title>
<style>body{font:16px system-ui;margin:3rem;max-width:50rem}progress{width:100%;height:2rem}
pre{background:#f4f6f8;padding:1rem;white-space:pre-wrap}</style></head>
<body><h1>Argo download progress</h1><progress id="bar" max="1" value="0"></progress>
<pre id="status">Waiting for status…</pre><script>
async function poll(){try{const r=await fetch('download_status.json?'+Date.now(),{cache:'no-store'});
const s=await r.json();const n=Number(s.total_files||0),d=Number(s.completed_files||0);
document.querySelector('#bar').max=Math.max(1,n);document.querySelector('#bar').value=d;
document.querySelector('#status').textContent=JSON.stringify(s,null,2);}catch(e){}
setTimeout(poll,1500)}poll();</script></body></html>"""


def launch_monitor(run_dir: str | Path, *, open_browser: bool = True) -> dict[str, Any]:
    """Serve the run directory on 127.0.0.1 for the lifetime of this process."""
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    html = root / HTML_NAME
    html.write_text(monitor_html(), encoding="utf-8")

    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=str(root), **k
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/{HTML_NAME}"
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    return {"server": server, "thread": thread, "url": url, "html": html.name}
