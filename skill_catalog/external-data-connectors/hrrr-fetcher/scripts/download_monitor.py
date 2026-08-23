#!/usr/bin/env python3
"""Atomic machine-readable progress for HRRR transfers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any


TERMINAL_STATES = {"complete", "failed", "cancelled"}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
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


class DownloadStatus:
    def __init__(self, path: str | Path, *, request_hash: str, total_chunks: int, expected_bytes: int) -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.data: dict[str, Any] = {
            "schema_version": "external_download_status_v1",
            "connector": "hrrr-fetcher",
            "request_hash": request_hash,
            "state": "planned",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "total_chunks": int(total_chunks),
            "completed_chunks": 0,
            "expected_bytes": int(expected_bytes),
            "completed_bytes": 0,
            "retries": 0,
            "failed_chunks": 0,
            "active_chunk": None,
            "recent_messages": [],
        }
        self._write()

    def _write(self) -> None:
        self.data["updated_utc"] = utc_now()
        self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        elapsed = float(self.data["elapsed_seconds"])
        completed = int(self.data.get("completed_bytes", 0))
        if elapsed > 0 and completed:
            rate = completed / elapsed
            self.data["measured_bytes_per_second"] = round(rate, 3)
            self.data["eta_seconds"] = round(max(0, int(self.data["expected_bytes"]) - completed) / rate, 3)
        atomic_write_json(self.path, self.data)

    def update(self, *, message: str | None = None, **values: Any) -> None:
        with self.lock:
            if message:
                rows = list(self.data["recent_messages"])
                rows.append({"utc": utc_now(), "message": str(message)[:1000]})
                self.data["recent_messages"] = rows[-16:]
            self.data.update(values)
            self._write()

    def start(self, message: str = "Starting HRRR transfer") -> None:
        self.update(state="running", started_utc=utc_now(), message=message)

    def finish(self, state: str, message: str) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"Invalid terminal state {state}")
        self.update(state=state, finished_utc=utc_now(), active_chunk=None, message=message)
