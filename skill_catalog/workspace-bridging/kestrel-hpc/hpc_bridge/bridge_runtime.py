"""Session-local request ledger and responsive command/transfer lanes.

No scheduler polling or remote command replay happens in this module.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REQUEST_ID = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")
MAX_OUTPUT = 1024 * 1024
MAX_LEDGER_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers can transiently prevent replacement; never truncate in place.
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.02)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class SessionLock:
    """OS releases this advisory lock on crash; stale PID files grant no ownership."""
    def __init__(self, root):
        self.path = Path(root) / "bridge.lock"

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            self.stream.seek(0)
            if self.stream.read(1) == b"":
                self.stream.write(b"0")
                self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("This session already has a bridge process") from None
        return self

    def __exit__(self, *args):
        self.stream.close()


class Operation:
    def __init__(self, root, payload):
        self.root = Path(root).resolve()
        self.id = payload["id"]
        self.payload = payload
        self.path = Path(root) / "progress" / f"{self.id}.json"
        self.started = time.monotonic()
        self.deadline = self.started + positive(payload.get("timeout", 3600), "timeout")
        if "expires_at" in payload:
            self.deadline = min(self.deadline, self.started + float(payload["expires_at"]) - time.time())
        self.stall_timeout = positive(payload.get("stall_timeout", 120), "stall_timeout")
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.channels = []
        self.reason = None
        self.last_activity = self.started
        self.last_write = 0.0
        self.state = {"id": self.id, "action": payload["action"], "state": "running",
                      "started_at": utc_now(), "phase": "starting", "bytes_done": 0}
        self.update(force=True)

    def bind(self, channel):
        with self.lock:
            self.channels.append(channel)
            if self.cancelled.is_set():
                channel.close()
        self.check()

    def abort(self, reason):
        with self.lock:
            if self.reason is None:
                self.reason = reason
            self.cancelled.set()
            self.close_channels()

    def close_channels(self):
        with self.lock:
            for channel in self.channels:
                try:
                    channel.close()
                except Exception:
                    pass

    def check(self):
        if time.monotonic() >= self.deadline:
            self.abort("deadline_exceeded")
        if self.cancelled.is_set():
            raise RuntimeError(self.reason or "cancelled")

    def watchdog(self):
        if time.monotonic() >= self.deadline:
            self.abort("deadline_exceeded")
        elif (self.state["phase"] == "transferring" and
              time.monotonic() - self.last_activity > self.stall_timeout):
            self.abort("transfer_stalled")

    def update(self, force=False, activity=False, **fields):
        with self.lock:
            if activity or fields.get("phase") != self.state.get("phase") and "phase" in fields:
                self.last_activity = time.monotonic()
            self.state.update(fields)
            if force or time.monotonic() - self.last_write >= 1:
                self.state["updated_at"] = utc_now()
                atomic_json(self.path, self.state)
                self.last_write = time.monotonic()


class BridgeRuntime:
    def __init__(self, root, handler, connector, identity):
        self.root = Path(root)
        self.handler, self.connector, self.identity = handler, connector, identity
        self.client = None
        self.active = {}
        self.stopping = False
        self.auth_thread = None
        self.auth_error = None
        self.last_health = 0.0
        self.storage_ok = True
        self.last_storage = 0.0
        self.receipt_failure = False
        for name in ("commands", "inflight", "results", "progress"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def result(self, request_id, data):
        atomic_json(self.root / "results" / f"{request_id}.json",
                    {"id": request_id, "finished_at": utc_now(), **data})

    def recover(self):
        for path in (self.root / "inflight").glob("*.json"):
            if not (self.root / "results" / path.name).exists():
                self.result(path.stem, {"status": "uncertain", "exit_status": 1,
                            "request": read_json(path),
                            "stderr": "Bridge restarted after claim. Reconcile original operation; never replay exec automatically."})
            path.unlink()

    def connected(self):
        transport = self.client.get_transport() if self.client else None
        return bool(transport and transport.is_active() and transport.is_authenticated())

    def authenticate(self):
        if self.auth_thread and self.auth_thread.is_alive():
            return
        if self.connected():
            return
        if self.active:
            raise RuntimeError("Wait for active requests to settle before reconnecting")
        self.auth_error = None

        def worker():
            try:
                if self.client:
                    self.client.close()
                client = self.connector()
                if self.stopping:
                    client.close()
                else:
                    self.client = client
            except Exception as exc:
                # Authentication exception text may contain server-specific data.
                self.auth_error = type(exc).__name__
        self.auth_thread = threading.Thread(target=worker, daemon=True)
        self.auth_thread.start()

    def execute(self, op):
        try:
            op.check()
            result = self.handler(self.client, op.payload, op)
        except Exception as exc:
            status = "uncertain" if op.payload["action"] == "exec" else "interrupted"
            result = {"status": status, "exit_status": 1, "stdout": "",
                      "stderr": str(exc)[:4000], "reason": op.reason or type(exc).__name__,
                      "recovery": "Reconcile remote effects before retrying exec; use explicit --resume for a transfer after restoring the connection."}
        try:
            self.result(op.id, {**result, "request": op.payload})
            op.update(force=True, state=result["status"], phase="finished")
            (self.root / "inflight" / f"{op.id}.json").unlink(missing_ok=True)
        except Exception:
            # Leave the claim on disk for restart reconciliation.
            op.state["state"] = "receipt_write_failed"
            self.receipt_failure = True
        finally:
            op.close_channels()

    def control(self, payload):
        action = payload["action"]
        if action == "stop":
            self.stopping = True
            for op, _ in self.active.values():
                op.abort("bridge_stopping")
            if self.client:
                self.client.close()
            return {"status": "ok", "exit_status": 0, "stdout": "Bridge stop requested; Slurm jobs are NOT cancelled. Check final bridge status and reconcile owned jobs separately."}
        if action == "reconnect":
            if self.stopping:
                raise RuntimeError("Bridge is stopping; no new authentication will be started")
            self.authenticate()
            return {"status": "ok", "exit_status": 0, "stdout": "Authentication requested in the visible bridge window. No failed command will be replayed."}
        target = payload.get("request_id", "")
        if not REQUEST_ID.fullmatch(target):
            raise ValueError("Invalid cancellation request ID")
        for op, _ in self.active.values():
            if op.id == target:
                op.abort("cancel_requested")
                return {"status": "ok", "exit_status": 0, "stdout": "Cancellation requested; read the original receipt. Remote exec effects may remain."}
        path = self.root / "commands" / f"{target}.json"
        if path.exists():
            self.result(target, {"status": "cancelled_before_start", "exit_status": 1})
            path.unlink()
            return {"status": "ok", "exit_status": 0, "stdout": "Queued request cancelled before dispatch."}
        return {"status": "ok", "exit_status": 0, "stdout": "No active request matched; inspect its existing receipt."}

    def tick(self):
        for lane, (op, thread) in list(self.active.items()):
            op.watchdog()
            if not thread.is_alive():
                del self.active[lane]
        if time.monotonic() - self.last_storage > 10:
            size = sum(p.stat().st_size for name in ("commands", "inflight", "results", "progress")
                       for p in (self.root / name).rglob("*") if p.is_file())
            self.storage_ok = size < MAX_LEDGER_BYTES
            self.last_storage = time.monotonic()
        paths = sorted((self.root / "commands").glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))
        # Control requests bypass both occupied lanes and the storage gate.
        pending = []
        for path in paths:
            try:
                if path.stat().st_size > MAX_REQUEST_BYTES:
                    raise ValueError("Request exceeds 64 KiB; use an uploaded script and a short command")
                payload = read_json(path)
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object")
                if not REQUEST_ID.fullmatch(path.stem) or payload.get("id") != path.stem:
                    raise ValueError("Request ID must match its filename")
                if (self.root / "results" / path.name).exists():
                    path.unlink()
                    continue
                action = payload.get("action")
                if action in {"stop", "cancel", "reconnect"}:
                    path.replace(self.root / "inflight" / path.name)
                    try:
                        result = self.control(payload)
                    except Exception as exc:
                        result = {"status": "error", "exit_status": 1, "stderr": str(exc)}
                    self.result(path.stem, result)
                    (self.root / "inflight" / path.name).unlink()
                elif action in {"exec", "upload", "download"}:
                    positive(payload.get("timeout", 3600), "timeout")
                    positive(payload.get("stall_timeout", 120), "stall_timeout")
                    if "expires_at" in payload:
                        positive(payload["expires_at"], "expires_at")
                    pending.append((path, payload))
                else:
                    raise ValueError("Unsupported action")
            except (ValueError, KeyError, TypeError) as exc:
                if REQUEST_ID.fullmatch(path.stem):
                    self.result(path.stem, {"status": "rejected", "exit_status": 1, "stderr": str(exc)})
                    path.unlink(missing_ok=True)
                else:
                    raise
        for path, payload in pending:
            if not path.exists():  # Cancelled by a control request in this tick.
                continue
            if self.stopping:
                self.result(path.stem, {"status": "cancelled_before_start", "exit_status": 1})
                path.unlink()
                continue
            lane = "exec" if payload["action"] == "exec" else "transfer"
            if payload.get("expires_at", float("inf")) < time.time():
                self.result(path.stem, {"status": "expired_before_start", "exit_status": 1})
                path.unlink()
                continue
            if lane in self.active or not self.connected() or not self.storage_ok or self.receipt_failure:
                continue
            path.replace(self.root / "inflight" / path.name)
            op = Operation(self.root, payload)
            thread = threading.Thread(target=self.execute, args=(op,), daemon=True)
            self.active[lane] = (op, thread)
            thread.start()
        if time.monotonic() - self.last_health >= 1 or self.stopping:
            authenticating = bool(self.auth_thread and self.auth_thread.is_alive())
            state = ("stopping" if self.stopping else "authenticating" if authenticating else
                     "receipt_write_failed" if self.receipt_failure else
                     "storage_limit" if not self.storage_ok else "connected" if self.connected() else "authentication_required")
            atomic_json(self.root / "bridge_status.json", {
                "state": state, "heartbeat_at": utc_now(), "pid": os.getpid(),
                "bridge_name": self.identity.get("bridge_name"), "auth_error": self.auth_error,
                "active": [op.state.copy() for op, _ in self.active.values()],
                "queued_count": len(list((self.root / "commands").glob("*.json"))),
                "ledger_limit_bytes": MAX_LEDGER_BYTES,
            })
            self.last_health = time.monotonic()

    def run(self):
        with SessionLock(self.root):
            self.recover()
            self.authenticate()
            try:
                while not self.stopping:
                    self.tick()
                    time.sleep(.1)
            finally:
                self.stopping = True
                for op, _ in self.active.values():
                    op.abort("bridge_stopping")
                if self.client:
                    self.client.close()
                end = time.monotonic() + 5
                for _, thread in self.active.values():
                    thread.join(max(0, end - time.monotonic()))
                unresolved = [op.id for op, thread in self.active.values() if thread.is_alive()]
                atomic_json(self.root / "bridge_status.json", {
                    "state": "stopped", "heartbeat_at": utc_now(), "pid": os.getpid(),
                    "unresolved_requests": unresolved, "remote_jobs_cancelled": False})
        return 0
