#!/usr/bin/env python3
"""Persistent local JSON bridge for Perlmutter SSH/SFTP commands."""

from __future__ import annotations

import getpass
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import paramiko

from bridge_runtime import BridgeRuntime, MAX_OUTPUT, positive
from bridge_transfer import transfer


ROOT = Path(__file__).resolve().parent
IDENTITY = ROOT / "bridge_identity.json"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 3600.0


def load_identity() -> dict:
    if not IDENTITY.exists():
        raise FileNotFoundError(
            f"Missing {IDENTITY.name}. Create a named bridge session with make_bridge_session.py first."
        )
    identity = json.loads(IDENTITY.read_text(encoding="utf-8-sig"))
    identity["startup_datetime_local"] = datetime.now().astimezone().replace(microsecond=0).isoformat()
    IDENTITY.write_text(json.dumps(identity, indent=2), encoding="utf-8", newline="\n")
    return identity


def print_identity(identity: dict) -> None:
    print("")
    print("Bridge identity")
    print(f"  name: {identity.get('bridge_name', '')}")
    print(f"  startup: {identity.get('startup_datetime_local', '')}")
    print(f"  purpose: {identity.get('purpose', '')}")
    print(f"  work_summary: {identity.get('work_summary', '')}")
    print(f"  local_project_root: {identity.get('local_project_root', '')}")
    print(f"  remote_target: {identity.get('remote_target', '')}")
    print("")


def connection_from_identity(identity: dict) -> tuple[str, str]:
    target = str(identity.get("remote_target", "")).strip()
    if target.count("@") != 1:
        raise ValueError("bridge_identity.json remote_target must be username@hostname")
    user, host = target.split("@", 1)
    if not user or not host:
        raise ValueError("bridge_identity.json remote_target must be username@hostname")
    return user, host


def connect() -> paramiko.SSHClient:
    from site_auth import connect_verified
    identity = load_identity()
    print_identity(identity)
    return connect_verified(identity)


def exec_command(client: paramiko.SSHClient, command: str, timeout: float | None, operation=None) -> dict:
    effective_timeout = timeout if timeout is not None else DEFAULT_COMMAND_TIMEOUT_SECONDS
    positive(effective_timeout, "command timeout")

    if operation is not None:
        # Bind before the exec acknowledgement so cancellation can close a server
        # channel which stalls before command startup, too.
        channel = client.get_transport().open_session(timeout=min(30, effective_timeout))
        operation.bind(channel)
        channel.settimeout(min(30, effective_timeout))
        channel.exec_command(command)
        channel.shutdown_write()
    else:
        stdin, stdout, stderr = client.exec_command(command, timeout=effective_timeout)
        stdin.close()
        channel = stdout.channel
    deadline = time.monotonic() + effective_timeout
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    counts = [0, 0]
    truncated = False
    while True:
        if operation is not None:
            operation.check()
        while channel.recv_ready():
            data = channel.recv(65536)
            available = max(0, MAX_OUTPUT - counts[0])
            if available:
                out_chunks.append(data[:available])
            counts[0] += len(data)
            truncated |= len(data) > available
            if operation is not None:
                operation.check()
            if time.monotonic() >= deadline:
                channel.close()
                raise TimeoutError("remote command output exceeded its deadline")
        while channel.recv_stderr_ready():
            data = channel.recv_stderr(65536)
            available = max(0, MAX_OUTPUT - counts[1])
            if available:
                err_chunks.append(data[:available])
            counts[1] += len(data)
            truncated |= len(data) > available
            if operation is not None:
                operation.check()
            if time.monotonic() >= deadline:
                channel.close()
                raise TimeoutError("remote command output exceeded its deadline")
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            # Exit status can arrive before the final stream data.  Only EOF
            # plus empty buffers establishes complete output capture.
            if channel.eof_received:
                break
            if channel.closed:
                raise EOFError("remote channel closed before output EOF; capture incomplete")
        if time.monotonic() >= deadline:
            channel.close()
            raise TimeoutError(f"remote command exceeded {effective_timeout:g} seconds")
        time.sleep(0.05)

    exit_status = channel.recv_exit_status()
    if exit_status < 0:
        raise EOFError("Remote command ended without an exit status; effects are uncertain")
    out = b"".join(out_chunks).decode("utf-8", "replace")
    err = b"".join(err_chunks).decode("utf-8", "replace")
    result = {
        "status": "ok",
        "exit_status": exit_status,
        "stdout": out,
        "stderr": err,
    }
    if truncated:
        result.update(status="output_limit", capture_complete=False, stdout_bytes=counts[0], stderr_bytes=counts[1])
    return result


def dispatch(client, payload, operation):
    if payload["action"] == "exec":
        operation.update(force=True, phase="remote_command")
        return exec_command(client, payload["command"], payload.get("timeout"), operation)
    return transfer(client, payload, operation, exec_command)


def main() -> int:
    return BridgeRuntime(ROOT, dispatch, connect, load_identity()).run()


if __name__ == "__main__":
    raise SystemExit(main())
