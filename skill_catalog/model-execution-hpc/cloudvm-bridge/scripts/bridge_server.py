#!/usr/bin/env python3
"""Persistent local JSON bridge for cloud VM SSH/SFTP commands."""

from __future__ import annotations

import getpass
import json
import posixpath
import shlex
import shutil
import time
import traceback
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parent
COMMANDS = ROOT / "commands"
PROCESSED = COMMANDS / "processed"
RESULTS = ROOT / "results"
STATUS = ROOT / "bridge_status.txt"
HOST = "automodeldev01.pnl.gov"
USER = "huan111"


def ensure_dirs() -> None:
    COMMANDS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_status(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    STATUS.write_text(f"{timestamp} {message}\n", encoding="utf-8", newline="\n")


def connect() -> paramiko.SSHClient:
    print(f"Connecting to {USER}@{HOST}")
    write_status("waiting_for_password")
    password = getpass.getpass("Password: ")
    write_status("connecting")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=HOST,
        username=USER,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    print("Connected. Watching JSON command files.")
    write_status("connected_watching_commands")
    return client


def read_command(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_result(command_id: str, result: dict) -> None:
    allowed = {
        "id": command_id,
        "status": result.get("status", "error"),
        "exit_status": int(result.get("exit_status", 1) or 0),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }
    tmp = RESULTS / f"{command_id}.json.tmp"
    final = RESULTS / f"{command_id}.json"
    tmp.write_text(json.dumps(allowed, indent=2), encoding="utf-8", newline="\n")
    tmp.replace(final)


def move_processed(path: Path) -> None:
    dest = PROCESSED / path.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(path), str(dest))


def exec_command(client: paramiko.SSHClient, command: str, timeout: float | None) -> dict:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    del stdin
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    exit_status = stdout.channel.recv_exit_status()
    return {
        "status": "ok",
        "exit_status": exit_status,
        "stdout": out,
        "stderr": err,
    }


def expand_remote_path(client: paramiko.SSHClient, remote_path: str) -> str:
    if remote_path == "~" or remote_path.startswith("~/"):
        home = exec_command(client, 'printf "%s" "$HOME"', timeout=30)["stdout"]
        suffix = remote_path[2:] if remote_path.startswith("~/") else ""
        return posixpath.join(home, suffix) if suffix else home
    return remote_path


def upload_file(client: paramiko.SSHClient, local_path: str, remote_path: str) -> dict:
    local = Path(local_path).expanduser().resolve()
    if not local.is_file():
        raise FileNotFoundError(f"Local file not found: {local}")
    remote = expand_remote_path(client, remote_path)
    parent = posixpath.dirname(remote)
    if parent:
        mkdir = exec_command(client, f"mkdir -p {shlex.quote(parent)}", timeout=60)
        if mkdir["exit_status"] != 0:
            raise RuntimeError(mkdir["stderr"] or f"Could not create remote directory: {parent}")
    sftp = client.open_sftp()
    try:
        sftp.put(str(local), remote)
    finally:
        sftp.close()
    return {
        "status": "ok",
        "exit_status": 0,
        "stdout": f"uploaded {local} -> {remote}\n",
        "stderr": "",
    }


def download_file(client: paramiko.SSHClient, remote_path: str, local_path: str) -> dict:
    remote = expand_remote_path(client, remote_path)
    local = Path(local_path).expanduser().resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote, str(local))
    finally:
        sftp.close()
    return {
        "status": "ok",
        "exit_status": 0,
        "stdout": f"downloaded {remote} -> {local}\n",
        "stderr": "",
    }


def handle_command(client: paramiko.SSHClient, payload: dict) -> tuple[dict, bool]:
    action = payload.get("action")
    timeout = payload.get("timeout")
    if timeout is not None:
        timeout = float(timeout)
    if action == "exec":
        return exec_command(client, payload["command"], timeout), False
    if action == "upload":
        return upload_file(client, payload["local_path"], payload["remote_path"]), False
    if action == "download":
        return download_file(client, payload["remote_path"], payload["local_path"]), False
    if action == "stop":
        return {"status": "ok", "exit_status": 0, "stdout": "stopping bridge\n", "stderr": ""}, True
    raise ValueError(f"Unsupported action: {action}")


def main() -> int:
    ensure_dirs()
    write_status("starting")
    try:
        client = connect()
    except Exception as exc:
        write_status(f"connect_error:{exc.__class__.__name__}")
        raise
    should_stop = False
    try:
        while not should_stop:
            for path in sorted(COMMANDS.glob("*.json")):
                try:
                    payload = read_command(path)
                    command_id = payload.get("id") or path.stem
                    print(f"Processing {path.name}: {payload.get('action')}")
                    write_status(f"processing:{payload.get('action')}:{command_id}")
                    result, should_stop = handle_command(client, payload)
                except Exception as exc:
                    command_id = path.stem
                    result = {
                        "status": "error",
                        "exit_status": 1,
                        "stdout": "",
                        "stderr": f"{exc}\n{traceback.format_exc()}",
                    }
                write_result(command_id, result)
                move_processed(path)
                write_status(f"processed:{command_id}")
                if should_stop:
                    break
            time.sleep(0.25)
    finally:
        client.close()
        write_status("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
