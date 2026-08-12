#!/usr/bin/env python3
"""Persistent local JSON bridge for read-only Constance SSH commands."""

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import shutil
import socket
import time
import traceback
from datetime import datetime
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parent
IDENTITY = ROOT / "bridge_identity.json"
COMMANDS = ROOT / "commands"
PROCESSED = COMMANDS / "processed"
RESULTS = ROOT / "results"
HOST = "constance.pnl.gov"
USER = "huan111"
EXPECTED_ED25519_SHA256 = "SHA256:tb23nJucub3vtSE3254A7U1AVajet/ITL3eiTE6zUtE"
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024


def ensure_dirs() -> None:
    COMMANDS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


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


def sha256_fingerprint(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


class StrictConstanceHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        fingerprint = sha256_fingerprint(key)
        if hostname == HOST and key.get_name() == "ssh-ed25519" and fingerprint == EXPECTED_ED25519_SHA256:
            client.get_host_keys().add(hostname, key.get_name(), key)
            return
        raise paramiko.SSHException(
            f"Unexpected host key for {hostname}: {key.get_name()} {fingerprint}"
        )


def connect() -> paramiko.SSHClient:
    print_identity(load_identity())
    print(f"Connecting to {USER}@{HOST}")
    print("Enter Constance password/MFA only in this window.")
    password = getpass.getpass("Constance password/MFA: ")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(StrictConstanceHostKeyPolicy())
    sock = socket.create_connection((HOST, 22), timeout=30)
    try:
        client.connect(
            hostname=HOST,
            username=USER,
            password=password,
            sock=sock,
            look_for_keys=True,
            allow_agent=True,
            timeout=30,
            auth_timeout=60,
            banner_timeout=30,
        )
    except Exception:
        sock.close()
        raise
    print("Connected. Watching JSON command files.")
    return client


def read_command(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_result(command_id: str, result: dict) -> None:
    result = {"id": command_id, **result}
    tmp = RESULTS / f"{command_id}.json.tmp"
    final = RESULTS / f"{command_id}.json"
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    tmp.replace(final)


def move_processed(path: Path) -> None:
    dest = PROCESSED / path.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(path), str(dest))


def exec_command(client: paramiko.SSHClient, command: str, timeout: float | None) -> dict:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    stdin.close()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    exit_status = stdout.channel.recv_exit_status()
    return {"status": "ok", "exit_status": exit_status, "stdout": out, "stderr": err}


def download_file(client: paramiko.SSHClient, remote_path: str, local_path: str) -> dict:
    local = Path(local_path).expanduser().resolve()
    sftp = client.open_sftp()
    try:
        stat = sftp.stat(remote_path)
        if stat.st_size > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"Refusing to download {stat.st_size} bytes; limit is {MAX_DOWNLOAD_BYTES} bytes"
            )
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(local))
    finally:
        sftp.close()
    return {
        "status": "ok",
        "exit_status": 0,
        "stdout": f"downloaded {remote_path} -> {local}\n",
        "stderr": "",
    }


def handle_command(client: paramiko.SSHClient, payload: dict) -> tuple[dict, bool]:
    action = payload.get("action")
    timeout = payload.get("timeout")
    if timeout is not None:
        timeout = float(timeout)
    if action == "exec":
        return exec_command(client, payload["command"], timeout), False
    if action == "download":
        return download_file(client, payload["remote_path"], payload["local_path"]), False
    if action == "stop":
        return {"status": "ok", "exit_status": 0, "stdout": "stopping bridge\n", "stderr": ""}, True
    raise ValueError(f"Unsupported action: {action}")


def main() -> int:
    ensure_dirs()
    client = connect()
    should_stop = False
    try:
        while not should_stop:
            for path in sorted(COMMANDS.glob("*.json")):
                try:
                    payload = read_command(path)
                    command_id = payload.get("id") or path.stem
                    print(f"Processing {path.name}: {payload.get('action')}")
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
                if should_stop:
                    break
            time.sleep(0.25)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
