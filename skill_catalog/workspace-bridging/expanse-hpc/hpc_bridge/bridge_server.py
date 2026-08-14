#!/usr/bin/env python3
"""Persistent local JSON bridge for Expanse SSH/SFTP commands."""

from __future__ import annotations

import getpass
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


def connection_from_identity(identity: dict) -> tuple[str, str]:
    target = str(identity.get("remote_target", "")).strip()
    if target.count("@") != 1:
        raise ValueError("bridge_identity.json remote_target must be username@hostname")
    user, host = target.split("@", 1)
    if not user or not host:
        raise ValueError("bridge_identity.json remote_target must be username@hostname")
    return user, host


def interactive_handler(title: str, instructions: str, prompts: list[tuple[str, bool]]) -> list[str]:
    """Answer server-driven password and TOTP prompts without echoing responses."""
    if title.strip():
        print(title.strip())
    if instructions.strip():
        print(instructions.strip())
    responses: list[str] = []
    for prompt, _echo in prompts:
        label = prompt.strip() or "Authentication response:"
        responses.append(getpass.getpass(f"{label} "))
    return responses


def try_agent_keys(transport: paramiko.Transport, user: str) -> bool:
    """Try keys already available through ssh-agent without reading key files."""
    agent = paramiko.Agent()
    try:
        for key in agent.get_keys():
            try:
                remaining = transport.auth_publickey(user, key)
            except paramiko.AuthenticationException:
                continue
            if transport.is_authenticated():
                return True
            if "keyboard-interactive" in remaining:
                return False
    finally:
        agent.close()
    return False


def authenticate(transport: paramiko.Transport, user: str) -> None:
    """Complete Expanse agent-or-password authentication followed by TOTP."""
    if try_agent_keys(transport, user):
        return

    try:
        remaining = transport.auth_interactive(user, interactive_handler)
    except paramiko.BadAuthenticationType as exc:
        if "password" not in exc.allowed_types:
            raise
        password = getpass.getpass("Password: ")
        try:
            remaining = transport.auth_password(user, password, fallback=False)
        finally:
            password = ""

        if not transport.is_authenticated():
            if "keyboard-interactive" not in remaining:
                raise paramiko.AuthenticationException(
                    "Password was accepted only partially, but no TOTP challenge was offered."
                )
            remaining = transport.auth_interactive(user, interactive_handler)

    if not transport.is_authenticated():
        raise paramiko.AuthenticationException(
            f"Expanse authentication incomplete; remaining methods: {remaining}"
        )


def connect() -> paramiko.SSHClient:
    identity = load_identity()
    print_identity(identity)
    user, host = connection_from_identity(identity)
    print(f"Connecting to {user}@{host}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    sock = socket.create_connection((host, 22), timeout=30)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=30)
    authenticate(transport, user)
    client._transport = transport  # Paramiko exposes no public setter for this path.
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
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    exit_status = stdout.channel.recv_exit_status()
    return {
        "status": "ok",
        "exit_status": exit_status,
        "stdout": out,
        "stderr": err,
    }


def upload_file(client: paramiko.SSHClient, local_path: str, remote_path: str) -> dict:
    local = Path(local_path).expanduser().resolve()
    if not local.is_file():
        raise FileNotFoundError(f"Local file not found: {local}")
    sftp = client.open_sftp()
    try:
        sftp.put(str(local), remote_path)
    finally:
        sftp.close()
    return {"status": "ok", "exit_status": 0, "stdout": f"uploaded {local} -> {remote_path}\n", "stderr": ""}


def download_file(client: paramiko.SSHClient, remote_path: str, local_path: str) -> dict:
    local = Path(local_path).expanduser().resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, str(local))
    finally:
        sftp.close()
    return {"status": "ok", "exit_status": 0, "stdout": f"downloaded {remote_path} -> {local}\n", "stderr": ""}


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
