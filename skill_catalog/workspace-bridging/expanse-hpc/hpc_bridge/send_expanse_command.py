#!/usr/bin/env python3
"""Submit a JSON command to the local Expanse bridge and wait for the result."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IDENTITY = ROOT / "bridge_identity.json"
COMMANDS = ROOT / "commands"
RESULTS = ROOT / "results"


def normalize_purpose(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_project_root(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def build_purpose_key(purpose: str, project_root: str) -> str:
    payload = f"{normalize_purpose(purpose)}\n{normalize_project_root(project_root)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_identity() -> dict:
    if not IDENTITY.exists():
        raise SystemExit(
            f"Missing {IDENTITY}. Create a named bridge session with make_bridge_session.py first."
        )
    try:
        return json.loads(IDENTITY.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid bridge identity file: {IDENTITY}: {exc}") from exc


def print_identity() -> int:
    print(json.dumps(read_identity(), indent=2))
    return 0


def validate_identity(args: argparse.Namespace) -> None:
    has_purpose_pair = bool(args.purpose) and bool(args.project_root)
    if bool(args.purpose) != bool(args.project_root):
        raise SystemExit("--purpose and --project-root must be provided together")
    if not args.bridge_name and not has_purpose_pair:
        raise SystemExit("Provide --bridge-name or both --purpose and --project-root before queueing a command")

    identity = read_identity()
    if args.bridge_name and identity.get("bridge_name") != args.bridge_name:
        raise SystemExit(
            "Bridge name mismatch; refusing to queue command.\n"
            + json.dumps(identity, indent=2)
        )
    if has_purpose_pair:
        expected = build_purpose_key(args.purpose, args.project_root)
        if identity.get("purpose_key") != expected:
            raise SystemExit(
                "Bridge purpose/project-root mismatch; refusing to queue command.\n"
                + json.dumps(identity, indent=2)
            )


def write_command(payload: dict) -> Path:
    COMMANDS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    command_id = uuid.uuid4().hex
    payload = {"id": command_id, **payload}
    final = COMMANDS / f"{command_id}.json"
    tmp = COMMANDS / f"{command_id}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    tmp.replace(final)
    return RESULTS / f"{command_id}.json"


def wait_result(result_path: Path, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if result_path.exists():
            try:
                return json.loads(result_path.read_text(encoding="utf-8-sig"))
            except (PermissionError, json.JSONDecodeError):
                time.sleep(0.25)
                continue
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {result_path}")


def print_result(result: dict) -> int:
    print(json.dumps(result, indent=2))
    status = result.get("status")
    if status == "ok":
        return int(result.get("exit_status", 0) or 0)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["identity", "exec", "upload", "download", "stop"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--bridge-name", help="Expected Japanese bridge name for this session.")
    parser.add_argument("--purpose", help="Expected bridge purpose text.")
    parser.add_argument("--project-root", help="Expected local project root for this bridge.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "identity":
        return print_identity()

    validate_identity(args)

    if args.action == "exec":
        if not args.args:
            raise SystemExit("exec requires a command string")
        payload = {"action": "exec", "command": " ".join(args.args)}
    elif args.action == "upload":
        if len(args.args) != 2:
            raise SystemExit("upload requires local_path remote_path")
        payload = {"action": "upload", "local_path": args.args[0], "remote_path": args.args[1]}
    elif args.action == "download":
        if len(args.args) != 2:
            raise SystemExit("download requires remote_path local_path")
        payload = {"action": "download", "remote_path": args.args[0], "local_path": args.args[1]}
    else:
        payload = {"action": "stop"}

    result_path = write_command(payload)
    result = wait_result(result_path, args.timeout)
    return print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
