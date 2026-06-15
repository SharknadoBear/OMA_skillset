#!/usr/bin/env python3
"""Submit a JSON command to the local cloud VM bridge and wait for the result."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = ROOT / "commands"
RESULTS = ROOT / "results"


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
    parser.add_argument("action", choices=["exec", "upload", "download", "stop"])
    parser.add_argument("args", nargs="*")
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Remote exec timeout in seconds, and base wait timeout for all actions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "exec":
        if not args.args:
            raise SystemExit("exec requires a command string")
        payload = {
            "action": "exec",
            "command": " ".join(args.args),
            "timeout": args.timeout,
        }
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
    wait_timeout = args.timeout + 30 if args.action == "exec" else args.timeout
    result = wait_result(result_path, wait_timeout)
    return print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
