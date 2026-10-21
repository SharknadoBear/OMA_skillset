#!/usr/bin/env python3
"""Submit a JSON command to the local Kestrel bridge and wait for the result."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bridge_runtime import atomic_json, positive, read_json, REQUEST_ID, MAX_REQUEST_BYTES


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
    payload = {**payload, "id": command_id}
    if len(json.dumps(payload, indent=2).encode("utf-8")) > MAX_REQUEST_BYTES:
        raise SystemExit("Request exceeds 64 KiB; upload a script and use a short command")
    final = COMMANDS / f"{command_id}.json"
    atomic_json(final, payload)
    return RESULTS / f"{command_id}.json"


def wait_result(result_path: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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


def snapshot(request_id=None):
    status_file = ROOT / "bridge_status.json"
    health = read_json(status_file) if status_file.exists() else {"state": "not_started"}
    if health.get("heartbeat_at"):
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(health["heartbeat_at"])).total_seconds()
        health["heartbeat_age_seconds"] = round(age, 1)
        health["heartbeat_stale"] = age > 10
    if request_id is None:
        return {"status": "ok", "bridge": health}
    if not REQUEST_ID.fullmatch(request_id):
        raise SystemExit("Invalid request ID")
    for directory in ("results", "progress", "inflight", "commands"):
        path = ROOT / directory / f"{request_id}.json"
        if path.exists():
            data = read_json(path)
            if directory in {"inflight", "commands"}:
                data = {"id": request_id, "action": data.get("action"),
                        "state": "claimed" if directory == "inflight" else "queued"}
            return {"status": "ok", "bridge": health, "request": data, "location": directory}
    return {"status": "error", "bridge": health, "stderr": "Unknown request ID"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["identity", "status", "result", "exec", "upload", "download", "cancel", "reconnect", "stop"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--wait", type=float, default=30, help="Caller wait only; expiry does not cancel or resubmit the request.")
    parser.add_argument("--no-wait", action="store_true", help="Return the durable request ID immediately.")
    parser.add_argument("--stall-timeout", type=float, default=120)
    parser.add_argument("--resume", action="store_true", help="Verify an existing partial prefix before resuming a transfer.")
    parser.add_argument("--overwrite", action="store_true", help="Allow atomic replacement of a different destination file.")
    parser.add_argument("--bridge-name", help="Expected Japanese bridge name for this session.")
    parser.add_argument("--purpose", help="Expected bridge purpose text.")
    parser.add_argument("--project-root", help="Expected local project root for this bridge.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.action == "identity":
        return print_identity()

    validate_identity(args)

    if args.action in {"status", "result"}:
        if len(args.args) > 1 or args.action == "result" and len(args.args) != 1:
            raise SystemExit("status takes an optional request ID; result requires one")
        if args.action == "result":
            request_id = args.args[0]
            if not REQUEST_ID.fullmatch(request_id):
                raise SystemExit("Invalid request ID")
            path = RESULTS / f"{request_id}.json"
            if path.exists():
                return print_result(read_json(path))
        return print_result(snapshot(args.args[0] if args.args else None))

    positive(args.timeout, "timeout")
    positive(args.wait, "wait")
    positive(args.stall_timeout, "stall timeout")

    if args.action == "exec":
        if not args.args:
            raise SystemExit("exec requires a command string")
        if args.timeout <= 0:
            raise SystemExit("--timeout must be positive")
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
    elif args.action == "cancel":
        if len(args.args) != 1 or not REQUEST_ID.fullmatch(args.args[0]):
            raise SystemExit("cancel requires the original request ID")
        payload = {"action": "cancel", "request_id": args.args[0]}
    else:
        if args.args:
            raise SystemExit(f"{args.action} takes no positional arguments")
        payload = {"action": args.action}

    payload.update(timeout=args.timeout, stall_timeout=args.stall_timeout,
                   submitted_at=datetime.now(timezone.utc).isoformat(),
                   expires_at=time.time() + args.timeout)
    if args.action in {"upload", "download"}:
        payload.update(resume=args.resume, overwrite=args.overwrite)

    result_path = write_command(payload)
    receipt = {"status": "submitted", "id": result_path.stem, "result_path": str(result_path),
               "note": "Retain this ID. Inspect status/result before any retry; caller wait expiry is not remote failure."}
    if args.no_wait:
        print(json.dumps(receipt, indent=2))
        return 0
    print(json.dumps(receipt), file=sys.stderr, flush=True)
    try:
        result = wait_result(result_path, args.wait)
    except TimeoutError:
        print(json.dumps({**receipt, "status": "pending"}, indent=2))
        return 2
    return print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
