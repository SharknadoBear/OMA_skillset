#!/usr/bin/env python3
"""Check whether rclone and an optional authenticated remote are available."""

from __future__ import annotations

import argparse
import json

from tpxo9v5.rclone import configured_remotes, find_rclone, normalize_remote, run_rclone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rclone", help="Optional rclone executable path or command name.")
    parser.add_argument("--remote", help="Optional configured remote name to require.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    executable = find_rclone(args.rclone)
    if executable is None:
        result = {
            "schema_version": "tpxo9v5_rclone_preflight_v1",
            "status": "install_required",
            "message": (
                "rclone is not installed on this equipment. Notify the user that rclone must be "
                "installed before automated Google Drive staging can continue."
            ),
        }
        exit_code = 2
    else:
        version_output = run_rclone(executable, ["version"], timeout=30).stdout.splitlines()
        result = {
            "schema_version": "tpxo9v5_rclone_preflight_v1",
            "status": "pass",
            "rclone_version": version_output[0].strip() if version_output else "unknown",
            "message": "rclone is installed.",
        }
        exit_code = 0
        if args.remote:
            remote = normalize_remote(args.remote)
            if remote not in configured_remotes(executable):
                result.update(
                    {
                        "status": "authorization_required",
                        "remote": remote,
                        "message": (
                            "rclone is installed, but the requested remote is not configured. "
                            "Notify the user that one-time Google Drive authorization is required."
                        ),
                    }
                )
                exit_code = 3
            else:
                result["remote"] = remote
                result["message"] = "rclone and the requested remote are available."

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
