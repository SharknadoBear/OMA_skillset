#!/usr/bin/env python3
"""Inventory or stage exact TPXO9v5 files from an authorized Drive folder with rclone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from tpxo9v5.rclone import (
    configured_remotes,
    find_rclone,
    normalize_remote,
    parse_drive_folder,
    run_rclone,
)

VALID_FIELDS = {"elevation", "transport"}


def parse_fields(value: str) -> list[str]:
    fields = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(fields) - VALID_FIELDS)
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported field(s): {', '.join(invalid)}")
    if not fields:
        raise argparse.ArgumentTypeError("At least one field is required.")
    return list(dict.fromkeys(fields))


def infer_role(name: str) -> str | None:
    lowered = name.lower()
    if "grid" in lowered:
        return "grid"
    if lowered.startswith("h_") or "elevation" in lowered:
        return "elevation"
    if lowered.startswith("u_") or "transport" in lowered or "current" in lowered:
        return "transport"
    return None


def _rclone_context(args: argparse.Namespace) -> tuple[Path, str, str]:
    executable = find_rclone(args.rclone)
    if executable is None:
        raise RuntimeError(
            "rclone is not installed on this equipment. Notify the user that rclone must be installed "
            "before automated Google Drive staging can continue."
        )
    remote = normalize_remote(args.remote)
    if remote not in configured_remotes(executable):
        raise RuntimeError(
            f"rclone remote {remote!r} is not configured. Notify the user that one-time read-only "
            "Google Drive authorization is required."
        )
    return executable, remote, parse_drive_folder(args.drive_folder)


def _remote_items(executable: Path, remote: str, folder_id: str, timeout: int) -> list[dict[str, Any]]:
    completed = run_rclone(
        executable,
        [
            "lsjson",
            f"{remote}:",
            "--drive-root-folder-id",
            folder_id,
            "--files-only",
            "--max-depth",
            "1",
            "--hash",
        ],
        timeout=timeout,
        redact=(folder_id,),
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise TypeError("Unexpected rclone lsjson response; expected a file list.")
    return [item for item in payload if isinstance(item, dict) and not item.get("IsDir")]


def _select(items: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    required = {"grid", *fields}
    selected: dict[str, dict[str, Any]] = {}
    for item in items:
        name = str(item.get("Name") or item.get("Path") or "")
        if not name or Path(name).name != name:
            continue
        role = infer_role(name)
        if role not in required:
            continue
        if role in selected:
            raise ValueError(f"Remote folder has multiple candidates for {role!r}.")
        selected[role] = item
    missing = sorted(required - selected.keys())
    if missing:
        raise FileNotFoundError(f"Remote folder is missing required role(s): {', '.join(missing)}")
    return [dict(selected[role], Role=role) for role in sorted(selected)]


def _remote_md5(item: dict[str, Any]) -> str | None:
    hashes = item.get("Hashes")
    if not isinstance(hashes, dict):
        return None
    lookup = {str(key).lower(): str(value).lower() for key, value in hashes.items()}
    return lookup.get("md5")


def _public_file_record(item: dict[str, Any]) -> dict[str, Any]:
    size = int(item.get("Size", -1))
    if size < 0:
        raise ValueError(f"Remote size is unavailable for {item.get('Name', 'file')!r}.")
    record: dict[str, Any] = {
        "role": item["Role"],
        "name": str(item["Name"]),
        "size_bytes": size,
    }
    md5 = _remote_md5(item)
    if md5:
        record["md5"] = md5
    return record


def _inventory(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str, str, list[dict[str, Any]]]:
    executable, remote, folder_id = _rclone_context(args)
    selected = _select(_remote_items(executable, remote, folder_id, args.timeout), args.fields)
    files = [_public_file_record(item) for item in selected]
    result = {
        "schema_version": "tpxo9v5_rclone_inventory_v1",
        "status": "pass",
        "requested_fields": args.fields,
        "required_roles": sorted({"grid", *args.fields}),
        "remote": remote,
        "remote_folder_fingerprint": hashlib.sha256(folder_id.encode("ascii")).hexdigest()[:16],
        "files": files,
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
    }
    return result, executable, remote, folder_id, selected


def _write_json(path: str | Path, result: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def _local_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_manifest_matches(path: str | Path, inventory: dict[str, Any]) -> None:
    expected = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    keys = ("requested_fields", "required_roles", "remote", "remote_folder_fingerprint", "files", "total_bytes")
    mismatches = [key for key in keys if expected.get(key) != inventory.get(key)]
    if mismatches:
        raise ValueError(f"Remote inventory changed after estimation: {', '.join(mismatches)}")


def _download(args: argparse.Namespace) -> dict[str, Any]:
    inventory, executable, remote, folder_id, selected = _inventory(args)
    _assert_manifest_matches(args.manifest, inventory)
    staging = Path(args.staging_dir).expanduser().resolve()
    staging.mkdir(parents=True, exist_ok=True)
    required = int(inventory["total_bytes"])
    free = int(shutil.disk_usage(staging).free)
    if free <= 4 * required:
        raise OSError(f"Insufficient working space: {free} free bytes; more than {4 * required} required.")

    staged: list[dict[str, Any]] = []
    for item in selected:
        record = _public_file_record(item)
        destination = staging / record["name"]
        if destination.parent != staging:
            raise ValueError(f"Unsafe remote basename: {record['name']!r}")
        expected_size = int(record["size_bytes"])
        expected_md5 = record.get("md5")
        reused = destination.is_file() and destination.stat().st_size == expected_size
        local_md5: str | None = None
        if reused and expected_md5:
            local_md5 = _local_md5(destination)
            reused = local_md5 == expected_md5
        if destination.exists() and not reused:
            raise FileExistsError(f"Refusing to overwrite mismatched staged file: {destination.name}")
        if not reused:
            partial = staging / f".{destination.name}.rclone-partial"
            if partial.exists():
                partial.unlink()
            run_rclone(
                executable,
                [
                    "copyto",
                    f"{remote}:{record['name']}",
                    str(partial),
                    "--drive-root-folder-id",
                    folder_id,
                    "--transfers",
                    "1",
                    "--retries",
                    "3",
                    "--low-level-retries",
                    "10",
                    "--stats",
                    "0",
                ],
                timeout=args.timeout,
                redact=(folder_id,),
            )
            if not partial.is_file() or partial.stat().st_size != expected_size:
                partial.unlink(missing_ok=True)
                raise OSError(f"Staged size verification failed for {destination.name}.")
            if expected_md5:
                local_md5 = _local_md5(partial)
                if local_md5 != expected_md5:
                    partial.unlink(missing_ok=True)
                    raise OSError(f"Staged MD5 verification failed for {destination.name}.")
            os.replace(partial, destination)
        staged.append(
            {
                **record,
                "verified_size": True,
                "verified_md5": bool(expected_md5),
                "reused": reused,
            }
        )
    return {
        "schema_version": "tpxo9v5_rclone_staging_v1",
        "status": "pass",
        "requested_fields": args.fields,
        "remote": remote,
        "remote_folder_fingerprint": inventory["remote_folder_fingerprint"],
        "staged_files": staged,
        "staged_bytes": sum(int(item["size_bytes"]) for item in staged),
        "cleanup_owner": "extract_tpxo9v5.py --staging-dir",
    }


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", required=True, help="Configured read-only rclone remote name.")
    parser.add_argument("--drive-folder", required=True, help="Runtime Google Drive folder URL or ID.")
    parser.add_argument("--fields", default="elevation", type=parse_fields, help="Elevation and/or transport.")
    parser.add_argument("--rclone", help="Optional rclone executable path or command name.")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-command timeout in seconds.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    inventory_parser = subparsers.add_parser("inventory", help="List exact required files without downloading.")
    _add_source_arguments(inventory_parser)
    inventory_parser.add_argument("--output", required=True, help="Size-only inventory JSON path.")
    download_parser = subparsers.add_parser("download", help="Download the files from a reviewed inventory.")
    _add_source_arguments(download_parser)
    download_parser.add_argument("--manifest", required=True, help="Inventory JSON used by the estimate gate.")
    download_parser.add_argument("--staging-dir", required=True, help="Dedicated raw staging directory.")
    download_parser.add_argument("--output", required=True, help="Staging report JSON path.")
    args = parser.parse_args()

    if args.action == "inventory":
        result, *_ = _inventory(args)
    else:
        result = _download(args)
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
