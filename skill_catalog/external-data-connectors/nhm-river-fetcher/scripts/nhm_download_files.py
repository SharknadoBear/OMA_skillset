#!/usr/bin/env python3
"""Download selected files from an NHM ScienceBase manifest."""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "selected"}


def read_manifest(path: Path) -> List[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def selected_rows(rows: Iterable[dict], include_unselected: bool = False) -> List[dict]:
    return [row for row in rows if include_unselected or truthy(row.get("selected"))]


def safe_name(name: str) -> str:
    return Path(name).name.replace("\\", "_").replace("/", "_")


def download_one(url: str, dest: Path, expected_size: int | None, retries: int = 3) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    if expected_size and existing == expected_size:
        return "skipped-complete"

    for attempt in range(1, retries + 1):
        headers = {"User-Agent": "nhm-river-fetcher/1.0"}
        mode = "wb"
        if existing and (not expected_size or existing < expected_size):
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if mode == "ab" and getattr(response, "status", None) != 206:
                    mode = "wb"
                    existing = 0
                with dest.open(mode + "") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            final_size = dest.stat().st_size
            if expected_size and final_size != expected_size:
                existing = final_size
                raise IOError(f"size mismatch: got {final_size}, expected {expected_size}")
            return "downloaded"
        except (urllib.error.URLError, OSError, IOError) as exc:
            if attempt == retries:
                raise
            print(f"retry {attempt}/{retries} for {dest.name}: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)
            existing = dest.stat().st_size if dest.exists() else 0

    return "failed"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--include-unselected", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-large", action="store_true", help="Allow selected total above --large-threshold-gb.")
    parser.add_argument("--large-threshold-gb", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    rows = selected_rows(read_manifest(args.manifest), args.include_unselected)
    total = sum(int(float(row.get("size_bytes") or 0)) for row in rows)
    if total > args.large_threshold_gb * 1e9 and not args.allow_large:
        print(
            f"selected total {total / 1e9:.3f} GB exceeds {args.large_threshold_gb:.3f} GB; "
            "rerun with --allow-large after storage preflight",
            file=sys.stderr,
        )
        return 2

    print(f"selected_files={len(rows)} selected_gb={total / 1e9:.6f}")
    for row in rows:
        url = row.get("download_uri") or row.get("url")
        if not url:
            print(f"missing URL for {row.get('file_name')}", file=sys.stderr)
            return 1
        dest = args.dest_dir / safe_name(row.get("file_name") or "download.bin")
        expected_size = int(float(row.get("size_bytes") or 0)) or None
        if args.dry_run:
            print(f"DRY {dest} <- {url}")
            continue
        status = download_one(url, dest, expected_size, args.retries)
        print(f"{status}: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
