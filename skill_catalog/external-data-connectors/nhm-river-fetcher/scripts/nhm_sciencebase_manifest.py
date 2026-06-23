#!/usr/bin/env python3
"""Build a ScienceBase file manifest for Alaska NHM/NHM-PRMS items."""

import argparse
import csv
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCIENCEBASE_URL = "https://www.sciencebase.gov/catalog/item/{item_id}?format=json"

DEFAULT_ITEMS: Dict[str, str] = {
    "parent": "64a84670d34e70357a27dd86",
    "output-data": "6723ad59d34e4f57573e8c57",
    "input-run-files": "64c1cbebd34e70357a32a300",
    "gage-simulated-flow": "65cbb0f9d34ef4b119cb3780",
    "huc12-aggregations": "667b23c8d34e6151c9d6be10",
    "geofabric": "6644f81ed34e1955f5a42db4",
}

FIELDNAMES = [
    "profile",
    "selected",
    "selected_reason",
    "item_role",
    "item_id",
    "item_title",
    "file_name",
    "file_title",
    "size_bytes",
    "size_gb",
    "size_gib",
    "content_type",
    "url",
    "download_uri",
]


def fetch_json(item_id: str) -> dict:
    req = urllib.request.Request(
        SCIENCEBASE_URL.format(item_id=item_id),
        headers={"User-Agent": "nhm-river-fetcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_item_arg(raw: str) -> Tuple[str, str]:
    if "=" in raw:
        role, item_id = raw.split("=", 1)
        return role.strip(), item_id.strip()
    return raw.strip(), raw.strip()


def item_roles_for_profile(profile: str) -> Dict[str, str]:
    if profile == "metadata-smoke":
        return {
            "parent": DEFAULT_ITEMS["parent"],
            "output-data": DEFAULT_ITEMS["output-data"],
        }
    if profile == "geofabric-only":
        return {"geofabric": DEFAULT_ITEMS["geofabric"]}
    if profile == "byPOIobs-seg-outflow":
        return {
            "parent": DEFAULT_ITEMS["parent"],
            "output-data": DEFAULT_ITEMS["output-data"],
            "geofabric": DEFAULT_ITEMS["geofabric"],
        }
    if profile == "molly-all-listed":
        return dict(DEFAULT_ITEMS)
    raise ValueError(f"unknown profile: {profile}")


def selection_for(profile: str, role: str, file_name: str, size_bytes: int) -> Tuple[bool, str]:
    name = file_name.lower()

    if profile == "metadata-smoke":
        wanted = {
            "ak_output_variables_data_dictionary.csv",
            "table_1_ak_gages.csv",
            "ak_nhm_prms_parent_page.xml",
            "ak_bypoiobs_nhm_prms_data_release.out",
            "4_ak_output_data.xml",
        }
        if name in wanted and size_bytes < 5_000_000:
            return True, "metadata-smoke tiny metadata"
        return False, ""

    if profile == "geofabric-only":
        if role == "geofabric":
            return True, "geofabric item file"
        return False, ""

    if profile == "byPOIobs-seg-outflow":
        if role == "output-data" and name in {
            "ak_bypoiobs_netcdf.zip",
            "ak_bypoiobs_nhm_prms_data_release.out",
            "4_ak_output_data.xml",
        }:
            return True, "byPOIobs output or metadata"
        if role == "geofabric":
            return True, "geofabric required for segment geometry"
        if role == "parent" and name in {
            "ak_output_variables_data_dictionary.csv",
            "table_1_ak_gages.csv",
            "ak_nhm_prms_parent_page.xml",
        }:
            return True, "supporting parent metadata"
        return False, ""

    if profile == "molly-all-listed":
        return True, "molly all-listed profile"

    return False, ""


def file_rows(profile: str, role: str, item_id: str, item_json: dict) -> List[dict]:
    rows: List[dict] = []
    for file_info in item_json.get("files") or []:
        size_bytes = int(file_info.get("size") or 0)
        selected, reason = selection_for(profile, role, file_info.get("name", ""), size_bytes)
        rows.append(
            {
                "profile": profile,
                "selected": "true" if selected else "false",
                "selected_reason": reason,
                "item_role": role,
                "item_id": item_id,
                "item_title": item_json.get("title", ""),
                "file_name": file_info.get("name", ""),
                "file_title": file_info.get("title", ""),
                "size_bytes": str(size_bytes),
                "size_gb": f"{size_bytes / 1e9:.6f}",
                "size_gib": f"{size_bytes / (1024 ** 3):.6f}",
                "content_type": file_info.get("contentType", ""),
                "url": file_info.get("url", ""),
                "download_uri": file_info.get("downloadUri") or file_info.get("url", ""),
            }
        )
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def apply_name_regex(rows: List[dict], pattern: Optional[str]) -> List[dict]:
    if not pattern:
        return rows
    regex = re.compile(pattern, re.IGNORECASE)
    updated: List[dict] = []
    for row in rows:
        row = dict(row)
        if regex.search(row["file_name"]):
            row["selected"] = "true"
            row["selected_reason"] = f"matched --select-regex {pattern}"
        updated.append(row)
    return updated


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="metadata-smoke",
        choices=["metadata-smoke", "geofabric-only", "byPOIobs-seg-outflow", "molly-all-listed"],
    )
    parser.add_argument(
        "--item",
        action="append",
        help="Override queried items as role=item_id or item_id. May be repeated.",
    )
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/nhm_prms_ak/metadata"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/nhm_prms_ak/tables/manifest.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/nhm_prms_ak/tables/manifest.json"))
    parser.add_argument("--select-regex", help="Additional case-insensitive filename regex to mark selected.")
    args = parser.parse_args(argv)

    items = item_roles_for_profile(args.profile)
    if args.item:
        items = dict(parse_item_arg(value) for value in args.item)

    all_rows: List[dict] = []
    for role, item_id in items.items():
        item_json = fetch_json(item_id)
        metadata_path = args.metadata_dir / f"{role}_{item_id}.json"
        save_json(metadata_path, item_json)
        all_rows.extend(file_rows(args.profile, role, item_id, item_json))

    all_rows = apply_name_regex(all_rows, args.select_regex)
    save_csv(args.out_csv, all_rows)
    save_json(args.out_json, all_rows)

    selected_bytes = sum(int(row["size_bytes"]) for row in all_rows if row["selected"] == "true")
    print(f"rows={len(all_rows)} selected_rows={sum(row['selected'] == 'true' for row in all_rows)}")
    print(f"selected_bytes={selected_bytes} selected_gb={selected_bytes / 1e9:.3f}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
