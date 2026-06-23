#!/usr/bin/env python3
"""List or selectively extract members from a ZIP archive."""

import argparse
import re
import zipfile
from pathlib import Path
from typing import List, Optional


def wanted_members(names: List[str], explicit: List[str], pattern: str | None) -> List[str]:
    wanted = set(explicit)
    if pattern:
        regex = re.compile(pattern)
        wanted.update(name for name in names if regex.search(name))
    return [name for name in names if name in wanted]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, dest="zip_path")
    parser.add_argument("--list", action="store_true", dest="list_members")
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--member-regex")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    with zipfile.ZipFile(args.zip_path) as archive:
        infos = archive.infolist()
        if args.list_members:
            for info in infos:
                print(f"{info.file_size}\t{info.filename}")
            return 0

        names = [info.filename for info in infos]
        selected = wanted_members(names, args.member, args.member_regex)
        if not selected:
            raise SystemExit("no matching ZIP members")
        if not args.out_dir:
            raise SystemExit("--out-dir is required for extraction")

        for name in selected:
            print(f"extract {name} -> {args.out_dir}")
            if not args.dry_run:
                archive.extract(name, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
