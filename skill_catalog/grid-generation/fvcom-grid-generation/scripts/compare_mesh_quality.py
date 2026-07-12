#!/usr/bin/env python3
"""Compare any before/after FVCOM mesh-quality JSON documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.comparison import compare_quality_documents, write_quality_comparison_plot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot")
    parser.add_argument("--title", default="FVCOM mesh post-generation quality comparison")
    args = parser.parse_args()
    before_path = Path(args.before).resolve()
    after_path = Path(args.after).resolve()
    before = json.loads(before_path.read_text(encoding="utf-8-sig"))
    after = json.loads(after_path.read_text(encoding="utf-8-sig"))
    comparison = compare_quality_documents(before, after)
    comparison["inputs"] = {
        "before": str(before_path),
        "before_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
        "after": str(after_path),
        "after_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    if args.plot:
        write_quality_comparison_plot(args.plot, before, after, args.title)
    print(json.dumps({"output": str(output), "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
