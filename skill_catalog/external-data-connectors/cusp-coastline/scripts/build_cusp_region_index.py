"""Build a NOAA CUSP regional ZIP source index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cusp_coastline.progress import ProgressReporter  # noqa: E402
from cusp_coastline.sources import build_region_index, save_region_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output cusp_region_index.json path.")
    parser.add_argument("--no-head", action="store_true", help="Skip HTTP HEAD metadata checks.")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--client-timeout-s", type=float, default=0.0, help="0 means no hard client timeout.")
    parser.add_argument("--progress-jsonl", help="Progress JSONL path. Defaults next to output.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress text; JSONL is still written.")
    args = parser.parse_args()

    output = Path(args.output)
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else output.with_name("build_cusp_region_index_progress.jsonl")
    reporter = ProgressReporter(progress_path, heartbeat_seconds=args.heartbeat_seconds, quiet=args.quiet)
    with reporter.stage("cusp-index", "build CUSP region index", output=str(output)):
        index = build_region_index(include_head=not args.no_head, client_timeout_s=args.client_timeout_s, reporter=reporter)
        path = save_region_index(index, output)
    print(
        json.dumps(
            {
                "output": str(path),
                "regions": len(index["regions"]),
                "progress_jsonl": str(progress_path),
                "stage_elapsed_seconds": reporter.stage_elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
