"""Record an agent visual review decision for a coastline diagnostic map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cusp_coastline.visual_qa import record_visual_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to *_visual_review.json.")
    parser.add_argument("--decision", required=True, choices=("pass", "fail", "needs_followup"))
    parser.add_argument("--reviewer", default="codex-agent")
    parser.add_argument("--notes", required=True)
    parser.add_argument(
        "--fail-reason",
        action="append",
        default=[],
        help="Repeatable reason for fail or follow-up decisions.",
    )
    args = parser.parse_args()

    manifest = record_visual_review(
        args.manifest,
        decision=args.decision,
        reviewer=args.reviewer,
        notes=args.notes,
        fail_reasons=args.fail_reason,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0 if args.decision == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
