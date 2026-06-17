"""Record a visual-review decision for a prepared coastline domain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark a coastline-domain visual-review manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decision", choices=["pass", "fail", "needs_followup"], required=True)
    parser.add_argument("--reviewer", default="codex-agent")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.manifest)
    review = json.loads(path.read_text(encoding="utf-8"))
    review.update(
        {
            "status": "reviewed",
            "decision": args.decision,
            "reviewer": args.reviewer,
            "notes": args.notes,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
