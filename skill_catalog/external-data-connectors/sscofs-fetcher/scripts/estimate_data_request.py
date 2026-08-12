#!/usr/bin/env python3
"""Plan an SSCOFS request and write the standard download estimate artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .sscofs_fetcher import plan_request, write_json_atomic
except ImportError:  # Support direct execution from the scripts directory.
    from sscofs_fetcher import plan_request, write_json_atomic


DEFAULT_SKILL_NAME = "sscofs-fetcher"
DEFAULT_OUTPUT_NAME = "download_estimate.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Path to an sscofs_request_v2 or migratable v1 JSON file.")
    parser.add_argument(
        "--run-dir",
        default=".",
        help="Run/cache directory used for free-space routing and the default output path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Estimate JSON path; defaults to <run-dir>/download_estimate.json.",
    )
    parser.add_argument(
        "--skill-name",
        default=DEFAULT_SKILL_NAME,
        help="Compatibility metadata override for external-data connector orchestration.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Compatibility run identifier used in estimate metadata and Kestrel routing.",
    )
    return parser


def _apply_compatibility_metadata(
    estimate: dict[str, Any], *, skill_name: str, run_id: str | None
) -> dict[str, Any]:
    """Add orchestration metadata without replacing planner-owned evidence."""

    result = dict(estimate)
    result["skill_name"] = skill_name
    if run_id:
        result["run_id"] = run_id
        scratch = f"/scratch/yhuang168/oma_external_data_connectors/{skill_name}/{run_id}"
        result["kestrel_scratch_path"] = scratch
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    # Preserve the caller's original schema/version for planner migration
    # lineage; plan_request performs the authoritative normalization.
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    # Pass run_dir so the canonical planner performs its exact local-free-space
    # check and writes the conventional artifact used by the main CLI.
    estimate = plan_request(request, run_dir=run_dir)
    estimate = _apply_compatibility_metadata(
        estimate,
        skill_name=args.skill_name,
        run_id=args.run_id,
    )

    output = Path(args.output) if args.output else run_dir / DEFAULT_OUTPUT_NAME
    write_json_atomic(output, estimate)
    print(json.dumps(estimate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
