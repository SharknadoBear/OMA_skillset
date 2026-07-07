from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .mesh_io import estimate_sponge, parse_2dm
except ImportError:  # pragma: no cover
    from mesh_io import estimate_sponge, parse_2dm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate initial FVCOM sponge parameters from a 2DM nodestring.")
    parser.add_argument("--mesh", required=True, help="Input SMS .2dm mesh")
    parser.add_argument("--nodestring", type=int, required=True, help="Nodestring id to inspect")
    parser.add_argument("--default-coeff", type=float, default=0.0025, help="Initial sponge damping coefficient")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mesh = parse_2dm(args.mesh)
    result = estimate_sponge(mesh, args.nodestring, default_coeff=args.default_coeff)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

