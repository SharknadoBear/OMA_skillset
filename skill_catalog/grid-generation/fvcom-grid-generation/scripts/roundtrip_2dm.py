"""Roundtrip an SMS 2DM file through the FVCOM reader/writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fvcom_grid_generation import read_2dm, write_2dm


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and rewrite a 2DM mesh.")
    parser.add_argument("input_2dm")
    parser.add_argument("--output-2dm", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    mesh = read_2dm(args.input_2dm)
    open_boundary = mesh.open_boundaries[0] if mesh.open_boundaries else None
    write_2dm(args.output_2dm, mesh.nodes, mesh.depths, mesh.triangles, open_boundary, mesh.name)
    reread = read_2dm(args.output_2dm)
    summary = {
        "input": str(args.input_2dm),
        "output": str(args.output_2dm),
        "nodes": int(len(reread.nodes)),
        "triangles": int(len(reread.triangles)),
        "open_boundaries": int(len(reread.open_boundaries)),
        "first_open_boundary_nodes": int(len(reread.open_boundaries[0])) if reread.open_boundaries else 0,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary_json:
        Path(args.summary_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
