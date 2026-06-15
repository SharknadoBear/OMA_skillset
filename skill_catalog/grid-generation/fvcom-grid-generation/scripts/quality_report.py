"""Evaluate FVCOM/SMS quality metrics for a 2DM mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fvcom_grid_generation import QualityThresholds, evaluate_mesh_quality, read_2dm


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-check an FVCOM/SMS 2DM mesh.")
    parser.add_argument("mesh")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    mesh = read_2dm(args.mesh)
    open_boundary = mesh.open_boundaries[0] if mesh.open_boundaries else None
    quality = evaluate_mesh_quality(mesh.nodes, mesh.depths, mesh.triangles, open_boundary, QualityThresholds())
    serializable = {
        key: (value.tolist() if hasattr(value, "tolist") else value)
        for key, value in quality.items()
    }
    text = json.dumps(serializable, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
