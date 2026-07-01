#!/usr/bin/env python3
"""Check GSHHS clipped vector product health and write a compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gshhs_coastline.quality import summarize_product  # noqa: E402
from gshhs_coastline.sources import write_json  # noqa: E402


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _find_gpkg(run_dir: Path, request: dict[str, Any]) -> Path | None:
    outputs = request.get("outputs") if isinstance(request, dict) else None
    if isinstance(outputs, dict) and outputs.get("gpkg"):
        candidate = Path(outputs["gpkg"])
        if candidate.exists():
            return candidate
    candidates = sorted(run_dir.glob("*_gshhs_land.gpkg"))
    return candidates[0] if candidates else None


def _plot_health(gpkg: Path, plots_dir: Path) -> str | None:
    try:
        land = gpd.read_file(gpkg, layer="land_polygons")
        coastline = gpd.read_file(gpkg, layer="coastline_lines")
    except Exception:
        return None
    if land.empty and coastline.empty:
        return None
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{gpkg.stem}_health.png"
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.set_facecolor("#d8edf7")
    if not land.empty:
        land.plot(ax=ax, facecolor="#eee6d6", edgecolor="#475569", linewidth=0.4)
    if not coastline.empty:
        coastline.plot(ax=ax, color="#111827", linewidth=0.7)
    ax.set_title(gpkg.name)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Optional fetch manifest JSON.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plots-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    request = _read_json(args.request)
    gpkg = _find_gpkg(run_dir, request)
    if gpkg is None:
        result = {
            "schema_version": "external_data_health_v1",
            "status": "fail",
            "run_dir": str(run_dir),
            "warnings": ["No *_gshhs_land.gpkg product was found."],
            "plots": [],
        }
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 1

    summary = summarize_product(gpkg)
    plot = _plot_health(gpkg, Path(args.plots_dir))
    result = {
        "schema_version": "external_data_health_v1",
        "status": summary["status"],
        "run_dir": str(run_dir),
        "gpkg": str(gpkg),
        "summary": summary,
        "warnings": summary["warnings"],
        "plots": [plot] if plot else [],
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] in {"pass", "needs_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
