#!/usr/bin/env python
"""Sample CUDEM, NBS BlueTopo, CRM, and ETOPO sources directly to FVCOM mesh nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cudem_bathy.mesh_compare import (  # noqa: E402
    anomaly_stats,
    read_2dm_nodes,
    sample_sources_to_mesh,
    selected_bathy_sources,
    source_counts,
    write_bathy_node_csv,
)
from cudem_bathy.sources import build_bathy_source_index, save_bathy_source_index  # noqa: E402


COLORS = {
    "cudem": "#0072b2",
    "nbs_bluetopo": "#cc79a7",
    "crm": "#009e73",
    "etopo": "#d55e00",
    "none": "#bdbdbd",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-2dm", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", default="mesh_bathy_sources")
    parser.add_argument("--index", required=True, help="Combined bathymetry source index JSON.")
    parser.add_argument(
        "--fallback-policy",
        default="cudem-nbs-crm-etopo",
        choices=("cudem-only", "cudem-crm", "cudem-crm-etopo", "cudem-nbs-crm-etopo"),
    )
    parser.add_argument(
        "--resolution-policy",
        default="source-priority",
        choices=("source-priority", "finest"),
        help=(
            "source-priority keeps CUDEM/NBS/CRM/ETOPO family order; finest "
            "lets the finest usable local native resolution win across sources."
        ),
    )
    parser.add_argument(
        "--max-nbs-sources",
        type=int,
        default=0,
        help=(
            "Optional smoke-test limiter for BlueTopo tiles. 0 means no limit. "
            "When set, keep the intersecting NBS tiles covering the most mesh nodes."
        ),
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.index)
    if args.rebuild_index or not index_path.exists():
        index = build_bathy_source_index()
        save_bathy_source_index(index, index_path)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh = read_2dm_nodes(args.mesh_2dm)
    sources = selected_bathy_sources(
        index_path,
        mesh.bbox,
        fallback_policy=args.fallback_policy,
        resolution_policy=args.resolution_policy,
    )
    source_limit_report = None
    if args.max_nbs_sources > 0:
        sources, source_limit_report = limit_nbs_sources_by_mesh_coverage(
            sources, mesh, max_nbs_sources=args.max_nbs_sources
        )
    result = sample_sources_to_mesh(
        mesh,
        sources,
        progress=not args.no_progress,
        resolution_policy=args.resolution_policy,
    )

    csv_path = write_bathy_node_csv(result, run_dir / f"{args.name}_node_bathy_sources.csv")
    source_map = plot_source_map(result, run_dir / f"{args.name}_source_map.png")
    hist_png = plot_anomaly_hist(result, run_dir / f"{args.name}_anomaly_hist.png")
    resolution_hist = plot_resolution_hist(result, run_dir / f"{args.name}_resolution_hist.png")
    uncertainty_map = plot_uncertainty_map(result, run_dir / f"{args.name}_nbs_uncertainty_map.png")
    summary = {
        "case": args.name,
        "mesh_2dm": args.mesh_2dm,
        "bbox_wsen": list(mesh.bbox),
        "n_nodes": int(mesh.lon.size),
        "fallback_policy": args.fallback_policy,
        "resolution_policy": args.resolution_policy,
        "nbs_source_limit": source_limit_report,
        "n_selected_sources": len(sources),
        "selected_sources": [source.to_dict() for source in sources],
        "source_counts": source_counts(result),
        "resolution_stats_m": resolution_stats(result["best_source_resolution_m"]),
        "nbs_uncertainty_stats_m": anomaly_stats(result["best_uncertainty_m"]),
        "anomaly_stats": anomaly_stats(result["depth_anomaly_m"]),
        "datum_warning": (
            "CUDEM, NBS BlueTopo, CRM, and ETOPO can use different vertical datums. "
            "Values were filled by source priority without vertical-datum harmonization."
        ),
        "warnings": [str(x) for x in result.get("warnings", [])],
        "outputs": {
            "node_csv": str(csv_path),
            "source_map": str(source_map),
            "anomaly_histogram": str(hist_png),
            "resolution_histogram": str(resolution_hist),
            "nbs_uncertainty_map": str(uncertainty_map),
            "summary_json": str(run_dir / f"{args.name}_summary.json"),
        },
    }
    summary_path = run_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def limit_nbs_sources_by_mesh_coverage(sources, mesh, *, max_nbs_sources: int):
    """Keep the BlueTopo tiles that cover the most mesh nodes for tractable smoke tests."""

    nbs = [source for source in sources if source.source_name == "nbs_bluetopo"]
    if len(nbs) <= max_nbs_sources:
        return sources, {
            "applied": False,
            "requested_max": max_nbs_sources,
            "available_nbs_sources": len(nbs),
            "kept_nbs_sources": len(nbs),
        }
    scored = []
    for source in nbs:
        mask = (
            (mesh.lon >= source.west)
            & (mesh.lon <= source.east)
            & (mesh.lat >= source.south)
            & (mesh.lat <= source.north)
        )
        scored.append((int(mask.sum()), source))
    keep = {source.name for _count, source in sorted(scored, key=lambda item: (-item[0], item[1].name))[:max_nbs_sources]}
    filtered = [source for source in sources if source.source_name != "nbs_bluetopo" or source.name in keep]
    return filtered, {
        "applied": True,
        "requested_max": max_nbs_sources,
        "available_nbs_sources": len(nbs),
        "kept_nbs_sources": len(keep),
        "dropped_nbs_sources": len(nbs) - len(keep),
        "selection_rule": "kept BlueTopo source bboxes covering the most mesh nodes",
    }


def plot_source_map(result: dict[str, np.ndarray], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = np.asarray(result["best_source"], dtype=object)
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    for source in ("none", "etopo", "crm", "nbs_bluetopo", "cudem"):
        mask = sources == source
        if source == "none":
            mask = sources == ""
        if not mask.any():
            continue
        ax.scatter(
            result["lon"][mask],
            result["lat"][mask],
            s=0.35,
            c=COLORS[source],
            label=source.upper(),
            linewidths=0,
            rasterized=True,
        )
    ax.set_title("Mesh-node bathymetry source assignment")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_anomaly_hist(result: dict[str, np.ndarray], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    anomaly = np.asarray(result["depth_anomaly_m"], dtype=float)
    anomaly = anomaly[np.isfinite(anomaly)]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    if anomaly.size:
        lo, hi = np.nanpercentile(anomaly, [1, 99])
        anomaly = anomaly[(anomaly >= lo) & (anomaly <= hi)]
        ax.hist(anomaly, bins=80, color="#4c78a8", edgecolor="white", linewidth=0.2)
    ax.set_title("Best-source depth anomaly: source_depth - original_depth")
    ax.set_xlabel("Depth anomaly (m)")
    ax.set_ylabel("Node count")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_resolution_hist(result: dict[str, np.ndarray], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    resolution = np.asarray(result["best_source_resolution_m"], dtype=float)
    resolution = resolution[np.isfinite(resolution)]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    if resolution.size:
        ax.hist(resolution, bins=60, color="#7f3c8d", edgecolor="white", linewidth=0.2)
    ax.set_title("Assigned source native resolution")
    ax.set_xlabel("Resolution estimate (m)")
    ax.set_ylabel("Node count")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_uncertainty_map(result: dict[str, np.ndarray], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    uncertainty = np.asarray(result["best_uncertainty_m"], dtype=float)
    source = np.asarray(result["best_source"], dtype=object)
    mask = (source == "nbs_bluetopo") & np.isfinite(uncertainty)
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    if mask.any():
        sc = ax.scatter(
            result["lon"][mask],
            result["lat"][mask],
            s=0.45,
            c=uncertainty[mask],
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, label="BlueTopo uncertainty (m)")
    else:
        ax.text(0.5, 0.5, "No NBS BlueTopo uncertainty assigned", transform=ax.transAxes, ha="center")
    ax.set_title("NBS BlueTopo uncertainty at assigned nodes")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def resolution_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


if __name__ == "__main__":
    main()
