"""Compare NOAA CUDEM bathymetry against the large SE-AK 2DM mesh."""

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

from cudem_bathy.catalog import build_tile_index, save_tile_index  # noqa: E402
from cudem_bathy.mesh_compare import (  # noqa: E402
    anomaly_stats,
    estimate_tile_sizes,
    read_2dm_nodes,
    region_mask,
    sample_tiles_to_mesh,
    selected_cudem_tiles,
    write_json,
    write_node_csv,
)


REGIONS = {
    "icy_strait": (-136.2, 58.1, -134.8, 58.6),
    "sumner_strait": (-133.5, 55.6, -131.7, 56.4),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-2dm", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--index")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "maps").mkdir(exist_ok=True)
    (run_dir / "histograms").mkdir(exist_ok=True)
    index_path = Path(args.index) if args.index else run_dir / "cudem_tile_index_expanded.json"

    if args.rebuild_index or not index_path.exists():
        index = build_tile_index()
        save_tile_index(index, index_path)

    mesh = read_2dm_nodes(args.mesh_2dm)
    tiles = selected_cudem_tiles(index_path, mesh.bbox)
    size_info = estimate_tile_sizes(tiles)
    coverage = _coverage_report(mesh, tiles, size_info)
    coverage["mesh"] = {
        "path": str(args.mesh_2dm),
        "nodes": int(mesh.lon.size),
        "bbox_wsen": list(mesh.bbox),
    }
    coverage_path = write_json(coverage, run_dir / "SE_AK_cudem_coverage.json")
    _plot_coverage(mesh, tiles, run_dir / "SE_AK_cudem_coverage.png")

    if args.dry_run or args.skip_sampling:
        print(json.dumps({"coverage": str(coverage_path), "dry_run": True}, indent=2))
        return 0

    result = sample_tiles_to_mesh(mesh, tiles)
    node_csv = write_node_csv(result, run_dir / "SE_AK_cudem_node_depths.csv")
    stats = _stats_by_region(result)

    map_paths = {
        "whole": str(_plot_anomaly(result, run_dir / "maps" / "SE_AK_anomaly_nodes.png")),
        "icy_strait": str(
            _plot_anomaly(
                result,
                run_dir / "maps" / "icy_strait_anomaly_zoom.png",
                bbox=REGIONS["icy_strait"],
                title="Icy Strait CUDEM - original depth anomaly",
            )
        ),
        "sumner_strait": str(
            _plot_anomaly(
                result,
                run_dir / "maps" / "sumner_strait_anomaly_zoom.png",
                bbox=REGIONS["sumner_strait"],
                title="Sumner Strait CUDEM - original depth anomaly",
            )
        ),
    }
    hist_paths = {
        "whole": str(
            _plot_histogram(
                result["depth_anomaly_m"],
                run_dir / "histograms" / "SE_AK_anomaly_hist.png",
                "Covered SE-AK depth anomaly",
            )
        ),
        "icy_strait": str(
            _plot_histogram(
                result["depth_anomaly_m"][
                    region_mask(result["lon"], result["lat"], REGIONS["icy_strait"])
                ],
                run_dir / "histograms" / "icy_strait_anomaly_hist.png",
                "Icy Strait depth anomaly",
            )
        ),
        "sumner_strait": str(
            _plot_histogram(
                result["depth_anomaly_m"][
                    region_mask(result["lon"], result["lat"], REGIONS["sumner_strait"])
                ],
                run_dir / "histograms" / "sumner_strait_anomaly_hist.png",
                "Sumner Strait depth anomaly",
            )
        ),
    }

    summary = {
        "coverage_report": str(coverage_path),
        "node_comparison_csv": str(node_csv),
        "maps": map_paths,
        "histograms": hist_paths,
        "statistics": stats,
        "coverage": _sample_coverage_summary(result),
        "warnings": [str(x) for x in result.get("warnings", [])],
        "resolution_note": (
            "Native per selected CUDEM source: 1/9 arc-sec where available, "
            "1/3 arc-sec elsewhere in CUDEM-covered SE-AK."
        ),
    }
    summary_path = write_json(summary, run_dir / "SE_AK_analysis_summary.json")
    print(json.dumps({"summary": str(summary_path)}, indent=2))
    return 0


def _coverage_report(mesh, tiles, size_info: dict) -> dict:
    resolution_counts = {}
    for tile in tiles:
        resolution_counts.setdefault(tile.collection, 0)
        resolution_counts[tile.collection] += 1
    node_counts = {"tiled_19as": 0, "tiled_13as": 0}
    for collection in node_counts:
        hits = [tile for tile in tiles if tile.collection == collection]
        mask = np.zeros(mesh.lon.size, dtype=bool)
        for tile in hits:
            mask |= (
                (mesh.lon >= tile.west)
                & (mesh.lon <= tile.east)
                & (mesh.lat >= tile.south)
                & (mesh.lat <= tile.north)
            )
        node_counts[collection] = int(mask.sum())
    return {
        "tile_counts": resolution_counts,
        "node_counts_by_tile_rectangles": node_counts,
        "estimated_download": size_info,
        "regions": REGIONS,
    }


def _sample_coverage_summary(result: dict[str, np.ndarray]) -> dict:
    status = result["coverage_status"]
    finite = np.isfinite(result["depth_anomaly_m"])
    by_resolution = {}
    for label in sorted(set(result["source_resolution"])):
        if not label:
            continue
        by_resolution[label] = int(np.count_nonzero(result["source_resolution"] == label))
    return {
        "total_nodes": int(status.size),
        "covered_nodes": int(np.count_nonzero(finite)),
        "uncovered_nodes": int(np.count_nonzero(~finite)),
        "covered_fraction": float(np.count_nonzero(finite) / status.size),
        "by_resolution": by_resolution,
    }


def _stats_by_region(result: dict[str, np.ndarray]) -> dict:
    stats = {"whole_cudem_covered": anomaly_stats(result["depth_anomaly_m"])}
    for name, bbox in REGIONS.items():
        mask = region_mask(result["lon"], result["lat"], bbox)
        stats[name] = anomaly_stats(result["depth_anomaly_m"][mask])
        stats[name]["bbox_wsen"] = list(bbox)
        stats[name]["nodes_in_bbox"] = int(mask.sum())
    return stats


def _plot_coverage(mesh, tiles, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    step = max(1, mesh.lon.size // 120000)
    ax.scatter(mesh.lon[::step], mesh.lat[::step], s=0.2, c="0.75", label="mesh nodes")
    colors = {"tiled_19as": "tab:red", "tiled_13as": "tab:blue"}
    labels = {"tiled_19as": "CUDEM 1/9 arc-sec", "tiled_13as": "CUDEM 1/3 arc-sec"}
    for tile in tiles:
        x = [tile.west, tile.east, tile.east, tile.west, tile.west]
        y = [tile.south, tile.south, tile.north, tile.north, tile.south]
        ax.plot(x, y, color=colors.get(tile.collection, "black"), linewidth=0.8, alpha=0.8)
    for collection, color in colors.items():
        if any(tile.collection == collection for tile in tiles):
            ax.plot([], [], color=color, label=labels[collection])
    for name, bbox in REGIONS.items():
        west, south, east, north = bbox
        ax.plot([west, east, east, west, west], [south, south, north, north, south], "k--", linewidth=1)
        ax.text(west, north, name.replace("_", " "), fontsize=8)
    ax.set_title("SE-AK mesh and selected CUDEM tile coverage")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", markerscale=6)
    ax.grid(True, linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _plot_anomaly(
    result: dict[str, np.ndarray],
    output: Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    title: str = "Covered SE-AK CUDEM - original depth anomaly",
) -> Path:
    lon = result["lon"]
    lat = result["lat"]
    anomaly = result["depth_anomaly_m"]
    mask = np.isfinite(anomaly)
    if bbox is not None:
        mask &= region_mask(lon, lat, bbox)
    vals = anomaly[mask]
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if vals.size:
        vmax = float(np.nanpercentile(np.abs(vals), 98))
        vmax = max(vmax, 1.0)
        sc = ax.scatter(lon[mask], lat[mask], c=vals, s=1.0, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
        cbar.set_label("CUDEM depth - original depth (m)")
    else:
        ax.text(0.5, 0.5, "No finite CUDEM anomaly values", transform=ax.transAxes, ha="center")
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _plot_histogram(values: np.ndarray, output: Path, title: str) -> Path:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if vals.size:
        lo, hi = np.nanpercentile(vals, [1, 99])
        ax.hist(vals, bins=80, range=(lo, hi), color="0.25", alpha=0.85)
        ax.axvline(np.mean(vals), color="tab:red", linewidth=1.2, label=f"mean {np.mean(vals):.2f} m")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite anomaly values", transform=ax.transAxes, ha="center")
    ax.set_title(title)
    ax.set_xlabel("CUDEM depth - original depth (m)")
    ax.set_ylabel("Node count")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
