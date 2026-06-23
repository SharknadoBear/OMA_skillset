#!/usr/bin/env python3
"""Check downloaded data coverage, gaps, NaNs, and basic visual health."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".csv", ".txt", ".json", ".geojson", ".nc", ".nc4", ".cdf", ".npz", ".npy", ".gpkg", ".shp"}


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _finite_fraction(values: list[float]) -> float | None:
    if not values:
        return None
    finite = sum(1 for value in values if math.isfinite(value))
    return finite / len(values)


def _try_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> datetime | None:
    if value in ("", None):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _plot_series(path: Path, x: list[Any], y: list[float], title: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not y:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(range(len(y)) if not x else x, y, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _plot_scatter(path: Path, xs: list[float], ys: list[float], title: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not xs or not ys:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(xs, ys, s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("x/lon")
    ax.set_ylabel("y/lat")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _check_csv(path: Path, plots_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        rows = list(reader)
    fields = list(rows[0].keys()) if rows else []
    numeric: dict[str, list[float]] = {}
    times: list[datetime] = []
    lon_values: list[float] = []
    lat_values: list[float] = []
    for row in rows:
        for key, value in row.items():
            parsed = _try_float(value)
            if parsed is not None:
                numeric.setdefault(key, []).append(parsed)
            if "time" in key.lower() or "date" in key.lower():
                t = _parse_time(value)
                if t:
                    times.append(t)
            lk = key.lower()
            if lk in {"lon", "longitude", "x"}:
                parsed_lon = _try_float(value)
                if parsed_lon is not None and math.isfinite(parsed_lon):
                    lon_values.append(parsed_lon)
            if lk in {"lat", "latitude", "y"}:
                parsed_lat = _try_float(value)
                if parsed_lat is not None and math.isfinite(parsed_lat):
                    lat_values.append(parsed_lat)
    fractions = {key: _finite_fraction(vals) for key, vals in numeric.items()}
    plots: list[str] = []
    if numeric:
        first = next(iter(numeric))
        plot = _plot_series(plots_dir / f"{path.stem}_timeseries.png", times[: len(numeric[first])], numeric[first], path.name)
        if plot:
            plots.append(plot)
    if lon_values and lat_values:
        plot = _plot_scatter(plots_dir / f"{path.stem}_scatter.png", lon_values, lat_values, path.name)
        if plot:
            plots.append(plot)
    return {
        "path": str(path),
        "kind": "table",
        "rows": len(rows),
        "columns": fields,
        "numeric_finite_fraction": fractions,
        "time_coverage": {
            "start": min(times).isoformat() if times else None,
            "end": max(times).isoformat() if times else None,
            "count": len(times),
        },
        "spatial_coverage": {
            "bbox": [min(lon_values), min(lat_values), max(lon_values), max(lat_values)] if lon_values and lat_values else None,
            "point_count": min(len(lon_values), len(lat_values)) if lon_values and lat_values else 0,
        },
        "plots": plots,
    }


def _check_json(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    size = path.stat().st_size
    if isinstance(data, dict):
        keys = sorted(data.keys())
        feature_count = len(data.get("features", [])) if isinstance(data.get("features"), list) else None
    else:
        keys = []
        feature_count = len(data) if isinstance(data, list) else None
    return {"path": str(path), "kind": "json", "bytes": size, "keys": keys[:40], "feature_count": feature_count, "plots": []}


def _check_np(path: Path, plots_dir: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        return {"path": str(path), "kind": "array", "error": f"numpy unavailable: {exc}", "plots": []}
    arrays: dict[str, Any] = {}
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
    else:
        arrays = {path.stem: np.load(path)}
    fractions: dict[str, float | None] = {}
    plots: list[str] = []
    for key, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.number):
            values = arr.astype(float).ravel()
            fractions[key] = float(np.isfinite(values).sum() / values.size) if values.size else None
            if values.size:
                plot = _plot_series(plots_dir / f"{path.stem}_{key}_series.png", [], values[: min(values.size, 500)].tolist(), f"{path.name}:{key}")
                if plot:
                    plots.append(plot)
    return {"path": str(path), "kind": "array", "arrays": {k: list(v.shape) for k, v in arrays.items()}, "numeric_finite_fraction": fractions, "plots": plots}


def _check_netcdf(path: Path, plots_dir: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import xarray as xr
    except Exception as exc:
        return {"path": str(path), "kind": "netcdf", "error": f"xarray/numpy unavailable: {exc}", "plots": []}
    plots: list[str] = []
    with xr.open_dataset(path) as ds:
        fractions: dict[str, float | None] = {}
        variables = list(ds.data_vars)
        for name in variables:
            arr = ds[name]
            if np.issubdtype(arr.dtype, np.number):
                values = arr.values.astype(float).ravel()
                fractions[name] = float(np.isfinite(values).sum() / values.size) if values.size else None
                if not plots and values.size:
                    plot = _plot_series(plots_dir / f"{path.stem}_{name}_series.png", [], values[: min(values.size, 500)].tolist(), f"{path.name}:{name}")
                    if plot:
                        plots.append(plot)
        coverage = {}
        for coord in ("time", "lon", "longitude", "lat", "latitude", "x", "y"):
            if coord in ds.coords:
                vals = ds[coord].values
                try:
                    coverage[coord] = {"min": str(vals.min()), "max": str(vals.max()), "count": int(vals.size)}
                except Exception:
                    pass
        return {"path": str(path), "kind": "netcdf", "dims": dict(ds.sizes), "variables": variables, "numeric_finite_fraction": fractions, "coverage": coverage, "plots": plots}


def _check_vector(path: Path, plots_dir: Path) -> dict[str, Any]:
    try:
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"path": str(path), "kind": "vector", "error": f"geopandas/matplotlib unavailable: {exc}", "plots": []}
    gdf = gpd.read_file(path)
    plots: list[str] = []
    plot_path = plots_dir / f"{path.stem}_vector.png"
    if not gdf.empty:
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        gdf.to_crs(4326).plot(ax=ax, linewidth=0.8, markersize=6)
        ax.set_title(path.name)
        ax.grid(True, alpha=0.25)
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        plots.append(str(plot_path))
    return {"path": str(path), "kind": "vector", "features": int(len(gdf)), "bbox": [float(x) for x in gdf.to_crs(4326).total_bounds] if not gdf.empty else None, "columns": list(gdf.columns), "plots": plots}


def _find_data_files(run_dir: Path, output: Path, plots_dir: Path) -> list[Path]:
    out_resolved = output.resolve()
    plots_resolved = plots_dir.resolve()
    files: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == out_resolved or str(resolved).startswith(str(plots_resolved)):
            continue
        suffix = path.suffix.lower()
        stem = path.stem.lower()
        if suffix == ".json" and any(token in stem for token in ("request", "estimate", "health_check")):
            continue
        if suffix in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


def _requested_values(request: Any, keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []

    def add(value: Any) -> None:
        if value in ("", None):
            return
        if isinstance(value, list):
            for item in value:
                add(item)
        else:
            values.append(str(value))

    if isinstance(request, dict):
        for key in keys:
            add(request.get(key))
    elif isinstance(request, list):
        has_selected_field = any(isinstance(item, dict) and "selected" in item for item in request)
        for item in request:
            if not isinstance(item, dict):
                continue
            if has_selected_field and str(item.get("selected", "")).strip().lower() not in {"1", "true", "yes", "y", "selected"}:
                continue
            for key in keys:
                add(item.get(key))
    return values


def _summarize_caveats(files: list[dict[str, Any]], request: Any) -> list[str]:
    caveats: list[str] = []
    if not files:
        caveats.append("No supported downloaded data files were found in the run directory.")
    observed_vars: set[str] = set()
    for item in files:
        if item.get("error"):
            caveats.append(f"{Path(item['path']).name}: {item['error']}")
        observed_vars.update(str(v).lower() for v in item.get("variables", []) or [])
        observed_vars.update(str(v).lower() for v in item.get("columns", []) or [])
        fractions = item.get("numeric_finite_fraction") or {}
        for name, frac in fractions.items():
            if frac is None:
                caveats.append(f"{Path(item['path']).name}:{name} has no numeric samples.")
            elif frac == 0:
                caveats.append(f"{Path(item['path']).name}:{name} is entirely non-finite.")
            elif frac < 0.95:
                caveats.append(f"{Path(item['path']).name}:{name} finite coverage is {frac:.3f}, below 0.950.")
        if item.get("kind") == "table" and item.get("rows") == 0:
            caveats.append(f"{Path(item['path']).name} has zero rows.")
        if item.get("kind") == "vector" and item.get("features") == 0:
            caveats.append(f"{Path(item['path']).name} has zero vector features.")
        if not item.get("plots"):
            caveats.append(f"{Path(item['path']).name} did not produce a diagnostic plot.")
    requested_vars = _requested_values(request, ("variables", "products"))
    missing = [str(v) for v in requested_vars if str(v).lower() not in observed_vars]
    if missing and observed_vars:
        caveats.append("Requested variables/products not directly observed in downloaded files: " + ", ".join(missing))
    return caveats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Request or manifest JSON used for coverage comparison.")
    parser.add_argument("--run-dir", required=True, help="Directory containing downloaded data products.")
    parser.add_argument("--output", required=True, help="Health-check JSON to write.")
    parser.add_argument("--plots-dir", required=True, help="Directory for diagnostic plots.")
    args = parser.parse_args()

    request = _read_json(args.request)
    run_dir = Path(args.run_dir)
    output = Path(args.output)
    plots_dir = Path(args.plots_dir)
    checked: list[dict[str, Any]] = []
    for path in _find_data_files(run_dir, output, plots_dir):
        suffix = path.suffix.lower()
        try:
            if suffix in {".csv", ".txt"}:
                checked.append(_check_csv(path, plots_dir))
            elif suffix in {".json", ".geojson"}:
                checked.append(_check_json(path))
            elif suffix in {".nc", ".nc4", ".cdf"}:
                checked.append(_check_netcdf(path, plots_dir))
            elif suffix in {".npz", ".npy"}:
                checked.append(_check_np(path, plots_dir))
            elif suffix in {".gpkg", ".shp"}:
                checked.append(_check_vector(path, plots_dir))
        except Exception as exc:
            checked.append({"path": str(path), "kind": suffix.lstrip("."), "error": str(exc), "plots": []})
    caveats = _summarize_caveats(checked, request)
    result = {
        "schema_version": "external_data_health_check_v1",
        "run_dir": str(run_dir),
        "request_path": str(args.request) if args.request else None,
        "checked_file_count": len(checked),
        "files": checked,
        "caveats": caveats,
        "reporting_policy": "Only surface this report in user-facing summaries when caveats are important.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checked_file_count": len(checked), "caveat_count": len(caveats), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
