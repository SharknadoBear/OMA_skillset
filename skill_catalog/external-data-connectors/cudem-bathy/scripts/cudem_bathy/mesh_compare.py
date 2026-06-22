"""Mesh-driven CUDEM sampling and anomaly analysis."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .catalog import load_tile_index
from .fetch import _find_elevation_var
from .normalize import elevation_to_depth
from .sources import (
    BathySourceRecord,
    SOURCE_ID,
    load_bathy_source_index,
    native_resolution_m,
    select_sources,
    source_sort_key,
)
from .tiles import TileRecord, bbox_intersects


@dataclass(frozen=True)
class MeshNodes:
    node_id: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    original_z_m: np.ndarray

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(np.nanmin(self.lon)),
            float(np.nanmin(self.lat)),
            float(np.nanmax(self.lon)),
            float(np.nanmax(self.lat)),
        )

    @property
    def original_depth_m(self) -> np.ndarray:
        return np.where(np.isfinite(self.original_z_m), np.maximum(-self.original_z_m, 0.0), np.nan)


def read_2dm_nodes(path: str | Path) -> MeshNodes:
    """Read ND lines from an SMS 2DM mesh."""

    node_id: list[int] = []
    lon: list[float] = []
    lat: list[float] = []
    z: list[float] = []
    with Path(path).open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ND"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            node_id.append(int(parts[1]))
            lon.append(float(parts[2]))
            lat.append(float(parts[3]))
            z.append(float(parts[4]))
    if not node_id:
        raise ValueError(f"No ND nodes with z values found in {path}")
    return MeshNodes(
        node_id=np.asarray(node_id, dtype=np.int64),
        lon=np.asarray(lon, dtype=np.float64),
        lat=np.asarray(lat, dtype=np.float64),
        original_z_m=np.asarray(z, dtype=np.float64),
    )


def selected_cudem_tiles(
    index_path: str | Path,
    bbox: Sequence[float],
    *,
    source_preference: Sequence[str] = ("https_geotiff", "opendap_netcdf"),
) -> list[TileRecord]:
    """Return native 1/9 CUDEM tiles plus 1/3 CUDEM tiles for broader coverage."""

    records = load_tile_index(index_path)
    source_rank = {name: i for i, name in enumerate(source_preference)}
    selected: list[TileRecord] = []
    for collection in ("tiled_19as", "tiled_13as"):
        candidates = [
            rec
            for rec in records
            if rec.collection == collection
            and rec.source_mode in set(source_preference)
            and bbox_intersects(rec.bbox, bbox, touch_counts=True)
        ]
        best: dict[tuple[str, float, float], TileRecord] = {}
        for rec in candidates:
            key = rec.tile_key
            old = best.get(key)
            if old is None or _tile_score(rec, source_rank) > _tile_score(old, source_rank):
                best[key] = rec
        selected.extend(sorted(best.values(), key=lambda t: (t.collection, t.north, t.west)))
    return selected


def estimate_tile_sizes(tiles: Sequence[TileRecord], *, timeout: int = 20) -> dict:
    """Estimate selected download volume with known sizes and HTTP HEAD fallback."""

    import requests

    total = 0.0
    by_collection: dict[str, dict] = {}
    enriched: list[dict] = []
    for tile in tiles:
        size_mb = tile.size_mb
        size_source = "catalog"
        if size_mb is None and tile.source_mode == "https_geotiff":
            try:
                resp = requests.head(tile.url, timeout=timeout, allow_redirects=True)
                length = resp.headers.get("content-length")
                if length:
                    size_mb = int(length) / 1024.0 / 1024.0
                    size_source = "http_head"
            except Exception:
                size_source = "unknown"
        if size_mb is not None:
            total += size_mb
        coll = by_collection.setdefault(tile.collection, {"n_tiles": 0, "size_mb": 0.0})
        coll["n_tiles"] += 1
        coll["size_mb"] += float(size_mb or 0.0)
        item = tile.to_dict()
        item["estimated_size_mb"] = size_mb
        item["size_source"] = size_source
        enriched.append(item)
    return {"total_size_mb": total, "by_collection": by_collection, "tiles": enriched}


def sample_tiles_to_mesh(
    mesh: MeshNodes,
    tiles: Sequence[TileRecord],
    *,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Sample CUDEM GeoTIFF tiles to mesh nodes, preferring finer coverage first."""

    n = mesh.lon.size
    cudem_elevation = np.full(n, np.nan, dtype=np.float64)
    source_resolution = np.full(n, "", dtype=object)
    source_tile = np.full(n, "", dtype=object)
    coverage_status = np.full(n, "no_cudem_coverage", dtype=object)
    warnings: list[str] = []

    ordered = sorted(
        [tile for tile in tiles if tile.source_mode == "https_geotiff"],
        key=lambda t: (0 if t.collection == "tiled_19as" else 1, t.north, t.west),
    )
    for i, tile in enumerate(ordered, start=1):
        mask = (
            np.isnan(cudem_elevation)
            & (mesh.lon >= tile.west)
            & (mesh.lon <= tile.east)
            & (mesh.lat >= tile.south)
            & (mesh.lat <= tile.north)
        )
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        if progress:
            print(f"[{i}/{len(ordered)}] {tile.name}: sampling {idx.size} nodes", flush=True)
        try:
            values = sample_geotiff_at_points(tile.url, mesh.lon[idx], mesh.lat[idx])
        except Exception as exc:
            warning = f"{tile.name}: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            if progress:
                print(f"  warning: {warning}", flush=True)
            continue
        valid = np.isfinite(values)
        if valid.any():
            assigned = idx[valid]
            cudem_elevation[assigned] = values[valid]
            source_resolution[assigned] = _resolution_label(tile)
            source_tile[assigned] = tile.name
            coverage_status[assigned] = "ok"

    cudem_depth, _wet = elevation_to_depth(cudem_elevation)
    original_depth = mesh.original_depth_m
    anomaly = cudem_depth.astype(np.float64) - original_depth
    anomaly[~np.isfinite(cudem_depth)] = np.nan
    return {
        "node_id": mesh.node_id,
        "lon": mesh.lon,
        "lat": mesh.lat,
        "original_z_m": mesh.original_z_m,
        "original_depth_m": original_depth,
        "cudem_elevation_m": cudem_elevation,
        "cudem_depth_m": cudem_depth.astype(np.float64),
        "depth_anomaly_m": anomaly,
        "source_resolution": source_resolution,
        "source_tile": source_tile,
        "coverage_status": coverage_status,
        "warnings": np.asarray(warnings, dtype=object),
    }


def selected_bathy_sources(
    index_path: str | Path,
    bbox: Sequence[float],
    *,
    fallback_policy: str = "cudem-crm-etopo",
    resolution_policy: str = "source-priority",
) -> list[BathySourceRecord]:
    """Return generic bathymetry fallback sources intersecting a mesh bbox."""

    policy = {
        "cudem-only": ("cudem",),
        "cudem-crm": ("cudem", "crm"),
        "cudem-crm-etopo": ("cudem", "crm", "etopo"),
        "cudem-nbs-crm-etopo": ("cudem", "nbs_bluetopo", "crm", "etopo"),
        "all": ("cudem", "nbs_bluetopo", "crm", "etopo"),
    }.get(fallback_policy)
    if policy is None:
        raise ValueError(
            "fallback_policy must be cudem-only, cudem-crm, cudem-crm-etopo, "
            "or cudem-nbs-crm-etopo"
        )
    records = load_bathy_source_index(index_path)
    return select_sources(
        records,
        tuple(float(x) for x in bbox),
        source_names=policy,
        resolution_policy=resolution_policy,
    )


def sample_sources_to_mesh(
    mesh: MeshNodes,
    sources: Sequence[BathySourceRecord],
    *,
    progress: bool = True,
    resolution_policy: str = "source-priority",
) -> dict[str, np.ndarray]:
    """Sample CUDEM, NBS BlueTopo, CRM, and ETOPO sources to mesh nodes."""

    n = mesh.lon.size
    best_elevation = np.full(n, np.nan, dtype=np.float64)
    best_uncertainty = np.full(n, np.nan, dtype=np.float64)
    best_contributor = np.full(n, "", dtype=object)
    best_source = np.full(n, "", dtype=object)
    best_source_resolution = np.full(n, np.nan, dtype=np.float64)
    best_source_resolution_m = np.full(n, np.nan, dtype=np.float64)
    source_dataset = np.full(n, "", dtype=object)
    coverage_status = np.full(n, "no_source_coverage", dtype=object)
    warnings: list[str] = []

    ordered = sorted(sources, key=lambda s: source_sort_key(s, resolution_policy=resolution_policy))
    for i, source in enumerate(ordered, start=1):
        idx = _nodes_in_source(mesh, source, unfilled=np.isnan(best_elevation))
        if idx.size == 0:
            continue
        if progress:
            print(
                f"[{i}/{len(ordered)}] {source.source_name}:{source.name}: "
                f"sampling {idx.size} nodes",
                flush=True,
            )
        try:
            sampled = sample_source_at_points_full(source, mesh.lon[idx], mesh.lat[idx])
        except Exception as exc:
            warning = f"{source.source_name}:{source.name}: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            if progress:
                print(f"  warning: {warning}", flush=True)
            continue
        values = sampled["elevation"]
        valid = np.isfinite(values)
        if valid.any():
            assigned = idx[valid]
            best_elevation[assigned] = values[valid]
            if "uncertainty" in sampled:
                best_uncertainty[assigned] = sampled["uncertainty"][valid]
            if "contributor" in sampled:
                best_contributor[assigned] = sampled["contributor"][valid]
            best_source[assigned] = source.source_name
            best_source_resolution[assigned] = source.resolution_arcsec
            best_source_resolution_m[assigned] = native_resolution_m(source)
            source_dataset[assigned] = source.name
            coverage_status[assigned] = "ok"

    best_depth, _wet = elevation_to_depth(best_elevation)
    original_depth = mesh.original_depth_m
    anomaly = best_depth.astype(np.float64) - original_depth
    anomaly[~np.isfinite(best_depth)] = np.nan
    return {
        "node_id": mesh.node_id,
        "lon": mesh.lon,
        "lat": mesh.lat,
        "original_z_m": mesh.original_z_m,
        "original_depth_m": original_depth,
        "best_elevation_m": best_elevation,
        "best_depth_m": best_depth.astype(np.float64),
        "depth_anomaly_m": anomaly,
        "best_uncertainty_m": best_uncertainty,
        "best_contributor": best_contributor,
        "best_source": best_source,
        "best_source_resolution": best_source_resolution,
        "best_source_resolution_m": best_source_resolution_m,
        "source_dataset": source_dataset,
        "coverage_status": coverage_status,
        "warnings": np.asarray(warnings, dtype=object),
    }


def sample_source_at_points(source: BathySourceRecord, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Sample one generic source at lon/lat points."""

    return sample_source_at_points_full(source, lon, lat)["elevation"]


def sample_source_at_points_full(
    source: BathySourceRecord, lon: np.ndarray, lat: np.ndarray
) -> dict[str, np.ndarray]:
    """Sample one generic source and preserve optional source-specific bands."""

    if lon.size == 0:
        return {
            "elevation": np.empty(0, dtype=np.float64),
            "uncertainty": np.empty(0, dtype=np.float64),
            "contributor": np.empty(0, dtype=object),
        }
    if source.source_mode == "nbs_bluetopo_geotiff":
        return sample_nbs_bluetopo_at_points(source, lon, lat)
    if source.source_mode == "https_geotiff":
        return {
            "elevation": sample_geotiff_at_points(source.url, lon, lat),
            "uncertainty": np.full(lon.size, np.nan, dtype=np.float64),
            "contributor": np.full(lon.size, "", dtype=object),
        }
    if source.source_mode == "opendap_netcdf":
        return {
            "elevation": sample_opendap_at_points(source, lon, lat),
            "uncertainty": np.full(lon.size, np.nan, dtype=np.float64),
            "contributor": np.full(lon.size, "", dtype=object),
        }
    raise ValueError(f"Unsupported source_mode: {source.source_mode}")


def sample_nbs_bluetopo_at_points(
    source: BathySourceRecord, lon: np.ndarray, lat: np.ndarray
) -> dict[str, np.ndarray]:
    """Sample NOAA NBS BlueTopo GeoTIFF elevation/uncertainty/contributor bands."""

    import rasterio
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import from_bounds

    n = lon.size
    out = {
        "elevation": np.full(n, np.nan, dtype=np.float64),
        "uncertainty": np.full(n, np.nan, dtype=np.float64),
        "contributor": np.full(n, "", dtype=object),
    }
    if n == 0:
        return out
    last_error: Exception | None = None
    for path in (source.url, f"/vsicurl/{source.url}"):
        try:
            with rasterio.Env(
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_MAX_RETRY="3",
                GDAL_HTTP_RETRY_DELAY="2",
            ):
                with rasterio.open(path) as src:
                    if src.crs is None:
                        raise ValueError("BlueTopo tile has no CRS.")
                    x, y = rio_transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
                    x_arr = np.asarray(x, dtype=np.float64)
                    y_arr = np.asarray(y, dtype=np.float64)
                    dx = abs(src.transform.a)
                    dy = abs(src.transform.e)
                    west = float(np.nanmin(x_arr)) - 2.0 * dx
                    east = float(np.nanmax(x_arr)) + 2.0 * dx
                    south = float(np.nanmin(y_arr)) - 2.0 * dy
                    north = float(np.nanmax(y_arr)) + 2.0 * dy
                    window = from_bounds(west, south, east, north, src.transform)
                    window = window.round_offsets().round_lengths()
                    if window.width < 2 or window.height < 2:
                        return out
                    elevation = _read_projected_geotiff_band(src, 1, window, x_arr, y_arr)
                    out["elevation"] = elevation
                    if src.count >= 2:
                        out["uncertainty"] = _read_projected_geotiff_band(src, 2, window, x_arr, y_arr)
                    if src.count >= 3:
                        contributor = _read_projected_geotiff_band(src, 3, window, x_arr, y_arr)
                        contributor_text = np.full(n, "", dtype=object)
                        valid_contributor = np.isfinite(contributor)
                        if valid_contributor.any():
                            contributor_text[valid_contributor] = (
                                np.round(contributor[valid_contributor]).astype(np.int64).astype(str)
                            )
                        out["contributor"] = contributor_text
                    return out
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not sample BlueTopo GeoTIFF {source.url}: {last_error}")


def _read_projected_geotiff_band(src, band: int, window, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from rasterio.windows import bounds as window_bounds

    data = src.read(band, window=window, masked=True).astype(np.float64)
    values = np.asarray(data.filled(np.nan), dtype=np.float64)
    if src.nodata is not None:
        values = np.where(values == src.nodata, np.nan, values)
    left, bottom, right, top = window_bounds(window, src.transform)
    height, width = values.shape
    if width < 2 or height < 2:
        return np.full(x.size, np.nan, dtype=np.float64)
    x_coords = np.linspace(
        left + (right - left) / (2 * width),
        right - (right - left) / (2 * width),
        width,
    )
    y_desc = np.linspace(
        top - (top - bottom) / (2 * height),
        bottom + (top - bottom) / (2 * height),
        height,
    )
    y_coords = y_desc[::-1]
    values = values[::-1, :]
    interp = RegularGridInterpolator(
        (y_coords, x_coords),
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(np.column_stack([y, x]))


def sample_opendap_at_points(source: BathySourceRecord, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Read an OPeNDAP point bounding window and interpolate to points."""

    import xarray as xr

    lon_query = _lon_for_source(source, lon)
    west = float(np.nanmin(lon_query))
    east = float(np.nanmax(lon_query))
    south = float(np.nanmin(lat))
    north = float(np.nanmax(lat))
    pad = max(source.resolution_arcsec / 3600.0 * 2.0, 1.0e-6)
    ds = xr.open_dataset(source.url, decode_times=False)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("OPeNDAP source has no lat/lon coordinates.")
    ds = ds.sortby("lat").sortby("lon")
    var = source.variable if source.variable != "auto" and source.variable in ds.data_vars else _find_elevation_var(ds)
    sub = ds.sel(lat=slice(south - pad, north + pad), lon=slice(west - pad, east + pad))
    if sub["lat"].size < 2 or sub["lon"].size < 2:
        return np.full(lon.size, np.nan, dtype=np.float64)
    values = sub[var].load().values.astype(np.float64)
    values = np.where(values <= -99990.0, np.nan, values)
    interp = RegularGridInterpolator(
        (sub["lat"].values.astype(float), sub["lon"].values.astype(float)),
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(np.column_stack([lat, lon_query]))


def _nodes_in_source(mesh: MeshNodes, source: BathySourceRecord, *, unfilled: np.ndarray) -> np.ndarray:
    lon = _lon_for_source(source, mesh.lon)
    return np.flatnonzero(
        unfilled
        & (lon >= source.west)
        & (lon <= source.east)
        & (mesh.lat >= source.south)
        & (mesh.lat <= source.north)
    )


def _lon_for_source(source: BathySourceRecord, lon: np.ndarray) -> np.ndarray:
    if source.west > 180.0 or source.east > 180.0:
        return np.where(lon < 0.0, lon + 360.0, lon)
    return lon


def sample_geotiff_at_points(url: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Read the point bounding window from one GeoTIFF and interpolate to points."""

    import rasterio
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import from_bounds

    if lon.size == 0:
        return np.empty(0, dtype=np.float64)
    last_error: Exception | None = None
    for path in (url, f"/vsicurl/{url}"):
        try:
            with rasterio.Env(
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_MAX_RETRY="3",
                GDAL_HTTP_RETRY_DELAY="2",
            ):
                with rasterio.open(path) as src:
                    dx = abs(src.transform.a)
                    dy = abs(src.transform.e)
                    west = float(np.nanmin(lon)) - 2 * dx
                    east = float(np.nanmax(lon)) + 2 * dx
                    south = float(np.nanmin(lat)) - 2 * dy
                    north = float(np.nanmax(lat)) + 2 * dy
                    window = from_bounds(west, south, east, north, src.transform)
                    window = window.round_offsets().round_lengths()
                    data = src.read(1, window=window, masked=True).astype(np.float64)
                    values = np.asarray(data.filled(np.nan), dtype=np.float64)
                    if src.nodata is not None:
                        values = np.where(values == src.nodata, np.nan, values)
                    left, bottom, right, top = window_bounds(window, src.transform)
                    width = values.shape[1]
                    height = values.shape[0]
                    if width < 2 or height < 2:
                        return np.full(lon.size, np.nan, dtype=np.float64)
                    x_coords = np.linspace(
                        left + (right - left) / (2 * width),
                        right - (right - left) / (2 * width),
                        width,
                    )
                    y_desc = np.linspace(
                        top - (top - bottom) / (2 * height),
                        bottom + (top - bottom) / (2 * height),
                        height,
                    )
                    y_coords = y_desc[::-1]
                    values = values[::-1, :]
                    interp = RegularGridInterpolator(
                        (y_coords, x_coords),
                        values,
                        method="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    return interp(np.column_stack([lat, lon]))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not sample GeoTIFF {url}: {last_error}")


def write_node_csv(result: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "node_id",
        "lon",
        "lat",
        "original_z_m",
        "original_depth_m",
        "cudem_elevation_m",
        "cudem_depth_m",
        "depth_anomaly_m",
        "source_resolution",
        "source_tile",
        "coverage_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        n = len(result["node_id"])
        for i in range(n):
            row = []
            for field in fields:
                value = result[field][i]
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    row.append("")
                else:
                    row.append(value)
            writer.writerow(row)
    return path


def write_bathy_node_csv(result: dict[str, np.ndarray], path: str | Path) -> Path:
    """Write generic CUDEM/CRM/ETOPO mesh sampling output."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "node_id",
        "lon",
        "lat",
        "original_z_m",
        "original_depth_m",
        "best_elevation_m",
        "best_depth_m",
        "depth_anomaly_m",
        "best_source",
        "best_source_resolution",
        "best_source_resolution_m",
        "best_uncertainty_m",
        "best_contributor",
        "source_dataset",
        "coverage_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        n = len(result["node_id"])
        for i in range(n):
            row = []
            for field in fields:
                value = result[field][i]
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    row.append("")
                else:
                    row.append(value)
            writer.writerow(row)
    return path


def source_counts(result: dict[str, np.ndarray]) -> dict:
    """Count mesh-node assignments by source and coverage status."""

    counts: dict[str, int] = {}
    for value in result["best_source"]:
        key = str(value) if str(value) else "none"
        counts[key] = counts.get(key, 0) + 1
    status_counts: dict[str, int] = {}
    for value in result["coverage_status"]:
        key = str(value)
        status_counts[key] = status_counts.get(key, 0) + 1
    return {"by_source": counts, "by_status": status_counts}


def anomaly_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "rmse": None,
            "bias": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "bias": float(np.mean(arr)),
    }


def region_mask(
    lon: np.ndarray, lat: np.ndarray, bbox: Sequence[float]
) -> np.ndarray:
    west, south, east, north = (float(x) for x in bbox)
    return (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _tile_score(rec: TileRecord, source_rank: dict[str, int]) -> tuple[int, int, int, int]:
    return (
        -source_rank.get(rec.source_mode, 99),
        1 if rec.size_mb is not None else 0,
        rec.year,
        rec.version,
    )


def _resolution_label(tile: TileRecord) -> str:
    if tile.collection == "tiled_19as":
        return "1/9 arc-sec"
    if tile.collection == "tiled_13as":
        return "1/3 arc-sec"
    if tile.collection == "tiled_1as":
        return "1 arc-sec"
    if tile.collection == "tiled_3as":
        return "3 arc-sec"
    return tile.collection
