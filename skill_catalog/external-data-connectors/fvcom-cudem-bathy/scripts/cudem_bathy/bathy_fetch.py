"""CUDEM-first bathymetry fetching with CRM and ETOPO gap filling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from .fetch import _find_elevation_var, _read_geotiff_tile, _target_grid
from .normalize import elevation_to_depth, finite_coverage_fraction
from .plot import plot_bathymetry_map, plot_source_id_map
from .sources import (
    BathySourceRecord,
    SOURCE_ID,
    SOURCE_LABELS,
    load_bathy_source_index,
    select_sources,
)
from .tiles import normalize_bbox


@dataclass(frozen=True)
class BathyFetchResult:
    """Output paths and metadata for a fallback bathymetry bbox fetch."""

    netcdf_path: Path
    png_path: Path
    source_png_path: Path
    metadata_path: Path
    metadata: dict


def fetch_bathy_bbox(
    index: str | Path | Sequence[BathySourceRecord | dict],
    bbox: Sequence[float],
    *,
    run_dir: str | Path,
    name: str,
    fallback_policy: str = "cudem-crm-etopo",
    target_spacing_arcsec: float | None = 3.0,
    max_sources: int = 256,
    make_plot: bool = True,
) -> BathyFetchResult:
    """Fetch bathymetry sources by priority and fill spatial/finite gaps."""

    bbox = normalize_bbox(bbox)
    records = _load_records(index)
    source_names = _policy_sources(fallback_policy)
    selected = select_sources(records, bbox, source_names=source_names)
    if len(selected) > max_sources:
        raise RuntimeError(
            f"{len(selected)} sources intersect bbox, exceeding max_sources={max_sources}. "
            "Choose a coarser policy or raise --max-sources."
        )
    if not selected:
        raise RuntimeError(f"No bathymetry source coverage for bbox {bbox} with {fallback_policy}.")

    native_arcsec = min(src.resolution_arcsec for src in selected)
    spacing_arcsec = native_arcsec if target_spacing_arcsec is None else max(
        float(target_spacing_arcsec), native_arcsec
    )
    target_lon, target_lat = _target_grid(bbox, spacing_arcsec)
    elevation = np.full((target_lat.size, target_lon.size), np.nan, dtype=np.float32)
    source_id = np.zeros(elevation.shape, dtype=np.int16)
    source_resolution = np.full(elevation.shape, np.nan, dtype=np.float32)

    warnings: list[str] = []
    attempted: list[dict] = []
    for source in selected:
        before = int(np.isfinite(elevation).sum())
        try:
            lon, lat, values = read_source_window(
                source, bbox=bbox, target_spacing_arcsec=spacing_arcsec
            )
            filled = burn_source_into_target(
                target_lon,
                target_lat,
                elevation,
                source_id,
                source_resolution,
                lon,
                lat,
                values,
                source=source,
            )
        except Exception as exc:
            filled = 0
            warnings.append(f"{source.source_name}:{source.name}: {type(exc).__name__}: {exc}")
        after = int(np.isfinite(elevation).sum())
        item = source.to_dict()
        item["filled_cells"] = int(filled)
        item["finite_cells_before"] = before
        item["finite_cells_after"] = after
        attempted.append(item)
        if np.isfinite(elevation).all():
            break

    if not np.isfinite(elevation).any():
        raise RuntimeError(
            "Bathymetry sources were selected, but no finite elevation values were read. "
            f"Warnings: {warnings}"
        )

    depth, wet = elevation_to_depth(elevation)
    ds = xr.Dataset(
        data_vars={
            "elevation_m": (("lat", "lon"), elevation),
            "depth_m": (("lat", "lon"), depth),
            "wet_mask": (("lat", "lon"), wet.astype(np.int8)),
            "source_id": (("lat", "lon"), source_id),
            "source_resolution_arcsec": (("lat", "lon"), source_resolution),
        },
        coords={"lat": target_lat.astype(np.float64), "lon": target_lon.astype(np.float64)},
        attrs={
            "title": f"CUDEM-first bathymetry fallback mosaic for {name}",
            "summary": "FVCOM positive-down bathymetry with CUDEM, CRM, and ETOPO fallback.",
            "source_priority": "CUDEM -> NOAA Coastal Relief Model -> ETOPO 2022",
            "source_legend": json.dumps(SOURCE_LABELS, sort_keys=True),
            "bbox_wsen": json.dumps(list(bbox)),
            "target_spacing_arcsec": spacing_arcsec,
            "vertical_units": "meters",
            "datum_warning": (
                "Sources may use different vertical datums. This product preserves "
                "source provenance and does not perform vertical-datum harmonization."
            ),
        },
    )
    ds["elevation_m"].attrs.update({"long_name": "source elevation", "units": "m", "positive": "up"})
    ds["depth_m"].attrs.update({"long_name": "FVCOM positive-down depth", "units": "m", "positive": "down"})
    ds["source_id"].attrs.update({"long_name": "bathymetry source id", "legend": json.dumps(SOURCE_LABELS)})
    ds["source_resolution_arcsec"].attrs.update({"long_name": "source native resolution", "units": "arc-second"})

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    nc_path = run_dir / f"{name}_bathy_sources.nc"
    png_path = run_dir / f"{name}_bathy_sources.png"
    source_png_path = run_dir / f"{name}_bathy_source_id.png"
    metadata_path = run_dir / f"{name}_metadata.json"
    ds.to_netcdf(nc_path)
    if make_plot:
        plot_bathymetry_map(ds, png_path, title=f"{name} bathymetry fallback mosaic", bbox=bbox)
        plot_source_id_map(ds, source_png_path, title=f"{name} bathymetry source coverage", bbox=bbox)

    metadata = metadata_for_bathy_fetch(
        name=name,
        bbox=bbox,
        fallback_policy=fallback_policy,
        spacing_arcsec=spacing_arcsec,
        elevation=elevation,
        source_id=source_id,
        attempted=attempted,
        warnings=warnings,
        outputs={
            "netcdf": str(nc_path),
            "png": str(png_path),
            "source_png": str(source_png_path),
            "metadata": str(metadata_path),
        },
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return BathyFetchResult(nc_path, png_path, source_png_path, metadata_path, metadata)


def read_source_window(
    source: BathySourceRecord,
    *,
    bbox: tuple[float, float, float, float],
    target_spacing_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one source into lon, lat, elevation arrays."""

    if source.source_mode == "https_geotiff":
        return _read_geotiff_tile(source, bbox=bbox, target_spacing_arcsec=target_spacing_arcsec)
    if source.source_mode == "opendap_netcdf":
        return _read_opendap_source(source, bbox=bbox, target_spacing_arcsec=target_spacing_arcsec)
    raise ValueError(f"Unsupported source mode: {source.source_mode}")


def _read_opendap_source(
    source: BathySourceRecord,
    *,
    bbox: tuple[float, float, float, float],
    target_spacing_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(source.url, decode_times=False)
    ds = _normalize_source_lon(ds, bbox=bbox)
    ds = ds.sortby("lat").sortby("lon")
    west, south, east, north = _coord_bbox_intersection(ds, bbox)
    var = source.variable if source.variable != "auto" and source.variable in ds.data_vars else _find_elevation_var(ds)
    sub = ds.sel(lat=slice(south, north), lon=slice(west, east))
    lat = sub["lat"].values.astype(float)
    lon = sub["lon"].values.astype(float)
    if lat.size == 0 or lon.size == 0:
        raise ValueError("No OPeNDAP coordinates overlap bbox.")
    step_lat = _stride(lat, target_spacing_arcsec)
    step_lon = _stride(lon, target_spacing_arcsec)
    sub = sub.isel(lat=slice(None, None, step_lat), lon=slice(None, None, step_lon))
    values = sub[var].load().values.astype(np.float32)
    values = np.where(values <= -99990.0, np.nan, values)
    lon_values = sub["lon"].values.astype(float)
    lon_values = np.where(lon_values > 180.0, lon_values - 360.0, lon_values)
    order = np.argsort(lon_values)
    return lon_values[order], sub["lat"].values.astype(float), values[:, order]


def burn_source_into_target(
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    target_elevation: np.ndarray,
    target_source_id: np.ndarray,
    target_source_resolution: np.ndarray,
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    source_elevation: np.ndarray,
    *,
    source: BathySourceRecord,
) -> int:
    """Fill only empty cells from a source grid and track source provenance."""

    if source_lon.size < 2 or source_lat.size < 2:
        return 0
    lon_tol = float(np.nanmedian(np.abs(np.diff(source_lon))))
    lat_tol = float(np.nanmedian(np.abs(np.diff(source_lat))))
    lon_mask = (target_lon >= source_lon.min() - lon_tol) & (target_lon <= source_lon.max() + lon_tol)
    lat_mask = (target_lat >= source_lat.min() - lat_tol) & (target_lat <= source_lat.max() + lat_tol)
    if not lon_mask.any() or not lat_mask.any():
        return 0

    interpolator = RegularGridInterpolator(
        (source_lat, source_lon),
        source_elevation,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    yy, xx = np.meshgrid(target_lat[lat_mask], target_lon[lon_mask], indexing="ij")
    yy_clip = np.clip(yy, source_lat.min(), source_lat.max())
    xx_clip = np.clip(xx, source_lon.min(), source_lon.max())
    vals = interpolator(np.column_stack([yy_clip.ravel(), xx_clip.ravel()])).reshape(yy.shape)
    current = target_elevation[np.ix_(lat_mask, lon_mask)]
    fill = np.isfinite(vals) & ~np.isfinite(current)
    filled = int(fill.sum())
    if filled:
        current[fill] = vals[fill]
        target_elevation[np.ix_(lat_mask, lon_mask)] = current
        sid = target_source_id[np.ix_(lat_mask, lon_mask)]
        sid[fill] = source.source_id or SOURCE_ID.get(source.source_name, -1)
        target_source_id[np.ix_(lat_mask, lon_mask)] = sid
        res = target_source_resolution[np.ix_(lat_mask, lon_mask)]
        res[fill] = source.resolution_arcsec
        target_source_resolution[np.ix_(lat_mask, lon_mask)] = res
    return filled


def metadata_for_bathy_fetch(
    *,
    name: str,
    bbox: tuple[float, float, float, float],
    fallback_policy: str,
    spacing_arcsec: float,
    elevation: np.ndarray,
    source_id: np.ndarray,
    attempted: list[dict],
    warnings: list[str],
    outputs: dict,
) -> dict:
    total = int(source_id.size)
    counts = {
        SOURCE_LABELS.get(int(sid), str(int(sid))): int((source_id == sid).sum())
        for sid in np.unique(source_id)
    }
    coverage_by_source = {
        key: {"cells": value, "fraction": float(value / total) if total else 0.0}
        for key, value in counts.items()
    }
    return {
        "case": name,
        "bbox_wsen": list(bbox),
        "fallback_policy": fallback_policy,
        "target_spacing_arcsec": spacing_arcsec,
        "finite_output_fraction": finite_coverage_fraction(elevation),
        "no_data_cells": int((source_id == 0).sum()),
        "coverage_by_source": coverage_by_source,
        "source_legend": SOURCE_LABELS,
        "datum_notes": {
            "warning": (
                "CUDEM, CRM, and ETOPO can use different vertical datums. "
                "Values were gap-filled by priority without vertical-datum harmonization."
            ),
            "depth_conversion": "depth_m = max(-elevation_m, 0)",
        },
        "attempted_sources": attempted,
        "warnings": warnings,
        "outputs": outputs,
    }


def _load_records(index: str | Path | Sequence[BathySourceRecord | dict]) -> list[BathySourceRecord]:
    if isinstance(index, (str, Path)):
        return load_bathy_source_index(index)
    return [x if isinstance(x, BathySourceRecord) else BathySourceRecord.from_dict(x) for x in index]


def _policy_sources(policy: str) -> tuple[str, ...]:
    aliases = {
        "cudem-only": ("cudem",),
        "cudem": ("cudem",),
        "cudem-crm": ("cudem", "crm"),
        "cudem-crm-etopo": ("cudem", "crm", "etopo"),
        "all": ("cudem", "crm", "etopo"),
    }
    key = policy.lower().strip()
    if key not in aliases:
        raise ValueError("fallback_policy must be cudem-only, cudem-crm, or cudem-crm-etopo")
    return aliases[key]


def _coord_bbox_intersection(
    ds: xr.Dataset, bbox: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    lon = ds["lon"].values.astype(float)
    lat = ds["lat"].values.astype(float)
    sw = float(np.nanmin(lon))
    se = float(np.nanmax(lon))
    ss = float(np.nanmin(lat))
    sn = float(np.nanmax(lat))
    iw = max(sw, west)
    ie = min(se, east)
    isouth = max(ss, south)
    inorth = min(sn, north)
    if iw >= ie or isouth >= inorth:
        raise ValueError("Source does not overlap bbox.")
    return iw, isouth, ie, inorth


def _normalize_source_lon(ds: xr.Dataset, *, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"].values.astype(float)
    if np.nanmax(lon) > 180.0 and bbox[0] < 0.0:
        return ds.assign_coords(lon=np.where(lon > 180.0, lon - 360.0, lon))
    return ds


def _stride(coord: np.ndarray, target_spacing_arcsec: float) -> int:
    if coord.size < 2:
        return 1
    native = float(np.nanmedian(np.abs(np.diff(coord)))) * 3600.0
    if native <= 0:
        return 1
    return max(1, int(round(target_spacing_arcsec / native)))
