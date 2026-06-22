"""Fetch and mosaic NOAA CUDEM tiles for FVCOM preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from .catalog import load_tile_index
from .normalize import elevation_to_depth, finite_coverage_fraction
from .plot import plot_bathymetry_map
from .tiles import TileRecord, coverage_fraction, normalize_bbox, select_tiles

NOAA_CUDEM_CITATION = (
    "Cooperative Institute for Research in Environmental Sciences (CIRES) at "
    "the University of Colorado, Boulder. Continuously Updated Digital "
    "Elevation Model (CUDEM). NOAA National Centers for Environmental "
    "Information. Not for navigation."
)


@dataclass(frozen=True)
class FetchResult:
    """Output paths and metadata for one CUDEM bbox fetch."""

    netcdf_path: Path
    png_path: Path
    metadata_path: Path
    metadata: dict


def fetch_cudem_bbox(
    index: str | Path | Sequence[TileRecord | dict],
    bbox: Sequence[float],
    *,
    run_dir: str | Path,
    name: str,
    resolution: str = "auto",
    max_tiles: int = 48,
    target_spacing_arcsec: float | None = 3.0,
    source_preference: Sequence[str] = ("opendap_netcdf", "https_geotiff"),
    make_plot: bool = True,
) -> FetchResult:
    """Fetch selected CUDEM tiles, mosaic them, and write NetCDF/PNG/JSON."""

    bbox = normalize_bbox(bbox)
    records = load_tile_index(index) if isinstance(index, (str, Path)) else list(index)
    selected = select_tiles(
        records,
        bbox,
        resolution=resolution,
        max_tiles=max_tiles,
        source_preference=source_preference,
    )
    native_arcsec = min(tile.resolution_arcsec for tile in selected)
    spacing_arcsec = native_arcsec if target_spacing_arcsec is None else max(
        float(target_spacing_arcsec), native_arcsec
    )
    target_lon, target_lat = _target_grid(bbox, spacing_arcsec)
    elevation = np.full((target_lat.size, target_lon.size), np.nan, dtype=np.float32)

    warnings: list[str] = []
    for tile in selected:
        try:
            tile_lon, tile_lat, tile_elevation = _read_tile(
                tile, bbox=bbox, target_spacing_arcsec=spacing_arcsec
            )
            _burn_tile_into_target(
                target_lon,
                target_lat,
                elevation,
                tile_lon,
                tile_lat,
                tile_elevation,
            )
        except Exception as exc:
            warnings.append(f"{tile.name}: {type(exc).__name__}: {exc}")

    if not np.isfinite(elevation).any():
        raise RuntimeError(
            "Selected CUDEM tiles were found, but no finite elevation values were read. "
            f"Warnings: {warnings}"
        )

    depth, wet = elevation_to_depth(elevation)
    ds = xr.Dataset(
        data_vars={
            "elevation_m": (("lat", "lon"), elevation),
            "depth_m": (("lat", "lon"), depth),
            "wet_mask": (("lat", "lon"), wet.astype(np.int8)),
        },
        coords={"lat": target_lat.astype(np.float64), "lon": target_lon.astype(np.float64)},
        attrs={
            "title": f"NOAA CUDEM bathymetry subset for {name}",
            "summary": "CUDEM elevation and FVCOM positive-down depth mosaic.",
            "source": "NOAA CUDEM via THREDDS OPeNDAP and/or Digital Coast HTTPS",
            "citation": NOAA_CUDEM_CITATION,
            "horizontal_datum": "NAD83 / EPSG:4269 where provided by CUDEM tiles",
            "vertical_datum": "NAVD88 where provided by CUDEM tiles",
            "vertical_units": "meters",
            "bbox_wsen": json.dumps(list(bbox)),
            "target_spacing_arcsec": spacing_arcsec,
            "selected_collection": selected[0].collection,
        },
    )
    ds["elevation_m"].attrs.update(
        {"long_name": "CUDEM elevation", "units": "m", "positive": "up"}
    )
    ds["depth_m"].attrs.update(
        {"long_name": "FVCOM positive-down depth from CUDEM", "units": "m", "positive": "down"}
    )
    ds["wet_mask"].attrs.update({"long_name": "Water mask from elevation_m < 0"})

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    nc_path = run_dir / f"{name}_cudem_bathy.nc"
    png_path = run_dir / f"{name}_cudem_bathy.png"
    metadata_path = run_dir / f"{name}_metadata.json"
    ds.to_netcdf(nc_path)
    if make_plot:
        plot_bathymetry_map(ds, png_path, title=f"{name} CUDEM elevation", bbox=bbox)

    metadata = _metadata(
        name=name,
        bbox=bbox,
        selected=selected,
        spacing_arcsec=spacing_arcsec,
        finite_fraction=finite_coverage_fraction(elevation),
        tile_fraction=coverage_fraction(selected, bbox),
        warnings=warnings,
        outputs={"netcdf": str(nc_path), "png": str(png_path), "metadata": str(metadata_path)},
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return FetchResult(nc_path, png_path, metadata_path, metadata)


def _read_tile(
    tile: TileRecord,
    *,
    bbox: tuple[float, float, float, float],
    target_spacing_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tile.source_mode == "opendap_netcdf":
        return _read_opendap_tile(tile, bbox=bbox, target_spacing_arcsec=target_spacing_arcsec)
    if tile.source_mode == "https_geotiff":
        return _read_geotiff_tile(tile, bbox=bbox, target_spacing_arcsec=target_spacing_arcsec)
    raise ValueError(f"Unsupported CUDEM source_mode: {tile.source_mode}")


def _read_opendap_tile(
    tile: TileRecord,
    *,
    bbox: tuple[float, float, float, float],
    target_spacing_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    west, south, east, north = _intersection(tile.bbox, bbox)
    ds = xr.open_dataset(tile.url, decode_times=False)
    ds = ds.sortby("lat").sortby("lon")
    var = _find_elevation_var(ds)
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
    return sub["lon"].values.astype(float), sub["lat"].values.astype(float), values


def _read_geotiff_tile(
    tile: TileRecord,
    *,
    bbox: tuple[float, float, float, float],
    target_spacing_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import from_bounds

    west, south, east, north = _intersection(tile.bbox, bbox)
    spacing_deg = target_spacing_arcsec / 3600.0
    open_candidates = (tile.url, f"/vsicurl/{tile.url}")
    last_error: Exception | None = None
    for src_path in open_candidates:
        try:
            with rasterio.Env(
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_MAX_RETRY="3",
                GDAL_HTTP_RETRY_DELAY="2",
            ):
                with rasterio.open(src_path) as src:
                    window = from_bounds(west, south, east, north, src.transform)
                    window = window.round_offsets().round_lengths()
                    if window.width <= 0 or window.height <= 0:
                        raise ValueError("GeoTIFF window is empty.")
                    out_w = max(2, int(np.ceil((east - west) / spacing_deg)) + 1)
                    out_h = max(2, int(np.ceil((north - south) / spacing_deg)) + 1)
                    data = src.read(
                        1,
                        window=window,
                        out_shape=(out_h, out_w),
                        resampling=Resampling.bilinear,
                        masked=True,
                    ).astype(np.float32)
                    values = np.asarray(data.filled(np.nan), dtype=np.float32)
                    if src.nodata is not None:
                        values = np.where(values == src.nodata, np.nan, values)
                    left, bottom, right, top = window_bounds(window, src.transform)
                    lon = np.linspace(
                        left + (right - left) / (2 * out_w),
                        right - (right - left) / (2 * out_w),
                        out_w,
                    )
                    lat_desc = np.linspace(
                        top - (top - bottom) / (2 * out_h),
                        bottom + (top - bottom) / (2 * out_h),
                        out_h,
                    )
                    lat = lat_desc[::-1]
                    values = values[::-1, :]
                    return lon.astype(float), lat.astype(float), values
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read GeoTIFF tile {tile.url}: {last_error}")


def _burn_tile_into_target(
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    target_elevation: np.ndarray,
    tile_lon: np.ndarray,
    tile_lat: np.ndarray,
    tile_elevation: np.ndarray,
) -> None:
    if tile_lon.size < 2 or tile_lat.size < 2:
        return
    lon_mask = (target_lon >= tile_lon.min()) & (target_lon <= tile_lon.max())
    lat_mask = (target_lat >= tile_lat.min()) & (target_lat <= tile_lat.max())
    if not lon_mask.any() or not lat_mask.any():
        return

    interpolator = RegularGridInterpolator(
        (tile_lat, tile_lon),
        tile_elevation,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    yy, xx = np.meshgrid(target_lat[lat_mask], target_lon[lon_mask], indexing="ij")
    vals = interpolator(np.column_stack([yy.ravel(), xx.ravel()])).reshape(yy.shape)
    current = target_elevation[np.ix_(lat_mask, lon_mask)]
    fill = np.isfinite(vals) & ~np.isfinite(current)
    current[fill] = vals[fill]
    target_elevation[np.ix_(lat_mask, lon_mask)] = current


def _target_grid(
    bbox: tuple[float, float, float, float], spacing_arcsec: float
) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = bbox
    spacing = spacing_arcsec / 3600.0
    lon = np.arange(west, east + 0.5 * spacing, spacing, dtype=np.float64)
    lat = np.arange(south, north + 0.5 * spacing, spacing, dtype=np.float64)
    if lon[-1] > east:
        lon[-1] = east
    if lat[-1] > north:
        lat[-1] = north
    return lon, lat


def _find_elevation_var(ds: xr.Dataset) -> str:
    if "Band1" in ds.data_vars:
        return "Band1"
    preferred = ("elevation", "elevation_m", "z", "depth", "bathymetry")
    for name in preferred:
        if name in ds.data_vars and {"lat", "lon"}.issubset(ds[name].dims):
            return name
    for name, da in ds.data_vars.items():
        if {"lat", "lon"}.issubset(da.dims) and da.ndim == 2:
            return name
    raise ValueError("Could not identify a 2D lat/lon elevation variable.")


def _stride(coord: np.ndarray, target_spacing_arcsec: float) -> int:
    if coord.size < 2:
        return 1
    native = float(np.nanmedian(np.abs(np.diff(coord)))) * 3600.0
    if native <= 0:
        return 1
    return max(1, int(round(target_spacing_arcsec / native)))


def _intersection(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    west = max(a[0], b[0])
    south = max(a[1], b[1])
    east = min(a[2], b[2])
    north = min(a[3], b[3])
    if west >= east or south >= north:
        raise ValueError("Tile does not overlap bbox.")
    return west, south, east, north


def _metadata(
    *,
    name: str,
    bbox: tuple[float, float, float, float],
    selected: Sequence[TileRecord],
    spacing_arcsec: float,
    finite_fraction: float,
    tile_fraction: float,
    warnings: list[str],
    outputs: dict,
) -> dict:
    source_modes = sorted({tile.source_mode for tile in selected})
    return {
        "case": name,
        "bbox_wsen": list(bbox),
        "selected_collection": selected[0].collection,
        "selected_resolution_arcsec": selected[0].resolution_arcsec,
        "target_spacing_arcsec": spacing_arcsec,
        "source_modes": source_modes,
        "n_tiles": len(selected),
        "tile_coverage_fraction": tile_fraction,
        "finite_output_fraction": finite_fraction,
        "datum_notes": {
            "horizontal": "NAD83 / EPSG:4269 where provided by CUDEM tiles",
            "vertical": "NAVD88 where provided by CUDEM tiles",
            "units": "meters",
            "depth_conversion": "depth_m = max(-elevation_m, 0)",
        },
        "citation": NOAA_CUDEM_CITATION,
        "tiles": [tile.to_dict() for tile in selected],
        "warnings": warnings,
        "outputs": outputs,
    }
