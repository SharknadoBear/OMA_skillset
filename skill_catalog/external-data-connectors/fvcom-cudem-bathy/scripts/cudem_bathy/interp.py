"""Interpolate CUDEM bathymetry products to FVCOM mesh nodes or points."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


def read_points(
    *,
    mesh_2dm: str | Path | None = None,
    grid_dat: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read target points and return node_id, lon, lat arrays."""

    given = [x is not None for x in (mesh_2dm, grid_dat, csv_path)]
    if sum(given) != 1:
        raise ValueError("Provide exactly one of mesh_2dm, grid_dat, or csv_path.")
    if mesh_2dm is not None:
        return _read_2dm_nodes(mesh_2dm)
    if grid_dat is not None:
        return _read_grid_dat_nodes(grid_dat)
    return _read_csv_points(csv_path)  # type: ignore[arg-type]


def interpolate_to_points(
    bathy_netcdf: str | Path,
    *,
    mesh_2dm: str | Path | None = None,
    grid_dat: str | Path | None = None,
    csv_path: str | Path | None = None,
    method: str = "linear",
) -> dict[str, np.ndarray]:
    """Interpolate elevation and depth fields to requested points."""

    node_id, lon, lat = read_points(mesh_2dm=mesh_2dm, grid_dat=grid_dat, csv_path=csv_path)
    ds = xr.open_dataset(bathy_netcdf)
    lat_src = ds["lat"].values.astype(float)
    lon_src = ds["lon"].values.astype(float)
    elevation = ds["elevation_m"].values.astype(float)
    depth = ds["depth_m"].values.astype(float)

    pts = np.column_stack([lat, lon])
    out: dict[str, np.ndarray] = {"node_id": node_id, "lon": lon, "lat": lat}
    for name, values in (("elevation_m", elevation), ("depth_m", depth)):
        interp = RegularGridInterpolator(
            (lat_src, lon_src),
            values,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        out[name] = interp(pts)
    out["status"] = np.where(np.isfinite(out["depth_m"]), "ok", "outside_or_missing")
    return out


def write_points_csv(result: dict[str, np.ndarray], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["node_id", "lon", "lat", "elevation_m", "depth_m", "status"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        n = len(result["lon"])
        for i in range(n):
            writer.writerow([result[field][i] for field in fields])
    return output_csv


def _read_2dm_nodes(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[int] = []
    lon: list[float] = []
    lat: list[float] = []
    with Path(path).open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ND"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                ids.append(int(parts[1]))
                lon.append(float(parts[2]))
                lat.append(float(parts[3]))
    if not ids:
        raise ValueError(f"No ND nodes found in {path}")
    return np.asarray(ids, dtype=np.int64), np.asarray(lon), np.asarray(lat)


def _read_grid_dat_nodes(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Path(path).open(encoding="utf-8", errors="ignore") as f:
        nvert = int(f.readline().split("=")[-1].strip())
        nelem = int(f.readline().split("=")[-1].strip())
        for _ in range(nelem):
            f.readline()
        ids = np.arange(1, nvert + 1, dtype=np.int64)
        lon = np.empty(nvert, dtype=float)
        lat = np.empty(nvert, dtype=float)
        for i in range(nvert):
            parts = f.readline().split()
            lon[i] = float(parts[1])
            lat[i] = float(parts[2])
    return ids, lon, lat


def _read_csv_points(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No points found in {path}")
    lon_key = _find_key(rows[0], ("lon", "longitude", "x"))
    lat_key = _find_key(rows[0], ("lat", "latitude", "y"))
    id_key = _find_key(rows[0], ("node_id", "id", "point_id"), required=False)
    lon = np.asarray([float(row[lon_key]) for row in rows], dtype=float)
    lat = np.asarray([float(row[lat_key]) for row in rows], dtype=float)
    if id_key is None:
        node_id = np.arange(1, len(rows) + 1, dtype=np.int64)
    else:
        node_id = np.asarray([int(float(row[id_key])) for row in rows], dtype=np.int64)
    return node_id, lon, lat


def _find_key(row: dict, names: tuple[str, ...], *, required: bool = True) -> str | None:
    lower = {key.lower(): key for key in row.keys()}
    for name in names:
        if name in lower:
            return lower[name]
    if required:
        raise ValueError(f"CSV must contain one of: {', '.join(names)}")
    return None
