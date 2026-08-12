"""FVCOM mesh helpers for postprocessing maps and sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from netCDF4 import Dataset


def read_fvcom_mesh_dat(grid_dat: str | Path, dep_dat: str | Path | None = None) -> dict[str, np.ndarray]:
    """Read FVCOM ASCII grid/depth files into a small mesh dictionary."""

    grid_dat = Path(grid_dat)
    with grid_dat.open() as f:
        nvert = int(f.readline().split("=")[-1].strip())
        nelem = int(f.readline().split("=")[-1].strip())
        tri = np.zeros((nelem, 3), dtype=np.int32)
        for i in range(nelem):
            parts = f.readline().split()
            tri[i] = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]
        lon = np.empty(nvert)
        lat = np.empty(nvert)
        for i in range(nvert):
            parts = f.readline().split()
            lon[i] = float(parts[1])
            lat[i] = float(parts[2])

    mesh: dict[str, np.ndarray] = {"lon": lon, "lat": lat, "tri": tri}
    if dep_dat is not None and Path(dep_dat).exists():
        h = np.empty(nvert)
        with Path(dep_dat).open() as f:
            f.readline()
            for i in range(nvert):
                parts = f.readline().split()
                h[i] = float(parts[2])
        mesh["h"] = h
    return mesh


def _as_1d(var) -> np.ndarray:
    arr = np.asarray(var[:], dtype=float)
    return np.ravel(arr)


def mesh_from_output(path: str | Path) -> dict[str, np.ndarray]:
    """Load mesh coordinates/connectivity from an FVCOM output NetCDF file."""

    with Dataset(path) as ds:
        lon_name = "lon" if "lon" in ds.variables else "x"
        lat_name = "lat" if "lat" in ds.variables else "y"
        if lon_name not in ds.variables or lat_name not in ds.variables:
            raise KeyError(f"{path} does not contain lon/lat or x/y variables.")
        lon = _as_1d(ds.variables[lon_name])
        lat = _as_1d(ds.variables[lat_name])
        if "nv" not in ds.variables:
            raise KeyError(f"{path} does not contain FVCOM connectivity variable 'nv'.")
        nv = np.asarray(ds.variables["nv"][:], dtype=np.int64)
        tri = nv.T if nv.shape[0] == 3 else nv
        if tri.min() == 1:
            tri = tri - 1
        mesh = {"lon": lon, "lat": lat, "tri": tri.astype(np.int32)}
        for name in ("h", "lonc", "latc"):
            if name in ds.variables:
                mesh[name] = _as_1d(ds.variables[name])
        return mesh


def load_case_mesh(
    case: str,
    workspace: str | Path | None = None,
    output_file: str | Path | None = None,
    input_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load a case mesh from output first, then from FVCOM ASCII input files."""

    try:
        from .fvcom_output import discover_output_stacks, workspace_dir
    except ImportError:
        from fvcom_output import discover_output_stacks, workspace_dir

    ws = Path(workspace) if workspace is not None else workspace_dir()
    if output_file is not None:
        return mesh_from_output(output_file)

    files = discover_output_stacks(case, ws)
    for path in files:
        try:
            return mesh_from_output(path)
        except Exception:
            continue

    if input_dir is None:
        candidates = [ws / f"INPUT_{case.upper()}", ws / "INPUT" / case.upper()]
    else:
        candidates = [Path(input_dir)]

    for folder in candidates:
        grid = folder / "waterPACT_grd.dat"
        dep = folder / "waterPACT_dep.dat"
        if grid.exists():
            return read_fvcom_mesh_dat(grid, dep if dep.exists() else None)
    raise FileNotFoundError(f"Could not find mesh for {case} in outputs or input folders.")


def build_triangulation(mesh: Mapping[str, np.ndarray]):
    """Return a Matplotlib triangulation for a mesh dictionary."""

    import matplotlib.tri as mtri

    return mtri.Triangulation(np.asarray(mesh["lon"]), np.asarray(mesh["lat"]), np.asarray(mesh["tri"], dtype=np.int32))


def mesh_extent(mesh: Mapping[str, np.ndarray], pad_fraction: float = 0.02) -> tuple[float, float, float, float]:
    """Return padded lon/lat bounds as ``(xmin, xmax, ymin, ymax)``."""

    lon = np.asarray(mesh["lon"], dtype=float)
    lat = np.asarray(mesh["lat"], dtype=float)
    xmin, xmax = float(np.nanmin(lon)), float(np.nanmax(lon))
    ymin, ymax = float(np.nanmin(lat)), float(np.nanmax(lat))
    dx = max(xmax - xmin, 1.0e-6) * pad_fraction
    dy = max(ymax - ymin, 1.0e-6) * pad_fraction
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def auto_zoom_boxes(mesh: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float, float, float]]:
    """Build reusable zoom boxes from the mesh lon/lat distribution."""

    lon = np.asarray(mesh["lon"], dtype=float)
    lat = np.asarray(mesh["lat"], dtype=float)
    full = mesh_extent(mesh)
    qlon = np.nanquantile(lon, [0.02, 0.98])
    qlat = np.nanquantile(lat, [0.05, 0.35, 0.55, 0.75, 0.95])
    return {
        "full": full,
        "upper_estuary": (float(qlon[0]), float(qlon[1]), float(qlat[2]), float(qlat[4])),
        "lower_estuary": (float(qlon[0]), float(qlon[1]), float(qlat[1]), float(qlat[3])),
        "mouth_shelf": (float(qlon[0]), float(qlon[1]), float(qlat[0]), float(qlat[2])),
    }


def resolve_zoom(mesh: Mapping[str, np.ndarray], zoom: str | Sequence[float] = "full") -> tuple[float, float, float, float]:
    """Resolve a named or explicit zoom box."""

    if isinstance(zoom, str):
        boxes = auto_zoom_boxes(mesh)
        if zoom not in boxes:
            raise KeyError(f"Unknown zoom {zoom!r}. Available: {sorted(boxes)}")
        return boxes[zoom]
    vals = tuple(float(v) for v in zoom)
    if len(vals) != 4:
        raise ValueError("Explicit zoom must be xmin,xmax,ymin,ymax.")
    return vals


def apply_zoom(ax, mesh: Mapping[str, np.ndarray], zoom: str | Sequence[float] = "full") -> None:
    """Apply a named or explicit zoom box to a Matplotlib axes."""

    xmin, xmax, ymin, ymax = resolve_zoom(mesh, zoom)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def element_to_node_average(mesh: Mapping[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    """Average element-centered values to nodes."""

    tri = np.asarray(mesh["tri"], dtype=np.int32)
    values = np.asarray(values, dtype=float)
    nnode = len(mesh["lon"])
    out = np.zeros(nnode, dtype=float)
    count = np.zeros(nnode, dtype=float)
    for elem, value in zip(tri, values):
        valid = np.isfinite(value)
        if not valid:
            continue
        out[elem] += value
        count[elem] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / count
    out[count == 0] = np.nan
    return out
