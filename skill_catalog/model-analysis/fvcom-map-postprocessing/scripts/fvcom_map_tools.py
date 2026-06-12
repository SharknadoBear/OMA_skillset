"""Gridded FVCOM scalar map helpers for loaded scatter data.

The public functions here operate on already-loaded FVCOM mesh/scalar arrays:
``lon``, ``lat``, ``nv``/triangles, and one scalar field.  They are intentionally
usable with compact MAT products as well as raw NetCDF-derived arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Colormap, TwoSlopeNorm
import numpy as np

try:
    import cmocean
except ImportError:  # pragma: no cover - expected in some lightweight envs.
    cmocean = None


V007_ZOOM = (-76.0, -74.5, 38.5, 40.5)
EARTH_KM_PER_DEG = 111.32


@dataclass
class ContourGrid:
    """Regular lon/lat contour grid and interpolation metadata."""

    lon: np.ndarray
    lat: np.ndarray
    values: np.ndarray
    inside: np.ndarray
    metadata: dict[str, Any]


@dataclass
class ScalarMapResult:
    """Return object for scalar map plots."""

    fig: Any
    ax: Any
    artist: Any
    colorbar: Any
    grid: ContourGrid
    metadata: dict[str, Any]


def normalize_triangles(nv: np.ndarray) -> np.ndarray:
    """Return zero-based triangle connectivity as ``(nele, 3)``."""

    tri = np.asarray(nv, dtype=np.int64)
    tri = np.squeeze(tri)
    if tri.ndim != 2:
        raise ValueError(f"nv/triangles must be 2-D; got shape {tri.shape}")
    if tri.shape[0] == 3 and tri.shape[1] != 3:
        tri = tri.T
    if tri.shape[1] != 3:
        raise ValueError(f"triangle connectivity must have three columns; got shape {tri.shape}")
    if np.nanmin(tri) == 1:
        tri = tri - 1
    return tri.astype(np.int32, copy=False)


def standardize_longitudes(lon: np.ndarray, wrap: bool = True) -> np.ndarray:
    """Convert [0, 360] longitudes to [-180, 180] when requested."""

    out = np.asarray(lon, dtype=float).ravel().copy()
    if wrap and np.nanmax(out) > 180.0:
        out = ((out + 180.0) % 360.0) - 180.0
    return out


def _polygon_area(ring: np.ndarray) -> float:
    x = ring[:, 0]
    y = ring[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def boundary_rings_from_triangles(
    lon: np.ndarray,
    lat: np.ndarray,
    nv: np.ndarray,
    *,
    wrap_lon: bool = True,
) -> list[np.ndarray]:
    """Extract exact one-sided-edge boundary rings from FVCOM triangles.

    Returns a list of closed ``(n, 2)`` lon/lat rings sorted by descending
    absolute polygon area.  The largest ring is usually the full model
    circumference; smaller rings can represent islands or internal holes.
    """

    x = standardize_longitudes(lon, wrap=wrap_lon)
    y = np.asarray(lat, dtype=float).ravel()
    tri = normalize_triangles(nv)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    adjacency: dict[int, list[int]] = {}
    unvisited: set[tuple[int, int]] = set()
    for a_raw, b_raw in boundary_edges:
        a = int(a_raw)
        b = int(b_raw)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        unvisited.add((min(a, b), max(a, b)))

    rings: list[np.ndarray] = []
    while unvisited:
        start, nxt = next(iter(unvisited))
        ring_nodes = [start, nxt]
        unvisited.remove((start, nxt))
        prev = start
        current = nxt
        guard = 0
        while current != start and guard <= len(boundary_edges) + 5:
            guard += 1
            candidates = [node for node in adjacency.get(current, []) if node != prev]
            if not candidates:
                break
            chosen = None
            for candidate in candidates:
                edge = (min(current, candidate), max(current, candidate))
                if edge in unvisited:
                    chosen = candidate
                    break
            if chosen is None and start in candidates:
                chosen = start
            if chosen is None:
                break
            edge = (min(current, chosen), max(current, chosen))
            unvisited.discard(edge)
            ring_nodes.append(chosen)
            prev, current = current, chosen

        if ring_nodes[-1] != ring_nodes[0]:
            ring_nodes.append(ring_nodes[0])
        coords = np.column_stack([x[ring_nodes], y[ring_nodes]])
        if coords.shape[0] >= 4:
            rings.append(coords)

    rings.sort(key=lambda arr: abs(_polygon_area(arr)), reverse=True)
    return rings


def resolve_map_zoom(
    lon: np.ndarray,
    lat: np.ndarray,
    zoom: str | Sequence[float] = "full",
    *,
    pad_fraction: float = 0.02,
) -> tuple[float, float, float, float]:
    """Resolve named or explicit lon/lat zoom bounds."""

    x = np.asarray(lon, dtype=float).ravel()
    y = np.asarray(lat, dtype=float).ravel()
    if isinstance(zoom, str):
        key = zoom.lower()
        if key == "v007":
            return V007_ZOOM
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if key == "full":
            dx = max((xmax - xmin) * pad_fraction, 1.0e-6)
            dy = max((ymax - ymin) * pad_fraction, 1.0e-6)
            return xmin - dx, xmax + dx, ymin - dy, ymax + dy
        qlon = np.nanquantile(x, [0.02, 0.98])
        qlat = np.nanquantile(y, [0.05, 0.35, 0.55, 0.75, 0.95])
        boxes = {
            "upper_estuary": (float(qlon[0]), float(qlon[1]), float(qlat[2]), float(qlat[4])),
            "lower_estuary": (float(qlon[0]), float(qlon[1]), float(qlat[1]), float(qlat[3])),
            "mouth_shelf": (float(qlon[0]), float(qlon[1]), float(qlat[0]), float(qlat[2])),
        }
        if key not in boxes:
            raise KeyError(f"Unknown zoom {zoom!r}. Use full, v007, upper_estuary, lower_estuary, mouth_shelf, or bounds.")
        return boxes[key]

    vals = tuple(float(v) for v in zoom)
    if len(vals) != 4:
        raise ValueError("Explicit zoom must be xmin, xmax, ymin, ymax.")
    return vals


def _unique_edges(tri: np.ndarray) -> np.ndarray:
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _edge_lengths_km(lon: np.ndarray, lat: np.ndarray, tri: np.ndarray, zoom_bounds: tuple[float, float, float, float]) -> np.ndarray:
    edges = _unique_edges(tri)
    xmin, xmax, ymin, ymax = zoom_bounds
    visible = (
        (lon[edges[:, 0]] >= xmin)
        & (lon[edges[:, 0]] <= xmax)
        & (lat[edges[:, 0]] >= ymin)
        & (lat[edges[:, 0]] <= ymax)
        & (lon[edges[:, 1]] >= xmin)
        & (lon[edges[:, 1]] <= xmax)
        & (lat[edges[:, 1]] >= ymin)
        & (lat[edges[:, 1]] <= ymax)
    )
    if not np.any(visible):
        visible = np.ones(edges.shape[0], dtype=bool)
    e = edges[visible]
    mid_lat = np.deg2rad((lat[e[:, 0]] + lat[e[:, 1]]) * 0.5)
    dx = (lon[e[:, 1]] - lon[e[:, 0]]) * EARTH_KM_PER_DEG * np.cos(mid_lat)
    dy = (lat[e[:, 1]] - lat[e[:, 0]]) * EARTH_KM_PER_DEG
    length = np.sqrt(dx * dx + dy * dy)
    return length[np.isfinite(length) & (length > 0)]


def _grid_shape_for_spacing(
    bounds: tuple[float, float, float, float],
    spacing_km: float,
) -> tuple[int, int, float, float]:
    xmin, xmax, ymin, ymax = bounds
    mean_lat = 0.5 * (ymin + ymax)
    coslat = max(abs(math.cos(math.radians(mean_lat))), 0.05)
    dlat = spacing_km / EARTH_KM_PER_DEG
    dlon = spacing_km / (EARTH_KM_PER_DEG * coslat)
    nx = max(int(math.ceil((xmax - xmin) / dlon)) + 1, 2)
    ny = max(int(math.ceil((ymax - ymin) / dlat)) + 1, 2)
    return nx, ny, dlon, dlat


def _adaptive_grid(
    bounds: tuple[float, float, float, float],
    requested_spacing_km: float,
    max_grid_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spacing = float(requested_spacing_km)
    nx, ny, dlon, dlat = _grid_shape_for_spacing(bounds, spacing)
    cap_applied = False
    if max_grid_points and nx * ny > max_grid_points:
        cap_applied = True
        spacing *= math.sqrt((nx * ny) / float(max_grid_points)) * 1.001
        nx, ny, dlon, dlat = _grid_shape_for_spacing(bounds, spacing)
        while nx * ny > max_grid_points:
            spacing *= 1.02
            nx, ny, dlon, dlat = _grid_shape_for_spacing(bounds, spacing)
    xmin, xmax, ymin, ymax = bounds
    grid_lon = np.linspace(xmin, xmax, nx)
    grid_lat = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(grid_lon, grid_lat)
    meta = {
        "requested_spacing_km": float(requested_spacing_km),
        "actual_spacing_km": float(spacing),
        "grid_shape": [int(ny), int(nx)],
        "grid_point_count": int(nx * ny),
        "max_grid_points": int(max_grid_points),
        "cap_applied": bool(cap_applied),
        "dlon_degree": float(dlon),
        "dlat_degree": float(dlat),
    }
    return xx, yy, meta


def _element_to_node_average(tri: np.ndarray, values: np.ndarray, nnode: int) -> np.ndarray:
    out = np.zeros(nnode, dtype=float)
    count = np.zeros(nnode, dtype=float)
    for elem, value in zip(tri, values):
        if not np.isfinite(value):
            continue
        out[elem] += float(value)
        count[elem] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / count
    out[count == 0] = np.nan
    return out


def _values_to_nodes(values: np.ndarray, tri: np.ndarray, nnode: int) -> tuple[np.ndarray, str]:
    arr = np.asarray(values, dtype=float).squeeze()
    if arr.size == nnode:
        return arr.ravel(), "node"
    if arr.size == tri.shape[0]:
        return _element_to_node_average(tri, arr.ravel(), nnode), "element_to_node_average"
    raise ValueError(f"values size {arr.size} does not match node count {nnode} or element count {tri.shape[0]}.")


def scatter_to_contour_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    nv: np.ndarray,
    values: np.ndarray,
    zoom: str | Sequence[float] = "full",
    grid_spacing_km: float | None = None,
    max_grid_points: int = 1_500_000,
    edge_quantile: float = 5.0,
    *,
    wrap_lon: bool = True,
) -> ContourGrid:
    """Interpolate scattered FVCOM scalar data to a masked regular grid."""

    x = standardize_longitudes(lon, wrap=wrap_lon)
    y = np.asarray(lat, dtype=float).ravel()
    tri = normalize_triangles(nv)
    node_values, source_kind = _values_to_nodes(values, tri, x.size)
    bounds = resolve_map_zoom(x, y, zoom)

    if grid_spacing_km is None:
        lengths = _edge_lengths_km(x, y, tri, bounds)
        if lengths.size:
            requested_spacing = float(np.nanpercentile(lengths, edge_quantile))
        else:
            requested_spacing = 0.1
    else:
        requested_spacing = float(grid_spacing_km)

    grid_lon, grid_lat, grid_meta = _adaptive_grid(bounds, requested_spacing, max_grid_points)
    triangulation = mtri.Triangulation(x, y, tri)
    interpolator = mtri.LinearTriInterpolator(triangulation, node_values)
    grid_values = interpolator(grid_lon, grid_lat)
    if np.ma.isMaskedArray(grid_values):
        grid_values = grid_values.filled(np.nan)
    else:
        grid_values = np.asarray(grid_values, dtype=float)
    finder = triangulation.get_trifinder()
    inside = finder(grid_lon, grid_lat) >= 0
    grid_values = np.asarray(grid_values, dtype=float)
    grid_values[~inside] = np.nan

    metadata = {
        **grid_meta,
        "zoom_bounds": [float(v) for v in bounds],
        "edge_quantile": float(edge_quantile),
        "source_kind": source_kind,
        "node_count": int(x.size),
        "element_count": int(tri.shape[0]),
        "inside_grid_point_count": int(np.count_nonzero(inside)),
    }
    return ContourGrid(grid_lon, grid_lat, grid_values, inside, metadata)


def transform_values(values: np.ndarray, scale: str = "linear") -> tuple[np.ndarray, str]:
    """Apply display transform to scalar values."""

    arr = np.asarray(values, dtype=float)
    if scale == "linear":
        return arr, ""
    if scale == "log10":
        out = np.full(arr.shape, np.nan, dtype=float)
        mask = np.isfinite(arr) & (arr > 0)
        out[mask] = np.log10(arr[mask])
        return out, "log10"
    raise ValueError("scale must be 'linear' or 'log10'.")


def colormap_for_variable(variable_name: str | None = None, values: np.ndarray | None = None):
    """Return a cmocean-aware colormap for common FVCOM scalar names."""

    if cmocean is None:
        if values is not None:
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size and np.nanmin(finite) < 0 < np.nanmax(finite):
                return plt.get_cmap("seismic")
        return plt.get_cmap("viridis")

    name = (variable_name or "").lower()
    if "sal" in name or "salt" in name:
        return cmocean.cm.haline
    if name.startswith("coarse_sand") or "sediment" in name or "floc" in name:
        return cmocean.cm.turbid
    if name.startswith("mp") or "plastic" in name or "bot_mass" in name:
        return cmocean.cm.matter
    if any(token in name for token in ("u_", "v_", "velocity", "vel", "speed", "settle")):
        return cmocean.cm.balance
    if values is not None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size and np.nanmin(finite) < 0 < np.nanmax(finite):
            return cmocean.cm.balance
    return cmocean.cm.matter


def resolve_colormap(cmap: str | Colormap | None, variable_name: str | None, values: np.ndarray | None = None):
    if cmap is None or cmap == "auto":
        return colormap_for_variable(variable_name, values)
    if isinstance(cmap, str):
        if cmocean is not None and hasattr(cmocean.cm, cmap):
            return getattr(cmocean.cm, cmap)
        return plt.get_cmap(cmap)
    return cmap


def quantile_limits(
    values: np.ndarray,
    quantiles: tuple[float, float] | None = (2.0, 98.0),
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float | None, float | None]:
    """Resolve color limits from explicit values or display quantiles."""

    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return vmin, vmax
    if quantiles is None:
        qmin, qmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        qmin, qmax = (float(np.nanpercentile(finite, quantiles[0])), float(np.nanpercentile(finite, quantiles[1])))
    lo = float(vmin) if vmin is not None else qmin
    hi = float(vmax) if vmax is not None else qmax
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _projector():
    try:
        from pyproj import Transformer

        to_web = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        from_web = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        return to_web, from_web
    except Exception:
        return None, None


def _transform_xy(x: np.ndarray, y: np.ndarray, transformer) -> tuple[np.ndarray, np.ndarray]:
    if transformer is None:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xx, yy = transformer.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(xx, dtype=float), np.asarray(yy, dtype=float)


def _background_source(name: str):
    import contextily as ctx

    key = name.lower()
    if key in {"cartodb_positron", "positron"}:
        return ctx.providers.CartoDB.Positron
    if key in {"cartodb_voyager", "voyager"}:
        return ctx.providers.CartoDB.Voyager
    if key in {"osm", "openstreetmap"}:
        return ctx.providers.OpenStreetMap.Mapnik
    raise KeyError(f"Unknown background {name!r}. Use none, cartodb_positron, cartodb_voyager, or osm.")


def _add_background(ax, background: str, zoom: int | str = "auto") -> bool:
    if background.lower() in {"none", "off", "false"}:
        return False
    try:
        import contextily as ctx

        ctx.add_basemap(ax, source=_background_source(background), zoom=zoom, attribution_size=6)
        return True
    except Exception as exc:
        warnings.warn(f"Could not add background map {background!r}: {exc}", RuntimeWarning)
        return False


def _set_lonlat_ticks(ax, bounds: tuple[float, float, float, float], transformer) -> None:
    xmin, xmax, ymin, ymax = bounds
    if transformer is None:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return
    lon_ticks = np.arange(np.floor(xmin * 2.0) / 2.0, np.ceil(xmax * 2.0) / 2.0 + 0.1, 0.5)
    lat_ticks = np.arange(np.floor(ymin * 2.0) / 2.0, np.ceil(ymax * 2.0) / 2.0 + 0.1, 0.5)
    lon_ticks = lon_ticks[(lon_ticks >= xmin) & (lon_ticks <= xmax)]
    lat_ticks = lat_ticks[(lat_ticks >= ymin) & (lat_ticks <= ymax)]
    x_ticks, _ = _transform_xy(lon_ticks, np.full_like(lon_ticks, 0.5 * (ymin + ymax)), transformer)
    _, y_ticks = _transform_xy(np.full_like(lat_ticks, 0.5 * (xmin + xmax)), lat_ticks, transformer)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{abs(v):.1f}W" if v < 0 else f"{v:.1f}E" for v in lon_ticks])
    ax.set_yticklabels([f"{v:.1f}N" if v >= 0 else f"{abs(v):.1f}S" for v in lat_ticks])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _levels(vmin: float | None, vmax: float | None, n_levels: int) -> int | np.ndarray:
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return n_levels
    return np.linspace(float(vmin), float(vmax), n_levels + 1)


def plot_fvcom_scalar_map(
    lon: np.ndarray,
    lat: np.ndarray,
    nv: np.ndarray,
    values: np.ndarray,
    *,
    ax=None,
    variable_name: str | None = None,
    title: str | None = None,
    zoom: str | Sequence[float] = "full",
    grid_spacing_km: float | None = None,
    max_grid_points: int = 1_500_000,
    edge_quantile: float = 5.0,
    scale: str = "linear",
    cmap: str | Colormap | None = "auto",
    quantiles: tuple[float, float] | None = (2.0, 98.0),
    vmin: float | None = None,
    vmax: float | None = None,
    contour_lines: bool = False,
    contour_line_color: str = "0.2",
    contour_line_width: float = 0.35,
    filled_alpha: float | None = None,
    filled_levels: int = 40,
    line_levels: int = 10,
    boundary: bool = True,
    boundary_color: str = "black",
    boundary_linewidth: float = 0.8,
    background: str = "none",
    background_zoom: int | str = "auto",
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    figsize: tuple[float, float] = (8.0, 7.0),
    wrap_lon: bool = True,
) -> ScalarMapResult:
    """Plot a gridded, masked FVCOM scalar contour map."""

    x = standardize_longitudes(lon, wrap=wrap_lon)
    y = np.asarray(lat, dtype=float).ravel()
    tri = normalize_triangles(nv)
    display_values, transform_label = transform_values(values, scale)
    grid = scatter_to_contour_grid(
        x,
        y,
        tri,
        display_values,
        zoom=zoom,
        grid_spacing_km=grid_spacing_km,
        max_grid_points=max_grid_points,
        edge_quantile=edge_quantile,
        wrap_lon=False,
    )
    lo, hi = quantile_limits(grid.values, quantiles, vmin=vmin, vmax=vmax)
    colormap = resolve_colormap(cmap, variable_name, grid.values)
    norm = None
    finite = grid.values[np.isfinite(grid.values)]
    if finite.size and lo is not None and hi is not None and np.nanmin(finite) < 0 < np.nanmax(finite):
        limit = max(abs(float(lo)), abs(float(hi)))
        if limit > 0:
            lo, hi = -limit, limit
            norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)

    own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    use_background = background.lower() not in {"none", "off", "false"}
    to_web, _ = _projector() if use_background else (None, None)
    if use_background and to_web is None:
        warnings.warn("pyproj is unavailable; drawing without projected basemap.", RuntimeWarning)
        use_background = False
    transformer = to_web if use_background else None
    plot_x, plot_y = _transform_xy(grid.lon, grid.lat, transformer)
    bounds = tuple(grid.metadata["zoom_bounds"])
    xlim_raw, ylim_raw = _transform_xy(np.asarray(bounds[:2]), np.asarray(bounds[2:]), transformer)
    ax.set_xlim(float(np.nanmin(xlim_raw)), float(np.nanmax(xlim_raw)))
    ax.set_ylim(float(np.nanmin(ylim_raw)), float(np.nanmax(ylim_raw)))
    if use_background:
        _add_background(ax, background, background_zoom)

    contour_kwargs: dict[str, Any] = {"cmap": colormap, "levels": _levels(lo, hi, filled_levels), "extend": "both"}
    if filled_alpha is not None:
        contour_kwargs["alpha"] = float(filled_alpha)
    if norm is not None:
        contour_kwargs["norm"] = norm
    else:
        contour_kwargs["vmin"] = lo
        contour_kwargs["vmax"] = hi
    artist = ax.contourf(plot_x, plot_y, grid.values, **contour_kwargs)

    if contour_lines and np.isfinite(grid.values).any():
        try:
            ax.contour(
                plot_x,
                plot_y,
                grid.values,
                levels=_levels(lo, hi, line_levels),
                colors=contour_line_color,
                linewidths=contour_line_width,
                alpha=0.65,
            )
        except Exception as exc:
            warnings.warn(f"Could not draw contour lines: {exc}", RuntimeWarning)

    ring_count = 0
    if boundary:
        for ring in boundary_rings_from_triangles(x, y, tri, wrap_lon=False):
            bx, by = _transform_xy(ring[:, 0], ring[:, 1], transformer)
            ax.plot(bx, by, color=boundary_color, linewidth=boundary_linewidth, zorder=5)
            ring_count += 1

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title if title is not None else (variable_name or "FVCOM scalar field"))
    _set_lonlat_ticks(ax, bounds, transformer)
    colorbar = None
    if add_colorbar:
        label = colorbar_label
        if label is None:
            base = variable_name or "scalar"
            label = base if not transform_label else f"{transform_label}({base})"
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.85)
        colorbar.set_label(label)

    metadata = {
        **grid.metadata,
        "variable_name": variable_name,
        "scale": scale,
        "transform_label": transform_label,
        "vmin": None if lo is None else float(lo),
        "vmax": None if hi is None else float(hi),
        "quantiles": None if quantiles is None else [float(quantiles[0]), float(quantiles[1])],
        "contour_lines": bool(contour_lines),
        "filled_alpha": filled_alpha,
        "background": background,
        "boundary_ring_count": ring_count,
        "created_figure": bool(own_fig),
    }
    return ScalarMapResult(fig=fig, ax=ax, artist=artist, colorbar=colorbar, grid=grid, metadata=metadata)
