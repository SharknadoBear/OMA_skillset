from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np

from surface_fluxes_core import DAY_MS, MJD_EPOCH_MS, MeshGeometry, _parse_time_text, sha256_file


NON_FIELD_VARIABLES = {
    "iint",
    "time",
    "Itime",
    "Itime2",
    "Times",
    "XLAT",
    "XLONG",
    "lon",
    "lat",
    "lonc",
    "latc",
    "node_id",
    "element_id",
}


def _as_float(variable: Any) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def _decode_exact_time(dataset: nc4.Dataset) -> np.ndarray:
    days = np.asarray(dataset.variables["Itime"][:], dtype=np.int64)
    millis = np.asarray(dataset.variables["Itime2"][:], dtype=np.int64)
    exact = MJD_EPOCH_MS + days * DAY_MS + millis
    texts = nc4.chartostring(dataset.variables["Times"][:])
    text_ms = np.asarray([_parse_time_text(str(value), True) for value in texts], dtype=np.int64)
    if not np.array_equal(exact, text_ms):
        raise ValueError("Times and Itime/Itime2 disagree at millisecond precision")
    expected_mjd = (days.astype(np.float64) + millis.astype(np.float64) / DAY_MS).astype(np.float32)
    actual_mjd = np.asarray(dataset.variables["time"][:], dtype=np.float32)
    if not np.array_equal(expected_mjd, actual_mjd):
        raise ValueError("Legacy float32 time is not the canonical float32 MJD representation")
    if np.any(np.diff(exact) <= 0):
        raise ValueError("Forcing time is not strictly increasing")
    return exact


def _locations(dataset: nc4.Dataset, variable: Any) -> tuple[np.ndarray, np.ndarray]:
    dims = variable.dimensions
    if "south_north" in dims:
        return _as_float(dataset.variables["XLONG"]), _as_float(dataset.variables["XLAT"])
    if "nele" in dims:
        return _as_float(dataset.variables["lonc"]), _as_float(dataset.variables["latc"])
    return _as_float(dataset.variables["lon"]), _as_float(dataset.variables["lat"])


def _representatives(width: int, count: int = 5) -> np.ndarray:
    return np.unique(np.linspace(0, width - 1, min(count, width), dtype=int))


def _flatten_time_space(values: np.ndarray) -> np.ndarray:
    return values.reshape(values.shape[0], -1)


def _time_objects(times_ms: np.ndarray) -> np.ndarray:
    return times_ms.astype("datetime64[ms]").astype(object)


def _wind_diagnostics_to_earth(dataset, loaded):
    """Convert only the plot inputs after publishing native component statistics."""
    if getattr(dataset, "wind_vector_frame", "earth") != "model_xy":
        return
    from pyproj import Proj

    u = loaded["uwind_stress"][0]
    v = loaded["vwind_stress"][0]
    lon, lat = _locations(dataset, dataset["uwind_stress"])
    convergence = (
        Proj(dataset.model_horizontal_crs).get_factors(lon, lat, errcheck=True).meridian_convergence
    )
    c, s = np.cos(np.deg2rad(convergence)), np.sin(np.deg2rad(convergence))
    for ux, vy in zip(u, v):
        east = c * ux + s * vy
        vy[:] = -s * ux + c * vy
        ux[:] = east


def _snapshot_plot(path: Path, name: str, values: np.ndarray, lon: np.ndarray, lat: np.ndarray, units: str) -> str:
    field = _flatten_time_space(values)
    x, y = lon.ravel(), lat.ravel()
    indices = (0, len(field) // 2, len(field) - 1)
    finite = field[np.isfinite(field)]
    lo, hi = np.nanpercentile(finite, [2.0, 98.0]) if finite.size else (0.0, 1.0)
    if math.isclose(float(lo), float(hi)):
        lo, hi = float(lo) - 0.5, float(hi) + 0.5
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, index, label in zip(axes, indices, ("start", "midpoint", "end")):
        image = axis.scatter(x, y, c=field[index], s=7, cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
        axis.set_title(label)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=axis, label=units)
    fig.suptitle(f"{name}: start, midpoint, and end")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.resolve())


def _series_plot(path: Path, name: str, values: np.ndarray, times_ms: np.ndarray, units: str, *, locations=None) -> str:
    matrix = _flatten_time_space(values)
    points = _representatives(matrix.shape[1])
    fig, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    times = _time_objects(times_ms)
    for point in points:
        axis.plot(times, matrix[:, point], linewidth=0.9, label=f"index {point if locations is None else locations[point]}")
    axis.set_title(f"{name}: representative locations")
    axis.set_ylabel(units)
    axis.set_xlabel("UTC")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=min(5, len(points)), fontsize=8)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.resolve())


def _package_plot(
    path: Path,
    package: str,
    variables: dict[str, tuple[np.ndarray, str]],
    times_ms: np.ndarray,
) -> str | None:
    names = set(variables)
    times = _time_objects(times_ms)
    if package == "wind":
        pair = next(
            ((left, right) for left, right in (("U10", "V10"), ("uwind_speed", "vwind_speed"), ("uwind_stress", "vwind_stress")) if left in names and right in names),
            None,
        )
        if not pair:
            return None
        u = np.nanmedian(_flatten_time_space(variables[pair[0]][0]), axis=1)
        v = np.nanmedian(_flatten_time_space(variables[pair[1]][0]), axis=1)
        magnitude = np.hypot(u, v)
        stress = pair[0] == "uwind_stress"
        direction = ((90.0 if stress else 270.0) - np.degrees(np.arctan2(v, u))) % 360.0
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
        axes[0].plot(times, magnitude, color="tab:blue")
        axes[0].set_ylabel(variables[pair[0]][1])
        axes[0].set_title("Domain-median wind stress" if stress else "Domain-median wind magnitude")
        axes[1].plot(times, direction, color="tab:orange")
        axes[1].set_ylabel("direction toward (° true)" if stress else "direction from (° true)")
        axes[1].set_ylim(0, 360)
        axes[1].set_xlabel("UTC")
    elif package == "heat":
        heat_names = [name for name in ("short_wave", "net_heat_flux", "air_temperature", "relative_humidity", "air_pressure", "long_wave") if name in names]
        if not heat_names:
            return None
        fig, axes = plt.subplots(len(heat_names), 1, figsize=(11, 2.3 * len(heat_names)), sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for axis, name in zip(axes, heat_names):
            median = np.nanmedian(_flatten_time_space(variables[name][0]), axis=1)
            axis.plot(times, median)
            axis.set_ylabel(variables[name][1])
            axis.set_title(name)
        axes[-1].set_xlabel("UTC")
    elif package == "freshwater":
        p_name = "Precipitation" if "Precipitation" in names else "precip"
        e_name = "Evaporation" if "Evaporation" in names else "evap"
        if p_name not in names or e_name not in names:
            return None
        precipitation = np.nanmedian(_flatten_time_space(variables[p_name][0]), axis=1)
        evaporation = np.nanmedian(_flatten_time_space(variables[e_name][0]), axis=1)
        net = precipitation + evaporation
        elapsed = np.diff(times_ms, prepend=times_ms[0]) / 1000.0
        cumulative = np.cumsum(net * elapsed)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
        axes[0].plot(times, precipitation, label="precipitation")
        axes[0].plot(times, evaporation, label="evaporation (negative)")
        axes[0].plot(times, net, label="P + E")
        axes[0].set_ylabel("m s-1")
        axes[0].legend()
        axes[1].plot(times, cumulative)
        axes[1].set_ylabel("cumulative water depth (m)")
        axes[1].set_xlabel("UTC")
    else:
        if "air_pressure" not in names:
            return None
        pressure, units = variables["air_pressure"]
        median = np.nanmedian(_flatten_time_space(pressure), axis=1)
        pressure_pa = median * 100.0 if units.lower() == "hpa" else median
        inverse_barometer = -(pressure_pa - 101_325.0) / (1025.0 * 9.81)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
        axes[0].plot(times, pressure_pa / 100.0)
        axes[0].set_ylabel("absolute pressure (hPa)")
        axes[1].plot(times, inverse_barometer)
        axes[1].set_ylabel("IB-equivalent elevation (m)")
        axes[1].set_xlabel("UTC")
    for axis in np.atleast_1d(axes):
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle(f"{package.capitalize()} forcing diagnostics")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.resolve())


def _wind_quiver_plot(
    path: Path,
    variables: dict[str, tuple[np.ndarray, str]],
    lon: np.ndarray,
    lat: np.ndarray,
) -> str | None:
    pair = next(
        ((left, right) for left, right in (("U10", "V10"), ("uwind_speed", "vwind_speed"), ("uwind_stress", "vwind_stress")) if left in variables and right in variables),
        None,
    )
    if pair is None:
        return None
    u = _flatten_time_space(variables[pair[0]][0])
    v = _flatten_time_space(variables[pair[1]][0])
    index = len(u) // 2
    x, y = lon.ravel(), lat.ravel()
    stride = max(1, len(x) // 1000)
    fig, axis = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    magnitude = np.hypot(u[index], v[index])
    image = axis.scatter(x, y, c=magnitude, s=6, cmap="viridis", rasterized=True)
    axis.quiver(x[::stride], y[::stride], u[index, ::stride], v[index, ::stride], color="black", alpha=0.65)
    axis.set_title("Wind vector field at record midpoint")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_aspect("equal", adjustable="box")
    fig.colorbar(image, ax=axis, label=variables[pair[0]][1])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.resolve())


def _diurnal_plot(
    path: Path,
    variables: dict[str, tuple[np.ndarray, str]],
    times_ms: np.ndarray,
) -> str | None:
    names = [name for name in ("short_wave", "net_heat_flux", "air_temperature", "relative_humidity", "long_wave") if name in variables]
    if not names:
        return None
    hours = np.asarray([int(str(np.datetime64(int(value), "ms"))[11:13]) for value in times_ms])
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.3 * len(names)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        median = np.nanmedian(_flatten_time_space(variables[name][0]), axis=1)
        cycle = np.asarray([np.nanmean(median[hours == hour]) if np.any(hours == hour) else np.nan for hour in range(24)])
        axis.plot(np.arange(24), cycle, marker="o", markersize=3)
        axis.set_ylabel(variables[name][1])
        axis.set_title(name)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("UTC hour")
    axes[-1].set_xticks(np.arange(0, 24, 3))
    fig.suptitle("Heat-input diurnal cycle")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.resolve())


def _native_atmosphere_qa(path, ds, output, start, end, mesh):
    """Scan every value in bounded blocks and retain only plotting samples."""
    from pyproj import Proj

    times = _decode_exact_time(ds)
    if ds.data_model != "NETCDF3_CLASSIC" or not ds.dimensions["time"].isunlimited():
        raise ValueError("Expected classic native forcing with unlimited time")
    if ds.source != "FVCOM grid (unstructured) surface forcing" or set(
        ds.active_packages.split(",")
    ) != {"wind", "pressure"}:
        raise ValueError("Expected native wind and pressure")
    if start is not None and times[0] > start or end is not None and times[-1] < end:
        raise ValueError("Forcing does not cover model endpoints")
    if mesh is not None:
        for name, values in (
            ("node_id", mesh.node_ids),
            ("element_id", mesh.element_ids),
            ("lon", mesh.lon),
            ("lat", mesh.lat),
            ("lonc", mesh.lonc),
            ("latc", mesh.latc),
        ):
            if not np.allclose(ds[name][:], values, rtol=0, atol=2e-5):
                raise ValueError(f"Native {name} disagrees with model mesh")
    names = ("uwind_stress", "vwind_stress", "air_pressure")
    tracks, snapshots, limits, medians = {}, {}, {}, {}
    points, coords = {}, {}
    indices = np.array([0, len(times) // 2, len(times) - 1])
    for name in names:
        var = ds[name]
        dim = "node" if name == "air_pressure" else "nele"
        if var.units != "Pa" or var.dimensions != ("time", dim):
            raise ValueError(f"Invalid dimensions or units for {name}")
        width = len(ds.dimensions[dim])
        points[name] = _representatives(width)
        coords[name] = _locations(ds, var)
        tracks[name] = np.empty((len(times), len(points[name])))
        snapshots[name] = np.asarray(var[indices, :], dtype=float)
        limits[name] = [math.inf, -math.inf]
        medians[name] = np.empty((len(times), 1))
    lon, lat = coords["uwind_stress"]
    angle = np.deg2rad(
        Proj(ds.model_horizontal_crs).get_factors(lon, lat, errcheck=True).meridian_convergence
    )
    c, s = np.cos(angle), np.sin(angle)
    peak = [0.0, 0, 0]
    for start_row in range(0, len(times), 6):
        rows = slice(start_row, min(start_row + 6, len(times)))
        fields = {}
        for name in names:
            values = np.ma.asarray(ds[name][rows, :], dtype=float).filled(np.nan)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {name} in forcing")
            limits[name][0] = min(limits[name][0], float(values.min()))
            limits[name][1] = max(limits[name][1], float(values.max()))
            tracks[name][rows] = values[:, points[name]]
            fields[name] = values
        x, y, p = (fields[name] for name in names)
        medians[names[0]][rows, 0] = np.median(c * x + s * y, axis=1)
        medians[names[1]][rows, 0] = np.median(-s * x + c * y, axis=1)
        medians[names[2]][rows, 0] = np.median(p, axis=1)
        stress = np.hypot(x, y)
        row, elem = np.unravel_index(np.argmax(stress), stress.shape)
        if stress[row, elem] >= peak[0]:
            peak = [float(stress[row, elem]), int(start_row + row), int(elem + 1)]
    if limits["air_pressure"][0] < 50_000 or limits["air_pressure"][1] > 120_000:
        raise ValueError("Implausible absolute pressure in Pa")
    artifacts = {}
    for name in names:
        artifacts[name + "_maps"] = _snapshot_plot(
            output / (name + "_maps.png"), name, snapshots[name], *coords[name], "Pa"
        )
        artifacts[name + "_series"] = _series_plot(
            output / (name + "_series.png"), name, tracks[name], times, "Pa", locations=points[name]
        )
    x, y, _ = (snapshots[name][1] for name in names)
    arrows = {names[0]: ((x * c + y * s)[None, :], "Pa"), names[1]: ((-x * s + y * c)[None, :], "Pa")}
    artifacts["package_wind_quiver"] = _wind_quiver_plot(
        output / "package_wind_quiver.png", arrows, lon, lat
    )
    for kind in ("wind", "pressure"):
        artifacts["package_" + kind] = _package_plot(
            output / f"package_{kind}_diagnostics.png",
            kind,
            {k: (v, "Pa") for k, v in medians.items()},
            times,
        )
    cadence = np.diff(times) / 1000
    report = dict(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        source=ds.source,
        active_packages=["wind", "pressure"],
        dimensions={k: len(v) for k, v in ds.dimensions.items()},
        time_start_utc=str(np.datetime64(int(times[0]), "ms")) + "Z",
        time_end_utc=str(np.datetime64(int(times[-1]), "ms")) + "Z",
        regular_time=bool(not len(cadence) or np.all(cadence == cadence[0])),
        cadence_seconds=float(cadence[0]) if len(cadence) else None,
        variables={
            k: dict(min=v[0], max=v[1], finite_fraction=1.0, units="Pa") for k, v in limits.items()
        },
        peak_wind_stress_pa=peak[0],
        peak_stress_time_utc=str(np.datetime64(int(times[peak[1]]), "ms")) + "Z",
        peak_stress_element=peak[2],
    )
    return dict(
        status="pass",
        structural_checks={
            k: True
            for k in (
                "netcdf3_classic",
                "exact_time_representations",
                "strictly_monotonic_time",
                "zero_nan_gate",
                "bundle_time_alignment",
                "fvcom_source_contract",
            )
        },
        files=[report],
        qa_artifacts=artifacts,
    )


def validate_forcing_files(
    paths: Iterable[str | Path],
    qa_dir: str | Path,
    *,
    model_start_ms: int | None = None,
    model_end_ms: int | None = None,
    mesh: MeshGeometry | None = None,
) -> dict[str, Any]:
    files = [Path(value) for value in paths]
    if not files:
        raise ValueError("At least one forcing file is required")
    output_dir = Path(qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(files)==1:
        with nc4.Dataset(files[0]) as ds:
            if getattr(ds,"wind_vector_frame","")=="model_xy" and set(getattr(ds,"active_packages","").split(",")) == {"wind","pressure"}:
                return _native_atmosphere_qa(files[0],ds,output_dir,model_start_ms,model_end_ms,mesh)
    reports: list[dict[str, Any]] = []
    all_times: list[np.ndarray] = []
    source_contracts: list[str] = []
    artifacts: dict[str, str] = {}
    package_variables: dict[str, dict[str, tuple[np.ndarray, str]]] = {}
    package_times: dict[str, np.ndarray] = {}
    package_coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for file_index, path in enumerate(files):
        with nc4.Dataset(path) as dataset:
            if dataset.data_model != "NETCDF3_CLASSIC":
                raise ValueError(f"{path.name} must be NETCDF3_CLASSIC")
            source_contract = str(getattr(dataset, "source", ""))
            if source_contract not in {
                "wrf grid (structured) surface forcing",
                "FVCOM grid (unstructured) surface forcing",
            }:
                raise ValueError(f"{path.name} has an unrecognized FVCOM surface source attribute")
            if source_contracts and source_contract != source_contracts[0]:
                raise ValueError("Bundle members mix structured and FVCOM-native source contracts")
            source_contracts.append(source_contract)
            if source_contract == "FVCOM grid (unstructured) surface forcing" and mesh is not None:
                if not np.array_equal(np.asarray(dataset.variables["node_id"][:], dtype=np.int64), mesh.node_ids):
                    raise ValueError(f"{path.name} node IDs do not match the supplied mesh")
                if not np.array_equal(np.asarray(dataset.variables["element_id"][:], dtype=np.int64), mesh.element_ids):
                    raise ValueError(f"{path.name} element IDs do not match the supplied mesh")
                for name, expected in (("lon", mesh.lon), ("lat", mesh.lat), ("lonc", mesh.lonc), ("latc", mesh.latc)):
                    if not np.allclose(_as_float(dataset.variables[name]), expected, atol=2e-5, rtol=0.0):
                        raise ValueError(f"{path.name}:{name} does not match the supplied mesh")
            required = {"iint", "time", "Itime", "Itime2", "Times"}
            missing = sorted(required - set(dataset.variables))
            if missing:
                raise ValueError(f"{path.name} is missing required time variables {missing}")
            if not dataset.dimensions["time"].isunlimited():
                raise ValueError(f"{path.name} time dimension is not unlimited")
            times_ms = _decode_exact_time(dataset)
            all_times.append(times_ms)
            if model_start_ms is not None and times_ms[0] > model_start_ms:
                raise ValueError(f"{path.name} begins after the requested model start")
            if model_end_ms is not None and times_ms[-1] < model_end_ms:
                raise ValueError(f"{path.name} ends before the requested model end")
            names = [name for name in dataset.variables if name not in NON_FIELD_VARIABLES]
            if not names:
                raise ValueError(f"{path.name} contains no surface forcing fields")
            variable_report: dict[str, Any] = {}
            loaded: dict[str, tuple[np.ndarray, str]] = {}
            for name in names:
                variable = dataset.variables[name]
                if not variable.dimensions or variable.dimensions[0] != "time":
                    continue
                values = _as_float(variable)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{path.name}:{name} contains NaN, infinity, or a fill sentinel")
                units = str(getattr(variable, "units", ""))
                if name == "air_pressure":
                    if units == "Pa" and (np.min(values) < 50_000.0 or np.max(values) > 120_000.0):
                        raise ValueError(f"{path.name}:air_pressure is not plausible absolute Pa")
                    if units == "hPa" and (np.min(values) < 500.0 or np.max(values) > 1200.0):
                        raise ValueError(f"{path.name}:air_pressure is not plausible absolute hPa")
                if name in {"relative_humidity"} and (np.min(values) < 0.0 or np.max(values) > 100.0):
                    raise ValueError(f"{path.name}:relative_humidity is outside 0-100 percent")
                if name in {"Precipitation", "precip"} and np.min(values) < 0.0:
                    raise ValueError(f"{path.name}:{name} must be non-negative")
                if name in {"Evaporation", "evap"} and np.max(values) > 0.0:
                    raise ValueError(f"{path.name}:{name} must be non-positive in FVCOM output")
                loaded[name] = (values, units)
                variable_report[name] = {
                    "dimensions": list(variable.dimensions),
                    "units": units,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "p02": float(np.percentile(values, 2.0)),
                    "p98": float(np.percentile(values, 98.0)),
                    "finite_fraction": 1.0,
                }
                lon, lat = _locations(dataset, variable)
                stem = f"{file_index:02d}_{path.stem}_{name}"
                artifacts[f"{stem}_maps"] = _snapshot_plot(
                    output_dir / f"{stem}_maps.png", name, values, lon, lat, units
                )
                artifacts[f"{stem}_series"] = _series_plot(
                    output_dir / f"{stem}_series.png", name, values, times_ms, units
                )
            active = [value.strip() for value in str(getattr(dataset, "active_packages", "")).split(",") if value.strip()]
            if "wind" in active:
                _wind_diagnostics_to_earth(dataset, loaded)
            for package in active:
                package_variables.setdefault(package, {}).update(loaded)
                package_times[package] = times_ms
                if loaded:
                    first = dataset.variables[next(iter(loaded))]
                    package_coordinates[package] = _locations(dataset, first)
            cadence = np.diff(times_ms) / 1000.0
            reports.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "source": getattr(dataset, "source"),
                    "active_packages": active,
                    "dimensions": {name: len(value) for name, value in dataset.dimensions.items()},
                    "time_start_utc": str(np.datetime64(int(times_ms[0]), "ms")) + "Z",
                    "time_end_utc": str(np.datetime64(int(times_ms[-1]), "ms")) + "Z",
                    "cadence_seconds": float(np.median(cadence)) if len(cadence) else None,
                    "regular_time": bool(not len(cadence) or np.all(cadence == cadence[0])),
                    "variables": variable_report,
                }
            )
    for times in all_times[1:]:
        if not np.array_equal(times, all_times[0]):
            raise ValueError("Bundle members do not share the exact same UTC time axis")
    if len(set(source_contracts)) != 1:
        raise ValueError("Bundle members mix structured and FVCOM-native source contracts")
    for package, variables in package_variables.items():
        plot = _package_plot(output_dir / f"package_{package}_diagnostics.png", package, variables, package_times[package])
        if plot:
            artifacts[f"package_{package}"] = plot
        if package == "wind":
            lon, lat = package_coordinates[package]
            quiver = _wind_quiver_plot(output_dir / "package_wind_quiver.png", variables, lon, lat)
            if quiver:
                artifacts["package_wind_quiver"] = quiver
        if package == "heat":
            diurnal = _diurnal_plot(output_dir / "package_heat_diurnal.png", variables, package_times[package])
            if diurnal:
                artifacts["package_heat_diurnal"] = diurnal
    return {
        "status": "pass",
        "structural_checks": {
            "netcdf3_classic": True,
            "exact_time_representations": True,
            "strictly_monotonic_time": True,
            "zero_nan_gate": True,
            "bundle_time_alignment": True,
            "fvcom_source_contract": True,
        },
        "files": reports,
        "qa_artifacts": artifacts,
    }
