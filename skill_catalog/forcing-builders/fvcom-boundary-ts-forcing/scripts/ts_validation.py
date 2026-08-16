from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np

from ts_core import (
    DAY_MS,
    MJD_EPOCH_MS,
    Boundary,
    arc_distances_km,
    decode_netcdf_times,
    format_times,
    iso_utc,
    sha256_file,
    validate_sigma,
)


REQUIRED_VARIABLES = (
    "obc_nodes",
    "obc_h",
    "iint",
    "time",
    "Itime",
    "Itime2",
    "Times",
    "siglay",
    "siglev",
    "obc_temp",
    "obc_salinity",
)


def _sample_positions(boundary: Boundary, count_per_arc: int = 5) -> list[int]:
    samples: list[int] = []
    for arc in boundary.arcs:
        local = np.unique(np.rint(np.linspace(0, len(arc) - 1, min(count_per_arc, len(arc)))).astype(int))
        samples.extend(int(arc[index]) for index in local)
    return samples


def _level_indices(siglay: np.ndarray) -> dict[str, int]:
    representative = np.nanmedian(siglay, axis=1)
    return {
        "surface": int(np.argmax(representative)),
        "middle": int(np.argmin(np.abs(representative + 0.5))),
        "bottom": int(np.argmin(representative)),
    }


def _dates(times_ms: np.ndarray) -> list[Any]:
    return [np.datetime64(int(value), "ms").astype("datetime64[ms]").astype(object) for value in times_ms]


def _aggregate_time(data: np.ndarray, times_ms: np.ndarray, max_rows: int = 2000) -> tuple[np.ndarray, np.ndarray, int]:
    if len(times_ms) <= max_rows:
        return data, times_ms, 1
    block = int(np.ceil(len(times_ms) / max_rows))
    count = len(times_ms) // block
    trimmed = count * block
    aggregate = data[:trimmed].reshape((count, block) + data.shape[1:]).mean(axis=1)
    times = np.rint(times_ms[:trimmed].reshape(count, block).mean(axis=1)).astype(np.int64)
    if trimmed < len(times_ms):
        aggregate = np.concatenate([aggregate, data[trimmed:].mean(axis=0, keepdims=True)], axis=0)
        times = np.concatenate([times, [int(round(float(times_ms[trimmed:].mean())))]]).astype(np.int64)
    return aggregate, times, block


def _plot_boundary_map(boundary: Boundary, samples: list[int], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    for arc_number, arc in enumerate(boundary.arcs, start=1):
        axis.plot(boundary.lon[arc], boundary.lat[arc], "-o", ms=2.5, lw=1.2, label=f"arc {arc_number}")
    axis.scatter(boundary.lon[samples], boundary.lat[samples], c="red", marker="x", s=40, label="QA samples", zorder=5)
    for position in samples:
        axis.annotate(str(int(boundary.node_ids[position])), (boundary.lon[position], boundary.lat[position]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("FVCOM open boundary and representative QA nodes")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_sample_series(temperature: np.ndarray, salinity: np.ndarray, times_ms: np.ndarray, boundary: Boundary, samples: list[int], levels: dict[str, int], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    dates = _dates(times_ms)
    for column, (level_name, layer) in enumerate(levels.items()):
        for position in samples:
            label = f"node {int(boundary.node_ids[position])}"
            axes[0, column].plot(dates, temperature[:, layer, position], lw=0.8, label=label)
            axes[1, column].plot(dates, salinity[:, layer, position], lw=0.8, label=label)
        axes[0, column].set_title(f"{level_name.title()} temperature")
        axes[1, column].set_title(f"{level_name.title()} salinity")
        axes[1, column].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[1, column].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1, column].xaxis.get_major_locator()))
    axes[0, 0].set_ylabel("Temperature (Celsius)")
    axes[1, 0].set_ylabel("Practical salinity (PSU)")
    axes[0, -1].legend(fontsize=6, ncol=2, loc="best")
    figure.suptitle("Boundary-node temperature and salinity through the deployment")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_vertical_curtains(temperature: np.ndarray, salinity: np.ndarray, times_ms: np.ndarray, siglay: np.ndarray, boundary: Boundary, output: Path) -> None:
    rows = len(boundary.arcs)
    figure, axes = plt.subplots(rows, 2, figsize=(16, max(4.5, 4.2 * rows)), squeeze=False, sharex=True)
    reduced_temp, reduced_times, block = _aggregate_time(temperature, times_ms)
    reduced_salt, _, _ = _aggregate_time(salinity, times_ms)
    dates = _dates(reduced_times)
    for row, arc in enumerate(boundary.arcs):
        position = int(arc[len(arc) // 2])
        depth = -siglay[:, position] * boundary.depth_m[position]
        order = np.argsort(depth)
        for column, (data, label, units, cmap) in enumerate(((reduced_temp, "temperature", "Celsius", "coolwarm"), (reduced_salt, "salinity", "PSU", "viridis"))):
            mesh = axes[row, column].pcolormesh(dates, depth[order], data[:, order, position].T, shading="auto", cmap=cmap)
            axes[row, column].invert_yaxis()
            axes[row, column].set_ylabel("Depth below surface (m)")
            axes[row, column].set_title(f"Arc {row + 1}, node {int(boundary.node_ids[position])}: {label}")
            figure.colorbar(mesh, ax=axes[row, column], label=units)
            axes[row, column].xaxis.set_major_locator(mdates.AutoDateLocator())
            axes[row, column].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[row, column].xaxis.get_major_locator()))
    figure.suptitle(f"Time-depth boundary curtains (time display aggregation: {block} record(s)/bin)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_hovmoller(data: np.ndarray, times_ms: np.ndarray, boundary: Boundary, levels: dict[str, int], label: str, units: str, cmap: str, output: Path) -> int:
    reduced, reduced_times, block = _aggregate_time(data, times_ms)
    rows = len(boundary.arcs)
    figure, axes = plt.subplots(rows, 3, figsize=(18, max(4.5, 4.0 * rows)), squeeze=False, sharex=True)
    dates = _dates(reduced_times)
    for row, arc in enumerate(boundary.arcs):
        distance = arc_distances_km(boundary, arc)
        for column, (level_name, layer) in enumerate(levels.items()):
            mesh = axes[row, column].pcolormesh(dates, distance, reduced[:, layer, arc].T, shading="auto", cmap=cmap)
            axes[row, column].set_ylabel("Cumulative distance (km)")
            axes[row, column].set_title(f"Arc {row + 1}, {level_name}")
            axes[row, column].xaxis.set_major_locator(mdates.AutoDateLocator())
            axes[row, column].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[row, column].xaxis.get_major_locator()))
            figure.colorbar(mesh, ax=axes[row, column], label=units)
    figure.suptitle(f"Complete-boundary {label} Hovmöller diagrams ({block} record(s)/display bin)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return block


def _plot_transect_snapshots(data: np.ndarray, times_ms: np.ndarray, siglay: np.ndarray, boundary: Boundary, label: str, units: str, cmap: str, output: Path) -> None:
    snapshots = [0, len(times_ms) // 2, len(times_ms) - 1]
    rows = len(boundary.arcs)
    figure, axes = plt.subplots(rows, 3, figsize=(18, max(4.5, 4.0 * rows)), squeeze=False)
    for row, arc in enumerate(boundary.arcs):
        distance = arc_distances_km(boundary, arc)
        x = np.repeat(distance[None, :], siglay.shape[0], axis=0)
        depth = -siglay[:, arc] * boundary.depth_m[arc][None, :]
        for column, time_index in enumerate(snapshots):
            axis = axes[row, column]
            if len(arc) > 1:
                contour = axis.contourf(x, depth, data[time_index][:, arc], levels=20, cmap=cmap)
                figure.colorbar(contour, ax=axis, label=units)
            else:
                axis.plot(data[time_index, :, arc[0]], depth[:, 0])
                axis.set_xlabel(units)
            axis.invert_yaxis()
            axis.set_ylabel("Depth below surface (m)")
            axis.set_title(f"Arc {row + 1}: {iso_utc(times_ms[[time_index]])[0]}")
            if len(arc) > 1:
                axis.set_xlabel("Cumulative distance (km)")
    figure.suptitle(f"Along-boundary vertical {label} transect snapshots")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_missing_diagnostics(repair_codes: dict[str, np.ndarray] | None, repair_times_ms: np.ndarray | None, siglay_count: int, node_count: int, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17, 8))
    for row, variable in enumerate(("temperature", "salinity")):
        codes = None if repair_codes is None else repair_codes.get(variable)
        if codes is None:
            codes = np.zeros((1, siglay_count, node_count), dtype=np.uint8)
            dates = [0]
        else:
            dates = _dates(repair_times_ms) if repair_times_ms is not None else np.arange(codes.shape[0])
        original = codes > 0
        axes[row, 0].plot(dates, original.sum(axis=(1, 2)), lw=1.0)
        if repair_times_ms is not None:
            locator = mdates.AutoDateLocator()
            axes[row, 0].xaxis.set_major_locator(locator)
            axes[row, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[row, 0].set_title(f"{variable.title()}: original missing by time")
        axes[row, 0].set_ylabel("Values repaired")
        axes[row, 1].bar(np.arange(codes.shape[1]), original.sum(axis=(0, 2)))
        axes[row, 1].set_title("Original missing by sigma layer")
        axes[row, 1].set_xlabel("Layer index")
        axes[row, 2].bar(np.arange(codes.shape[2]), original.sum(axis=(0, 1)))
        axes[row, 2].set_title("Original missing by boundary position")
        axes[row, 2].set_xlabel("Boundary position")
        axes[row, 2].text(0.98, 0.95, "Final forcing NaNs: 0", ha="right", va="top", transform=axes[row, 2].transAxes, color="darkgreen", weight="bold")
    figure.suptitle("Missing-data repair audit (zero-NaN output gate)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def validate_forcing(
    forcing_path: str | Path,
    boundary: Boundary,
    qa_dir: str | Path,
    *,
    temperature_min: float = -5.0,
    temperature_max: float = 45.0,
    salinity_min: float = 0.0,
    salinity_max: float = 50.0,
    repair_codes: dict[str, np.ndarray] | None = None,
    repair_times_ms: np.ndarray | None = None,
    repair_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    forcing = Path(forcing_path)
    output_dir = Path(qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with nc4.Dataset(forcing) as dataset:
        if dataset.data_model != "NETCDF3_CLASSIC":
            raise ValueError(f"Expected NETCDF3_CLASSIC, found {dataset.data_model}")
        if str(getattr(dataset, "type", "")) != "FVCOM TIME SERIES OBC TS FILE":
            raise ValueError("FVCOM T/S global type attribute is absent or incorrect")
        for dimension in ("nobc", "siglay", "siglev", "time", "DateStrLen"):
            if dimension not in dataset.dimensions:
                raise ValueError(f"Missing required dimension {dimension!r}")
        if not dataset.dimensions["time"].isunlimited() or len(dataset.dimensions["DateStrLen"]) != 26:
            raise ValueError("time must be unlimited and DateStrLen must equal 26")
        absent = [name for name in REQUIRED_VARIABLES if name not in dataset.variables]
        if absent:
            raise ValueError(f"Missing required forcing variables: {absent}")
        expected_dimensions = {
            "obc_nodes": ("nobc",), "obc_h": ("nobc",), "iint": ("time",), "time": ("time",),
            "Itime": ("time",), "Itime2": ("time",), "Times": ("time", "DateStrLen"),
            "siglay": ("siglay", "nobc"), "siglev": ("siglev", "nobc"),
            "obc_temp": ("time", "siglay", "nobc"), "obc_salinity": ("time", "siglay", "nobc"),
        }
        for name, dimensions in expected_dimensions.items():
            if dataset.variables[name].dimensions != dimensions:
                raise ValueError(f"Variable {name!r} dimensions must be {dimensions}, found {dataset.variables[name].dimensions}")
        nodes = np.asarray(dataset.variables["obc_nodes"][:], dtype=np.int64)
        if not np.array_equal(nodes, boundary.node_ids):
            raise ValueError("Forcing obc_nodes do not exactly match the selected boundary order")
        depth = np.ma.filled(dataset.variables["obc_h"][:], np.nan).astype(np.float64)
        if not np.all(np.isfinite(depth)) or not np.allclose(depth, boundary.depth_m, rtol=1e-6, atol=1e-4):
            raise ValueError("Forcing obc_h does not match positive-down boundary depth")
        siglay = np.ma.filled(dataset.variables["siglay"][:], np.nan).astype(np.float64)
        siglev = np.ma.filled(dataset.variables["siglev"][:], np.nan).astype(np.float64)
        orientation = validate_sigma(siglay, siglev, tolerance=2.0e-5)
        temperature = np.ma.filled(dataset.variables["obc_temp"][:], np.nan).astype(np.float64)
        salinity = np.ma.filled(dataset.variables["obc_salinity"][:], np.nan).astype(np.float64)
        expected_shape = (len(dataset.dimensions["time"]), len(dataset.dimensions["siglay"]), len(nodes))
        if temperature.shape != expected_shape or salinity.shape != expected_shape:
            raise ValueError("T/S variables do not have exact (time, siglay, nobc) shape")
        if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(salinity)):
            raise ValueError("Zero-NaN gate failed: forcing T/S contains missing or non-finite values")
        if str(getattr(dataset.variables["obc_temp"], "units", "")) != "Celsius":
            raise ValueError("obc_temp units must be 'Celsius'")
        if str(getattr(dataset.variables["obc_salinity"], "units", "")) != "PSU":
            raise ValueError("obc_salinity units must be 'PSU'")
        if np.min(temperature) < temperature_min or np.max(temperature) > temperature_max:
            raise ValueError(f"Temperature exceeds reviewed bounds [{temperature_min}, {temperature_max}] Celsius")
        if np.min(salinity) < salinity_min or np.max(salinity) > salinity_max:
            raise ValueError(f"Salinity exceeds reviewed bounds [{salinity_min}, {salinity_max}] PSU")
        times_ms, _, _ = decode_netcdf_times(dataset)
        integer_times = MJD_EPOCH_MS + np.asarray(dataset.variables["Itime"][:], dtype=np.int64) * DAY_MS + np.asarray(dataset.variables["Itime2"][:], dtype=np.int64)
        if not np.array_equal(times_ms, integer_times):
            raise ValueError("Times and Itime/Itime2 disagree at millisecond precision")
        if not np.array_equal(np.asarray(dataset.variables["Times"][:]), format_times(times_ms)):
            raise ValueError("Times strings are not canonical UTC millisecond timestamps")
        expected_float = (np.floor_divide(times_ms - MJD_EPOCH_MS, DAY_MS).astype(np.float64) + np.mod(times_ms - MJD_EPOCH_MS, DAY_MS).astype(np.float64) / DAY_MS).astype(np.float32)
        if not np.array_equal(np.asarray(dataset.variables["time"][:], dtype=np.float32), expected_float):
            raise ValueError("Float MJD time is inconsistent with the canonical exact time axis")
        if not np.array_equal(np.asarray(dataset.variables["iint"][:], dtype=np.int64), np.arange(1, len(times_ms) + 1)):
            raise ValueError("iint must be one-based and contiguous")
        title = str(getattr(dataset, "title", ""))
    if len(times_ms) < 2 or np.any(np.diff(times_ms) <= 0):
        raise ValueError("Forcing timestamps must contain at least two strictly increasing values")
    differences = np.diff(times_ms)
    cadence_ms = int(np.median(differences))
    if np.max(np.abs(differences - cadence_ms)) > 1:
        raise ValueError("Forcing timestamps are not regular to millisecond precision")
    levels = _level_indices(siglay)
    samples = _sample_positions(boundary)
    artifacts = {
        "boundary_map": output_dir / "boundary_sample_map.png",
        "sample_timeseries": output_dir / "boundary_sample_timeseries.png",
        "vertical_time_curtains": output_dir / "boundary_vertical_time_curtains.png",
        "temperature_hovmoller": output_dir / "temperature_boundary_hovmoller.png",
        "salinity_hovmoller": output_dir / "salinity_boundary_hovmoller.png",
        "temperature_transects": output_dir / "temperature_vertical_transect_snapshots.png",
        "salinity_transects": output_dir / "salinity_vertical_transect_snapshots.png",
        "missing_data_diagnostics": output_dir / "missing_data_repair_diagnostics.png",
    }
    _plot_boundary_map(boundary, samples, artifacts["boundary_map"])
    _plot_sample_series(temperature, salinity, times_ms, boundary, samples, levels, artifacts["sample_timeseries"])
    _plot_vertical_curtains(temperature, salinity, times_ms, siglay, boundary, artifacts["vertical_time_curtains"])
    temp_block = _plot_hovmoller(temperature, times_ms, boundary, levels, "temperature", "Celsius", "coolwarm", artifacts["temperature_hovmoller"])
    salt_block = _plot_hovmoller(salinity, times_ms, boundary, levels, "salinity", "PSU", "viridis", artifacts["salinity_hovmoller"])
    _plot_transect_snapshots(temperature, times_ms, siglay, boundary, "temperature", "Celsius", "coolwarm", artifacts["temperature_transects"])
    _plot_transect_snapshots(salinity, times_ms, siglay, boundary, "salinity", "PSU", "viridis", artifacts["salinity_transects"])
    _plot_missing_diagnostics(repair_codes, repair_times_ms, siglay.shape[0], len(nodes), artifacts["missing_data_diagnostics"])
    repaired_total = 0 if not repair_reports else sum(int(report.get("repaired_total", 0)) for report in repair_reports.values())
    status = "pass_with_repairs" if repaired_total else "pass"
    warnings = [f"Automatically repaired {repaired_total} source T/S values; inspect the repair audit."] if repaired_total else []
    return {
        "status": status,
        "forcing_file": str(forcing.resolve()),
        "forcing_sha256": sha256_file(forcing),
        "case_name": title,
        "netcdf_format": "NETCDF3_CLASSIC",
        "node_count": int(len(nodes)),
        "arc_count": int(len(boundary.arcs)),
        "sigma_layer_count": int(siglay.shape[0]),
        "sigma_level_count": int(siglev.shape[0]),
        "sigma_orientation": orientation,
        "time_count": int(len(times_ms)),
        "time_start_utc": iso_utc(times_ms[[0]])[0],
        "time_end_utc": iso_utc(times_ms[[-1]])[0],
        "cadence_seconds": cadence_ms / 1000.0,
        "temperature_celsius": {"minimum": float(np.min(temperature)), "maximum": float(np.max(temperature)), "mean": float(np.mean(temperature))},
        "salinity_psu": {"minimum": float(np.min(salinity)), "maximum": float(np.max(salinity)), "mean": float(np.mean(salinity))},
        "missing_data": {"final_temperature_nan_count": 0, "final_salinity_nan_count": 0, "source_repair": repair_reports or "not_available_for_existing_file"},
        "structural_checks": {
            "netcdf3_classic": True,
            "required_dimensions_present": True,
            "time_dimension_unlimited": True,
            "date_strlen_26": True,
            "required_variables_present": True,
            "variable_dimensions_exact": True,
            "boundary_node_order_exact": True,
            "boundary_depth_exact": True,
            "sigma_valid_and_orientation_preserved": True,
            "temperature_units_celsius": True,
            "salinity_units_psu": True,
            "physical_bounds_pass": True,
            "zero_nan_gate_pass": True,
            "iint_one_based": True,
            "time_strictly_monotonic": True,
            "time_regular_to_millisecond": True,
            "times_matches_itime_itime2": True,
            "legacy_float_mjd_matches_expected_float32": True,
        },
        "sample_nodes": [{"boundary_position": int(position), "node_id": int(boundary.node_ids[position]), "longitude": float(boundary.lon[position]), "latitude": float(boundary.lat[position]), "depth_m": float(boundary.depth_m[position])} for position in samples],
        "representative_layers": levels,
        "display_aggregation": {"temperature_records_per_bin": temp_block, "salinity_records_per_bin": salt_block},
        "qa_artifacts": {name: str(path.resolve()) for name, path in artifacts.items()},
        "warnings": warnings,
    }
