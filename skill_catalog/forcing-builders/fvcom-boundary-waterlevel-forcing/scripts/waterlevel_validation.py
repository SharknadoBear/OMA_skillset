from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt, welch

from waterlevel_core import (
    DAY_MS,
    MJD_EPOCH_MS,
    Boundary,
    arc_distances_km,
    decode_netcdf_times,
    format_times,
    iso_utc,
)


REQUIRED_VARIABLES = ("obc_nodes", "iint", "time", "Itime", "Itime2", "Times", "elevation")
TIDAL_MARKERS = {"M4": 6.2103, "S2": 12.0, "M2": 12.4206, "K1": 23.9345, "O1": 25.8193}


def _sample_positions(boundary: Boundary, count: int = 5) -> list[int]:
    selected: list[int] = []
    for arc in boundary.arcs:
        indices = np.unique(np.rint(np.linspace(0, len(arc) - 1, min(count, len(arc)))).astype(int))
        selected.extend(int(arc[index]) for index in indices)
    return selected


def _aggregate_time(data: np.ndarray, times_ms: np.ndarray, max_rows: int) -> tuple[np.ndarray, np.ndarray, int]:
    if len(times_ms) <= max_rows:
        return data, times_ms, 1
    block = int(math.ceil(len(times_ms) / max_rows))
    usable = len(times_ms) // block * block
    aggregated = np.nanmean(data[:usable].reshape((-1, block, data.shape[1])), axis=1)
    stamps = times_ms[:usable].reshape((-1, block))[:, block // 2]
    if usable < len(times_ms):
        aggregated = np.vstack((aggregated, np.nanmean(data[usable:], axis=0)))
        stamps = np.append(stamps, times_ms[usable + (len(times_ms) - usable) // 2])
    return aggregated, stamps, block


def _filter_bands(
    elevation: np.ndarray,
    dt_hours: float,
    tidal_min_hours: float,
    tidal_max_hours: float,
    vlf_min_days: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    warnings: list[str] = []
    sample_rate = 1.0 / dt_hours
    nyquist = sample_rate / 2.0
    tidal_low = 1.0 / tidal_max_hours
    tidal_high = 1.0 / tidal_min_hours
    subtidal_low = 1.0 / (vlf_min_days * 24.0)
    subtidal_high = tidal_low
    if tidal_low >= nyquist:
        raise ValueError("Output cadence cannot resolve the configured tidal band")
    if tidal_high >= nyquist:
        warnings.append(
            f"Tidal upper frequency was clipped below Nyquist; periods shorter than {2 * dt_hours:.3f} h are unresolved"
        )
        tidal_high = nyquist * 0.95
    if subtidal_low >= subtidal_high:
        raise ValueError("VLF cutoff must exceed the tidal maximum period")

    bands: dict[str, np.ndarray] = {}
    tidal_sos = butter(4, [tidal_low, tidal_high], btype="bandpass", fs=sample_rate, output="sos")
    subtidal_sos = butter(4, [subtidal_low, subtidal_high], btype="bandpass", fs=sample_rate, output="sos")
    vlf_sos = butter(4, subtidal_low, btype="lowpass", fs=sample_rate, output="sos")
    try:
        bands["tidal"] = sosfiltfilt(tidal_sos, elevation, axis=0)
        bands["subtidal"] = sosfiltfilt(subtidal_sos, elevation, axis=0)
        bands["vlf"] = sosfiltfilt(vlf_sos, elevation, axis=0)
    except ValueError as exc:
        raise ValueError(f"Record is too short for zero-phase band validation: {exc}") from exc
    return bands, warnings


def _spectrum(
    elevation: np.ndarray,
    dt_hours: float,
    sample_positions: list[int],
    tidal_min_hours: float,
    tidal_max_hours: float,
    vlf_min_days: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float] | None], list[np.ndarray]]:
    target_segment = max(256, int(round(365.0 * 24.0 / dt_hours)))
    nperseg = min(len(elevation), target_segment)
    median_series = np.nanmedian(elevation, axis=1)
    frequency, median_psd = welch(
        detrend(median_series, type="linear"), fs=1.0 / dt_hours, nperseg=nperseg, scaling="density"
    )
    sample_psd: list[np.ndarray] = []
    for position in sample_positions:
        _, power = welch(
            detrend(elevation[:, position], type="linear"),
            fs=1.0 / dt_hours,
            nperseg=nperseg,
            scaling="density",
        )
        sample_psd.append(power)

    ranges = {
        "tidal": (1.0 / tidal_max_hours, 1.0 / tidal_min_hours),
        "subtidal": (1.0 / (vlf_min_days * 24.0), 1.0 / tidal_max_hours),
        "vlf_slr_scale": (0.0, 1.0 / (vlf_min_days * 24.0)),
    }
    peaks: dict[str, dict[str, float] | None] = {}
    for name, (low, high) in ranges.items():
        mask = (frequency > low) & (frequency <= high)
        if not np.any(mask):
            peaks[name] = None
            continue
        candidates = np.where(mask)[0]
        index = int(candidates[np.nanargmax(median_psd[candidates])])
        peaks[name] = {
            "frequency_cycles_per_hour": float(frequency[index]),
            "period_hours": float(1.0 / frequency[index]),
            "power_m2_per_cph": float(median_psd[index]),
        }
    return frequency, median_psd, peaks, sample_psd


def _plot_boundary_map(boundary: Boundary, samples: list[int], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for arc_number, arc in enumerate(boundary.arcs, start=1):
        ax.plot(boundary.lon[arc], boundary.lat[arc], "-", lw=1.2, label=f"OBC arc {arc_number}")
    full_lon_span = max(float(np.ptp(boundary.lon)), 1e-12)
    full_lat_span = max(float(np.ptp(boundary.lat)), 1e-12)
    sample_number = 1
    sample_set = set(samples)
    for arc in boundary.arcs:
        arc_samples = [int(position) for position in arc if int(position) in sample_set]
        compact = (
            float(np.ptp(boundary.lon[arc])) < 0.05 * full_lon_span
            and float(np.ptp(boundary.lat[arc])) < 0.05 * full_lat_span
            and len(arc_samples) > 1
        )
        vertical_offsets = np.linspace(44.0, -44.0, len(arc_samples)) if compact else np.full(len(arc_samples), 4.0)
        for position, vertical_offset in zip(arc_samples, vertical_offsets):
            ax.scatter(boundary.lon[position], boundary.lat[position], s=32, zorder=4)
            ax.annotate(
                f"S{sample_number}\nnode {int(boundary.node_ids[position])}",
                (boundary.lon[position], boundary.lat[position]),
                xytext=(5, float(vertical_offset)),
                textcoords="offset points",
                fontsize=7,
                va="center" if compact else "bottom",
                annotation_clip=False,
            )
            sample_number += 1
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title("FVCOM open-boundary QA sample nodes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.margins(x=0.08, y=0.08)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_sample_series(
    elevation: np.ndarray,
    times_ms: np.ndarray,
    boundary: Boundary,
    samples: list[int],
    output: Path,
    title: str,
    max_rows: int = 4000,
) -> None:
    data, stamps, block = _aggregate_time(elevation, times_ms, max_rows)
    dates = stamps.astype("datetime64[ms]")
    fig, ax = plt.subplots(figsize=(13, 5))
    for sample_number, position in enumerate(samples, start=1):
        ax.plot(dates, data[:, position], lw=0.8, label=f"S{sample_number}: node {int(boundary.node_ids[position])}")
    suffix = "" if block == 1 else f"; {block}-record means for display"
    ax.set_title(title + suffix)
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Elevation (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _energetic_window(data: np.ndarray, times_ms: np.ndarray, dt_hours: float, days: float = 30.0) -> slice:
    length = min(len(times_ms), max(2, int(round(days * 24.0 / dt_hours))))
    if length >= len(times_ms):
        return slice(0, len(times_ms))
    energy = np.nanmean(data * data, axis=1)
    rms_window = max(1, int(round(24.0 / dt_hours)))
    running = np.convolve(energy, np.ones(rms_window) / rms_window, mode="same")
    half = length // 2
    valid = running.copy()
    valid[:half] = -np.inf
    valid[len(valid) - (length - half) :] = -np.inf
    center = int(np.argmax(valid))
    start = max(0, min(center - half, len(times_ms) - length))
    return slice(start, start + length)


def _plot_spectrum(
    frequency: np.ndarray,
    median_psd: np.ndarray,
    sample_psd: list[np.ndarray],
    tidal_min_hours: float,
    tidal_max_hours: float,
    vlf_min_days: float,
    output: Path,
) -> None:
    valid = frequency > 0
    fig, ax = plt.subplots(figsize=(11, 6))
    for power in sample_psd:
        ax.loglog(frequency[valid], power[valid], color="0.7", lw=0.5, alpha=0.55)
    ax.loglog(frequency[valid], median_psd[valid], color="navy", lw=1.5, label="Boundary-median series")
    for name, period in TIDAL_MARKERS.items():
        marker = 1.0 / period
        ax.axvline(marker, color="tab:red", lw=0.7, ls=":")
        ax.text(marker, ax.get_ylim()[1] * 0.35, name, rotation=90, fontsize=7, ha="right")
    for period, label in ((tidal_min_hours, "4 h"), (tidal_max_hours, "34 h"), (vlf_min_days * 24, f"{vlf_min_days:g} d")):
        ax.axvline(1.0 / period, color="0.25", lw=0.8, ls="--", label=label)
    ax.set_xlabel("Frequency (cycles per hour)")
    ax.set_ylabel("Power spectral density (m² per cycles/hour)")
    ax.set_title("FVCOM boundary elevation Welch spectrum")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_hovmoller(
    data: np.ndarray,
    times_ms: np.ndarray,
    boundary: Boundary,
    output: Path,
    title: str,
    max_rows: int,
    edge_hours: float | None = None,
) -> int:
    plot_data, stamps, block = _aggregate_time(data, times_ms, max_rows)
    dates = stamps.astype("datetime64[ms]")
    limit = float(np.nanpercentile(np.abs(plot_data), 99.0))
    if not np.isfinite(limit) or limit == 0:
        limit = 1e-6
    fig, axes = plt.subplots(
        len(boundary.arcs), 1, figsize=(13, max(4.0, 3.1 * len(boundary.arcs))), squeeze=False, sharex=True
    )
    image = None
    for arc_number, (axis, arc) in enumerate(zip(axes[:, 0], boundary.arcs), start=1):
        distance = arc_distances_km(boundary, arc)
        image = axis.pcolormesh(
            dates, distance, plot_data[:, arc].T, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        if edge_hours is not None:
            edge = np.timedelta64(int(round(edge_hours * 3_600_000.0)), "ms")
            axis.axvspan(dates[0], min(dates[-1], dates[0] + edge), color="black", alpha=0.07, lw=0)
            axis.axvspan(max(dates[0], dates[-1] - edge), dates[-1], color="black", alpha=0.07, lw=0)
            axis.text(
                0.01,
                0.98,
                "Shaded endpoints are filter-sensitive",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            )
        axis.set_ylabel(f"Arc {arc_number}\ndistance (km)")
    axes[-1, 0].set_xlabel("UTC time")
    suffix = "" if block == 1 else f" ({block}-record means for display)"
    fig.suptitle(title + suffix)
    fig.autofmt_xdate()
    bottom = 0.18 if len(boundary.arcs) == 1 else 0.10
    top = 0.82 if "\n" in title else 0.90
    fig.subplots_adjust(left=0.10, right=0.88, bottom=bottom, top=top, hspace=0.18)
    if image is not None:
        fig.colorbar(image, ax=axes[:, 0].tolist(), label="Elevation (m)", pad=0.02)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return block


def validate_forcing(
    forcing_path: str | Path,
    boundary: Boundary,
    qa_dir: str | Path,
    *,
    tidal_min_hours: float = 4.0,
    tidal_max_hours: float = 34.0,
    vlf_min_days: float = 90.0,
) -> dict[str, Any]:
    forcing = Path(forcing_path)
    output_dir = Path(qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    with nc4.Dataset(forcing) as dataset:
        missing_dimensions = [name for name in ("nobc", "time", "DateStrLen") if name not in dataset.dimensions]
        if missing_dimensions:
            raise ValueError(f"Forcing file is missing dimensions {missing_dimensions}")
        if not dataset.dimensions["time"].isunlimited():
            raise ValueError("FVCOM time dimension must be unlimited")
        if len(dataset.dimensions["DateStrLen"]) != 26:
            raise ValueError("FVCOM DateStrLen dimension must be 26")
        missing = [name for name in REQUIRED_VARIABLES if name not in dataset.variables]
        if missing:
            raise ValueError(f"Forcing file is missing variables {missing}")
        expected_dimensions = {
            "obc_nodes": ("nobc",),
            "iint": ("time",),
            "time": ("time",),
            "Itime": ("time",),
            "Itime2": ("time",),
            "Times": ("time", "DateStrLen"),
            "elevation": ("time", "nobc"),
        }
        wrong_dimensions = {
            name: dataset.variables[name].dimensions
            for name, expected in expected_dimensions.items()
            if dataset.variables[name].dimensions != expected
        }
        if wrong_dimensions:
            raise ValueError(f"FVCOM variable dimensions are incorrect: {wrong_dimensions}")
        if str(getattr(dataset, "type", "")) != "FVCOM TIME SERIES ELEVATION FORCING FILE":
            raise ValueError("FVCOM forcing type attribute is absent or incorrect")
        if dataset.data_model != "NETCDF3_CLASSIC":
            raise ValueError(f"Expected NETCDF3_CLASSIC, found {dataset.data_model}")
        nodes = np.asarray(dataset.variables["obc_nodes"][:], dtype=np.int64)
        if not np.array_equal(nodes, boundary.node_ids):
            raise ValueError("Forcing obc_nodes do not exactly match the selected boundary order")
        elevation = np.ma.filled(dataset.variables["elevation"][:], np.nan).astype(np.float64)
        if elevation.shape != (len(dataset.dimensions["time"]), len(nodes)):
            raise ValueError("Elevation does not have shape (time, nobc)")
        if not np.all(np.isfinite(elevation)):
            raise ValueError("Forcing elevation contains missing or non-finite values")
        if str(getattr(dataset.variables["elevation"], "units", "")) != "meters":
            raise ValueError("FVCOM elevation units must be 'meters'")
        times_ms, _, _ = decode_netcdf_times(dataset)
        int_times = (
            MJD_EPOCH_MS
            + np.asarray(dataset.variables["Itime"][:], dtype=np.int64) * DAY_MS
            + np.asarray(dataset.variables["Itime2"][:], dtype=np.int64)
        )
        if not np.array_equal(times_ms, int_times):
            raise ValueError("Times and Itime/Itime2 disagree at millisecond precision")
        expected_chars = format_times(times_ms)
        actual_chars = np.asarray(dataset.variables["Times"][:])
        if not np.array_equal(actual_chars, expected_chars):
            raise ValueError("Times strings are not canonical UTC millisecond timestamps")
        expected_float = (
            np.floor_divide(times_ms - MJD_EPOCH_MS, DAY_MS).astype(np.float64)
            + np.mod(times_ms - MJD_EPOCH_MS, DAY_MS).astype(np.float64) / DAY_MS
        ).astype(np.float32)
        actual_float = np.asarray(dataset.variables["time"][:], dtype=np.float32)
        if not np.array_equal(actual_float, expected_float):
            raise ValueError("Float MJD time is inconsistent with the canonical exact time axis")
        iint = np.asarray(dataset.variables["iint"][:], dtype=np.int64)
        if not np.array_equal(iint, np.arange(1, len(times_ms) + 1)):
            raise ValueError("iint must be one-based and contiguous")
        title = str(getattr(dataset, "title", ""))
        datum = str(getattr(dataset, "vertical_datum", "unspecified"))

    differences = np.diff(times_ms)
    if len(times_ms) < 2:
        raise ValueError("Forcing QA requires at least two timestamps")
    if np.any(differences <= 0):
        raise ValueError("Forcing timestamps are not strictly increasing")
    cadence_ms = int(np.median(differences))
    if np.max(np.abs(differences - cadence_ms)) > 1:
        raise ValueError("Forcing timestamps are not regular to millisecond precision")
    dt_hours = cadence_ms / 3_600_000.0
    record_days = (times_ms[-1] - times_ms[0]) / DAY_MS
    if record_days < 2.0 * vlf_min_days:
        warnings.append(
            f"The {record_days:.1f}-day record is shorter than two {vlf_min_days:g}-day VLF periods; VLF interpretation is limited"
        )

    bands, filter_warnings = _filter_bands(
        elevation, dt_hours, tidal_min_hours, tidal_max_hours, vlf_min_days
    )
    warnings.extend(filter_warnings)
    samples = _sample_positions(boundary)
    frequency, median_psd, peaks, sample_psd = _spectrum(
        elevation, dt_hours, samples, tidal_min_hours, tidal_max_hours, vlf_min_days
    )
    elapsed_years = (times_ms - times_ms[0]) / (365.2425 * DAY_MS)
    slopes = np.asarray([np.polyfit(elapsed_years, elevation[:, column], 1)[0] for column in range(elevation.shape[1])])

    artifacts = {
        "boundary_map": output_dir / "boundary_sample_map.png",
        "sample_timeseries": output_dir / "boundary_sample_timeseries.png",
        "sample_tidal_window": output_dir / "boundary_sample_tidal_window.png",
        "spectrum": output_dir / "boundary_power_spectrum.png",
        "hovmoller_total": output_dir / "boundary_hovmoller_total.png",
        "hovmoller_tidal": output_dir / "boundary_hovmoller_tidal.png",
        "hovmoller_subtidal": output_dir / "boundary_hovmoller_subtidal.png",
        "hovmoller_vlf": output_dir / "boundary_hovmoller_vlf_slr_scale.png",
    }
    _plot_boundary_map(boundary, samples, artifacts["boundary_map"])
    _plot_sample_series(
        elevation, times_ms, boundary, samples, artifacts["sample_timeseries"], "FVCOM boundary elevation sample nodes"
    )
    tidal_window = _energetic_window(bands["tidal"], times_ms, dt_hours)
    _plot_sample_series(
        elevation[tidal_window], times_ms[tidal_window], boundary, samples,
        artifacts["sample_tidal_window"], "Energetic 30-day boundary elevation window", max_rows=8000,
    )
    _plot_spectrum(
        frequency, median_psd, sample_psd, tidal_min_hours, tidal_max_hours, vlf_min_days, artifacts["spectrum"]
    )
    aggregation = {
        "total_records_per_display_bin": _plot_hovmoller(
            elevation, times_ms, boundary, artifacts["hovmoller_total"], "Complete-boundary total elevation", 2000
        ),
        "tidal_records_per_display_bin": _plot_hovmoller(
            bands["tidal"][tidal_window], times_ms[tidal_window], boundary,
            artifacts["hovmoller_tidal"], f"Complete-boundary tidal band ({tidal_min_hours:g}-{tidal_max_hours:g} h)",
            8000, edge_hours=tidal_max_hours,
        ),
        "subtidal_records_per_display_bin": _plot_hovmoller(
            bands["subtidal"], times_ms, boundary, artifacts["hovmoller_subtidal"],
            f"Complete-boundary subtidal band ({tidal_max_hours:g} h-{vlf_min_days:g} d)",
            2000, edge_hours=vlf_min_days * 24.0,
        ),
        "vlf_records_per_display_bin": _plot_hovmoller(
            bands["vlf"], times_ms, boundary, artifacts["hovmoller_vlf"],
            (
                f"Complete-boundary VLF/SLR-scale variation (>{vlf_min_days:g} d; not an SLR attribution)\n"
                f"Fitted total-record median trend: {np.median(slopes) * 1000.0:+.2f} mm/year"
            ),
            2000, edge_hours=vlf_min_days * 24.0,
        ),
    }

    return {
        "status": "pass",
        "forcing_file": str(forcing.resolve()),
        "case_name": title,
        "vertical_datum": datum,
        "netcdf_format": "NETCDF3_CLASSIC",
        "node_count": int(len(nodes)),
        "arc_count": int(len(boundary.arcs)),
        "time_count": int(len(times_ms)),
        "time_start_utc": iso_utc(times_ms[[0]])[0],
        "time_end_utc": iso_utc(times_ms[[-1]])[0],
        "cadence_seconds": cadence_ms / 1000.0,
        "record_days": float(record_days),
        "elevation_min_m": float(np.min(elevation)),
        "elevation_max_m": float(np.max(elevation)),
        "structural_checks": {
            "netcdf3_classic": True,
            "required_dimensions_present": True,
            "time_dimension_unlimited": True,
            "date_strlen_26": True,
            "required_variables_present": True,
            "variable_dimensions_exact": True,
            "boundary_node_order_exact": True,
            "elevation_shape_exact": True,
            "elevation_finite": True,
            "elevation_units_meters": True,
            "iint_one_based": True,
            "time_strictly_monotonic": True,
            "time_regular_to_millisecond": True,
            "times_matches_itime_itime2": True,
            "legacy_float_mjd_matches_expected_float32": True,
        },
        "sample_nodes": [
            {
                "sample": index + 1,
                "boundary_position": int(position),
                "node_id": int(boundary.node_ids[position]),
                "longitude": float(boundary.lon[position]),
                "latitude": float(boundary.lat[position]),
            }
            for index, position in enumerate(samples)
        ],
        "bands": {
            "tidal_period_hours": [float(tidal_min_hours), float(tidal_max_hours)],
            "subtidal_period_hours_to_days": [float(tidal_max_hours), float(vlf_min_days)],
            "vlf_slr_scale_period_days_minimum": float(vlf_min_days),
            "slr_attribution": False,
        },
        "prominent_spectral_peaks": peaks,
        "linear_trend_mm_per_year": {
            "boundary_median": float(np.median(slopes) * 1000.0),
            "minimum": float(np.min(slopes) * 1000.0),
            "maximum": float(np.max(slopes) * 1000.0),
        },
        "display_aggregation": aggregation,
        "qa_artifacts": {name: str(path.resolve()) for name, path in artifacts.items()},
        "warnings": warnings,
    }
