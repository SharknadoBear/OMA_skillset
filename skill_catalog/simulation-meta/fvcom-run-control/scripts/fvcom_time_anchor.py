"""Shared, externally anchored FVCOM IINT time checks (frozen 4.3.1 semantics)."""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np

UTC = dt.timezone.utc
MJD_EPOCH = dt.datetime(1858, 11, 17, tzinfo=UTC)
ANCHOR_SCHEMA = "fvcom_iint_time_anchor_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def iso(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def finite_array(variable: Any) -> np.ndarray:
    values = np.ma.asarray(variable[:])
    if np.ma.getmaskarray(values).any() or not np.isfinite(values.data).all():
        raise ValueError(f"{variable.name} has masked or nonfinite values")
    return np.asarray(values.data)


def read_iint(ds: Any) -> np.ndarray:
    if "iint" not in ds.variables:
        raise ValueError("NetCDF is missing iint")
    raw = finite_array(ds["iint"])
    if raw.ndim != 1 or not len(raw) or np.any(raw != np.floor(raw)):
        raise ValueError("iint must contain a nonempty vector of integer steps")
    values = raw.astype(np.int64)
    if np.any(values != raw) or np.any(np.diff(values) <= 0):
        raise ValueError("iint must be strictly increasing within each file")
    return values


def exact_clock(ds: Any) -> list[dt.datetime]:
    missing = [name for name in ("Times", "Itime", "Itime2") if name not in ds.variables]
    if missing:
        raise ValueError(f"exact FVCOM clock is missing {missing}")
    raw = np.ma.asarray(ds["Times"][:])
    # NetCDF character padding may be masked; filled NUL padding is harmless.
    chars = np.asarray(raw.filled(b"\0") if np.ma.isMaskedArray(raw) else raw)
    if chars.ndim != 2:
        raise ValueError("Times must be a two-dimensional character array")
    times = [parse_utc(b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip("\0 ")) for row in chars]
    days, millis = finite_array(ds["Itime"]), finite_array(ds["Itime2"])
    if days.shape != (len(times),) or millis.shape != days.shape or not len(times):
        raise ValueError("Times and integer clock dimensions differ")
    if np.any(days != np.floor(days)) or np.any(millis != np.floor(millis)) or np.any((millis < 0) | (millis >= 86400000)):
        raise ValueError("Itime/Itime2 must be integer days and valid milliseconds of day")
    integer_times = [MJD_EPOCH + dt.timedelta(days=int(day), milliseconds=int(ms)) for day, ms in zip(days, millis)]
    if times != integer_times:
        raise ValueError("Times disagrees with Itime/Itime2")
    return times


def restart_anchor(path: Path, start: dt.datetime, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("startup restart differs from frozen SHA-256")
    with nc4.Dataset(path) as ds:
        steps, times = read_iint(ds), exact_clock(ds)
    if len(steps) != len(times):
        raise ValueError("startup restart iint/clock lengths differ")
    indices = [index for index, time in enumerate(times) if time == start]
    if len(indices) != 1:
        raise ValueError("startup restart must contain exactly one record at START_DATE")
    index = indices[0]
    return {"schema": ANCHOR_SCHEMA, "status": "verified", "method": "startup_restart",
            "start_utc": iso(start), "iint_at_start": int(steps[index]),
            "startup_restart_path": str(path), "startup_restart_sha256": digest,
            "startup_restart_record_index": index, "startup_restart_time_utc": iso(times[index])}


def resolve_anchor(values: dict[str, str], namelist_path: Path, *, startup_restart: Path | None = None,
                   expected_sha256: str | None = None, supplied: dict[str, Any] | None = None) -> dict[str, Any]:
    start = parse_utc(values["START_DATE"])
    kind = values.get("STARTUP_TYPE", "").lower()
    if kind == "hotstart":
        if startup_restart is None:
            candidate = Path(values.get("STARTUP_FILE", ""))
            if not values.get("STARTUP_FILE"):
                raise ValueError("hotstart requires STARTUP_FILE or explicit startup restart")
            input_dir = Path(values.get("INPUT_DIR", "."))
            if not input_dir.is_absolute():
                input_dir = Path(namelist_path).resolve().parent / input_dir
            startup_restart = candidate if candidate.is_absolute() else input_dir / candidate
        anchor = restart_anchor(startup_restart, start, expected_sha256)
    elif kind == "coldstart":
        anchor = {"schema": ANCHOR_SCHEMA, "status": "verified", "method": "coldstart_zero",
                  "start_utc": iso(start), "iint_at_start": 0}
    else:
        raise ValueError("an explicit hotstart or coldstart namelist is required; first_iint is not an absolute time anchor")
    if supplied is not None:
        for key in ("schema", "status", "method", "start_utc", "iint_at_start", "startup_restart_sha256", "startup_restart_record_index"):
            if supplied.get(key) != anchor.get(key):
                raise ValueError(f"supplied time anchor disagrees with startup evidence: {key}")
    return anchor


def anchored_times(steps: np.ndarray, anchor: dict[str, Any], internal_step: float) -> list[dt.datetime]:
    if anchor.get("schema") != ANCHOR_SCHEMA or anchor.get("status") != "verified":
        raise ValueError("verified external time anchor is required")
    origin = anchor.get("iint_at_start")
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise ValueError("time anchor iint_at_start must be an integer")
    if not np.isfinite(internal_step) or internal_step <= 0:
        raise ValueError("internal step must be positive and finite")
    micros = internal_step * 1e6
    if not np.isclose(micros, round(micros), rtol=0, atol=1e-6):
        raise ValueError("internal step is not representable to one microsecond")
    start = parse_utc(anchor["start_utc"])
    return [start + dt.timedelta(microseconds=(int(step) - origin) * int(round(micros))) for step in steps]


def check_file_clock(ds: Any, expected: list[dt.datetime], *, require_exact: bool = False) -> dict[str, Any]:
    present = any(name in ds.variables for name in ("Times", "Itime", "Itime2"))
    if require_exact or present:
        if exact_clock(ds) != expected:
            raise ValueError("exact NetCDF clock disagrees with startup-anchored iint")
    diagnostic = {"exact_clock_verified": bool(require_exact or present)}
    if "time" in ds.variables:
        var = ds["time"]
        raw = finite_array(var)
        if raw.shape != (len(expected),):
            raise ValueError("floating time has the wrong shape")
        units = getattr(var, "units", "")
        calendar = getattr(var, "calendar", "standard")
        if not units.lower().startswith("days since"):
            raise ValueError("floating FVCOM time must declare days-since units")
        exact = nc4.date2num(expected, units=units, calendar=calendar)
        # FVCOM writes single-precision MJD; half an ULP is unavoidable.
        tolerance = np.abs(np.spacing(raw)) / 2 + 1e-10
        error = np.abs(raw.astype(np.float64) - np.asarray(exact, dtype=np.float64))
        if np.any(error > tolerance):
            raise ValueError("floating NetCDF time disagrees with anchored UTC beyond its precision")
        diagnostic["maximum_float_time_error_seconds"] = float(np.max(error) * 86400)
    return diagnostic


def audit_restart(path: Path, anchor: dict[str, Any], internal_step: float, end: dt.datetime) -> dict[str, Any]:
    with nc4.Dataset(path) as ds:
        steps = read_iint(ds)
        times = anchored_times(steps, anchor, internal_step)
        check_file_clock(ds, times, require_exact=True)
    if times[-1] != end or any(time < parse_utc(anchor["start_utc"]) or time > end for time in times):
        raise ValueError("restart coverage does not end at the exact requested END_DATE")
    return {"record_count": len(times), "iint_last": int(steps[-1]), "decoded_end_utc": iso(times[-1]),
            "exact_clock_verified": True, "startup_anchored": True}
