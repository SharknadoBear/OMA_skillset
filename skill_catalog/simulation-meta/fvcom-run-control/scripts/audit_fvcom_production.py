from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np

from fvcom_time_anchor import (resolve_anchor, anchored_times, read_iint, check_file_clock, audit_restart)


FAILURE = re.compile(r"blow\s*up|fatal|segmentation|mpi_abort|forrtl:|floating invalid|nan detected", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_namelist(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*,?\s*(?:!.*)?$", line)
        if not match:
            continue
        value = match.group(2).rstrip(",").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[match.group(1).upper()] = value
    for name in ("START_DATE", "END_DATE", "EXTSTEP_SECONDS", "ISPLIT"):
        if name not in values:
            raise ValueError(f"namelist is missing {name}")
    return values


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def names(variable: Any) -> list[str]:
    return [
        b"".join(np.asarray(row, dtype="S1").tolist()).decode("utf-8", errors="replace").rstrip("\x00 ")
        for row in np.asarray(variable[:])
    ]


def finite_variable(variable: Any, chunk: int = 8) -> bool:
    def valid(values: Any) -> bool:
        array = np.ma.asarray(values)
        return not np.ma.getmaskarray(array).any() and bool(np.isfinite(array.data).all())
    if variable.ndim == 0:
        return valid(variable[...])
    for start in range(0, variable.shape[0], chunk):
        if not valid(variable[start:start + chunk]):
            return False
    return True


def time_audit(paths: list[Path], start: dt.datetime, end: dt.datetime, internal_step: float,
               expected_count: int, interval_seconds: float | None = None, *,
               anchor: dict[str, Any], require_exact_clock: bool = False,
               duplicate_variables: tuple[str, ...] = ("zeta", "ua", "va")) -> dict[str, Any]:
    if not np.isfinite(internal_step) or internal_step <= 0 or expected_count < 2:
        raise ValueError("monthly output requires a positive internal step and at least two records")
    if interval_seconds is None:
        interval_seconds = (end - start).total_seconds() / (expected_count - 1)
    interval_steps = interval_seconds / internal_step
    if not np.isfinite(interval_steps) or interval_steps < 1 or not np.isclose(interval_steps, round(interval_steps), rtol=0, atol=1e-7):
        raise ValueError("expected cadence must divide the internal step exactly")
    parts: list[np.ndarray] = []
    rows: dict[int, tuple[Path, int]] = {}
    duplicate_count = 0
    for path in paths:
        with nc4.Dataset(path) as ds:
            values = read_iint(ds)
            check_file_clock(ds, anchored_times(values, anchor, internal_step), require_exact=require_exact_clock)
            for index, step in enumerate(values):
                if int(step) in rows:
                    prior_path, prior_index = rows[int(step)]
                    with nc4.Dataset(prior_path) as previous:
                        dynamic = {name for name in PHYSICAL_FIELDS & set(ds.variables)
                                   if ds[name].dimensions and ds[name].dimensions[0] == "time"}
                        prior_dynamic = {name for name in PHYSICAL_FIELDS & set(previous.variables)
                                         if previous[name].dimensions and previous[name].dimensions[0] == "time"}
                        for name in sorted(set(duplicate_variables) | dynamic | prior_dynamic):
                            if name not in previous.variables or name not in ds.variables:
                                raise ValueError(f"cannot verify duplicate iint {step}: missing {name}")
                            if not np.array_equal(np.asarray(previous[name][prior_index]), np.asarray(ds[name][index])):
                                raise ValueError(f"conflicting {name} at duplicate iint {step}")
                    duplicate_count += 1
                else:
                    rows[int(step)] = (path, index)
            parts.append(values)
    values = np.unique(np.concatenate(parts))
    if len(values) != expected_count:
        raise ValueError(f"expected {expected_count} unique output records, found {len(values)}")
    if not np.all(np.diff(values) == round(interval_steps)):
        raise ValueError(f"output iint cadence differs from {interval_seconds:g} seconds")
    times = anchored_times(values, anchor, internal_step)
    if times[0] != start or abs((times[-1] - end).total_seconds()) > 1.0e-6:
        raise ValueError(f"decoded output coverage is {times[0].isoformat()} through {times[-1].isoformat()}")
    return {
        "startup_anchored": True,
        "exact_clock_verified": require_exact_clock,
        "iint_at_start": anchor["iint_at_start"],
        "record_count": int(len(values)),
        "iint_first": int(values[0]),
        "iint_last": int(values[-1]),
        "interval_seconds": interval_seconds,
        "interval_internal_steps": int(round(interval_steps)),
        "verified_duplicate_boundary_records": duplicate_count,
        "decoded_start_utc": times[0].isoformat().replace("+00:00", "Z"),
        "decoded_end_utc": times[-1].isoformat().replace("+00:00", "Z"),
    }


# Actual frozen mod_ncdio.F names: vertical velocity is ww, not w.
PHYSICAL_FIELDS = {"zeta", "ua", "va", "u", "v", "tauc", "ww", "omega", "temp", "salinity",
                   "viscofm", "viscofh", "km", "kh", "kq", "q2", "q2l", "l", "et",
                   "rho1", "rmean1", "tmean1", "smean1"}
TURBULENCE_FIELDS = ["viscofm", "viscofh", "km", "kh", "kq", "q2", "q2l", "l"]


def required_fields(values: dict[str, str], kind: str) -> list[str]:
    result = ["iint", "zeta", "ua", "va"]
    def enabled(name: str) -> bool:
        value = values.get(name, "").strip().lower()
        if value not in {"t", ".true.", "f", ".false."}:
            raise ValueError(f"explicit logical {name} is required for the field audit")
        return value in {"t", ".true."}
    if kind == "restart":
        result += ["u", "v", "tauc", "ww", "omega", "temp", "salinity"] + TURBULENCE_FIELDS
    elif kind == "full":
        for flag, fields in (("NC_VELOCITY", ["u", "v", "tauc"]), ("NC_VERTICAL_VEL", ["ww", "omega"]),
                             ("NC_SALT_TEMP", ["temp", "salinity"]), ("NC_TURBULENCE", TURBULENCE_FIELDS)):
            if enabled(flag):
                result += fields
    elif kind == "station":
        if enabled("OUT_VELOCITY_3D"):
            result += ["u", "v", "ww"]
        if enabled("OUT_SALT_TEMP"):
            result += ["temp", "salinity"]
    else:
        raise ValueError(f"unknown output kind {kind}")
    return result


def nc_audit(paths: list[Path], required: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with nc4.Dataset(path) as ds:
            missing = [item for item in required if item not in ds.variables]
            if missing:
                raise ValueError(f"{path} is missing variables {missing}")
            checked = sorted(set(required) | (PHYSICAL_FIELDS & set(ds.variables)))
            nonfinite = [item for item in checked if not finite_variable(ds[item])]
            if nonfinite:
                raise ValueError(f"{path} has non-finite variables {nonfinite}")
            rows.append({
                "path": str(path),
                "sha256": sha256(path),
                "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
                "required_variables": required,
                "checked_variables": checked,
                "required_variables_finite_and_unmasked": True,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an immutable FVCOM monthly production package")
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--namelist", required=True)
    parser.add_argument("--station-netcdf", action="append", required=True)
    parser.add_argument("--full-grid-netcdf", action="append", required=True)
    parser.add_argument("--restart-netcdf", required=True)
    parser.add_argument("--startup-restart-netcdf", type=Path,
                        help="Actual frozen startup restart; otherwise resolve INPUT_DIR/STARTUP_FILE")
    parser.add_argument("--station-mapping", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--expected-station-records", type=int, default=7201)
    parser.add_argument("--expected-full-grid-records", type=int, default=241)
    parser.add_argument("--station-interval-seconds", type=float, default=360)
    parser.add_argument("--full-grid-interval-seconds", type=float, default=10800)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stdout_path, stderr_path = Path(args.stdout), Path(args.stderr)
    namelist_path = Path(args.namelist)
    station_paths = [Path(item) for item in args.station_netcdf]
    full_paths = [Path(item) for item in args.full_grid_netcdf]
    restart_path = Path(args.restart_netcdf)
    mapping_path, lineage_path = Path(args.station_mapping), Path(args.lineage)
    values = parse_namelist(namelist_path)
    start, end = parse_utc(values["START_DATE"]), parse_utc(values["END_DATE"])
    extstep, isplit = float(values["EXTSTEP_SECONDS"]), int(values["ISPLIT"])
    internal_step = extstep * isplit
    stdout, stderr = stdout_path.read_text(encoding="utf-8", errors="replace"), stderr_path.read_text(encoding="utf-8", errors="replace")
    problems = []
    if args.exit_code != 0:
        problems.append(f"nonzero exit code {args.exit_code}")
    if "TADA!" not in stdout:
        problems.append("stdout lacks TADA!")
    if end.strftime("%Y-%m-%dT%H:%M:%S") not in stdout:
        problems.append("stdout lacks exact requested final timestamp")
    if FAILURE.search(stdout + "\n" + stderr):
        problems.append("failure marker found in runtime logs")
    if problems:
        raise ValueError("; ".join(problems))

    lineage = json.loads(lineage_path.read_text(encoding="utf-8-sig"))
    startup_digest = lineage.get("files", {}).get(Path(values.get("STARTUP_FILE", "")).name)
    if values.get("STARTUP_TYPE", "").lower() == "hotstart" and not startup_digest:
        raise ValueError("hotstart time anchoring requires the startup restart in the frozen file map")
    anchor = resolve_anchor(values, namelist_path, startup_restart=args.startup_restart_netcdf,
                            expected_sha256=startup_digest)
    station_required, full_required = required_fields(values, "station"), required_fields(values, "full")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    expected_names = [str(row["station_id"]) for row in mapping["stations"]]
    station_nc = nc_audit(station_paths, station_required)
    for path in station_paths:
        with nc4.Dataset(path) as ds:
            found = names(ds["name_station"])
        if found != expected_names:
            raise ValueError(f"station IDs/order differ in {path}: {found} != {expected_names}")
    full_nc = nc_audit(full_paths, full_required)
    restart_nc = nc_audit([restart_path], required_fields(values, "restart"))[0]

    restart_time = audit_restart(restart_path, anchor, internal_step, end)

    mapping_digest = lineage.get("station_mapping_sha256") or lineage.get("files", {}).get(mapping_path.name)
    if mapping_digest != sha256(mapping_path):
        raise ValueError("station-mapping hash differs from frozen lineage")
    if lineage.get("run_namelist_sha256") and lineage["run_namelist_sha256"] != sha256(namelist_path):
        raise ValueError("run-namelist hash differs from frozen lineage")

    result = {
        "schema": "fvcom_monthly_production_audit_v1",
        "audit_rules_version": "startup_anchor_exact_clock_3d_v1",
        "time_anchor": anchor,
        "restart_time": restart_time,
        "status": "passed",
        "audited_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "period": {"start_utc": start.isoformat().replace("+00:00", "Z"), "end_utc": end.isoformat().replace("+00:00", "Z")},
        "numerics": {"extstep_seconds": extstep, "isplit": isplit, "internal_step_seconds": internal_step},
        "run_namelist": str(namelist_path),
        "run_namelist_sha256": sha256(namelist_path),
        "logs": {"stdout_sha256": sha256(stdout_path), "stderr_sha256": sha256(stderr_path), "exit_code": args.exit_code, "tada": True},
        "station_time": time_audit(station_paths, start, end, internal_step, args.expected_station_records, args.station_interval_seconds,
                                   anchor=anchor, duplicate_variables=tuple(x for x in station_required if x != "iint")),
        "full_grid_time": time_audit(full_paths, start, end, internal_step, args.expected_full_grid_records, args.full_grid_interval_seconds, anchor=anchor,
                                     require_exact_clock=True, duplicate_variables=tuple(x for x in full_required if x != "iint")),
        "station_netcdf": station_nc,
        "full_grid_netcdf": full_nc,
        "restart_netcdf": restart_nc,
        "station_ids_in_order": expected_names,
        "station_mapping_sha256": sha256(mapping_path),
        "lineage": lineage,
        "lineage_manifest": str(lineage_path),
        "lineage_manifest_sha256": sha256(lineage_path),
        "blocking_reasons": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
