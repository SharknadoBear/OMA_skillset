from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np

# Installed skills and the simulation-meta source catalog both use sibling skills.
_script_path = Path(__file__).resolve()
for _run_control in (_script_path.parents[2] / "fvcom-run-control" / "scripts",
                     _script_path.parents[3] / "simulation-meta" / "fvcom-run-control" / "scripts"):
    if (_run_control / "fvcom_time_anchor.py").is_file():
        sys.path.insert(0, str(_run_control))
        break
from fvcom_time_anchor import resolve_anchor, anchored_times, read_iint, finite_array, check_file_clock


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_namelist(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*,?\s*(?:!.*)?$", raw)
        if match:
            value = match.group(2).rstrip(",").strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[match.group(1).upper()] = value
    for required in ("START_DATE", "END_DATE", "EXTSTEP_SECONDS", "ISPLIT"):
        if required not in values:
            raise ValueError(f"run namelist is missing {required}")
    return values


def parse_utc(value: str) -> dt.datetime:
    clean = value.strip().replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def station_names(variable: Any) -> list[str]:
    array = np.asarray(variable[:])
    if array.ndim != 2:
        raise ValueError("name_station must be a two-dimensional character array")
    result = []
    for row in array:
        value = b"".join(np.asarray(row, dtype="S1").tolist()).decode("utf-8", errors="replace")
        result.append(value.rstrip("\x00 "))
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError("station names are empty or duplicated")
    return result


def load_stack(paths: list[Path]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    reference_names: list[str] | None = None
    iint_parts: list[np.ndarray] = []
    fields: dict[str, list[np.ndarray]] = {"zeta": [], "ua": [], "va": []}
    source_rows: list[dict[str, Any]] = []
    for path in paths:
        with nc4.Dataset(path) as ds:
            missing = [name for name in ("name_station", "iint", "zeta", "ua", "va") if name not in ds.variables]
            if missing:
                raise ValueError(f"{path} is missing station variables: {', '.join(missing)}")
            names = station_names(ds["name_station"])
            if reference_names is None:
                reference_names = names
            elif names != reference_names:
                raise ValueError(f"station order differs across NetCDF stacks: {path}")
            iint = read_iint(ds)
            if iint.ndim != 1 or not len(iint):
                raise ValueError(f"{path} has no station records")
            iint_parts.append(iint)
            for field in fields:
                data = finite_array(ds[field]).astype(np.float64)
                if data.shape != (len(iint), len(names)):
                    raise ValueError(f"{path}:{field} shape {data.shape} does not match time/station dimensions")
                fields[field].append(data)
            quantization = None
            if "time" in ds.variables and len(ds["time"][:]) == len(iint):
                raw = finite_array(ds["time"]).astype(np.float64)
                raw_delta = (raw - raw[0]) * 86400.0
                exact_delta = (iint - iint[0]).astype(np.float64)
                quantization = {"raw_time_dtype": str(ds["time"].dtype), "raw_relative_seconds": raw_delta.tolist(),
                                "iint_relative_steps": exact_delta.tolist()}
            source_rows.append({"path": str(path), "sha256": sha256(path), "record_count": int(len(iint)),
                                "iint_first": int(iint[0]), "iint_last": int(iint[-1]),
                                "time_diagnostic": quantization})
    assert reference_names is not None
    all_iint = np.concatenate(iint_parts)
    all_fields = {name: np.concatenate(parts, axis=0) for name, parts in fields.items()}
    order = np.argsort(all_iint, kind="stable")
    all_iint = all_iint[order]
    all_fields = {name: values[order] for name, values in all_fields.items()}
    unique, first = np.unique(all_iint, return_index=True)
    if len(unique) != len(all_iint):
        for index in np.flatnonzero(np.diff(all_iint) == 0):
            for field, values in all_fields.items():
                if not np.array_equal(values[index], values[index + 1]):
                    raise ValueError(f"conflicting {field} at duplicate iint {all_iint[index]}")
        all_iint = unique
        all_fields = {name: values[first] for name, values in all_fields.items()}
    if np.any(np.diff(all_iint) <= 0):
        raise ValueError("iint is not strictly increasing after stack de-duplication")
    return reference_names, all_iint, all_fields, source_rows


def iso_time(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def condense(netcdf_paths: list[Path], mapping_path: Path, namelist_path: Path,
             output_dir: Path, manifest_path: Path, *, time_anchor: dict[str, Any] | None = None,
             startup_restart_path: Path | None = None) -> dict[str, Any]:
    if not netcdf_paths:
        raise ValueError("at least one --station-netcdf is required")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    if mapping.get("status") != "ready" or not mapping.get("stations"):
        raise ValueError("station mapping is not ready")
    expected_names = [str(row["station_id"]) for row in mapping["stations"]]
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("station mapping contains duplicate station IDs")
    mapped = {str(row["station_id"]): row for row in mapping["stations"]}
    names, iint, fields, sources = load_stack(netcdf_paths)
    if names != expected_names:
        raise ValueError(f"NetCDF station IDs/order {names} do not exactly match the station mapping {expected_names}")
    values = parse_namelist(namelist_path)
    start = parse_utc(values["START_DATE"])
    expected_end = parse_utc(values["END_DATE"])
    extstep = float(values["EXTSTEP_SECONDS"])
    if not np.isfinite(extstep) or extstep <= 0:
        raise ValueError("EXTSTEP_SECONDS must be positive")
    isplit_value = float(values["ISPLIT"])
    if not np.isfinite(isplit_value) or isplit_value <= 0 or not isplit_value.is_integer():
        raise ValueError("ISPLIT must be a positive integer")
    isplit = int(isplit_value)
    internal_step = extstep * isplit
    anchor = resolve_anchor(values, namelist_path, startup_restart=startup_restart_path, supplied=time_anchor)
    iint_origin = anchor["iint_at_start"]
    timestamps = anchored_times(iint, anchor, internal_step)
    if timestamps[0] != start or timestamps[-1] != expected_end:
        raise ValueError(f"station coverage is {iso_time(timestamps[0])} through {iso_time(timestamps[-1])}; expected {iso_time(start)} through {iso_time(expected_end)}")
    for path in netcdf_paths:
        with nc4.Dataset(path) as ds:
            check_file_clock(ds, anchored_times(read_iint(ds), anchor, internal_step))
    output_dir.mkdir(parents=True, exist_ok=True)
    index_by_name = {name: index for index, name in enumerate(names)}
    products = []
    diagnostic_products = []
    for station_id in expected_names:
        station = mapped[station_id]
        index = index_by_name[station_id]
        role = str(station["role"])
        if role == "water_level":
            path = output_dir / f"{station_id}_model_waterlevel.csv"
            write_csv(path, ["time", "model"], [[iso_time(time), f"{fields['zeta'][row, index]:.10g}"] for row, time in enumerate(timestamps)])
        elif role == "current":
            path = output_dir / f"{station_id}_model_current.csv"
            write_csv(path, ["time", "model_u", "model_v"], [[iso_time(time), f"{fields['ua'][row, index]:.10g}", f"{fields['va'][row, index]:.10g}"] for row, time in enumerate(timestamps)])
        elif role == "model_diagnostic" and station.get("validation_eligible") is False:
            path = output_dir / f"{station_id}_model_diagnostic.csv"
            write_csv(path, ["time", "model", "model_u", "model_v"], [[iso_time(time),
                f"{fields['zeta'][row, index]:.10g}", f"{fields['ua'][row, index]:.10g}",
                f"{fields['va'][row, index]:.10g}"] for row, time in enumerate(timestamps)])
        else:
            raise ValueError(f"unsupported station role {role!r} for {station_id}")
        destination = diagnostic_products if role == "model_diagnostic" else products
        destination.append({"station_id": station_id, "role": role, "path": str(path), "sha256": sha256(path),
                         "cell_id": int(station["cell_id"]), "record_count": int(len(iint)),
                         "spatial_mapping": station.get("spatial_mapping")})
    raw_time_warnings = []
    for source in sources:
        diagnostic = source.get("time_diagnostic")
        if diagnostic:
            raw = np.asarray(diagnostic.pop("raw_relative_seconds"), dtype=np.float64)
            raw_iint = np.asarray(diagnostic.pop("iint_relative_steps"), dtype=np.float64)
            expected = raw_iint * internal_step
            diagnostic["maximum_float_time_error_seconds"] = float(np.max(np.abs(raw - expected)))
            if diagnostic["maximum_float_time_error_seconds"] > 1.0:
                raw_time_warnings.append(
                    f"{source['path']} float time is quantized by up to {diagnostic['maximum_float_time_error_seconds']:.3f} s; exact UTC uses iint."
                )
    result = {
        "schema": "fvcom_station_condensation_v1", "status": "ready", "generated_at": utcnow(),
        "diagnostic_products": diagnostic_products,
        "time_reconstruction": {"method": "START_DATE + (iint - startup_iint) * EXTSTEP_SECONDS * ISPLIT",
                                "time_anchor": anchor,
                                "start_utc": iso_time(start), "end_utc": iso_time(timestamps[-1]),
                                "expected_end_utc": iso_time(expected_end), "extstep_seconds": extstep,
                                "isplit": isplit, "internal_step_seconds": internal_step,
                                "iint_origin": iint_origin, "iint_last": int(iint[-1]), "record_count": int(len(iint))},
        "station_mapping": str(mapping_path), "station_mapping_sha256": sha256(mapping_path),
        "run_namelist": str(namelist_path), "run_namelist_sha256": sha256(namelist_path),
        "sources": sources, "products": products, "warnings": raw_time_warnings,
        "blocking_reasons": [], "resume_token": f"station_condensed:{sha256(namelist_path)}:{int(iint[-1])}",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Condense FVCOM station NetCDF with exact iint-based UTC reconstruction")
    parser.add_argument("--station-netcdf", action="append", required=True)
    parser.add_argument("--station-mapping", required=True)
    parser.add_argument("--run-namelist", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--startup-restart-netcdf", type=Path)
    parser.add_argument("--time-anchor", type=Path, help="Audited anchor JSON; rechecked against actual startup restart")
    args = parser.parse_args()
    result = condense([Path(item) for item in args.station_netcdf], Path(args.station_mapping), Path(args.run_namelist),
                      Path(args.output_dir), Path(args.manifest), startup_restart_path=args.startup_restart_netcdf,
                      time_anchor=json.loads(args.time_anchor.read_text(encoding="utf-8-sig")) if args.time_anchor else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
