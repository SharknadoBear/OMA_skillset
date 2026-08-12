#!/usr/bin/env python3
"""Plan, fetch, inspect, extract, and health-check NOAA DBOFS ROMS data."""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:  # Package import (``import scripts``).
    from .roms_aws_core import (
        S3_ENDPOINT, ModelConfig, discovery_prefixes as _discovery_prefixes,
        discover_objects as _discover_objects, download_object,
        fetch_from_plan as _fetch_from_plan, fetch_request as _fetch_request,
        inventory_request as _inventory_request, iso_utc, json_clean,
        list_s3_objects, load_request as _load_request, manifest_paths,
        parse_object_key as _parse_object_key, parse_utc,
        plan_request as _plan_request, read_json, sha256_file,
        select_objects as _select_objects, validate_request as _validate_request,
        verify_transfers,
    )
    from .roms_processing import (
        _dataset_scalar, _filled, _variable_data, decode_ocean_times,
        evaluate_health as _evaluate_health, extract_fields, inspect_file,
        geometry_snapshot, inspect_paths, roms_depths,
    )
except ImportError:  # Direct script execution.
    from roms_aws_core import (
        S3_ENDPOINT, ModelConfig, discovery_prefixes as _discovery_prefixes,
        discover_objects as _discover_objects, download_object,
        fetch_from_plan as _fetch_from_plan, fetch_request as _fetch_request,
        inventory_request as _inventory_request, iso_utc, json_clean,
        list_s3_objects, load_request as _load_request, manifest_paths,
        parse_object_key as _parse_object_key, parse_utc,
        plan_request as _plan_request, read_json, sha256_file,
        select_objects as _select_objects, validate_request as _validate_request,
        verify_transfers,
    )
    from roms_processing import (
        _dataset_scalar, _filled, _variable_data, decode_ocean_times,
        evaluate_health as _evaluate_health, extract_fields, inspect_file,
        geometry_snapshot, inspect_paths, roms_depths,
    )

CONFIG = ModelConfig(
    model="dbofs",
    request_schema="dbofs_request_v1",
    connector_name="dbofs-fetcher",
)
UTC = timezone.utc


def load_request(path: str | Path) -> dict[str, Any]:
    return _load_request(path, CONFIG)


def validate_request(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_request(mapping, CONFIG)


def parse_object_key(key: str) -> dict[str, Any] | None:
    return _parse_object_key(key, CONFIG)


def discovery_prefixes(request: Mapping[str, Any]) -> list[str]:
    return _discovery_prefixes(validate_request(request), CONFIG)


def discover_objects(request: Mapping[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
    return _discover_objects(validate_request(request), CONFIG, **kwargs)


def select_objects(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _select_objects(validate_request(request), objects, CONFIG)


def inventory_request(
    request: Mapping[str, Any] | str | Path,
    *,
    output: str | Path | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _inventory_request(request, CONFIG, output=output, objects=objects, **kwargs)


def plan_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _plan_request(request, run_dir, CONFIG, output=output, objects=objects, **kwargs)


def fetch_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    objects: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Disabled: transfer requires an explicit reviewed plan."""
    return _fetch_request(request, run_dir, CONFIG, objects=objects, **kwargs)


def fetch_plan(
    plan: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return _fetch_from_plan(plan, run_dir, CONFIG, **kwargs)


def inspect_request(
    inputs: Sequence[str | Path] | None = None,
    *,
    run_dir: str | Path | None = None,
    product: str | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    paths = [Path(path) for path in inputs] if inputs else manifest_paths(run_dir or ".")
    return inspect_paths(paths, product=product, output=output)


def extract_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path | None = None,
    *,
    inputs: Sequence[str | Path] | None = None,
    manifest: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    if normalized["product"] != "fields":
        raise ValueError(f"{normalized['product']} is passthrough-only and cannot be extracted")
    if inputs:
        paths = [Path(path) for path in inputs]
    elif manifest:
        if run_dir is None:
            run_dir = Path(manifest).resolve().parent
        paths = manifest_paths(run_dir, normalized, manifest_path=manifest)
    elif run_dir:
        paths = manifest_paths(run_dir, normalized)
    else:
        raise ValueError("extract requires --input, --manifest, or --run-dir")
    if not paths:
        raise ValueError("no successful local field inputs were found")
    if output is None:
        if run_dir is None:
            raise ValueError("--output is required when --run-dir is not supplied")
        output = Path(run_dir) / "dbofs_fields.nc"
    return extract_fields(
        normalized,
        paths,
        output,
        CONFIG,
        manifest_output=(Path(run_dir) / "extraction_manifest.json") if run_dir else None,
        transfer_provenance=(
            verify_transfers(run_dir, normalized, manifest_path=manifest)
            if run_dir else None
        ),
    )


def evaluate_health(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    return _evaluate_health(request, run_dir, CONFIG, output=output, plots_dir=plots_dir)


# ---------------------------------------------------------------------------
# Deprecated dbofs-boundary callables
# ---------------------------------------------------------------------------

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

FVCOM_ITS_ZSL = None if np is None else np.array([
    0., -5., -10., -20., -30., -40., -50., -100., -150., -200.,
    -250., -300., -500., -700., -900., -1100., -1300., -1500.,
    -1700., -1900., -2100., -2300., -2500., -2700., -2900., -3100.,
    -5600., -6600., -7600., -8600.,
], dtype=np.float32)
ALL_CYCLES = ("t00z", "t06z", "t12z", "t18z")


def _deprecated(name: str) -> None:
    warnings.warn(
        f"{name} is a deprecated dbofs-boundary compatibility helper; use the "
        "dbofs-fetcher request/CLI workflow",
        DeprecationWarning,
        stacklevel=2,
    )


def _cycle_parts(date_str: str, cycle: str) -> tuple[datetime, str]:
    if cycle not in ALL_CYCLES:
        raise ValueError(f"cycle must be one of {ALL_CYCLES}")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("date_str must be YYYY-MM-DD") from exc
    hour = int(cycle[1:3])
    return date.replace(hour=hour), f"{hour:02d}"


def _dbofs_url(date_str: str, cycle: str = "t00z", fhour: int = 0) -> str:
    """Return the current public-AWS URL used by the legacy snapshot wrapper."""
    run, hour = _cycle_parts(date_str, cycle)
    if isinstance(fhour, bool) or not isinstance(fhour, int) or fhour < 0:
        raise ValueError("fhour must be a non-negative integer")
    code = "n006" if fhour == 0 else f"f{fhour:03d}"
    key = f"dbofs/netcdf/{run:%Y/%m/%d}/dbofs.t{hour}z.{run:%Y%m%d}.fields.{code}.nc"
    return f"{S3_ENDPOINT}/{key}"


def _legacy_object(date_str: str, cycle: str, fhour: int) -> dict[str, Any]:
    run, _ = _cycle_parts(date_str, cycle)
    guidance = "nowcast" if fhour == 0 else "forecast"
    target_lead = 6 if fhour == 0 else fhour
    prefixes = [f"dbofs/netcdf/{run:%Y/%m/%d}/", f"dbofs/netcdf/{run:%Y%m}/"]
    candidates: list[dict[str, Any]] = []
    for prefix in prefixes:
        for raw in list_s3_objects(prefix):
            parsed = parse_object_key(raw["key"])
            if not parsed:
                continue
            if (
                parsed["product"] == "fields"
                and parsed["guidance"] == guidance
                and parsed["run_time"] == iso_utc(run)
                and parsed["lead"] == target_lead
            ):
                candidates.append({**raw, **parsed})
    if not candidates:
        raise RuntimeError(f"DBOFS object is unavailable for {date_str} {cycle} fhour={fhour}")
    def rank(item: Mapping[str, Any]) -> tuple[int, int]:
        return (
            1 if item["naming"] == "current" else 0,
            1 if item["layout"] == "daily" else 0,
        )

    best_rank = max(rank(item) for item in candidates)
    winners = [item for item in candidates if rank(item) == best_rank]
    if len(winners) > 1:
        reference = winners[0]
        if any(
            item.get("size") != reference.get("size")
            or str(item.get("etag", "")) != str(reference.get("etag", ""))
            for item in winners[1:]
        ):
            keys = ", ".join(str(item["key"]) for item in winners)
            raise RuntimeError(f"equal-rank conflicting legacy wrapper objects: {keys}")
    return sorted(winners, key=lambda item: item["key"])[0]


def fetch_dbofs_field(
    date_str: str,
    cycle: str = "t00z",
    fhour: int = 0,
    work_dir: Optional[str | Path] = None,
) -> Path:
    """Fetch one bounded snapshot; `fhour=0` means cycle-time `n006`."""
    _deprecated("fetch_dbofs_field")
    item = _legacy_object(date_str, cycle, fhour)
    directory = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir()) / "dbofs_cache"
    plan = plan_request(
        {
            "schema_version": "dbofs_request_v1",
            "start_utc": item["expected_start_utc"],
            "end_utc_exclusive": item["expected_end_utc_exclusive"],
            "product": "fields",
            "guidance": item["guidance"],
            **({"run_cycle_utc": item["run_time"]} if item["guidance"] == "forecast" else {}),
            "variables": ["salt"],
            "vertical_views": ["surface"],
            "missing_policy": "error",
            "cache_policy": "keep",
            "max_workers": 1,
        },
        directory,
        objects=[item],
    )
    result = fetch_plan(plan, directory)["outcomes"][0]
    if result["status"] == "failed":
        raise RuntimeError("DBOFS transfer failed: " + "; ".join(result.get("errors", [])))
    return Path(result["local_path"])


def roms_depths_2d(s_rho, Cs_r, hc: float, h_2d, vtransform: int = 2):
    """Legacy zeta-zero rho-depth calculation; production extraction reads metadata."""
    _deprecated("roms_depths_2d")
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required")
    return roms_depths(s_rho, Cs_r, hc, np.asarray(h_2d, dtype=float), np.zeros_like(h_2d, dtype=float), vtransform)


def extract_dbofs_field(nc_path: str | Path, variables: tuple[str, ...] = ("salt",)) -> dict[str, Any]:
    """Return the historical snapshot dictionary with metadata-aware live depths."""
    _deprecated("extract_dbofs_field")
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required")
    import netCDF4

    result: dict[str, Any] = {}
    with netCDF4.Dataset(nc_path) as ds:
        geometry_snapshot(ds)
        for name in ("lon_rho", "lat_rho", "h", "s_rho", "Cs_r"):
            result[name] = _variable_data(ds.variables[name])
        result["mask_rho"] = (_variable_data(ds.variables["mask_rho"]) == 1).astype(np.int32)
        result["hc"] = float(_dataset_scalar(ds, "hc"))
        result["vtransform"] = int(_dataset_scalar(ds, "Vtransform"))
        zeta = _variable_data(ds.variables["zeta"], 0) if "zeta" in ds.variables else np.zeros_like(result["h"])
        result["zeta"] = zeta
        result["z"] = roms_depths(result["s_rho"], result["Cs_r"], result["hc"], result["h"], zeta, result["vtransform"])
        for name in variables:
            if name in ds.variables:
                result[name] = _variable_data(ds.variables[name], 0)
        result["ocean_time"] = np.datetime64(decode_ocean_times(ds)[0].replace(tzinfo=None), "s")
    return result


def interp_dbofs_to_fvcom(
    dbofs_data: dict[str, Any],
    fvcom_lon,
    fvcom_lat,
    fvcom_h,
    zsl=FVCOM_ITS_ZSL,
    variables: tuple[str, ...] = ("salt",),
    n_nearest: int = 1,
) -> dict[str, Any]:
    """Retain the historical nearest-wet-point and vertical interpolation helper."""
    _deprecated("interp_dbofs_to_fvcom")
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required")
    from scipy.interpolate import interp1d
    from scipy.spatial import cKDTree

    water = np.asarray(dbofs_data["mask_rho"]) == 1
    lon_rho, lat_rho = np.asarray(dbofs_data["lon_rho"]), np.asarray(dbofs_data["lat_rho"])
    eta_water, xi_water = np.where(water)
    tree = cKDTree(np.column_stack([lat_rho[water], lon_rho[water]]))
    _, indices = tree.query(np.column_stack([fvcom_lat, fvcom_lon]), k=max(1, n_nearest))
    if np.asarray(indices).ndim == 1:
        indices = np.asarray(indices)[:, None]
    result: dict[str, Any] = {"ocean_time": dbofs_data["ocean_time"]}
    for name in variables:
        if name not in dbofs_data:
            continue
        field = np.asarray(dbofs_data[name], dtype=float)
        depths = np.asarray(dbofs_data["z"], dtype=float)
        out = np.full((len(zsl), len(fvcom_lon)), np.nan, dtype=float)
        for node, nearest in enumerate(indices[:, 0]):
            eta, xi = eta_water[int(nearest)], xi_water[int(nearest)]
            profile, profile_z = field[:, eta, xi], depths[:, eta, xi]
            valid = np.isfinite(profile) & np.isfinite(profile_z)
            if valid.sum() == 1:
                out[:, node] = profile[valid][0]
            elif valid.sum() >= 2:
                order = np.argsort(profile_z[valid])
                zp, fp = profile_z[valid][order], profile[valid][order]
                out[:, node] = interp1d(zp, fp, bounds_error=False, fill_value=(fp[0], fp[-1]))(zsl)
        result[name] = out.astype(np.float32)
    return result


def fetch_and_interp_dbofs(
    date_str: str,
    fvcom_lon,
    fvcom_lat,
    fvcom_h,
    zsl=FVCOM_ITS_ZSL,
    variables: tuple[str, ...] = ("salt",),
    cycle: str = "t00z",
    fhour: int = 0,
    work_dir: Optional[str | Path] = None,
    keep_file: bool = False,
) -> dict[str, Any]:
    _deprecated("fetch_and_interp_dbofs")
    path = fetch_dbofs_field(date_str, cycle=cycle, fhour=fhour, work_dir=work_dir)
    try:
        source = extract_dbofs_field(path, variables=variables)
        return interp_dbofs_to_fvcom(source, fvcom_lon, fvcom_lat, fvcom_h, zsl=zsl, variables=variables)
    finally:
        if not keep_file:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".download.json").unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="List matching public S3 objects")
    inventory.add_argument("--request", required=True); inventory.add_argument("--output")
    plan = sub.add_parser("plan", help="Select objects and write exact storage estimate")
    plan.add_argument("--request", required=True); plan.add_argument("--run-dir", required=True); plan.add_argument("--output")
    fetch = sub.add_parser("fetch", help="Transfer a previously reviewed local plan")
    fetch.add_argument("--plan", required=True)
    fetch.add_argument("--run-dir", required=True)
    inspect = sub.add_parser("inspect", help="Inspect one or more downloaded NetCDF files")
    inspect.add_argument("--input", action="append", required=True); inspect.add_argument("--output"); inspect.add_argument("--product", choices=("fields", "stations", "regulargrid"))
    extract = sub.add_parser("extract", help="Crop, concatenate, and derive compact ROMS fields")
    extract.add_argument("--request", required=True)
    source = extract.add_mutually_exclusive_group(required=True); source.add_argument("--input", action="append"); source.add_argument("--manifest"); source.add_argument("--run-dir")
    extract.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory_request(args.request, output=args.output)
        elif args.command == "plan":
            result = plan_request(args.request, args.run_dir, output=args.output)
        elif args.command == "fetch":
            result = fetch_plan(args.plan, args.run_dir)
        elif args.command == "inspect":
            result = inspect_request(args.input, product=args.product, output=args.output)
        elif args.command == "extract":
            result = extract_request(args.request, args.run_dir, inputs=args.input, manifest=args.manifest, output=args.output)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), file=__import__("sys").stderr)
        return 2
    print(json.dumps({"status": "ok", "command": args.command, "summary": json_clean(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
