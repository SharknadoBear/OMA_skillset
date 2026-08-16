#!/usr/bin/env python3
"""Fetch request-bounded CFSv2 atmospheric fields from HYCOM OPeNDAP.

The public ``fetch_cfsv2_window`` API is the preferred entry point.  Legacy
annual helpers remain available for notebooks that already use them.  The
connector preserves native CFSv2 fields; downstream model writers own model-
specific naming, interpolation, and packaging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

try:
    from tqdm.std import tqdm as _Tqdm
except ImportError:  # pragma: no cover
    _Tqdm = None  # type: ignore[assignment]


log = logging.getLogger(__name__)
_HYCOM_BASE = "https://tds.hycom.org/thredds/dodsC/datasets/force/ncep_cfsv2/netcdf/"
_URL_PREFIXES = ["cfsv2-sec2", "cfsv2-sec", "cfsv2-sea"]
_MT_EPOCH = np.datetime64("1900-12-31T00:00:00", "ms")
DEFAULT_LON_RANGE: tuple[float, float] = (283.0, 288.0)
DEFAULT_LAT_RANGE: tuple[float, float] = (36.0, 41.0)

SUBDATASET_VARIABLES: dict[str, list[str]] = {
    "uv-10m": ["wndewd", "wndnwd"],
    "sfcprs": ["airprs"],
    "dlwsfc": ["dlwflx"],
    "dswsfc": ["dswflx"],
    "strblk": ["tauewd", "taunwd"],
    "TaqaQrQp": ["airtmp", "vapmix", "radflx", "shwflx"],
    "precip": ["precip"],
    "surtmp": ["surtmp"],
}
SUBDATASET_ALIASES = {"dlwflx": "dlwsfc"}
CFSV2_PRESSURE_BASE_HPA = 1000.0


def normalize_subdataset(name: str) -> str:
    """Return a canonical HYCOM CFSv2 subdataset name."""
    value = SUBDATASET_ALIASES.get(str(name), str(name))
    if value not in SUBDATASET_VARIABLES:
        choices = ", ".join(sorted(set(SUBDATASET_VARIABLES) | set(SUBDATASET_ALIASES)))
        raise ValueError(f"Unknown CFSv2 subdataset {name!r}; choose one of: {choices}")
    return value


def _mt_to_datetime64(mt_values: np.ndarray) -> np.ndarray:
    """Convert MT days since 1900-12-31 to ``datetime64[ms]``."""
    milliseconds = np.rint(np.asarray(mt_values, dtype=np.float64) * 86_400_000.0).astype(np.int64)
    return _MT_EPOCH + milliseconds.astype("timedelta64[ms]")


def _parse_utc(value: str | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("Timestamp cannot be NaT")
        return value.astype("datetime64[ms]")
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} has no timezone; use an explicit UTC offset or Z")
    parsed = parsed.astimezone(timezone.utc)
    return np.datetime64(parsed.replace(tzinfo=None), "ms")


def cfsv2_airprs_to_absolute_pa(
    values: Any,
    *,
    source_units: str = "hPa",
    base_hpa: float = CFSV2_PRESSURE_BASE_HPA,
) -> np.ndarray:
    """Convert HYCOM CFSv2 ``airprs`` departures to absolute pressure in Pa.

    HYCOM atmospheric forcing stores this field relative to a 1000 hPa base.
    Conversion is explicit because treating the source values as absolute
    pressure would produce scientifically invalid surface forcing.
    """
    units = str(source_units).strip().lower().replace(" ", "")
    data = np.asanyarray(values, dtype=np.float64)
    if units in {"hpa", "hectopascal", "hectopascals", "mb", "mbar"}:
        departure_hpa = data
    elif units in {"pa", "pascal", "pascals"}:
        departure_hpa = data / 100.0
    else:
        raise ValueError(f"Unsupported CFSv2 airprs units {source_units!r}; expected hPa or Pa")
    result = (departure_hpa + float(base_hpa)) * 100.0
    if not np.all(np.isfinite(result)):
        raise ValueError("Converted CFSv2 pressure contains non-finite values")
    return result


def _try_cfsv2_url(year: int, subdataset_name: str) -> tuple[str, xr.Dataset]:
    name = normalize_subdataset(subdataset_name)
    errors: list[str] = []
    for prefix in _URL_PREFIXES:
        base_url = f"{_HYCOM_BASE}{prefix}_{year}_01hr_{name}.nc"
        try:
            dataset = xr.open_dataset(
                f"{base_url}?MT,Latitude,Longitude",
                engine="pydap",
                decode_times=False,
            )
            _ = dataset["Latitude"].values
            log.info("CFSv2 connected: %s", base_url)
            return base_url, dataset
        except Exception as exc:  # pragma: no cover - source failures vary
            errors.append(f"  {prefix}: {exc}")
    raise RuntimeError(
        f"Could not connect to CFSv2 {year}/{name} via any URL prefix.\n" + "\n".join(errors)
    )


def _download_chunk(
    base_url: str,
    variables: list[str],
    t0_idx: int,
    t1_idx: int,
    lat0: int,
    lat1: int,
    lon0: int,
    lon1: int,
    max_retries: int = 5,
) -> xr.Dataset:
    parts = [
        f"{name}[{t0_idx}:1:{t1_idx}][{lat0}:1:{lat1}][{lon0}:1:{lon1}]"
        for name in variables
    ]
    parts.extend(
        (
            f"MT[{t0_idx}:1:{t1_idx}]",
            f"Latitude[{lat0}:1:{lat1}]",
            f"Longitude[{lon0}:1:{lon1}]",
        )
    )
    url = f"{base_url}?{','.join(parts)}"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            dataset = xr.open_dataset(url, engine="pydap", decode_times=False)
            dataset.load()
            return dataset
        except Exception as exc:  # pragma: no cover - source failures vary
            last_error = exc
            wait = 2**attempt
            log.warning(
                "Chunk t[%d:%d] attempt %d/%d failed: %s; retry in %ds",
                t0_idx,
                t1_idx,
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            _time.sleep(wait)
    raise RuntimeError(f"Chunk t[{t0_idx}:{t1_idx}] failed after {max_retries} retries: {last_error}")


def _sanitize_nc_attrs(dataset: xr.Dataset) -> xr.Dataset:
    valid = (str, bytes, int, float, np.integer, np.floating, np.ndarray, list, tuple)
    dataset.attrs = {key: value for key, value in dataset.attrs.items() if isinstance(value, valid)}
    for variable in dataset.variables.values():
        variable.attrs = {
            key: value for key, value in variable.attrs.items() if isinstance(value, valid)
        }
        # Pydap exposes the source packing as xarray encoding. Reusing that
        # encoding can silently coerce decoded floats back to integers and is
        # unsafe when a bounded subset contains NaNs. Cache decoded products.
        variable.encoding = {}
    for name in dataset.data_vars:
        if np.issubdtype(dataset[name].dtype, np.floating):
            dataset[name] = dataset[name].astype(np.float32)
            dataset[name].encoding = {}
    return dataset


def _bounds(values: np.ndarray, requested: tuple[float, float], label: str) -> tuple[int, int]:
    if requested[0] > requested[1]:
        raise ValueError(f"{label} range must be increasing")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.diff(array) > 0):
        raise ValueError(f"CFSv2 {label} coordinate must be strictly increasing and one-dimensional")
    if requested[1] < array[0] or requested[0] > array[-1]:
        raise ValueError(f"Requested {label} range {requested} is outside source coverage")
    start = max(0, int(np.searchsorted(array, requested[0])) - 1)
    stop = min(len(array) - 1, int(np.searchsorted(array, requested[1], side="right")))
    return start, stop


def _download_year_window(
    year: int,
    subdataset_name: str,
    variables: list[str],
    start: np.datetime64,
    end: np.datetime64,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    chunk_hours: int,
    max_retries: int,
) -> tuple[xr.Dataset | None, str | None]:
    base_url, coords = _try_cfsv2_url(year, subdataset_name)
    try:
        lats = np.asarray(coords["Latitude"].values, dtype=np.float64)
        lons = np.asarray(coords["Longitude"].values, dtype=np.float64)
        times = _mt_to_datetime64(np.asarray(coords["MT"].values, dtype=np.float64))
    finally:
        coords.close()
    lat0, lat1 = _bounds(lats, lat_range, "latitude")
    lon0, lon1 = _bounds(lons, lon_range, "longitude")
    indices = np.where((times >= start) & (times <= end))[0]
    if not len(indices):
        return None, None
    if np.any(np.diff(indices) != 1):
        raise ValueError("Requested CFSv2 time indices are not contiguous")
    chunks: list[xr.Dataset] = []
    chunk_size = max(1, int(chunk_hours))
    for offset in range(0, len(indices), chunk_size):
        subset = indices[offset : offset + chunk_size]
        chunks.append(
            _download_chunk(
                base_url,
                variables,
                int(subset[0]),
                int(subset[-1]),
                lat0,
                lat1,
                lon0,
                lon1,
                max_retries,
            )
        )
    dataset = xr.concat(chunks, dim="MT") if len(chunks) > 1 else chunks[0]
    return dataset, base_url


def fetch_cfsv2_window(
    start: str | np.datetime64,
    end: str | np.datetime64,
    subdataset_name: str,
    variables: list[str] | None = None,
    *,
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    output: str | Path,
    chunk_hours: int = 168,
    max_retries: int = 5,
    overwrite: bool = False,
) -> Path:
    """Fetch an inclusive UTC time window and bounded native-grid region."""
    start_utc = _parse_utc(start)
    end_utc = _parse_utc(end)
    if end_utc < start_utc:
        raise ValueError("End time must not precede start time")
    name = normalize_subdataset(subdataset_name)
    selected = list(variables or SUBDATASET_VARIABLES[name])
    unknown = [value for value in selected if value not in SUBDATASET_VARIABLES[name]]
    if unknown:
        raise ValueError(f"Variables {unknown} are not available in CFSv2 subdataset {name}")
    destination = Path(output)
    if destination.exists() and not overwrite:
        return destination

    years = range(int(str(start_utc)[:4]), int(str(end_utc)[:4]) + 1)
    pieces: list[xr.Dataset] = []
    source_urls: list[str] = []
    for year in years:
        piece, url = _download_year_window(
            year,
            name,
            selected,
            start_utc,
            end_utc,
            lon_range,
            lat_range,
            chunk_hours,
            max_retries,
        )
        if piece is not None:
            pieces.append(piece)
            source_urls.append(str(url))
    if not pieces:
        raise ValueError(f"No CFSv2 records found from {start_utc} through {end_utc}")
    combined = xr.concat(pieces, dim="MT") if len(pieces) > 1 else pieces[0]
    combined = _sanitize_nc_attrs(combined)
    combined.attrs.update(
        {
            "connector": "cfsv2-atmosphere",
            "requested_start_utc": f"{start_utc}Z",
            "requested_end_utc": f"{end_utc}Z",
            "requested_bbox_0_360": json.dumps([*lon_range, *lat_range]),
            "source_urls": "\n".join(source_urls),
            "subdataset": name,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        combined.to_netcdf(temporary, engine="netcdf4", format="NETCDF4_CLASSIC")
        with xr.open_dataset(temporary, decode_times=False) as check:
            if int(check.sizes.get("MT", 0)) == 0:
                raise ValueError("Staged CFSv2 output has no time records")
            for variable in selected:
                if variable not in check:
                    raise ValueError(f"Staged CFSv2 output is missing {variable}")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        combined.close()
    return destination


def fetch_cfsv2_year(
    year: int,
    subdataset_name: str,
    variables: list[str],
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    cache_dir: str | Path = ".",
    chunk_days: int = 30,
    max_retries: int = 5,
    overwrite: bool = False,
) -> Path:
    """Backward-compatible full-calendar-year fetch helper."""
    canonical = normalize_subdataset(subdataset_name)
    output = Path(cache_dir) / f"{subdataset_name}_{year}.nc"
    return fetch_cfsv2_window(
        f"{year}-01-01T00:00:00Z",
        f"{year}-12-31T23:59:59Z",
        canonical,
        variables,
        lon_range=lon_range,
        lat_range=lat_range,
        output=output,
        chunk_hours=max(1, int(chunk_days)) * 24,
        max_retries=max_retries,
        overwrite=overwrite,
    )


def fetch_wind_year(year: int, lon_range=DEFAULT_LON_RANGE, lat_range=DEFAULT_LAT_RANGE, cache_dir=".", **kwargs) -> Path:
    return fetch_cfsv2_year(
        year, "uv-10m", ["wndewd", "wndnwd"], lon_range, lat_range, cache_dir, **kwargs
    )


def fetch_pressure_year(year: int, lon_range=DEFAULT_LON_RANGE, lat_range=DEFAULT_LAT_RANGE, cache_dir=".", **kwargs) -> Path:
    return fetch_cfsv2_year(year, "sfcprs", ["airprs"], lon_range, lat_range, cache_dir, **kwargs)


def load_and_concat_years(subdataset_name: str, years: list[int], cache_dir: str | Path) -> xr.Dataset:
    cache = Path(cache_dir)
    datasets: list[xr.Dataset] = []
    for year in years:
        path = cache / f"{subdataset_name}_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        datasets.append(xr.open_dataset(path, decode_times=False))
    return xr.concat(datasets, dim="MT")


def load_cfsv2_wind(*_args, **_kwargs):
    raise NotImplementedError("Use fetch_wind_year() or fetch_cfsv2_window()")


def load_cfsv2_pressure(*_args, **_kwargs):
    raise NotImplementedError("Use fetch_pressure_year() or fetch_cfsv2_window()")


def regrid_to_fvcom(*_args, **_kwargs):
    raise NotImplementedError("Keep connector output on the native grid; interpolate downstream")


def _write_report(path: Path, output: Path, args: argparse.Namespace) -> None:
    with xr.open_dataset(output, decode_times=False) as dataset:
        payload = {
            "schema_version": "cfsv2_window_fetch_v1",
            "output": str(output.resolve()),
            "bytes": output.stat().st_size,
            "subdataset": dataset.attrs.get("subdataset"),
            "variables": list(dataset.data_vars),
            "dimensions": dict(dataset.sizes),
            "start_utc": args.start,
            "end_utc": args.end,
            "bbox_0_360": [args.lon_min, args.lon_max, args.lat_min, args.lat_max],
            "source_urls": dataset.attrs.get("source_urls", "").splitlines(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    window = subparsers.add_parser("window", help="Fetch a bounded inclusive UTC time window")
    window.add_argument("--start", required=True, help="UTC start, including Z or an offset")
    window.add_argument("--end", required=True, help="UTC end, including Z or an offset")
    window.add_argument("--subdataset", required=True, choices=sorted(set(SUBDATASET_VARIABLES) | set(SUBDATASET_ALIASES)))
    window.add_argument("--variables", nargs="+", help="Subset variables; defaults to every field in the subdataset")
    window.add_argument("--lon-min", type=float, required=True, help="Western longitude in 0-360 degrees east")
    window.add_argument("--lon-max", type=float, required=True, help="Eastern longitude in 0-360 degrees east")
    window.add_argument("--lat-min", type=float, required=True)
    window.add_argument("--lat-max", type=float, required=True)
    window.add_argument("--output", required=True)
    window.add_argument("--report", help="Optional JSON fetch report")
    window.add_argument("--chunk-hours", type=int, default=168)
    window.add_argument("--max-retries", type=int, default=5)
    window.add_argument("--overwrite", action="store_true")

    pressure = subparsers.add_parser("pressure-to-pa", help="Document the CFSv2 pressure conversion")
    pressure.add_argument("--value", type=float, required=True, help="CFSv2 airprs departure")
    pressure.add_argument("--units", default="hPa", choices=("hPa", "Pa"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pressure-to-pa":
        value = float(cfsv2_airprs_to_absolute_pa([args.value], source_units=args.units)[0])
        print(json.dumps({"airprs_departure": args.value, "source_units": args.units, "absolute_pressure_pa": value}, indent=2))
        return 0
    output = fetch_cfsv2_window(
        args.start,
        args.end,
        args.subdataset,
        args.variables,
        lon_range=(args.lon_min, args.lon_max),
        lat_range=(args.lat_min, args.lat_max),
        output=args.output,
        chunk_hours=args.chunk_hours,
        max_retries=args.max_retries,
        overwrite=args.overwrite,
    )
    if args.report:
        _write_report(Path(args.report), output, args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
