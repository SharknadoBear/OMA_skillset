"""FVCOM output discovery, time decoding, and selected-range concatenation.

The functions in this module are intentionally path-driven and runtime-neutral:
they can run on a laptop for small products or on Kestrel next to full FVCOM
output stacks.  Heavy work is exposed through ordinary Python functions so the
same API can be wrapped by notebooks, command-line scripts, or Slurm jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Iterable, Sequence
import warnings

import numpy as np
from netCDF4 import Dataset, num2date


MJD_EPOCH = np.datetime64("1858-11-17T00:00:00", "ms")
TIME_DIM_CANDIDATES = ("time", "Time")
DEFAULT_CASES = tuple(f"P{i}" for i in range(8))
SPINUP_START = "2018-10-01T00:00:00"
SPINUP_END = "2019-01-01T00:00:00"
FORMAL_START = "2019-01-01T00:00:00"
FORMAL_END = "2021-01-01T00:00:00"


@dataclass(frozen=True)
class OutputStackInfo:
    """Compact inventory record for one FVCOM output file."""

    path: Path
    bytes: int
    n_time: int
    start: str | None
    end: str | None
    variables: tuple[str, ...]


def workspace_dir() -> Path:
    """Return the local ``Workspace`` directory inferred from this module."""

    return Path(__file__).resolve().parents[2]


def _strip_value(value: str) -> str:
    value = value.strip().rstrip(",")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_run_nml(path: str | Path) -> dict[str, str]:
    """Parse simple ``KEY = VALUE`` entries from an FVCOM run namelist.

    Parameters
    ----------
    path : str or Path
        FVCOM namelist path, for example ``Workspace/RUN_P0/waterPACT_P0_run.nml``.

    Returns
    -------
    dict
        Upper-case namelist keys mapped to stripped string values.
    """

    path = Path(path)
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        if key:
            values[key] = _strip_value(value)
    return values


def candidate_output_dirs(
    case: str,
    workspace: str | Path | None = None,
    override: str | Path | None = None,
) -> list[Path]:
    """Return possible FVCOM output directories for a case in priority order.

    The current Delaware workspace has used both ``OUTPUT/Pn`` and
    ``OUTPUT_Pn`` layouts.  The run namelist output directory is preferred when
    it is available because it records what FVCOM was asked to use.
    """

    ws = Path(workspace) if workspace is not None else workspace_dir()
    case = case.upper()
    candidates: list[Path] = []

    if override is not None:
        candidates.append(Path(override).expanduser())

    run_dir = ws / f"RUN_{case}"
    nml = run_dir / f"waterPACT_{case}_run.nml"
    cfg = parse_run_nml(nml)
    out_dir = cfg.get("OUTPUT_DIR")
    if out_dir:
        p = Path(out_dir)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        candidates.append(p)

    candidates.extend([(ws / f"OUTPUT_{case}").resolve(), (ws / "OUTPUT" / case).resolve()])

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique


def find_case_output_dir(
    case: str,
    workspace: str | Path | None = None,
    override: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    """Return the preferred output directory for ``case``.

    Existing directories are preferred.  If none exists and ``require_exists``
    is false, the first candidate is returned so callers can create outputs in a
    predictable place.
    """

    candidates = candidate_output_dirs(case, workspace, override)
    existing = [path for path in candidates if path.exists()]
    if len(existing) > 1:
        warnings.warn(
            f"Multiple output directories exist for {case}: "
            + ", ".join(str(p) for p in existing)
            + f"; using {existing[0]}",
            RuntimeWarning,
        )
    if existing:
        return existing[0]
    if require_exists:
        raise FileNotFoundError(f"No output directory found for {case}: {candidates}")
    return candidates[0]


def discover_output_stacks(
    case: str,
    workspace: str | Path | None = None,
    pattern: str = "*.nc",
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Find FVCOM NetCDF output stacks for a case.

    Empty or missing output directories return an empty list rather than an
    exception, which lets inventory scripts run before production files arrive.
    """

    out_dir = find_case_output_dir(case, workspace, override=output_dir)
    if not out_dir.exists():
        return []
    return sorted(path for path in out_dir.glob(pattern) if path.is_file())


def mjd_to_datetime64(mjd: np.ndarray | float) -> np.ndarray:
    """Convert Modified Julian Day values to ``datetime64[ms]``."""

    arr = np.asarray(mjd, dtype=float)
    delta = np.rint(arr * 86_400_000.0).astype("timedelta64[ms]")
    return MJD_EPOCH + delta


def datetime64_to_mjd(times: np.ndarray | str | datetime) -> np.ndarray:
    """Convert datetime-like values to Modified Julian Day."""

    arr = np.asarray(times, dtype="datetime64[ms]")
    delta_ms = (arr - MJD_EPOCH).astype("timedelta64[ms]").astype(float)
    return delta_ms / 86_400_000.0


def parse_time_like(value: str | datetime | np.datetime64 | None) -> np.datetime64 | None:
    """Parse ISO datetimes or FVCOM ``days=<mjd>`` strings."""

    if value is None:
        return None
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ms]")
    if isinstance(value, datetime):
        return np.datetime64(value, "ms")
    text = str(value).strip().strip("'").strip('"')
    match = re.match(r"^days\s*=\s*([-+0-9.eE]+)$", text)
    if match:
        return mjd_to_datetime64(float(match.group(1))).astype("datetime64[ms]")
    return np.datetime64(text, "ms")


def _decode_times_char(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values)
    decoded: list[str] = []
    for row in rows:
        chars: list[str] = []
        for item in np.ravel(row):
            if isinstance(item, bytes):
                chars.append(item.decode("ascii", errors="ignore"))
            else:
                chars.append(str(item))
        decoded.append("".join(chars).strip().replace(" ", "T", 1))
    return np.asarray(decoded, dtype="datetime64[ms]")


def decode_fvcom_time(ds: Dataset | str | Path) -> np.ndarray:
    """Decode FVCOM time variables from an open dataset or NetCDF path.

    The decoder supports common FVCOM forms: ``Times`` character arrays,
    ``time`` in MJD days, ``time`` with CF-style units, and ``Itime``/``Itime2``.
    """

    close = False
    if isinstance(ds, (str, Path)):
        ds = Dataset(ds)
        close = True
    try:
        if "Times" in ds.variables:
            return _decode_times_char(ds.variables["Times"][:])

        if "time" in ds.variables:
            var = ds.variables["time"]
            values = np.asarray(var[:], dtype=float)
            units = getattr(var, "units", "")
            if units and "since" in units.lower():
                try:
                    dates = num2date(values, units=units, only_use_cftime_datetimes=False)
                    return np.asarray([np.datetime64(d, "ms") for d in dates])
                except Exception:
                    pass
            return mjd_to_datetime64(values)

        if "Itime" in ds.variables and "Itime2" in ds.variables:
            day = np.asarray(ds.variables["Itime"][:], dtype=float)
            msec = np.asarray(ds.variables["Itime2"][:], dtype=float)
            return mjd_to_datetime64(day + msec / 86_400_000.0)

        raise KeyError("No recognizable FVCOM time variables found.")
    finally:
        if close:
            ds.close()


def _find_time_dim(ds: Dataset) -> str:
    for name in TIME_DIM_CANDIDATES:
        if name in ds.dimensions:
            return name
    for name in ("time", "Times", "Itime"):
        if name in ds.variables and ds.variables[name].dimensions:
            return ds.variables[name].dimensions[0]
    raise KeyError("No FVCOM time dimension found.")


def _subset_indices(times: np.ndarray, start: object = None, end: object = None) -> np.ndarray:
    start64 = parse_time_like(start)
    end64 = parse_time_like(end)
    mask = np.ones(times.shape, dtype=bool)
    if start64 is not None:
        mask &= times >= start64
    if end64 is not None:
        mask &= times < end64
    return np.flatnonzero(mask)


def stack_file_info(path: str | Path) -> OutputStackInfo:
    """Summarize one NetCDF output stack without loading large variables."""

    path = Path(path)
    with Dataset(path) as ds:
        variables = tuple(ds.variables.keys())
        try:
            times = decode_fvcom_time(ds)
            n_time = len(times)
            start = str(times[0]) if n_time else None
            end = str(times[-1]) if n_time else None
        except Exception:
            time_dim = next((d for d in TIME_DIM_CANDIDATES if d in ds.dimensions), None)
            n_time = len(ds.dimensions[time_dim]) if time_dim else 0
            start = None
            end = None
    return OutputStackInfo(path, path.stat().st_size, n_time, start, end, variables)


def inventory_cases(
    cases: Sequence[str] = DEFAULT_CASES,
    workspace: str | Path | None = None,
    pattern: str = "*.nc",
) -> list[dict[str, object]]:
    """Return a compact inventory for multiple production cases."""

    records: list[dict[str, object]] = []
    for case in cases:
        case = case.upper()
        out_dir = find_case_output_dir(case, workspace)
        files = discover_output_stacks(case, workspace, pattern=pattern)
        n_time = 0
        start = None
        end = None
        variables: set[str] = set()
        total_bytes = 0
        for path in files:
            total_bytes += path.stat().st_size
            try:
                info = stack_file_info(path)
                n_time += info.n_time
                variables.update(info.variables)
                start = min(filter(None, [start, info.start])) if start else info.start
                end = max(filter(None, [end, info.end])) if end else info.end
            except Exception as exc:
                warnings.warn(f"Could not inventory {path}: {exc}", RuntimeWarning)
        records.append(
            {
                "case": case,
                "output_dir": str(out_dir),
                "n_files": len(files),
                "total_bytes": total_bytes,
                "n_time_records": n_time,
                "start": start,
                "end": end,
                "variables": ",".join(sorted(variables)),
            }
        )
    return records


def _normalize_variables(variables: Iterable[str] | None) -> list[str] | None:
    if variables is None:
        return None
    out = []
    for item in variables:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out or None


def _copy_variable_attrs(src, dst) -> None:
    for attr in src.ncattrs():
        if attr == "_FillValue":
            continue
        dst.setncattr(attr, src.getncattr(attr))


def _variable_fill_value(var):
    return getattr(var, "_FillValue", None)


def _copy_selected_to_netcdf(
    files: Sequence[Path],
    selections: list[tuple[Path, np.ndarray]],
    variables: list[str] | None,
    output_path: Path,
) -> Path:
    first = Dataset(files[0])
    try:
        time_dim = _find_time_dim(first)
        requested = set(variables or first.variables.keys())
        always = {"time", "Times", "Itime", "Itime2", "lon", "lat", "x", "y", "nv", "h", "siglay", "siglev", "lonc", "latc", "xc", "yc"}

        selected_names: list[str] = []
        for name, var in first.variables.items():
            dims = var.dimensions
            if name in always or variables is None or name in requested or time_dim not in dims:
                selected_names.append(name)

        missing = sorted(name for name in requested if name not in first.variables)
        if variables is not None and missing:
            warnings.warn(f"Requested variables missing from first stack and skipped: {missing}", RuntimeWarning)

        total_time = int(sum(len(idx) for _, idx in selections))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(output_path, "w", format=getattr(first, "data_model", "NETCDF4")) as out:
            for attr in first.ncattrs():
                out.setncattr(attr, first.getncattr(attr))
            out.setncattr("postprocessing_source_files", json.dumps([str(p) for p, _ in selections]))
            out.setncattr("postprocessing_toolkit", "Workspace/Postprocessing/toolkit/fvcom_output.py")

            for name, dim in first.dimensions.items():
                out.createDimension(name, total_time if name == time_dim else len(dim))

            for name in selected_names:
                src = first.variables[name]
                fill_value = _variable_fill_value(src)
                kwargs = {"fill_value": fill_value} if fill_value is not None else {}
                dst = out.createVariable(name, src.datatype, src.dimensions, **kwargs)
                _copy_variable_attrs(src, dst)

                if time_dim not in src.dimensions:
                    dst[:] = src[:]
                    continue

                offset = 0
                axis = src.dimensions.index(time_dim)
                for path, indices in selections:
                    with Dataset(path) as ds:
                        data = ds.variables[name][:]
                        part = np.take(data, indices, axis=axis)
                    slc = [slice(None)] * part.ndim
                    slc[axis] = slice(offset, offset + len(indices))
                    dst[tuple(slc)] = part
                    offset += len(indices)
    finally:
        first.close()
    return output_path


def concat_output_range(
    files: Sequence[str | Path],
    start: object = None,
    end: object = None,
    variables: Iterable[str] | None = None,
    output_path: str | Path | None = None,
) -> Path | dict[str, np.ndarray]:
    """Concatenate selected FVCOM output records across stack files.

    Parameters
    ----------
    files : sequence of str or Path
        FVCOM NetCDF output stack files in chronological order.
    start, end : datetime-like or ``days=<mjd>`` strings, optional
        Inclusive start and exclusive end of the selected range.
    variables : iterable of str, optional
        Variables to prioritize in the output.  Non-time mesh coordinates are
        also copied so plotting remains possible.
    output_path : str or Path, optional
        If given, write a compact NetCDF file and return the path.  If omitted,
        return a dictionary containing only decoded times and selected records.

    Returns
    -------
    Path or dict
        Output NetCDF path, or an in-memory dictionary for small diagnostics.
    """

    paths = [Path(path) for path in files]
    if not paths:
        raise FileNotFoundError("No FVCOM output files were provided.")

    selections: list[tuple[Path, np.ndarray]] = []
    all_times: list[np.ndarray] = []
    for path in paths:
        with Dataset(path) as ds:
            times = decode_fvcom_time(ds)
            idx = _subset_indices(times, start, end)
            if len(idx):
                selections.append((path, idx))
                all_times.append(times[idx])

    if not selections:
        raise ValueError(f"No records found in requested range: start={start}, end={end}")

    if output_path is not None:
        return _copy_selected_to_netcdf(paths, selections, _normalize_variables(variables), Path(output_path))

    selected_times = np.concatenate(all_times)
    return {"time": selected_times, "selections": np.asarray([(str(p), len(idx)) for p, idx in selections], dtype=object)}


def write_case_subset(
    case: str,
    start: object = SPINUP_START,
    end: object = SPINUP_END,
    variables: Iterable[str] | None = None,
    out_dir: str | Path | None = None,
    workspace: str | Path | None = None,
    pattern: str = "*.nc",
) -> Path:
    """Discover a case's output stacks and write a selected compact subset."""

    ws = Path(workspace) if workspace is not None else workspace_dir()
    case = case.upper()
    files = discover_output_stacks(case, ws, pattern=pattern)
    if not files:
        raise FileNotFoundError(f"No FVCOM output stacks found for {case}.")

    target_dir = Path(out_dir) if out_dir is not None else ws / "Postprocessing" / "data_processed" / "concatenated" / case
    target_dir.mkdir(parents=True, exist_ok=True)
    start_label = str(parse_time_like(start)).replace(":", "").replace("-", "").replace("T", "_")
    end_label = str(parse_time_like(end)).replace(":", "").replace("-", "").replace("T", "_")
    out_path = target_dir / f"{case}_{start_label}_to_{end_label}.nc"
    return Path(concat_output_range(files, start=start, end=end, variables=variables, output_path=out_path))
