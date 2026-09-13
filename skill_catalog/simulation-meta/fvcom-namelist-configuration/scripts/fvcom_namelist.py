#!/usr/bin/env python3
"""Render and audit FVCOM 4.3.1 tide-only namelists and station files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


STAGES = {"smoke", "canary", "spinup", "benchmark", "production", "full"}
REQUIRED_FILES = {
    "NML_GRID_COORDINATES": ["GRID_FILE", "SIGMA_LEVELS_FILE", "DEPTH_FILE", "CORIOLIS_FILE", "SPONGE_FILE"],
    "NML_OPEN_BOUNDARY_CONTROL": ["OBC_NODE_LIST_FILE", "OBC_ELEVATION_FILE"],
    "NML_STATION_TIMESERIES": ["STATION_FILE"],
}
FALSE_KEYS = {
    "TEMPERATURE_ACTIVE",
    "SALINITY_ACTIVE",
    "SURFACE_WAVE_MIXING",
    "WIND_ON",
    "HEATING_ON",
    "PRECIPITATION_ON",
    "AIRPRESSURE_ON",
    "WAVE_ON",
    "OBC_TEMP_NUDGING",
    "OBC_SALT_NUDGING",
    "OBC_MEANFLOW",
    "OBC_LONGSHORE_FLOW_ON",
    "OBC_DEPTH_CONTROL_ON",
    "GROUNDWATER_ON",
    "GROUNDWATER_TEMP_ON",
    "GROUNDWATER_SALT_ON",
    "DATA_ASSIMILATION",
    "BIOLOGICAL_MODEL",
    "SEDIMENT_MODEL",
    "ICING_MODEL",
    "ICE_MODEL",
    "HIGH_LATITUDE_WAVE",
    "FLUME_BOTTOM",
    "PROBES_ON",
    "BOUNDSCHK_ON",
    "NCNEST_ON",
    "NESTING_ON",
    "OUT_VELOCITY_3D",
    "OUT_WIND_VELOCITY",
    "OUT_SALT_TEMP",
    "NC_VELOCITY",
    "NC_SALT_TEMP",
    "NC_TURBULENCE",
    "NC_VERTICAL_VEL",
    "NC_WIND_VEL",
    "NC_WIND_STRESS",
    "NC_EVAP_PRECIP",
    "NC_SURFACE_HEAT",
    "NC_GROUNDWATER",
    "NC_BIO",
    "NC_WQM",
}
TRUE_KEYS = {
    "BAROTROPIC",
    "WETTING_DRYING_ON",
    "OBC_ON",
    "OBC_ELEVATION_FORCING_ON",
    "NC_ON",
    "NC_AVERAGE_VEL",
    "OUT_STATION_TIMESERIES_ON",
    "OUT_ELEVATION",
    "OUT_VELOCITY_2D",
}
GROUP_ORDER = [
    "NML_CASE",
    "NML_STARTUP",
    "NML_IO",
    "NML_INTEGRATION",
    "NML_RESTART",
    "NML_NETCDF",
    "NML_NETCDF_SURFACE",
    "NML_NETCDF_AV",
    "NML_PHYSICS",
    "NML_SURFACE_FORCING",
    "NML_RIVER_TYPE",
    "NML_OPEN_BOUNDARY_CONTROL",
    "NML_GRID_COORDINATES",
    "NML_GROUNDWATER",
    "NML_LAG",
    "NML_ADDITIONAL_MODELS",
    "NML_PROBES",
    "NML_BOUNDSCHK",
    "NML_NCNEST",
    "NML_NESTING",
    "NML_STATION_TIMESERIES",
]


class ConfigError(ValueError):
    pass


def q(value: str) -> str:
    if "'" in value:
        raise ConfigError("single quotes are not allowed in namelist strings")
    return f"'{value}'"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a UTC timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ConfigError(f"invalid {label}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed


def nml_date(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def validate_request(request: Mapping[str, Any]) -> Tuple[datetime, datetime, int, float, float, int]:
    if request.get("mode") not in {"barotropic_tide", "benchmark"}:
        raise ConfigError("mode must be barotropic_tide or benchmark")
    if not isinstance(request.get("case_id"), str) or not request["case_id"]:
        raise ConfigError("case_id is required")
    period = request.get("period")
    if not isinstance(period, Mapping):
        raise ConfigError("period object is required")
    start = parse_utc(period.get("analysis_start"), "period.analysis_start")
    end = parse_utc(period.get("analysis_end"), "period.analysis_end")
    if end <= start:
        raise ConfigError("analysis_end must follow analysis_start")
    spinup = int(period.get("spinup_days", 7))
    if spinup < 2:
        raise ConfigError("spinup_days must be at least 2 so the tidal ramp fits")
    physics = request.get("physics")
    if not isinstance(physics, Mapping) or physics.get("formulation") != "three_dimensional_barotropic":
        raise ConfigError("physics.formulation must be three_dimensional_barotropic")
    temperature = float(physics.get("temperature_c", 20.0))
    salinity = float(physics.get("salinity_psu", 30.0))
    layers = int(physics.get("sigma_levels", physics.get("sigma_layers", 10)))
    if "sigma_levels" in physics and "sigma_layers" in physics and int(physics["sigma_layers"]) != layers:
        raise ConfigError("sigma_levels conflicts with legacy sigma_layers (which counts levels)")
    if layers != 10:
        raise ConfigError("this scenario requires exactly ten sigma levels (nine layers)")
    return start, end, spinup, temperature, salinity, layers


def validate_numerics(numerics: Mapping[str, Any]) -> Tuple[float, int]:
    extstep = float(numerics.get("extstep_seconds", 0.0))
    isplit = int(numerics.get("isplit", 0))
    if not math.isfinite(extstep) or extstep <= 0:
        raise ConfigError("numerics.extstep_seconds must be finite and positive")
    if isplit < 1:
        raise ConfigError("numerics.isplit must be at least one")
    return extstep, isplit


def stage_times(request: Mapping[str, Any], stage: str) -> Tuple[datetime, datetime, datetime, bool]:
    analysis_start, analysis_end, spinup_days, _, _, _ = validate_request(request)
    spinup_start = analysis_start - timedelta(days=spinup_days)
    if stage == "smoke":
        return spinup_start, spinup_start + timedelta(hours=1), analysis_start, False
    if stage == "canary":
        return spinup_start, spinup_start + timedelta(days=3), analysis_start, False
    if stage == "spinup":
        return spinup_start, analysis_start, analysis_start, False
    if stage == "benchmark":
        return analysis_start, min(analysis_start + timedelta(days=1), analysis_end), analysis_start, True
    if stage == "production":
        return analysis_start, analysis_end, analysis_start, True
    if stage == "full":
        return spinup_start, analysis_end, analysis_start, False
    raise ConfigError(f"unsupported stage: {stage}")


def bindings_required(bindings: Mapping[str, Any]) -> None:
    required_strings = {
        "input_dir", "output_dir", "grid_file", "depth_file", "coriolis_file", "sigma_file",
        "sponge_file", "obc_file", "elevation_file", "station_file", "grid_edge_file", "restart_file",
    }
    missing = sorted(key for key in required_strings if not isinstance(bindings.get(key), str) or not bindings[key])
    if missing:
        raise ConfigError("missing artifact bindings: " + ", ".join(missing))
    if not isinstance(bindings.get("grid_edge_read_from_file"), bool):
        raise ConfigError("grid_edge_read_from_file binding must be true or false")


def configuration_values(
    request: Mapping[str, Any], bindings: Mapping[str, Any], numerics: Mapping[str, Any], stage: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    bindings_required(bindings)
    analysis_start, _, _, temperature, salinity, _ = validate_request(request)
    extstep, isplit = validate_numerics(numerics)
    start, end, restart_at, hotstart = stage_times(request, stage)
    internal_step = extstep * isplit
    intervals = {"station/forcing cadence": 360, "full-grid cadence": 10800,
                 "stage duration": (end - start).total_seconds(), "tidal ramp": 172800}
    for label, seconds in intervals.items():
        steps = seconds / internal_step
        if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-7):
            raise ConfigError(f"{label} ({seconds}s) is not divisible by internal step {internal_step:g}s; derive time-aligned controls before submission")
    grid_edge_read = bindings["grid_edge_read_from_file"]
    if hotstart and not grid_edge_read:
        raise ConfigError("hot-start benchmark and production stages require a frozen grid-edge file")
    # Divisibility is already checked above; ceil can add a spurious step
    # when binary arithmetic puts an exact integer ratio just above it.
    ramp_steps = 0 if hotstart else int(round(172800 / internal_step))
    startup_file = bindings["restart_file"] if hotstart else "none"
    restart_on = stage not in {"smoke", "canary"}
    restart_first = (restart_at if not hotstart else end) if restart_on else end
    values: Dict[str, Dict[str, str]] = {
        "NML_CASE": {
            "CASE_TITLE": q(f"{request['case_id']} {stage} tide-only"),
            "TIMEZONE": q("UTC"),
            "DATE_FORMAT": q("YMD"),
            "DATE_REFERENCE": q("default"),
            "START_DATE": q(nml_date(start)),
            "END_DATE": q(nml_date(end)),
        },
        "NML_STARTUP": {
            "STARTUP_TYPE": q("hotstart" if hotstart else "coldstart"),
            "STARTUP_FILE": q(startup_file),
            "STARTUP_UV_TYPE": q("set values" if hotstart else "default"),
            "STARTUP_TURB_TYPE": q("set values" if hotstart else "default"),
            "STARTUP_TS_TYPE": q("set values" if hotstart else "constant"),
            # FVCOM 4.3.1 uses the second array member as a mode-specific
            # parameter/sentinel. A scalar assignment preserves its -99
            # default and is the proven constant-startup reader syntax.
            "STARTUP_T_VALS": f"{temperature:.8g}",
            "STARTUP_S_VALS": f"{salinity:.8g}",
            "STARTUP_U_VALS": "0.0",
            "STARTUP_V_VALS": "0.0",
        },
        "NML_IO": {
            "INPUT_DIR": q(bindings["input_dir"]),
            "OUTPUT_DIR": q(bindings["output_dir"]),
            "IREPORT": "60",
            "VISIT_ALL_VARS": "F",
            "WAIT_FOR_VISIT": "F",
            "USE_MPI_IO_MODE": "F",
        },
        "NML_INTEGRATION": {
            "EXTSTEP_SECONDS": f"{extstep:.10g}",
            "ISPLIT": str(isplit),
            "IRAMP": str(ramp_steps),
            "MIN_DEPTH": "0.05",
            "STATIC_SSH_ADJ": "0.0",
        },
        "NML_RESTART": {
            "RST_ON": "T" if restart_on else "F",
            "RST_FIRST_OUT": q(nml_date(restart_first)),
            "RST_OUT_INTERVAL": q("days=30.0"),
            "RST_OUTPUT_STACK": "0",
        },
        "NML_NETCDF": {
            "NC_ON": "T",
            "NC_FIRST_OUT": q(nml_date(start)),
            "NC_OUT_INTERVAL": q("seconds=10800.0"),
            "NC_OUTPUT_STACK": "0",
            # SETUP_SUBDOMAINS always consumes this value when NC_ON is true.
            # FVCOM 4.3.1 uses the case-sensitive sentinel "FVCOM" for the
            # complete model domain; "none" is interpreted as a filename.
            "NC_SUBDOMAIN_FILES": q("FVCOM"),
            "NC_GRID_METRICS": "T",
            "NC_FILE_DATE": "T",
            "NC_VELOCITY": "F",
            "NC_SALT_TEMP": "F",
            "NC_TURBULENCE": "F",
            "NC_VERTICAL_VEL": "F",
            "NC_AVERAGE_VEL": "T",
            "NC_VORTICITY": "F",
            "NC_WIND_VEL": "F",
            "NC_WIND_STRESS": "F",
            "NC_EVAP_PRECIP": "F",
            "NC_SURFACE_HEAT": "F",
            "NC_GROUNDWATER": "F",
            "NC_BIO": "F",
            "NC_WQM": "F",
        },
        "NML_NETCDF_SURFACE": {"NCSF_ON": "F"},
        "NML_NETCDF_AV": {"NCAV_ON": "F"},
        "NML_PHYSICS": {
            "HORIZONTAL_MIXING_TYPE": q("closure"),
            "HORIZONTAL_MIXING_KIND": q("constant"),
            "HORIZONTAL_MIXING_COEFFICIENT": "0.2",
            "HORIZONTAL_PRANDTL_NUMBER": "1.0",
            "VERTICAL_MIXING_TYPE": q("closure"),
            "VERTICAL_MIXING_COEFFICIENT": "1.0e-5",
            "VERTICAL_PRANDTL_NUMBER": "1.0",
            "BOTTOM_ROUGHNESS_TYPE": q("orig"),
            "BOTTOM_ROUGHNESS_KIND": q("constant"),
            "BOTTOM_ROUGHNESS_LENGTHSCALE": "0.005",
            "FLUME_BOTTOM": "F",
            "FLUME_DRAG_COEF": "0.0",
            "BOTTOM_ROUGHNESS_MINIMUM": "0.0025",
            "CONVECTIVE_OVERTURNING": "F",
            "SCALAR_POSITIVITY_CONTROL": "T",
            "BAROTROPIC": "T",
            "BAROCLINIC_PRESSURE_GRADIENT": q("sigma levels"),
            "SEA_WATER_DENSITY_FUNCTION": q("dens2"),
            "TEMPERATURE_ACTIVE": "F",
            "SALINITY_ACTIVE": "F",
            "SURFACE_WAVE_MIXING": "F",
            "WETTING_DRYING_ON": "T",
            "RECALCULATE_RHO_MEAN": "F",
            "INTERVAL_RHO_MEAN": q("seconds=21600.0"),
            "BACKWARD_ADVECTION": "F",
            "BACKWARD_STEP": "-1",
            "ADCOR_ON": "T",
            "EQUATOR_BETA_PLANE": "F",
            "NOFLUX_BOT_CONDITION": "T",
        },
        "NML_SURFACE_FORCING": {
            "WIND_ON": "F",
            "HEATING_ON": "F",
            "PRECIPITATION_ON": "F",
            "AIRPRESSURE_ON": "F",
            "WAVE_ON": "F",
        },
        "NML_RIVER_TYPE": {"RIVER_NUMBER": "0"},
        "NML_OPEN_BOUNDARY_CONTROL": {
            "OBC_ON": "T",
            "OBC_NODE_LIST_FILE": q(bindings["obc_file"]),
            "OBC_ELEVATION_FORCING_ON": "T",
            "OBC_ELEVATION_FILE": q(bindings["elevation_file"]),
            "OBC_TS_TYPE": "3",
            "OBC_TEMP_NUDGING": "F",
            "OBC_TEMP_FILE": q("none"),
            "OBC_TEMP_NUDGING_TIMESCALE": "0.0",
            "OBC_SALT_NUDGING": "F",
            "OBC_SALT_FILE": q("none"),
            "OBC_SALT_NUDGING_TIMESCALE": "0.0",
            "OBC_MEANFLOW": "F",
            "OBC_MEANFLOW_FILE": q("none"),
            "OBC_LONGSHORE_FLOW_ON": "F",
            "OBC_LONGSHORE_FLOW_FILE": q("none"),
            "OBC_DEPTH_CONTROL_ON": "F",
        },
        "NML_GRID_COORDINATES": {
            "GRID_FILE": q(bindings["grid_file"]),
            "GRID_FILE_UNITS": q("meters"),
            "PROJECTION_REFERENCE": q("none"),
            "SIGMA_LEVELS_FILE": q(bindings["sigma_file"]),
            "DEPTH_FILE": q(bindings["depth_file"]),
            "CORIOLIS_FILE": q(bindings["coriolis_file"]),
            "SPONGE_FILE": q(bindings["sponge_file"]),
        },
        "NML_GROUNDWATER": {
            "GROUNDWATER_ON": "F",
            "GROUNDWATER_TEMP_ON": "F",
            "GROUNDWATER_SALT_ON": "F",
        },
        "NML_LAG": {"LAG_PARTICLES_ON": "F"},
        "NML_ADDITIONAL_MODELS": {
            "DATA_ASSIMILATION": "F",
            "BIOLOGICAL_MODEL": "F",
            "SEDIMENT_MODEL": "F",
            "ICING_MODEL": "F",
            "ICE_MODEL": "F",
            "HIGH_LATITUDE_WAVE": "F",
        },
        "NML_PROBES": {"PROBES_ON": "F", "PROBES_NUMBER": "0", "PROBES_FILE": q("none")},
        "NML_BOUNDSCHK": {
            "BOUNDSCHK_ON": "F",
            "CHK_INTERVAL": "0",
            "VELOC_MAG_MAX": "0.0",
            "ZETA_MAG_MAX": "0.0",
            "TEMP_MAX": "0.0",
            "TEMP_MIN": "0.0",
            "SALT_MAX": "0.0",
            "SALT_MIN": "0.0",
        },
        "NML_NCNEST": {
            "NCNEST_ON": "F",
            "NCNEST_BLOCKSIZE": "1",
            "NCNEST_NODE_FILES": q("none"),
            "NCNEST_OUT_INTERVAL": q("seconds=3600.0"),
        },
        "NML_NESTING": {
            "NESTING_ON": "F",
            "NESTING_TYPE": q("1"),
            "NESTING_BLOCKSIZE": "1",
            "NESTING_FILE_NAME": q("none"),
        },
        "NML_STATION_TIMESERIES": {
            "OUT_STATION_TIMESERIES_ON": "T",
            "STATION_FILE": q(bindings["station_file"]),
            "LOCATION_TYPE": q("cell"),
            # FVCOM's TRIANGLE_GRID_EDGE_GL writes this file when false and
            # reads it when true. The first cold smoke may generate the named
            # artifact; later cold stages and all hot starts can consume its
            # frozen copy so immutable input directories remain read-only.
            "READ_GRID_EDGE_FROM_FILE": "T" if grid_edge_read else "F",
            "GRID_EDGE_FILE_NAME": q(bindings["grid_edge_file"]),
            "OUT_ELEVATION": "T",
            "OUT_VELOCITY_3D": "F",
            "OUT_VELOCITY_2D": "T",
            "OUT_WIND_VELOCITY": "F",
            "OUT_SALT_TEMP": "F",
            "OUT_INTERVAL": q("seconds=360.0"),
        },
    }
    timing = {
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "end_utc": end.isoformat().replace("+00:00", "Z"),
        "analysis_start_utc": analysis_start.isoformat().replace("+00:00", "Z"),
        "restart_first_out_utc": restart_first.isoformat().replace("+00:00", "Z"),
        "ramp_days": 0 if hotstart else 2,
        "ramp_internal_steps": ramp_steps,
        "hotstart": hotstart,
        "grid_edge_read_from_file": grid_edge_read,
    }
    return values, timing


def group_ranges(lines: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    ranges: Dict[str, Tuple[int, int]] = {}
    active: Optional[Tuple[str, int]] = None
    for index, raw in enumerate(lines):
        start = re.match(r"^\s*&([A-Za-z0-9_]+)\b", raw)
        if start:
            if active:
                raise ConfigError(f"nested namelist group near line {index + 1}")
            active = (start.group(1).upper(), index)
        elif active and re.match(r"^\s*/\s*(?:!.*)?$", raw):
            name, begin = active
            if name in ranges:
                raise ConfigError(f"duplicate namelist group: {name}")
            ranges[name] = (begin, index)
            active = None
    if active:
        raise ConfigError(f"unterminated namelist group: {active[0]}")
    return ranges


def render_template(template: str, values: Mapping[str, Mapping[str, str]]) -> str:
    lines = template.splitlines()
    ranges = group_ranges(lines)
    template_keys: Dict[str, set[str]] = {}
    template_order = sorted(ranges, key=lambda group: ranges[group][0])
    for group, (begin, end) in ranges.items():
        template_keys[group] = set()
        for index in range(begin + 1, end):
            match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", lines[index])
            if match and not lines[index].lstrip().startswith("!"):
                template_keys[group].add(match.group(1).upper())
    for group, updates in values.items():
        if group not in ranges:
            raise ConfigError(f"required group missing from executable template: {group}")
        for key, value in updates.items():
            if key not in template_keys[group]:
                raise ConfigError(f"required key missing from executable template: {group}.{key}")
    rendered = [
        "! FVCOM 4.3.1 namelist normalized from the frozen executable's blank template.",
        "! Omitted optional keys retain FVCOM's source-initialized defaults; disabled modules are explicit.",
    ]
    for group in template_order:
        updates = values.get(group)
        if not updates:
            continue
        rendered.append(f"&{group}")
        rendered.extend(f" {key} = {value}," for key, value in updates.items())
        rendered.append("/")
    return "\n".join(rendered) + "\n"


def write_overlay(path: Path, values: Mapping[str, Mapping[str, str]]) -> None:
    blocks: List[str] = ["! Provisional overlay; not executable until applied to the frozen binary's blank namelist."]
    for group in GROUP_ORDER:
        updates = values.get(group)
        if not updates:
            continue
        blocks.append(f"&{group}")
        blocks.extend(f" {key} = {value}," for key, value in updates.items())
        blocks.append("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8", newline="\n")


def parse_values(text: str) -> Dict[str, Dict[str, str]]:
    lines = text.splitlines()
    ranges = group_ranges(lines)
    result: Dict[str, Dict[str, str]] = {}
    for group, (begin, end) in ranges.items():
        result[group] = {}
        for raw in lines[begin + 1:end]:
            match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*,?\s*(?:!.*)?$", raw)
            if match:
                result[group][match.group(1).upper()] = match.group(2).rstrip(",").strip()
    return result


def scalar(value: str) -> str:
    clean = value.strip().rstrip(",").strip()
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
        return clean[1:-1]
    return clean


def logical(value: str) -> Optional[bool]:
    clean = scalar(value).strip().upper().strip(".")
    if clean in {"T", "TRUE"}:
        return True
    if clean in {"F", "FALSE"}:
        return False
    return None


def audit_duration_token(value: str) -> Optional[str]:
    """Check a scalar IDEAL_TIME_STRING2TIME token, not a calendar date.

    FVCOM 4.3.1 GET_VALUE recognizes reals by a literal decimal point;
    its numeric alphabet includes E/e but excludes Fortran D/d exponents.
    Keep this lexical check separate from physical cadence/alignment checks.
    """
    match = re.fullmatch(r"(seconds|days|cycles) *= *(.+?) *", scalar(value))
    if match:
        unit, number = match.groups()
        if unit == "cycles":
            valid = re.fullmatch(r"[+-]?[0-9]+", number)
        else:
            valid = re.fullmatch(
                r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)"
                r"(?:[Ee][+-]?[0-9]+|[+-][0-9]+)?", number
            )
        if valid:
            return None
    return (
        "FVCOM duration must use lowercase seconds/days with a decimal point "
        "(for example seconds=36.0 or seconds=3.6e1), or cycles with an integer"
    )


def audit_duration_values(parsed: Mapping[str, Mapping[str, str]]) -> List[str]:
    fields = (
        ("NML_RESTART", "RST_ON", "RST_OUT_INTERVAL"),
        ("NML_NETCDF", "NC_ON", "NC_OUT_INTERVAL"),
        ("NML_NETCDF_AV", "NCAV_ON", "NCAV_OUT_INTERVAL"),
        ("NML_NETCDF_SURFACE", "NCSF_ON", "NCSF_OUT_INTERVAL"),
        ("NML_PHYSICS", "RECALCULATE_RHO_MEAN", "INTERVAL_RHO_MEAN"),
        ("NML_STATION_TIMESERIES", "OUT_STATION_TIMESERIES_ON", "OUT_INTERVAL"),
    )
    problems = []
    for group, enabled, key in fields:
        values = parsed.get(group, {})
        if logical(values.get(enabled, "")) is True:
            problem = audit_duration_token(values.get(key, ""))
            if problem:
                problems.append(f"{group}.{key}: {problem}")
    return problems


def audit_values(parsed: Mapping[str, Mapping[str, str]]) -> List[str]:
    problems: List[str] = audit_duration_values(parsed)
    flat = {key: value for group in parsed.values() for key, value in group.items()}
    for key in sorted(TRUE_KEYS):
        if logical(flat.get(key, "")) is not True:
            problems.append(f"{key} must be true")
    for key in sorted(FALSE_KEYS):
        if key in flat and logical(flat[key]) is not False:
            problems.append(f"{key} must be false")
    if scalar(flat.get("LOCATION_TYPE", "")).lower() != "cell":
        problems.append("LOCATION_TYPE must be cell")
    if scalar(flat.get("GRID_EDGE_FILE_NAME", "")).lower() in {"", "none"}:
        problems.append("GRID_EDGE_FILE_NAME must name a real attempt-bound adjacency artifact")
    if scalar(flat.get("OUT_INTERVAL", "")).lower() != "seconds=360.0":
        problems.append("station OUT_INTERVAL must be seconds=360.0")
    if scalar(flat.get("NC_OUT_INTERVAL", "")).lower() != "seconds=10800.0":
        problems.append("NC_OUT_INTERVAL must be seconds=10800.0")
    if scalar(flat.get("NC_SUBDOMAIN_FILES", "")) != "FVCOM":
        problems.append("NC_SUBDOMAIN_FILES must be the case-sensitive FVCOM full-domain sentinel")
    if scalar(flat.get("GRID_FILE_UNITS", "")).lower() != "meters":
        problems.append("GRID_FILE_UNITS must be meters")
    if scalar(flat.get("RIVER_NUMBER", "")) != "0":
        problems.append("RIVER_NUMBER must be zero")
    if scalar(flat.get("STARTUP_TS_TYPE", "")).lower() == "constant":
        t_values = [float(x) for x in flat.get("STARTUP_T_VALS", "").split(",")]
        s_values = [float(x) for x in flat.get("STARTUP_S_VALS", "").split(",")]
        if not t_values or t_values[0] != 20.0 or any(value != -99.0 for value in t_values[1:]):
            problems.append("cold-start temperature must be uniformly 20 C")
        if not s_values or s_values[0] != 30.0 or any(value != -99.0 for value in s_values[1:]):
            problems.append("cold-start salinity must be uniformly 30 PSU")
    return problems


def audit_stage_values(parsed: Mapping[str, Mapping[str, str]], stage: str) -> List[str]:
    problems: List[str] = []
    restart_on = logical(parsed.get("NML_RESTART", {}).get("RST_ON", ""))
    expected = stage not in {"smoke", "canary"}
    if restart_on is not expected:
        problems.append(f"RST_ON must be {'true' if expected else 'false'} for {stage}")
    grid_edge_read = logical(parsed.get("NML_STATION_TIMESERIES", {}).get("READ_GRID_EDGE_FROM_FILE", ""))
    if grid_edge_read is None:
        problems.append("READ_GRID_EDGE_FROM_FILE must be a logical value")
    elif stage in {"benchmark", "production"} and not grid_edge_read:
        problems.append(
            f"READ_GRID_EDGE_FROM_FILE must be true for hot-start {stage}"
        )
    return problems


def station_rows(data: Any) -> List[Mapping[str, Any]]:
    rows = data.get("stations") if isinstance(data, Mapping) else data
    if not isinstance(rows, list):
        raise ConfigError("station JSON must be an array or an object with a stations array")
    return rows


def preconfiguration_sigma_layers(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return None
    direct = data.get("sigma_levels", data.get("sigma_layers", data.get("n_sigma_layers")))
    if direct is not None:
        return direct
    sigma = data.get("sigma")
    return sigma.get("levels") if isinstance(sigma, Mapping) else None


def write_station_file(path: Path, data: Any, provisional: bool = False) -> int:
    rows = station_rows(data)
    if not rows and not provisional:
        raise ConfigError("a final station file requires at least one station")
    seen_ids = set()
    rendered = ["No. Longitude Latitude Cell Depth Station"]
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ConfigError(f"station {index} is not an object")
        station_id = str(row.get("station_id", "")).strip()
        if not station_id or station_id in seen_ids or re.search(r"\s", station_id):
            raise ConfigError(f"station {index} has an empty, duplicate, or whitespace-containing station_id")
        seen_ids.add(station_id)
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        cell = int(row["cell_id"])
        depth = float(row["depth_m"])
        if not (-180 <= lon <= 180 and -90 <= lat <= 90 and cell >= 1 and math.isfinite(depth) and depth > 0):
            raise ConfigError(f"station {station_id} contains invalid coordinates, cell_id, or depth")
        rendered.append(f"{index:d} {lon:.10f} {lat:.10f} {cell:d} {depth:.6f} {station_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8", newline="\n")
    return len(rows)


def file_checks(parsed: Mapping[str, Mapping[str, str]], input_root: Optional[Path], hotstart: bool) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for group, keys in REQUIRED_FILES.items():
        for key in keys:
            value = scalar(parsed.get(group, {}).get(key, ""))
            resolved = input_root / value if input_root is not None and value else None
            checks.append({
                "group": group,
                "key": key,
                "value": value,
                "resolved": str(resolved.resolve()) if resolved else None,
                "exists": bool(resolved and resolved.is_file()),
                "required": True,
            })
    grid_edge_read = logical(parsed.get("NML_STATION_TIMESERIES", {}).get("READ_GRID_EDGE_FROM_FILE", ""))
    if grid_edge_read:
        value = scalar(parsed.get("NML_STATION_TIMESERIES", {}).get("GRID_EDGE_FILE_NAME", ""))
        resolved = input_root / value if input_root is not None and value else None
        checks.append({
            "group": "NML_STATION_TIMESERIES", "key": "GRID_EDGE_FILE_NAME", "value": value,
            "resolved": str(resolved.resolve()) if resolved else None,
            "exists": bool(resolved and resolved.is_file()), "required": True,
        })
    if hotstart:
        value = scalar(parsed.get("NML_STARTUP", {}).get("STARTUP_FILE", ""))
        resolved = input_root / value if input_root is not None and value else None
        checks.append({
            "group": "NML_STARTUP", "key": "STARTUP_FILE", "value": value,
            "resolved": str(resolved.resolve()) if resolved else None,
            "exists": bool(resolved and resolved.is_file()), "required": True,
        })
    return checks


def manifest(
    request: Mapping[str, Any], stage: str, root: Path, hashes: Dict[str, str], timing: Dict[str, Any],
    numerics: Mapping[str, Any], checks: List[Dict[str, Any]], provisional: bool,
    audit_problems: List[str], preconfig_manifest: Optional[Path], station_count: int,
) -> Dict[str, Any]:
    blocking = list(audit_problems)
    warnings: List[str] = []
    if provisional:
        warnings.append("provisional overlay is not runnable until applied to the frozen executable template")
    else:
        missing = [item for item in checks if item["required"] and not item["exists"]]
        if missing:
            blocking.append("missing required input files: " + ", ".join(item["key"] for item in missing))
        if station_count < 1:
            blocking.append("no stations were written")
        if preconfig_manifest is None or not preconfig_manifest.is_file():
            blocking.append("final configuration requires a preconfiguration manifest")
        else:
            pre = load_json(preconfig_manifest)
            layers = preconfiguration_sigma_layers(pre)
            if layers != 10:
                blocking.append("preconfiguration manifest does not report ten sigma levels (nine layers)")
            hashes["preconfiguration_manifest_sha256"] = sha256(preconfig_manifest)
    status = "provisional_ready" if provisional and not blocking else "ready" if not blocking else "blocked"
    result = {
        "schema": "fvcom_namelist_configuration_manifest_v1",
        "status": status,
        "case_id": request["case_id"],
        "stage": stage,
        "artifact_root": str(root.resolve()),
        "hashes": hashes,
        "timing": timing,
        "numerics": {"extstep_seconds": float(numerics["extstep_seconds"]), "isplit": int(numerics["isplit"])},
        "file_checks": checks,
        "warnings": warnings,
        "blocking_reasons": blocking,
        "resume_token": f"namelist:{request['case_id']}:{stage}:{'provisional' if provisional else 'final'}",
    }
    write_json(root / "configuration_manifest.json", result)
    return result


def cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("provisional", "render"):
        item = sub.add_parser(command)
        item.add_argument("--request", type=Path, required=True)
        item.add_argument("--bindings", type=Path, required=True)
        item.add_argument("--numerics", type=Path, required=True)
        item.add_argument("--stage", choices=sorted(STAGES), required=True)
        item.add_argument("--artifact-root", type=Path, required=True)
        item.add_argument("--stations", type=Path, required=True)
        if command == "render":
            item.add_argument("--blank-template", type=Path, required=True)
            item.add_argument("--input-root", type=Path, required=True)
            item.add_argument("--preconfiguration-manifest", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--namelist", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "validate":
            parsed = parse_values(args.namelist.read_text(encoding="utf-8-sig"))
            problems = audit_values(parsed)
            print(json.dumps({"status": "pass" if not problems else "fail", "problems": problems}, indent=2))
            return 0 if not problems else 2
        request = load_json(args.request)
        bindings = load_json(args.bindings)
        numerics = load_json(args.numerics)
        root: Path = args.artifact_root
        root.mkdir(parents=True, exist_ok=True)
        values, timing = configuration_values(request, bindings, numerics, args.stage)
        station_path = root / bindings["station_file"]
        provisional = args.command == "provisional"
        station_count = write_station_file(station_path, load_json(args.stations), provisional=provisional)
        hashes = {"station_file_sha256": sha256(station_path)}
        if provisional:
            namelist_path = root / f"{request['case_id']}_{args.stage}_provisional_overlay.nml"
            write_overlay(namelist_path, values)
            parsed = parse_values(namelist_path.read_text(encoding="utf-8-sig"))
            checks = file_checks(parsed, None, timing["hotstart"])
            preconfig = None
        else:
            template_text = args.blank_template.read_text(encoding="utf-8-sig")
            rendered = render_template(template_text, values)
            namelist_path = root / f"{request['case_id']}_{args.stage}_run.nml"
            namelist_path.write_text(rendered, encoding="utf-8", newline="\n")
            hashes["blank_template_sha256"] = sha256(args.blank_template)
            parsed = parse_values(rendered)
            checks = file_checks(parsed, args.input_root, timing["hotstart"])
            preconfig = args.preconfiguration_manifest
        hashes["namelist_sha256"] = sha256(namelist_path)
        problems = audit_values(parsed) + audit_stage_values(parsed, args.stage)
        result = manifest(
            request, args.stage, root, hashes, timing, numerics, checks, provisional,
            problems, preconfig, station_count,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"provisional_ready", "ready"} else 3
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
