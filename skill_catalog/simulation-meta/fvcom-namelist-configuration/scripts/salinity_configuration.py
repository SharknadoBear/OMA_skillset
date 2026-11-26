"""Render a reader or one-day fixed-temperature, active-salinity estuary pilot."""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fvcom_namelist import parse_values, render_template, scalar


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(base, blank, stage, start="2025-03-18T00:00:00Z"):
    if stage == "smoke":
        stage = "pilot"
    clock = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if clock.tzinfo is None or clock.utcoffset() != timedelta(0) or clock.microsecond:
        raise ValueError("start must be an exact UTC second")
    seconds = {"reader": 720, "bottom-sensitivity": 720, "pilot": 86400, "bottom-sensitivity-day": 86400}[stage]
    fmt = lambda value: "'" + value.strftime("%Y-%m-%d %H:%M:%S") + "'"
    groups = parse_values(base.read_text())
    checks = {
        "NML_GRID_COORDINATES": {
            "GRID_FILE_UNITS": "meters",
            "GRID_FILE": "columbia_grd.dat",
        },
        "NML_RIVER_TYPE": {
            "RIVER_NUMBER": "19",
            "RIVER_INFLOW_LOCATION": "node",
            "RIVER_INFO_FILE": "river.nml",
        },
        "NML_OPEN_BOUNDARY_CONTROL": {
            "OBC_ELEVATION_FORCING_ON": "T",
            "OBC_ELEVATION_FILE": "columbia_elevtide.nc",
            "OBC_TS_TYPE": "3",
        },
        "NML_SURFACE_FORCING": {
            "WIND_ON": "T",
            "WIND_TYPE": "stress",
            "AIRPRESSURE_ON": "T",
            "WIND_FILE": "columbia_surface.nc",
            "AIRPRESSURE_FILE": "columbia_surface.nc",
            "HEATING_ON": "F",
            "PRECIPITATION_ON": "F",
        },
        "NML_PHYSICS": {
            "WETTING_DRYING_ON": "T",
            "SCALAR_POSITIVITY_CONTROL": "T",
            "VERTICAL_MIXING_TYPE": "closure",
        },
        "NML_STATION_TIMESERIES": {
            "LOCATION_TYPE": "cell",
            "READ_GRID_EDGE_FROM_FILE": "T",
        },
    }
    for group, entries in checks.items():
        for key, expected in entries.items():
            if scalar(groups[group][key]) != expected:
                raise ValueError("Incompatible retained Columbia setting " + key)
    updates = {
        "NML_CASE": {
            "START_DATE": fmt(clock),
            "END_DATE": fmt(clock + timedelta(seconds=seconds)),
            "CASE_TITLE": "'Columbia baroclinic salinity " + stage + "'",
        },
        "NML_IO": {"INPUT_DIR": "'./input/'", "OUTPUT_DIR": "'./'"},
        "NML_STARTUP": {
            "STARTUP_TYPE": "'hotstart'",
            "STARTUP_FILE": "'columbia_start.nc'",
            "STARTUP_UV_TYPE": "'set values'",
            "STARTUP_TS_TYPE": "'set values'",
            "STARTUP_TURB_TYPE": "'default'",
        },
        "NML_INTEGRATION": {"IRAMP": "0", "EXTSTEP_SECONDS": "1.2", "ISPLIT": "1"},
        "NML_RESTART": {
            "RST_ON": "T",
            "RST_FIRST_OUT": fmt(clock + timedelta(seconds=seconds)),
            "RST_OUT_INTERVAL": "'days=1.0'",
        },
        "NML_NETCDF": {
            "NC_FIRST_OUT": fmt(clock),
            "NC_OUT_INTERVAL": "'seconds="
            + ("3600.0" if seconds == 86400 else "360.0")
            + "'",
            "NC_GRID_METRICS": "T",
            "NC_VELOCITY": "T",
            "NC_SALT_TEMP": "T",
            "NC_VERTICAL_VEL": "T",
            "NC_AVERAGE_VEL": "T",
            "NC_WIND_STRESS": "T",
        },
        "NML_PHYSICS": {
            "BAROTROPIC": "F",
            "SEA_WATER_DENSITY_FUNCTION": "'dens2'",
            "TEMPERATURE_ACTIVE": "F",
            "SALINITY_ACTIVE": "T",
        },
        "NML_RIVER_TYPE": {"RIVER_TS_SETTING": "'calculated'"},
        "NML_OPEN_BOUNDARY_CONTROL": {
            "OBC_SALT_NUDGING": "T",
            "OBC_SALT_FILE": "'columbia_tsobc.nc'",
            "OBC_SALT_NUDGING_TIMESCALE": "1.0",
            "OBC_TEMP_NUDGING": "F",
            "OBC_TEMP_FILE": "'none'",
        },
        "NML_STATION_TIMESERIES": {
            "OUT_ELEVATION": "T",
            "OUT_VELOCITY_3D": "T",
            "OUT_SALT_TEMP": "T",
            "OUT_INTERVAL": "'seconds=360.0'",
        },
    }
    for group, values in updates.items():
        groups[group].update(values)
    if stage.startswith("bottom-sensitivity"):
        groups["NML_PHYSICS"]["NOFLUX_BOT_CONDITION"] = "F"
    return render_template(blank.read_text(), groups)


def cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "blank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["reader", "smoke", "pilot", "bottom-sensitivity", "bottom-sensitivity-day"],
        required=True,
    )
    parser.add_argument("--start", default="2025-03-18T00:00:00Z")
    a = parser.parse_args(argv)
    data = render(a.base, a.blank, a.stage, a.start).encode()
    a.output.parent.mkdir(exist_ok=True, parents=True)
    if a.output.exists() and a.output.read_bytes() != data:
        raise FileExistsError(a.output)
    a.output.write_bytes(data)
    print(
        json.dumps(
            {
                "stage": a.stage,
                "namelist_sha256": sha(a.output),
                "blank_sha256": sha(a.blank),
                "base_sha256": sha(a.base),
            }
        )
    )
    return 0
