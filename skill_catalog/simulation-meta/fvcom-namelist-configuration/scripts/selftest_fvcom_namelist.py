#!/usr/bin/env python3
"""Focused offline tests for the FVCOM namelist configurator."""

from __future__ import annotations

import tempfile
import hashlib
import json
from pathlib import Path

import fvcom_namelist as fn
from prepare_station_mapping import build as build_station_mapping


REQUEST = {
    "mode": "barotropic_tide",
    "case_id": "galveston_tide_2025",
    "period": {"analysis_start": "2025-04-01T00:00:00Z", "analysis_end": "2025-05-01T00:00:00Z", "spinup_days": 7},
    "physics": {"formulation": "three_dimensional_barotropic", "temperature_c": 20, "salinity_psu": 30, "sigma_layers": 10},
}
BINDINGS = {
    "input_dir": "../input", "output_dir": "../output", "grid_file": "g_grd.dat", "depth_file": "g_dep.dat",
    "coriolis_file": "g_cor.dat", "sigma_file": "g_sig.dat", "sponge_file": "g_spg.dat", "obc_file": "g_obc.dat",
    "elevation_file": "g_tide.nc", "station_file": "g_station.dat", "grid_edge_file": "g_grid_edge.dat",
    "grid_edge_read_from_file": False, "restart_file": "g_restart.nc",
}


def template(values):
    lines = []
    for group in fn.GROUP_ORDER:
        if group not in values:
            continue
        lines.append(f"&{group}")
        lines.extend(f" {key} = 'blank'," for key in values[group])
        lines.append("/")
    return "\n".join(lines) + "\n"


def main() -> int:
    levels_request = dict(REQUEST, physics=dict(REQUEST["physics"], sigma_levels=10))
    assert fn.validate_request(levels_request)[-1] == 10
    conflict_request = dict(REQUEST, physics=dict(REQUEST["physics"], sigma_levels=11))
    try:
        fn.validate_request(conflict_request)
    except fn.ConfigError:
        pass
    else:
        raise AssertionError("conflicting vertical-grid aliases accepted")
    numerical = {"extstep_seconds": 0.5, "isplit": 10}
    try:
        fn.configuration_values(REQUEST, BINDINGS, {"extstep_seconds": 1.3, "isplit": 4}, "smoke")
    except fn.ConfigError as exc:
        assert "not divisible" in str(exc)
    else:
        raise AssertionError("Columbia incompatible fixed-step output schedule accepted")
    fn.configuration_values(REQUEST, BINDINGS, {"extstep_seconds": 1.2, "isplit": 4}, "smoke")
    values, timing = fn.configuration_values(REQUEST, BINDINGS, numerical, "spinup")
    assert timing["start_utc"] == "2025-03-25T00:00:00Z"
    assert timing["end_utc"] == "2025-04-01T00:00:00Z"
    assert timing["ramp_internal_steps"] == 34560
    rendered = fn.render_template(template(values), values)
    assert "'blank'" not in rendered
    assert "NML_BOUNDSCHK" in rendered and "NML_NCNEST" in rendered and "NML_NESTING" in rendered
    assert "GRID_EDGE_FILE_NAME = 'g_grid_edge.dat'" in rendered
    assert "READ_GRID_EDGE_FROM_FILE = F" in rendered
    parsed = fn.parse_values(rendered)
    assert fn.audit_values(parsed) == [], fn.audit_values(parsed)
    assert parsed["NML_STARTUP"]["STARTUP_T_VALS"] == "20"
    assert parsed["NML_STARTUP"]["STARTUP_S_VALS"] == "30"
    assert parsed["NML_RESTART"]["RST_ON"] == "T"
    assert parsed["NML_NETCDF"]["NC_SUBDOMAIN_FILES"] == "'FVCOM'"
    smoke_values, smoke_timing = fn.configuration_values(REQUEST, BINDINGS, numerical, "smoke")
    assert smoke_values["NML_RESTART"]["RST_ON"] == "F"
    assert smoke_timing["restart_first_out_utc"] == smoke_timing["end_utc"]
    assert fn.audit_stage_values(smoke_values, "smoke") == []
    assert fn.audit_stage_values(values, "spinup") == []
    frozen_bindings = dict(BINDINGS, grid_edge_read_from_file=True)
    benchmark_values, _ = fn.configuration_values(REQUEST, frozen_bindings, numerical, "benchmark")
    assert benchmark_values["NML_STATION_TIMESERIES"]["READ_GRID_EDGE_FROM_FILE"] == "T"
    assert fn.audit_stage_values(benchmark_values, "benchmark") == []
    precomputed_canary, _ = fn.configuration_values(REQUEST, frozen_bindings, numerical, "canary")
    assert precomputed_canary["NML_STATION_TIMESERIES"]["READ_GRID_EDGE_FROM_FILE"] == "T"
    assert fn.audit_stage_values(precomputed_canary, "canary") == []
    try:
        fn.configuration_values(REQUEST, BINDINGS, numerical, "production")
    except fn.ConfigError as exc:
        assert "frozen grid-edge" in str(exc)
    else:
        raise AssertionError("hot start without frozen grid-edge input was accepted")
    assert fn.scalar(parsed["NML_PHYSICS"]["BAROTROPIC"]) == "T"
    assert fn.scalar(parsed["NML_STATION_TIMESERIES"]["LOCATION_TYPE"]) == "cell"
    assert fn.preconfiguration_sigma_layers({"sigma": {"levels": 10}}) == 10
    broken = dict(parsed)
    broken["NML_SURFACE_FORCING"] = dict(parsed["NML_SURFACE_FORCING"])
    broken["NML_SURFACE_FORCING"]["WIND_ON"] = "T"
    assert "WIND_ON must be false" in fn.audit_values(broken)
    bad_river = {group: dict(items) for group, items in parsed.items()}
    bad_river["NML_RIVER_TYPE"]["RIVER_NUMBER"] = "1"
    assert "RIVER_NUMBER must be zero" in fn.audit_values(bad_river)
    bad_subdomain = {group: dict(items) for group, items in parsed.items()}
    bad_subdomain["NML_NETCDF"]["NC_SUBDOMAIN_FILES"] = "'none'"
    assert any("case-sensitive FVCOM" in item for item in fn.audit_values(bad_subdomain))
    bad_grid_edge = {group: dict(items) for group, items in parsed.items()}
    bad_grid_edge["NML_STATION_TIMESERIES"]["GRID_EDGE_FILE_NAME"] = "'none'"
    assert any("adjacency artifact" in item for item in fn.audit_values(bad_grid_edge))
    bad_ts_obc = {group: dict(items) for group, items in parsed.items()}
    bad_ts_obc["NML_OPEN_BOUNDARY_CONTROL"]["OBC_TEMP_NUDGING"] = "T"
    bad_ts_obc["NML_OPEN_BOUNDARY_CONTROL"]["OBC_SALT_NUDGING"] = "T"
    ts_problems = fn.audit_values(bad_ts_obc)
    assert "OBC_TEMP_NUDGING must be false" in ts_problems
    assert "OBC_SALT_NUDGING must be false" in ts_problems
    bad_template = template(values).replace(" BAROTROPIC = 'blank',\n", "")
    try:
        fn.render_template(bad_template, values)
    except fn.ConfigError as exc:
        assert "BAROTROPIC" in str(exc)
    else:
        raise AssertionError("missing template key was accepted")
    with tempfile.TemporaryDirectory(prefix="fvcom-nml-test-") as tmp:
        root = Path(tmp)
        path = root / "station.dat"
        count = fn.write_station_file(path, {"stations": [
            {"station_id": "8771450", "longitude": -94.7933, "latitude": 29.3100, "cell_id": 42, "depth_m": 8.1}
        ]})
        assert count == 1
        assert "8771450" in path.read_text()
        mesh = root / "grid.2dm"
        mesh.write_text(
            "MESH2D\nE3T 8 1 2 3 1\nND 1 -95 29 -2\n"
            "ND 2 -94 29 -4\nND 3 -95 30 -6\n",
            encoding="utf-8",
        )
        mesh_sha = hashlib.sha256(mesh.read_bytes()).hexdigest()
        contract = root / "contract.json"
        contract.write_text(json.dumps({
            "schema_version": "fvcom_grid_delivery_contract_v1",
            "mesh": {"path": "grid.2dm", "sha256": mesh_sha, "node_count": 3,
                     "triangle_count": 1, "coordinate_crs": "EPSG:4326"}
        }), encoding="utf-8")
        inventory = root / "inventory.json"
        inventory.write_text(json.dumps({
            "schema": "noaa_coops_validation_station_inventory_v1", "mesh_sha256": mesh_sha,
            "stations": [{"id": "test", "role": "water_level", "eligible": True,
                          "longitude": -94.75, "latitude": 29.25, "name": "test station"}]
        }), encoding="utf-8")
        mapping = build_station_mapping(contract, inventory, root / "station_mapping.json")
        assert mapping["station_count"] == 1
        assert mapping["stations"][0]["cell_id"] == 8
        assert abs(mapping["stations"][0]["depth_m"] - 3.5) < 1.0e-12
    print("fvcom-namelist-configuration selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
