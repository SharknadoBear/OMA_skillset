"""Shared, explicit FVCOM experiment settings; no scheduling or monitoring."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

PAPER_PROFILE = "paper_barotropic"
PAIR_PROFILE = "columbia_river_pair"
WIND_PROFILE = "columbia_wind_pair"
WIND_MODE = "barotropic-tide-wind"
RIVER_MODE = "barotropic-tide-river"

FACTORIAL_PROFILE = "columbia_wind_river_factorial"
COMBINED_MODE = "barotropic-tide-wind-river"
BAROTROPIC_MODES = {"barotropic_tide", RIVER_MODE, WIND_MODE, COMBINED_MODE}


def has_wind(request: dict[str, Any]) -> bool:
    return request.get("mode") in {WIND_MODE, COMBINED_MODE}


def has_river(request: dict[str, Any]) -> bool:
    return request.get("mode") in {RIVER_MODE, COMBINED_MODE}


def needs_air_pressure(request: dict[str, Any]) -> bool:
    return has_wind(request) or request.get("profile") in {WIND_PROFILE, FACTORIAL_PROFILE}


def is_cold_pair_stage(request: dict[str, Any]) -> bool:
    """These profiles start smoke and production independently from rest."""
    return request.get("profile") in {PAPER_PROFILE, PAIR_PROFILE, WIND_PROFILE, FACTORIAL_PROFILE}


def validate_river_source(request: dict[str, Any]) -> None:
    source = request.get("river")
    if has_river(request):
        if not isinstance(source, dict) or not all(
            isinstance(source.get(key), str) and source[key].strip()
            for key in ("provider", "station_id", "parameter_code")
        ):
            raise ValueError("river mode requires river.provider, station_id and parameter_code")
    elif source is not None:
        raise ValueError("river source requires a mode with river forcing")


PAPER_SETTINGS = {
    "mode": "barotropic_tide",
    "grid_policy": "fresh_only",
    "period": {"analysis_start": "2025-04-01T00:00:00Z",
               "analysis_end": "2025-04-15T00:00:00Z", "spinup_days": 3},
    "physics": {"formulation": "three_dimensional_barotropic", "temperature_c": 20.0,
                "salinity_psu": 30.0, "sigma_levels": 10},
    "tpxo": {"constituents": "all_available"},
    "observations": {"provider": "noaa_coops", "current_policy": "downward_all_bin_profiles_only"},
    "benchmark": {"enabled": False, "ranks": [], "probe_ranks": [], "repeats": 0, "extend": False},
    "execution": {"stages": ["smoke", "production"], "smoke_simulated_hours": 24,
                  "max_smoke_attempts": 3, "max_production_attempts": 3,
                  "production_nodes": 2, "production_ranks": 208,
                  "production_ranks_per_node": 104},
    "kestrel": {"os_preference": "rh9",
                "source_repository": "https://github.com/SharknadoBear/FVCOM_MP.git",
                "source_commit": "f8e4b50d4fd51409febcefc763cbc5dca11870e2"},
}


def resolve_request(request: dict[str, Any]) -> dict[str, Any]:
    """Fill profile defaults and reject contradictory settings, without mutating input."""
    result = deepcopy(request)
    profile = result.get("profile")
    validate_river_source(result)
    if has_wind(result):
        result.setdefault("period", {}).setdefault("spinup_days", 14)
    if profile is None:
        return result
    if profile not in {PAPER_PROFILE, PAIR_PROFILE, WIND_PROFILE, FACTORIAL_PROFILE}:
        raise ValueError(f"unsupported simulation profile: {profile}")

    def merge(target, settings, prefix=""):
        for key, value in settings.items():
            label = prefix + key
            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                if not isinstance(target[key], dict):
                    raise ValueError(f"{label} must be an object")
                merge(target[key], value, label + ".")
            elif key in target and (target[key] != value or
                    (isinstance(value, (int, float)) and not isinstance(value, bool)
                     and isinstance(target[key], bool))):
                raise ValueError(f"{profile} fixes {label}={value!r}")
            else:
                target[key] = deepcopy(value)
    settings = PAPER_SETTINGS
    if profile == PAIR_PROFILE:
        if result.get("mode") not in {"barotropic_tide", RIVER_MODE}:
            raise ValueError("paired study requires an explicit tide-only or river mode")
        settings = {key: value for key, value in PAPER_SETTINGS.items() if key != "mode"}
        if result["mode"] == RIVER_MODE:
            settings["river"] = {"provider": "usgs", "station_id": "14246900", "parameter_code": "72137"}
    if profile in {WIND_PROFILE, FACTORIAL_PROFILE}:
        allowed = BAROTROPIC_MODES if profile == FACTORIAL_PROFILE else {"barotropic_tide", WIND_MODE}
        if result.get("mode") not in allowed:
            raise ValueError(f"mode is incompatible with {profile}")
        settings = {key: deepcopy(value) for key, value in PAPER_SETTINGS.items() if key != "mode"}
        if has_river(result):
            settings["river"] = {"provider": "usgs", "station_id": "14246900", "parameter_code": "72137"}
        settings["grid_policy"] = "retained_only"
        settings["period"]["spinup_days"] = 14
    merge(result, settings)
    if result["grid_policy"] == "fresh_only" and result.get("accepted_grid"):
        raise ValueError(f"{profile} requires a fresh grid")
    if result["physics"].get("sigma_layers", 10) != 10:
        raise ValueError("legacy sigma_layers counts ten interfaces, not nine physical layers")
    return result


def paper_run_start(request: dict[str, Any]) -> datetime:
    resolved = resolve_request(request)
    if resolved.get("profile") != PAPER_PROFILE:
        raise ValueError("paper_run_start requires paper_barotropic")
    return (datetime.fromisoformat(resolved["period"]["analysis_start"].replace("Z", "+00:00"))
            - timedelta(days=resolved["period"]["spinup_days"]))
