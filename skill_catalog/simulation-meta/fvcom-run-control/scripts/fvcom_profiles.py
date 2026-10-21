"""Shared, explicit FVCOM experiment settings; no scheduling or monitoring."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

PAPER_PROFILE = "paper_barotropic"
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
    if profile is None:
        return result
    if profile != PAPER_PROFILE:
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
                raise ValueError(f"{PAPER_PROFILE} fixes {label}={value!r}")
            else:
                target[key] = deepcopy(value)
    merge(result, PAPER_SETTINGS)
    if result.get("accepted_grid"):
        raise ValueError("paper_barotropic cannot reuse an accepted grid")
    if result["physics"].get("sigma_layers", 10) != 10:
        raise ValueError("legacy sigma_layers counts ten interfaces, not nine physical layers")
    return result


def paper_run_start(request: dict[str, Any]) -> datetime:
    resolved = resolve_request(request)
    if resolved.get("profile") != PAPER_PROFILE:
        raise ValueError("paper_run_start requires paper_barotropic")
    return (datetime.fromisoformat(resolved["period"]["analysis_start"].replace("Z", "+00:00"))
            - timedelta(days=resolved["period"]["spinup_days"]))
