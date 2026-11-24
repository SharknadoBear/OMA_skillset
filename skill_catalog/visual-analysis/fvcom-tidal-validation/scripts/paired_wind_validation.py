"""Compare audited Columbia tide-control and wind-treatment simulations at NOAA gauges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from condense_fvcom_station import load_stack
from run_case_validation import parse_namelist, read, sha256, verify_prerequisites
from fvcom_profiles import WIND_MODE, WIND_PROFILE


def records(project, attempt):
    return verify_prerequisites(argparse.Namespace(
        project=project, attempt=attempt, grid_case="retained_columbia", input_freeze=None,
        production_audit=None, station_netcdf=None, observation_manifest=None, forcing_manifest=None))


def check_pair(control, wind):
    for case, mode in ((control, "barotropic_tide"), (wind, WIND_MODE)):
        if case["request"].get("profile") != WIND_PROFILE or case["request"]["mode"] != mode:
            raise ValueError("expected the Columbia wind profile control and treatment")
    if (control["executable_hash"] != wind["executable_hash"]
            or control["station_keys"] != wind["station_keys"]
            or sha256(control["inventory_path"]) != sha256(wind["inventory_path"])):
        raise ValueError("executable or sampling geometry differs between members")
    left, right = (parse_namelist(c["namelist"]) for c in (control, wind))
    physics = {"CASE_TITLE", "INPUT_DIR", "OUTPUT_DIR", "OBC_ELEVATION_FILE",
               "WIND_ON", "WIND_TYPE", "WIND_KIND", "WIND_FILE",
               "AIRPRESSURE_KIND", "AIRPRESSURE_FILE", "AIRPRESSURE_VALUE"}
    for key in left.keys() | right.keys():
        if key not in physics and left.get(key) != right.get(key):
            raise ValueError(f"paired namelist setting differs: {key}")
    common = dict(control["freeze"]["files"])
    pure_sha = common.pop(left["OBC_ELEVATION_FILE"])
    treatment = dict(wind["freeze"]["files"])
    if treatment.get("pure_tpxo.nc") != pure_sha:
        raise ValueError("treatment and control have different pure tides")
    extras = {right["OBC_ELEVATION_FILE"], right["WIND_FILE"], right["AIRPRESSURE_FILE"],
              "pure_tpxo.nc", "wind_boundary_components.nc", "wind_boundary_manifest.json",
              "atmosphere_manifest.json", "ordered_obc_pressure.nc", "surface_validation.json"}
    if {k:v for k,v in treatment.items() if k not in extras} != common:
        raise ValueError("runtime inputs differ outside the wind forcing products")


def elevation(case):
    names, steps, fields, _ = load_stack(case["station_paths"])
    nml = parse_namelist(case["namelist"])
    seconds = steps * float(nml["EXTSTEP_SECONDS"]) * int(nml["ISPLIT"])
    clock = pd.DatetimeIndex(case["run_start"] + pd.to_timedelta(seconds, unit="s"))
    return pd.DataFrame(fields["zeta"], index=clock, columns=names)


def score(model, reference):
    error = model - reference
    corr = None
    if model.nunique() > 1 and reference.nunique() > 1:
        corr = float(model.corr(reference))
    return dict(samples=len(error), mean_error_m=float(error.mean()),
                rmse_m=float(np.sqrt(np.mean(error**2))),
                centered_rmse_m=float(np.sqrt(np.mean((error-error.mean())**2))),
                correlation=corr)


def residual_peak(series):
    peak = series.idxmax()
    return {"time_utc": peak.isoformat(), "elevation_m": float(series.at[peak])}


def station_scores(levels):
    # Use identical support for all weather comparisons, even when NOAA has gaps.
    common = levels[["control", "wind", "observed", "predicted"]].dropna()
    response = levels.wind - levels.control
    result = {"signed_mean_response_m": float(response.mean()),
              "response_range_m": [float(response.min()), float(response.max())],
              "residual_samples": len(common)}
    if common.empty:
        return result
    if len(common) < 3 or not np.isfinite(common.to_numpy()).all():
        raise ValueError("insufficient or invalid common observations")
    residual = common.observed - common.predicted
    result["noaa_residual_peak"] = residual_peak(residual)
    for member in ("control", "wind"):
        model_residual = common[member] - common.predicted
        result[member] = {"total": score(common[member], common.observed),
                          "residual": score(model_residual, residual),
                          "residual_peak": residual_peak(model_residual),
                          "peak_time_error_hours": (model_residual.idxmax()-residual.idxmax()).total_seconds()/3600}
    result["increment_vs_noaa_residual"] = score(common.wind-common.control, residual)
    return result


def compare(control, wind, output):
    check_pair(control, wind)
    first, second = elevation(control), elevation(wind)
    if not first.index.equals(second.index) or not first.columns.equals(second.columns):
        raise ValueError("station clocks or ordering disagree")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    gauges = [sid for sid, role in control["station_keys"] if role == "water_level"]
    start, end = wind["start"], wind["end"]
    spin = second[gauges] - first[gauges]
    daily = spin.loc[spin.index < end].resample("D").mean()
    checks = {}
    tables = {}
    fig, axes = plt.subplots(len(gauges), 1, sharex=True, squeeze=False,
                             figsize=(10, max(3, 2*len(gauges))), constrained_layout=True)
    for sid, ax in zip(gauges, axes[:, 0]):
        refs = [case["observation_path"].parent/"water_level"/f"{sid}_noaa_waterlevel.csv"
                for case in (control, wind)]
        if sha256(refs[0]) != sha256(refs[1]):
            raise ValueError(f"NOAA data differ at {sid}")
        ref = pd.read_csv(refs[0])
        ref.index = pd.to_datetime(ref.pop("time"), utc=True)
        if not ref.index.is_unique:
            raise ValueError(f"duplicate NOAA timestamps at {sid}")
        levels = pd.DataFrame({"control":first[sid], "wind":second[sid]}).join(ref[["observed","predicted"]])
        levels = levels.loc[(levels.index >= start) & (levels.index < end)].copy()
        levels["response"] = levels.wind-levels.control
        levels["noaa_residual"] = levels.observed-levels.predicted
        checks[sid] = station_scores(levels)
        checks[sid]["noaa_sha256"] = sha256(refs[0])
        if checks[sid]["residual_samples"] == 0:
            inventory = next(row for row in read(control["inventory_path"])["stations"] if str(row["id"]) == sid)
            if inventory.get("observation_eligible") is not False or not inventory.get("observation_absence_evidence"):
                raise ValueError(f"unexplained missing NOAA observations at {sid}")
        checks[sid]["last_spinup_day_minus_previous_day_m"] = float(daily.loc[daily.index < start, sid].diff().iloc[-1])
        tables[sid] = levels
        ax.plot(levels.index, levels.response, label="Wind case minus control", linewidth=1)
        if checks[sid]["residual_samples"]:
            ax.plot(levels.index, levels.noaa_residual, label="NOAA observed minus predicted", linewidth=.8)
        ax.set_ylabel(f"{sid}\nLevel (m)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1,0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1,0].set_xlabel("April 2025 (UTC)")
    summary = {"schema":"fvcom_wind_pair_validation_v1", "period_end_exclusive": end.isoformat(),
               "analysis_start": start.isoformat(), "control_freeze_sha256":sha256(control["freeze_path"]),
               "wind_freeze_sha256":sha256(wind["freeze_path"]),
               "control_native_audit_sha256":sha256(control["audit_path"]),
               "wind_native_audit_sha256":sha256(wind["audit_path"]),
               "stations":checks, "omitted_forcing":"river discharge",
               "residual_definition":"Member elevation minus NOAA astronomical prediction; raw positive peak on common observations. Increment is wind minus matched control.",
               "spinup_diagnostic":"Daily signed changes include weather variability and do not independently establish convergence."}
    output.mkdir(parents=True)
    fig.savefig(output/"paired_residuals.png", dpi=150)
    plt.close(fig)
    for sid, levels in tables.items():
        levels.to_csv(output/f"{sid}_paired_levels.csv", index_label="time_utc")
    daily.to_csv(output/"daily_mean_response.csv", index_label="day_utc")
    (output/"paired_wind_validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-project","control-attempt","wind-project","wind-attempt","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compare(records(args.control_project,args.control_attempt), records(args.wind_project,args.wind_attempt),
            args.output)
