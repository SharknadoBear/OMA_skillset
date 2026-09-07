from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# cycles per hour; forcing may contain more, but only named, Rayleigh-resolvable
# constituents are estimated here.
FREQUENCIES_CPH = {
    "M2": 0.0805114007, "S2": 0.0833333333, "N2": 0.0789992488,
    "K2": 0.0835614924, "2N2": 0.0774870970, "MU2": 0.0776894680,
    "NU2": 0.0792016198, "L2": 0.0820235525, "K1": 0.0417807462,
    "O1": 0.0387306544, "P1": 0.0415525871, "Q1": 0.0372185026,
    "2Q1": 0.0357063507, "J1": 0.0432928981, "OO1": 0.0448308380,
    "S1": 0.0416666721, "MF": 0.0030500918, "MM": 0.0015121518,
    "M4": 0.1610228013, "MN4": 0.1595106495, "MS4": 0.1638447340,
    "M3": 0.1207671010,
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def scalar_metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(model) & np.isfinite(reference)
    if mask.sum() < 3:
        raise ValueError("fewer than three paired finite samples")
    m, r = model[mask], reference[mask]
    raw_difference = m - r
    mc, rc = m - m.mean(), r - r.mean()
    centered_difference = mc - rc
    crmse = float(np.sqrt(np.mean(centered_difference ** 2)))
    scale = float(np.std(r))
    return {
        "n": int(mask.sum()), "bias": float(np.mean(raw_difference)),
        "mae": float(np.mean(np.abs(centered_difference))),
        "raw_unaligned_mae": float(np.mean(np.abs(raw_difference))),
        "centered_rmse": crmse, "normalized_centered_rmse": crmse / scale if scale > 0 else math.nan,
        "correlation": float(np.corrcoef(m, r)[0, 1]), "reference_std": scale,
        "alignment": "common-period mean removed before MAE and centered RMSE; bias is the disclosed raw offset",
    }


def vector_metrics(mu: np.ndarray, mv: np.ndarray, ou: np.ndarray, ov: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(mu) & np.isfinite(mv) & np.isfinite(ou) & np.isfinite(ov)
    if mask.sum() < 3:
        raise ValueError("fewer than three paired current vectors")
    zm = mu[mask] + 1j * mv[mask]
    zo = ou[mask] + 1j * ov[mask]
    dm = zm - zm.mean(); do = zo - zo.mean()
    denom = math.sqrt(float(np.vdot(dm, dm).real * np.vdot(do, do).real))
    cc = np.vdot(do, dm) / denom if denom else complex(math.nan, math.nan)
    vector_rmse = float(np.sqrt(np.mean(np.abs(zm - zo) ** 2)))
    reference_rms = float(np.sqrt(np.mean(np.abs(zo) ** 2)))
    return {
        "n": int(mask.sum()),
        "east_bias": float(np.mean(mu[mask] - ou[mask])),
        "north_bias": float(np.mean(mv[mask] - ov[mask])),
        "vector_rmse": vector_rmse,
        "reference_vector_rms": reference_rms,
        "normalized_vector_rmse": vector_rmse / reference_rms if reference_rms > 0 else math.nan,
        "complex_correlation_magnitude": float(abs(cc)),
        "complex_correlation_phase_deg": float(np.degrees(np.angle(cc))),
    }


def resolvability(names: list[str], duration_hours: float) -> tuple[list[str], list[dict[str, Any]]]:
    rayleigh = 1.0 / duration_hours
    known = [n.upper() for n in names if n.upper() in FREQUENCIES_CPH]
    resolved, unresolved = [], []
    # Input order is the scientific priority order. Retain the first member of
    # each unresolved cluster and label later aliases rather than dropping both.
    for n in known:
        nearest = min(
            ((abs(FREQUENCIES_CPH[n] - FREQUENCIES_CPH[m]), m) for m in resolved),
            default=(math.inf, None),
        )
        if nearest[0] < rayleigh:
            unresolved.append({
                "constituent": n, "nearest": nearest[1],
                "frequency_separation_cph": nearest[0], "rayleigh_cph": rayleigh,
                "interpretation": "not independently resolvable; retained constituent is a cluster representative",
            })
        else:
            resolved.append(n)
    return resolved, unresolved


def harmonic_fit(time: pd.Series, values: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    t = pd.to_datetime(time, utc=True)
    hours = (t - t.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    valid = np.isfinite(values)
    cols = [np.ones(valid.sum())]
    for n in names:
        w = 2 * np.pi * FREQUENCIES_CPH[n]
        cols += [np.cos(w * hours[valid]), np.sin(w * hours[valid])]
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), values[valid], rcond=None)
    out = {}
    for i, n in enumerate(names):
        a, b = float(beta[1 + 2 * i]), float(beta[2 + 2 * i])
        out[n] = {"amplitude": math.hypot(a, b), "phase_deg": float(np.degrees(np.arctan2(b, a)) % 360), "cos": a, "sin": b}
    return out


def ellipse_from_uv(u: dict[str, dict[str, float]], v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for n in sorted(set(u) & set(v)):
        cu = complex(u[n]["cos"], -u[n]["sin"])
        cv = complex(v[n]["cos"], -v[n]["sin"])
        rp = 0.5 * (cu + 1j * cv)
        rm = 0.5 * (cu - 1j * cv)
        out[n] = {
            "major": abs(rp) + abs(rm), "minor": abs(rp) - abs(rm),
            "inclination_deg": float((0.5 * np.degrees(np.angle(rp) - np.angle(rm))) % 180),
            "phase_deg": float((-0.5 * np.degrees(np.angle(rp) + np.angle(rm))) % 360),
        }
    return out


def common_frame(path: Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [x for x in ["time"] + required if x not in df]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time")
    if not df["time"].is_monotonic_increasing:
        raise ValueError("time is not monotonic")
    return df


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lineage(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "executable_sha256", "input_bundle_sha256", "model_output_sha256",
        "observation_manifest_sha256", "station_mapping_sha256",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"validation lineage is missing fields: {missing}")
    for key in required:
        digest = str(value[key]).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"validation lineage {key} is not a SHA-256 digest")
    return value


def plot_water(df: pd.DataFrame, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    for col, label in [("model", "FVCOM"), ("observed", "NOAA observed"), ("predicted", "NOAA predicted")]:
        if col in df:
            ax.plot(df["time"], df[col] - df[col].mean(), label=label, lw=0.8)
    ax.set_ylabel("mean-removed water level (m)"); ax.set_title(title); ax.legend(ncol=3); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(output, dpi=150); plt.close(fig)


def plot_current(df: pd.DataFrame, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, m, o, lab in [(axes[0], "model_u", "observed_u", "east"), (axes[1], "model_v", "observed_v", "north")]:
        ax.plot(df["time"], df[m], label="FVCOM", lw=0.8); ax.plot(df["time"], df[o], label="NOAA depth mean", lw=0.8)
        ax.set_ylabel(f"{lab} (m/s)"); ax.legend()
    axes[0].set_title(title); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def validate(water_files: list[Path], current_files: list[Path], output_dir: Path, report_path: Path,
             constituents: list[str], lineage_path: Path,
             station_inventory_path: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage = load_lineage(lineage_path)
    water_results, current_results = [], []
    for path in water_files:
        df = common_frame(path, ["model", "observed", "predicted"])
        finite = df.dropna(subset=["model", "observed", "predicted"])[["time", "model", "observed", "predicted"]]
        if len(finite) < 3:
            raise ValueError(f"insufficient water-level overlap: {path}")
        duration = (finite["time"].iloc[-1] - finite["time"].iloc[0]).total_seconds() / 3600
        resolved, unresolved = resolvability(constituents, duration)
        datum_offsets = {"observed": float((finite["model"] - finite["observed"]).mean()), "predicted": float((finite["model"] - finite["predicted"]).mean())}
        metrics = {}
        harmonics = {}
        for ref in ("observed", "predicted"):
            metrics[ref] = scalar_metrics(finite["model"].to_numpy(), finite[ref].to_numpy())
            harmonics[ref] = {
                "model": harmonic_fit(finite["time"], finite["model"].to_numpy(), resolved),
                "reference": harmonic_fit(finite["time"], finite[ref].to_numpy(), resolved),
            }
        image = output_dir / f"{path.stem}_water.png"; plot_water(finite, image, path.stem)
        water_results.append({"station": path.stem, "station_id": path.stem.split("_", 1)[0], "source": str(path), "source_sha256": file_sha256(path), "coverage_start": finite["time"].iloc[0].isoformat(), "coverage_end": finite["time"].iloc[-1].isoformat(), "datum_alignment": "independent common-period mean removal; not a datum conversion", "model_minus_reference_mean_m": datum_offsets, "metrics": metrics, "resolved_constituents": resolved, "unresolved": unresolved, "harmonics": harmonics, "plot": image.name, "total_observation_scoring": False, "prediction_scoring": True})
    for path in current_files:
        df = common_frame(path, ["model_u", "model_v", "observed_u", "observed_v"])
        finite = df.dropna(subset=["model_u", "model_v", "observed_u", "observed_v"])
        if len(finite) < 3:
            raise ValueError(f"insufficient current overlap: {path}")
        duration = (finite["time"].iloc[-1] - finite["time"].iloc[0]).total_seconds() / 3600
        resolved, unresolved = resolvability(constituents, duration)
        mh_u = harmonic_fit(finite["time"], finite["model_u"].to_numpy(), resolved)
        mh_v = harmonic_fit(finite["time"], finite["model_v"].to_numpy(), resolved)
        oh_u = harmonic_fit(finite["time"], finite["observed_u"].to_numpy(), resolved)
        oh_v = harmonic_fit(finite["time"], finite["observed_v"].to_numpy(), resolved)
        image = output_dir / f"{path.stem}_current.png"; plot_current(finite, image, path.stem)
        current_results.append({"station": path.stem, "station_id": path.stem.split("_", 1)[0], "source": str(path), "source_sha256": file_sha256(path), "metrics": vector_metrics(finite["model_u"].to_numpy(), finite["model_v"].to_numpy(), finite["observed_u"].to_numpy(), finite["observed_v"].to_numpy()), "resolved_constituents": resolved, "unresolved": unresolved, "model_ellipses": ellipse_from_uv(mh_u, mh_v), "observed_ellipses": ellipse_from_uv(oh_u, oh_v), "plot": image.name})
    station_policy_evidence = None
    missing_eligible_stations: list[dict[str, str]] = []
    required_roles = {"water_level"}
    if station_inventory_path is not None:
        inventory = json.loads(station_inventory_path.read_text(encoding="utf-8-sig"))
        eligible = {
            (str(item["id"]), str(item["role"]))
            for item in inventory.get("stations", []) if item.get("eligible")
        }
        required_roles |= {role for _, role in eligible}
        delivered = (
            {(item["station_id"], "water_level") for item in water_results}
            | {(item["station_id"], "current") for item in current_results}
        )
        missing_eligible_stations = [
            {"station_id": station_id, "role": role}
            for station_id, role in sorted(eligible - delivered)
        ]
        station_policy_evidence = {
            "path": str(station_inventory_path),
            "sha256": file_sha256(station_inventory_path),
            "counts": inventory.get("counts"),
            "eligible_station_ids": [
                {"station_id": station_id, "role": role} for station_id, role in sorted(eligible)
            ],
            "excluded_stations": [
                {
                    "station_id": str(item.get("id")), "role": str(item.get("role")),
                    "reason": str(item.get("exclusion_reason")),
                }
                for item in inventory.get("stations", []) if not item.get("eligible")
            ],
            "missing_eligible_stations": missing_eligible_stations,
        }
    completed_roles = ({"water_level"} if water_results else set()) | ({"current"} if current_results else set())
    workflow = "validation_complete" if required_roles <= completed_roles and not missing_eligible_stations else "invalid"
    threshold_values = {
        "prediction_normalized_centered_rmse_max": 0.25,
        "prediction_correlation_min": 0.90,
        "current_normalized_vector_rmse_max": 0.50,
        "current_complex_correlation_min": 0.80,
    }
    water_pass = all(
        item["metrics"]["predicted"]["normalized_centered_rmse"] <= threshold_values["prediction_normalized_centered_rmse_max"]
        and item["metrics"]["predicted"]["correlation"] >= threshold_values["prediction_correlation_min"]
        for item in water_results
    )
    current_pass = all(
        item["metrics"]["normalized_vector_rmse"] <= threshold_values["current_normalized_vector_rmse_max"]
        and item["metrics"]["complex_correlation_magnitude"] >= threshold_values["current_complex_correlation_min"]
        for item in current_results
    )
    assessment = (
        "invalid" if workflow == "invalid" else
        "diagnostic-pass" if water_pass and current_pass else "diagnostic-advisory"
    )
    result = {
        "schema": "fvcom_tidal_validation_v1", "generated_at": now(),
        "workflow_status": workflow, "scientific_assessment": assessment,
        "threshold_policy": "informational_for_initial_regional_accepted_and_fresh_runs; never a stability-tuning or completion gate",
        "informational_thresholds": threshold_values,
        "lineage": lineage, "lineage_manifest": str(lineage_path),
        "lineage_manifest_sha256": file_sha256(lineage_path),
        "station_policy_evidence": station_policy_evidence,
        "water_level": water_results, "current": current_results,
    }
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    sections = []
    for item in water_results:
        image_reference = Path(os.path.relpath(output_dir / item["plot"], report_path.parent)).as_posix()
        sections.append(f"<h2>Water level: {html.escape(item['station'])}</h2><p>{html.escape(item['datum_alignment'])}</p><img src='{html.escape(image_reference)}' style='max-width:100%'><h3>Metrics and datum offsets</h3><pre>{html.escape(json.dumps({'offsets': item['model_minus_reference_mean_m'], 'metrics': item['metrics']}, indent=2))}</pre><details><summary>Resolvable harmonics and unresolved aliases</summary><pre>{html.escape(json.dumps({'resolved': item['resolved_constituents'], 'unresolved': item['unresolved'], 'harmonics': item['harmonics']}, indent=2))}</pre></details>")
    for item in current_results:
        image_reference = Path(os.path.relpath(output_dir / item["plot"], report_path.parent)).as_posix()
        sections.append(f"<h2>Current: {html.escape(item['station'])}</h2><img src='{html.escape(image_reference)}' style='max-width:100%'><h3>Vector metrics</h3><pre>{html.escape(json.dumps(item['metrics'], indent=2))}</pre><details><summary>Tidal-current ellipses and unresolved aliases</summary><pre>{html.escape(json.dumps({'resolved': item['resolved_constituents'], 'unresolved': item['unresolved'], 'model_ellipses': item['model_ellipses'], 'observed_ellipses': item['observed_ellipses']}, indent=2))}</pre></details>")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    station_section = (
        "<h2>Station eligibility and exclusions</h2><pre>"
        + html.escape(json.dumps(station_policy_evidence, indent=2)) + "</pre>"
        if station_policy_evidence is not None else ""
    )
    report_path.write_text("<!doctype html><html><head><meta charset='utf-8'><title>FVCOM tidal validation</title><style>body{font-family:system-ui;margin:2rem;max-width:1200px}pre{background:#f5f5f5;padding:1rem;overflow:auto}details{margin:1rem 0}</style></head><body><h1>FVCOM tidal validation</h1><p>Workflow: <b>" + workflow + "</b>; scientific assessment: <b>" + assessment + "</b>.</p><p>Thresholds are informational for initial regional accepted and fresh cases and never control stability tuning or workflow completion. NOAA/model mean offsets are disclosed and are not datum conversions.</p><h2>Frozen lineage</h2><pre>" + html.escape(json.dumps(lineage, indent=2)) + "</pre>" + station_section + "".join(sections) + "</body></html>", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="FVCOM tide and depth-mean-current validation")
    p.add_argument("validate", nargs="?")
    p.add_argument("--water", action="append", default=[]); p.add_argument("--current", action="append", default=[])
    p.add_argument("--output-dir", required=True); p.add_argument("--report", required=True)
    p.add_argument("--lineage", required=True)
    p.add_argument("--station-inventory")
    p.add_argument("--constituents", default=",".join(FREQUENCIES_CPH))
    args = p.parse_args()
    result = validate([Path(x) for x in args.water], [Path(x) for x in args.current], Path(args.output_dir), Path(args.report), [x.strip().upper() for x in args.constituents.split(",") if x.strip()], Path(args.lineage), Path(args.station_inventory) if args.station_inventory else None)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
