"""Compare four audited tide, wind, river, and wind-plus-river production attempts."""

import argparse
import html
import json
import tempfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paired_river_budget as river_budget
import paired_wind_validation as wind_validation
from factorial_fields import CONTRASTS, MEMBERS, effects, spatial
from fvcom_profiles import COMBINED_MODE, FACTORIAL_PROFILE, RIVER_MODE, WIND_MODE, WIND_PROFILE
from run_case_validation import read, sha256, verify_prerequisites


def records(project, attempt):
    project = project.resolve()
    attempt = (project / attempt).resolve()
    grid_case = read(attempt / "attempt_manifest.json")["grid_case"]
    return verify_prerequisites(
        argparse.Namespace(
            project=project,
            attempt=attempt,
            grid_case=grid_case,
            input_freeze=None,
            production_audit=None,
            station_netcdf=None,
            observation_manifest=None,
            forcing_manifest=None,
        )
    )


def check_experiment(cases):
    for name, mode in zip(MEMBERS, ("barotropic_tide", WIND_MODE, RIVER_MODE, COMBINED_MODE)):
        case = cases[name]
        allowed = (FACTORIAL_PROFILE, WIND_PROFILE) if name in ("tide", "wind") else (FACTORIAL_PROFILE,)
        if case["request"].get("profile") not in allowed or case["request"]["mode"] != mode:
            raise ValueError(f"Incorrect factorial profile or forcing for {name}")
        tide = cases["tide"]
        if any(
            case[k] != tide[k] for k in ("run_start", "start", "end", "station_keys", "executable_hash")
        ):
            raise ValueError(f"Changed comparison interval, executable, or station order for {name}")
        if sha256(case["inventory_path"]) != sha256(tide["inventory_path"]):
            raise ValueError("Four-member comparisons require the same NOAA observation inventory")
    wind_validation.check_pair(cases["tide"], cases["wind"])
    river_budget.check_pair(cases["tide"], cases["river"])
    river_budget.check_pair(cases["wind"], cases["combined"])
    a, b = (cases[n] for n in ("river", "combined"))
    manifest = "river_manifest.json"
    river_files = read(a["input_dir"] / manifest)["files"]
    for name in (manifest, river_files["namelist"]["path"], river_files["netcdf"]["path"]):
        if a["freeze"]["files"][name] != b["freeze"]["files"][name]:
            raise ValueError("Different river source products in the two river treatments")


def common_scores(levels, keys):
    rows = []
    for reference in ("predicted", "observed", "residual"):
        needed = ["observed", "predicted"] if reference == "residual" else [reference]
        common = levels[list(MEMBERS) + needed].dropna()
        if common.empty:
            if reference == "predicted":
                raise ValueError("Missing astronomical predictions on common model support")
            continue
        if len(common) < 3 or not np.isfinite(common).all().all():
            raise ValueError("Insufficient or nonfinite common NOAA samples")
        actual = common.observed - common.predicted if reference == "residual" else common[reference]
        for member in MEMBERS:
            model = common[member] - common.predicted if reference == "residual" else common[member]
            metrics = wind_validation.score(model, actual)
            if reference == "residual":
                metrics["peak_model_residual_m"] = float(model.max())
                metrics["peak_observed_residual_m"] = float(actual.max())
                metrics["peak_timing_error_hours"] = (
                    model.idxmax() - actual.idxmax()
                ).total_seconds() / 3600
            rows.append(dict(**keys, member=member, reference=reference, **metrics))
    return rows


def stations(cases, output):
    raw = [wind_validation.elevation(cases[name]) for name in MEMBERS]
    if any(
        not frame.index.equals(raw[0].index) or not frame.columns.equals(raw[0].columns)
        for frame in raw[1:]
    ):
        raise ValueError("FVCOM station clocks or ordering differ")
    c = cases["tide"]
    inventory = {str(row["id"]): row for row in read(c["inventory_path"])["stations"]}
    gauges = [sid for sid, role in c["station_keys"] if role == "water_level"]
    scores, response, missing = [], [], []
    fig, axes = plt.subplots(
        len(gauges),
        1,
        figsize=(11, max(5, 2.4 * len(gauges))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    data_dir = output / "stations"
    data_dir.mkdir()
    for sid, ax in zip(gauges, axes.flat):
        files = [
            cases[k]["observation_path"].parent / "water_level" / f"{sid}_noaa_waterlevel.csv"
            for k in MEMBERS
        ]
        h = sha256(files[0])
        if any(sha256(p) != h for p in files[1:]):
            raise ValueError(f"Different NOAA records at {sid}")
        obs = pd.read_csv(files[0])
        obs.index = pd.to_datetime(obs.pop("time"), utc=True)
        if not obs.index.is_unique:
            raise ValueError(f"Duplicate NOAA samples at {sid}")
        levels = pd.DataFrame({k: f[sid] for k, f in zip(MEMBERS, raw)})
        levels = levels.loc[(levels.index >= c["start"]) & (levels.index < c["end"])].join(
            obs[["observed", "predicted"]]
        )
        scores.extend(common_scores(levels, dict(station=sid, reference_sha256=h)))
        if not levels.observed.notna().any():
            row = inventory[sid]
            if row.get("observation_eligible") is not False or not row.get(
                "observation_absence_evidence"
            ):
                raise ValueError(f"Missing NOAA observations without archived evidence for {sid}")
            missing.append(sid)
        for (name, (label, _)), diff in zip(CONTRASTS.items(), effects(levels[k] for k in MEMBERS)):
            levels[name] = diff
            response.append(
                dict(
                    station=sid,
                    contrast=name,
                    samples=len(diff),
                    signed_mean_m=diff.mean(),
                    minimum_m=diff.min(),
                    maximum_m=diff.max(),
                    rms_m=np.sqrt(np.mean(diff**2)),
                )
            )
            ax.plot(levels.index, diff, label=label, linewidth=0.85)
        levels["noaa_residual"] = levels.observed - levels.predicted
        levels.to_csv(data_dir / f"{sid}.csv", index_label="time_utc")
        ax.set_ylabel(f"{sid}\nLevel (m)")
        ax.set_title(inventory[sid].get("name", sid), loc="left", fontsize=9)
    axes[0, 0].legend(ncols=3, fontsize=7)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1, 0].set_xlabel("UTC")
    fig.savefig(output / "gauge_contrasts.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(scores).to_csv(output / "common_noaa_metrics.csv", index=False)
    pd.DataFrame(response).to_csv(output / "gauge_effects.csv", index=False)
    current_ids = [sid for sid, role in c["station_keys"] if role == "current"]
    return dict(
        water_level_gauges=gauges,
        prediction_only=missing,
        metrics="common_noaa_metrics.csv",
        current_station_ids=current_ids,
        eligible_current_profiles=len(current_ids),
        current_metrics=currents(cases, current_ids, output),
    )


def currents(cases, ids, output):
    if not ids:
        return None
    # Regional tables already interpolate FVCOM vectors to screened NOAA profile midpoints.
    rows = []
    for sid in ids:
        frames = []
        for k in MEMBERS:
            case = cases[k]
            grid_case = read(case["attempt"] / "attempt_manifest.json")["grid_case"]
            base = case["project"] / "analysis" / grid_case / ("validation_" + case["attempt"].name)
            lineage = read(base / "validation_lineage.json")
            if lineage["production_audit_sha256"] != sha256(case["audit_path"]):
                raise ValueError("Current table belongs to a different production audit")
            table = base / "tables" / f"{sid}_current.csv"
            manifest = read(base / "validation_tables_manifest.json")
            product = next(
                row
                for row in manifest["products"]
                if row["station_id"] == sid and row["role"] == "current"
            )
            if sha256(table) != product["sha256"]:
                raise ValueError(f"Changed validated current table at {sid}")
            data = pd.read_csv(table)
            data.index = pd.to_datetime(data.pop("time"), utc=True)
            if not data.index.is_unique:
                raise ValueError(f"Duplicate NOAA current times at {sid}")
            frames.append(data)
        obs = frames[0][["observed_u", "observed_v"]]
        if any(not obs.equals(frame[obs.columns]) for frame in frames[1:]):
            raise ValueError(f"Four NOAA current reference series disagree at {sid}")
        common = obs.join(
            [f[["model_u", "model_v"]].add_prefix(k + "_") for f, k in zip(frames, MEMBERS)]
        ).dropna()
        if len(common) < 3 or not np.isfinite(common).all().all():
            raise ValueError("Insufficient valid common NOAA current profiles")
        for k in MEMBERS:
            u = common[k + "_model_u"] - common.observed_u
            v = common[k + "_model_v"] - common.observed_v
            rows.append(
                dict(
                    station=sid,
                    member=k,
                    samples=len(common),
                    mean_u_error_m_s=u.mean(),
                    mean_v_error_m_s=v.mean(),
                    vector_rmse_m_s=np.sqrt(np.mean(u * u + v * v)),
                )
            )
        common.to_csv(output / "stations" / f"{sid}_current.csv", index_label="time_utc")
    filename = "common_noaa_current_metrics.csv"
    pd.DataFrame(rows).to_csv(output / filename, index=False)
    return filename


def report(output, summary):
    links = [
        ("Common NOAA metrics", "common_noaa_metrics.csv"),
        ("Gauge responses", "gauge_effects.csv"),
        ("Native wetness at gauge cells and vertices", "gauge_wetness.csv"),
        ("All wet-mask-aware native fields", "native_contrasts.nc"),
        ("Native extrema", "native_timeseries_extrema.csv"),
        ("Conditional river budget without wind", "river_without_wind/budget.json"),
        ("Conditional river budget with wind", "river_with_wind/budget.json"),
    ]
    if summary["stations"]["current_metrics"]:
        links.append(("Common NOAA vector metrics", summary["stations"]["current_metrics"]))
    budgets = summary["river_budgets"]
    metrics = pd.read_csv(output / "common_noaa_metrics.csv")
    score_table = metrics.loc[metrics.reference == "observed"].pivot(
        index="station", columns="member", values="rmse_m"
    )
    observed_table = (
        score_table.reindex(columns=MEMBERS).to_html(float_format=lambda value: f"{value:.3f}")
        if len(score_table)
        else "<p>No eligible total-level observations.</p>"
    )
    rows = "".join(
        f"<tr><th>{html.escape(n)}</th><td>{b['applied_m3']:,.0f}</td>"
        f"<td>{b['excess_export_m3']:,.0f}</td><td>{b['excess_storage_change_m3']:,.0f}</td>"
        f"<td>{b['residual_fraction']:.2%}</td></tr>"
        for n, item in budgets.items()
        for b in [item["analysis"]]
    )
    pics = "".join(
        f'<figure><img src="{n}_mean_fields.png" alt="{html.escape(label)} mean fields"></figure>'
        for n, (label, _) in CONTRASTS.items()
    )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Wind and river factorial validation</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1em}}img{{max-width:100%}}
td,th{{padding:.35rem;text-align:right}}figure{{margin:1rem 0}}</style><main>
<h1>Wind and river effects in the Columbia estuary</h1>
<p>T is the pure tide control. W adds surface wind stress, atmospheric pressure, the offshore sea-surface
height anomaly and the inverse-barometer correction at the open boundary. R adds river discharge to T,
and WR applies both treatments. The interaction is (WR − W) − (R − T).</p>
<p>{summary['spatial']['analysis_start']} to {summary['spatial']['analysis_end_exclusive']} UTC,
end excluded. The comparison includes {summary['spatial']['timestamps']} three-hour native fields and
{len(summary['stations']['water_level_gauges'])} six-minute tide gauges.
{summary['stations']['eligible_current_profiles']} current stations met the inventory criteria.</p>
<p>Gauge scores retain raw model-minus-reference mean offset. Centered RMSE removes that offset on common samples.
Native vector differences are model-x/y depth means on cells wet in all four members at each timestamp.</p>
<p>At mapped gauges the three-hour snapshots include {summary['spatial']['gauge_dry_cell_samples']} dry-cell
samples and {summary['spatial']['gauge_samples_with_any_dry_vertex']} samples with one or more dry vertices.
Between snapshots gauge wetness is unknown. Prediction-only gauges: {', '.join(summary['stations']['prediction_only']) or 'none'}.</p>
<ul>{''.join(f'<li><a href="{p}">{label}</a></li>' for label,p in links)}</ul>
<h2>Total observed water-level RMSE (m)</h2>{observed_table}
<h2>Gauge changes</h2><img src="gauge_contrasts.png" alt="Conditional water level effects at NOAA gauges">
<h2>Analysis-period river volume budget</h2><table><tr><th>Background</th><th>River input (m³)</th>
<th>Additional export (m³)</th><th>Additional storage (m³)</th><th>Residual / input</th></tr>{rows}</table>
<p>Transport uses output-sampled upstream-cell velocity and edge depth. The signed closure error includes
the effects of spatial and temporal quadrature.</p><h2>Mean spatial effects</h2>{pics}</main></html>"""
    (output / "index.html").write_text(content, encoding="utf-8")


def compare(cases, geometry, output):
    check_experiment(cases)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "results"
        stage.mkdir()
        gauges = stations(cases, stage)
        fields = spatial(cases, stage)
        budgets = {}
        for label, a, b in (
            ("river_without_wind", "tide", "river"),
            ("river_with_wind", "wind", "combined"),
        ):
            result = river_budget.compare(cases[a], cases[b], geometry, stage / label)
            budgets[label] = result["budgets"]
        lineage = {
            k: dict(
                project=str(c["project"]),
                attempt=str(c["attempt"]),
                freeze_sha256=sha256(c["freeze_path"]),
                production_audit_sha256=sha256(c["audit_path"]),
                executable_sha256=c["executable_hash"],
                station_mapping_sha256=sha256(c["mapping_path"]),
                inventory_sha256=sha256(c["inventory_path"]),
            )
            for k, c in cases.items()
        }
        result = dict(
            schema="fvcom_factorial_validation_v1",
            status="complete",
            members=lineage,
            contrasts={k: v[0] for k, v in CONTRASTS.items()},
            geometry_sha256=sha256(geometry),
            stations=gauges,
            spatial=fields,
            river_budgets=budgets,
            scientific_limits=[
                "Wind treatments jointly include surface stress, atmospheric pressure, offshore sea-surface height anomaly and the boundary inverse-barometer correction.",
                "Common wet support can change at each output time. Vector magnitudes use both velocity components.",
                "Six-minute section velocity and three-hour storage are output-sampled budget estimates.",
            ],
        )
        report(stage, result)
        result["products"] = {
            path.relative_to(stage).as_posix(): sha256(path)
            for path in stage.rglob("*")
            if path.is_file()
        }
        (stage / "factorial_validation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MEMBERS:
        for kind in ("project", "attempt"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {k: records(getattr(args, k + "_project"), getattr(args, k + "_attempt")) for k in MEMBERS}
    compare(cases, args.geometry, args.output)
    print(args.output / "factorial_validation.json")
