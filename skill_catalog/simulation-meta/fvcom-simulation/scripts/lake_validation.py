"""Validate a freshwater FVCOM hindcast or NOAA model reference against independent lake observations."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lake_native_io import (at_depth, compact, digest, file_record, on_clock, read_stations,
                            require, thermocline)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def publish(source, destination):
    for attempt in range(4):
        try:
            source.rename(destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 3:
                raise
            time.sleep(0.25 * 2**attempt)


def epoch(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.timestamp())


def epochs(values):
    return pd.to_datetime(values, utc=True).to_numpy(dtype="datetime64[s]").astype("i8")


def clock_coverage(times, start, end, step):
    require(step > 0 and end > start and (end-start) % step == 0, "invalid requested cadence or interval")
    expected = np.arange(start, end, step)
    times = np.asarray(times)
    nominal = start + np.rint((times-start)/step).astype("i8")*step
    use = (nominal >= start) & (nominal < end)
    native, nominal = times[use], nominal[use]
    offset = np.abs(native-nominal)
    # One archived LSOFS station record per day is encoded one second early.
    # Keep the actual times for all physical comparisons. Match only the schedule here.
    tolerance = min(1, (step-1)//2)
    accepted = nominal[offset <= tolerance]
    missing = np.setdiff1d(expected, accepted)
    off_clock = int(np.sum(offset > tolerance))
    duplicates = len(accepted)-len(np.unique(accepted))
    return {"expected": len(expected), "available": len(native), "missing": len(missing),
            "off_clock": off_clock, "duplicate_slots": duplicates, "jittered_records": int(np.sum(offset > 0)),
            "max_offset_seconds": int(offset.max()) if len(offset) else None, "tolerance_seconds": tolerance,
            "cadence_seconds": step, "complete": not (len(missing) or off_clock or duplicates)}


def score(predicted, observed):
    a, b = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if not len(a):
        return {"count": 0}
    error = a-b
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and min(a.std(), b.std()) > 1e-9 else None
    return {"count": len(a), "bias": float(error.mean()), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(abs(error).mean()), "correlation": corr}


def geometry(project, mapping):
    role = load(project/"agents/01_grid/role_manifest.json")
    require(role["status"] == "ready" and role["worker"] == "grid", "grid role incomplete")
    files = {}
    for key in ("runtime_grd", "preconfiguration", "metrics"):
        path = project/role["artifacts"][key]
        require(digest(path) == role["hashes"][key], f"changed grid role artifact {key}")
        files[key] = path
    require(role["hashes"]["runtime_grd"] == mapping["grd_sha256"], "station and grid hashes differ")
    meta = load(files["preconfiguration"])
    require(meta["geometry"] == "spherical" and meta["files"]["metrics"]["sha256"] == role["hashes"]["metrics"],
            "spherical area provenance differs")
    return files["metrics"]


def observations(directory):
    inventory = load(directory/"observations.json")
    frames, gauges = [], {}
    lake = None
    for record in inventory:
        site, variable = str(record["station"]), record["variable"]
        path = directory/(site+".csv")
        require(digest(path) == record["series_sha256"], f"observation hash differs for {site}")
        data = pd.read_csv(path)
        require(variable in data and not data.utc.duplicated().any(), f"invalid observation series for {site}")
        if variable == "elevation_m_IGLD85":
            gauges[site] = data
        elif variable == "water_temperature_c":
            frames.append(pd.DataFrame({"station": site, "utc": data.utc, "depth_m": record["sensor_depth_m"],
                                        "temperature_c": data[variable]}))
        elif variable == "lake_mean_surface_temperature_c":
            require(lake is None, "multiple independent lake surface products")
            lake = data.rename(columns={"utc": "date", variable: "observed"})
    require(gauges and frames and lake is not None, "independent gauge, buoy or GLSEA product missing")
    extra = directory/"supplemental_temperatures.csv"
    require(digest(extra) == load(directory/"supplemental_temperatures.json")["series_sha256"],
            "supplemental observations hash differs")
    frames.append(pd.read_csv(extra, dtype={"station": str}))
    temp = pd.concat(frames, ignore_index=True)
    temp["epoch"] = epochs(temp.utc)
    require(not temp.duplicated(["station", "epoch", "depth_m"]).any(), "duplicate station/time/depth observations")
    return gauges, temp, lake


def station_metrics(model, namespace, gauges, temp, start, end, cadence, offset):
    lookup = {name: i for i, name in enumerate(model["ids"])}
    scores, pairs, omitted = [], [], []
    for site, table in gauges.items():
        key = namespace+site
        if key not in lookup:
            omitted.append(site)
            continue
        i = lookup[key]
        clock = epochs(table.utc)
        inside = (clock >= start) & (clock < end)
        dates = clock[inside]
        observed = table.elevation_m_IGLD85.to_numpy()[inside]
        predicted = on_clock(model["time"], model["zeta"][:, i], dates, cadence)
        valid = np.isfinite(observed) & np.isfinite(predicted)
        if valid.any():
            anomaly = predicted-np.mean(predicted[valid])
            observed_anomaly = observed-np.mean(observed[valid])
            scores.append(dict(station=site, variable="common_mean_removed_elevation_m", **score(anomaly, observed_anomaly)))
        if offset is not None:
            predicted = predicted+offset
            label = "elevation_m_IGLD85"
            scores.append(dict(station=site, variable="absolute_elevation_m_IGLD85", **score(predicted, observed)))
        elif valid.any():
            predicted, observed, label = anomaly, observed_anomaly, "common_mean_removed_elevation_m"
        else:
            continue
        pairs.append(pd.DataFrame(dict(station=site, variable=label, epoch=dates, depth_m=np.nan,
                                       model=predicted, observed=observed)))
    window = temp[temp.epoch.between(start, end-1)]
    for (site, depth), group in window.groupby(["station", "depth_m"], dropna=False):
        key = namespace+site
        if key not in lookup:
            omitted.append(site)
            continue
        dates = group.epoch.to_numpy()
        predicted, outer = at_depth(model, lookup[key], dates, depth, cadence)
        observed = group.temperature_c.to_numpy()
        scores.append(dict(station=site, variable="water_temperature_degC", depth_m=depth,
                           outer_sigma_half_layer_samples=int(outer.sum()), **score(predicted, observed)))
        pairs.append(pd.DataFrame(dict(station=site, variable="temperature_degC", epoch=dates, depth_m=depth,
                                       model=predicted, observed=observed)))
    require(pairs and any(row.get("count", 0) for row in scores), "no overlapping independent observations")
    return scores, pd.concat(pairs, ignore_index=True), sorted(set(omitted))


def profile_metrics(paired, scores):
    tables = []
    temp = paired[paired.variable.eq("temperature_degC") & paired.depth_m.notna()]
    for site, group in temp.groupby("station"):
        if group.depth_m.nunique() < 3:
            continue
        model = group.pivot(index="epoch", columns="depth_m", values="model")
        obs = group.pivot(index="epoch", columns="depth_m", values="observed").reindex_like(model)
        a, b = model.to_numpy(), obs.to_numpy()
        valid = np.isfinite(a) & np.isfinite(b)
        ma = thermocline(model.columns.to_numpy(), np.where(valid, a, np.nan))
        ob = thermocline(model.columns.to_numpy(), np.where(valid, b, np.nan))
        scores.append(dict(station=site, variable="thermocline_depth_m",
                           model_only=int(np.sum(np.isfinite(ma) & ~np.isfinite(ob))),
                           observed_only=int(np.sum(~np.isfinite(ma) & np.isfinite(ob))), **score(ma, ob)))
        tables.append(pd.DataFrame(dict(station=site, epoch=model.index, model_depth_m=ma, observed_depth_m=ob)))
    return pd.concat(tables, ignore_index=True) if tables else None


def reference_metrics(model, reference, start, end, cadence):
    result = []
    selected = (reference["time"] >= start) & (reference["time"] < end)
    times = reference["time"][selected]
    for j, site in enumerate(reference["ids"]):
        i = model["ids"].index("R"+site)
        require(np.allclose(model["sigma"][:, i], reference["sigma"][:, j], rtol=0, atol=1e-6),
                f"different native sigma coordinates at {site}")
        eta = on_clock(model["time"], model["zeta"][:, i], times, cadence)
        result.append(dict(station=site, variable="elevation_m", **score(eta, reference["zeta"][selected, j])))
        temp = on_clock(model["time"], model["temp"][:, :, i], times, cadence)
        for layer in range(temp.shape[1]):
            result.append(dict(station=site, variable="temperature_degC", sigma_layer=layer+1,
                               **score(temp[:, layer], reference["temp"][selected, layer, j])))
    return pd.DataFrame(result)


def daily_surface(table, lake, start, end, step):
    needed = {"epoch", "temperature_c", "wet_area_fraction"}
    require(needed.issubset(table) and not table.epoch.duplicated().any() and
            np.isfinite(table[list(needed)]).all().all() and
            table.wet_area_fraction.between(0, 1, inclusive="right").all(), "invalid compact surface records")
    table = table[table.epoch.between(start, end-1)].copy()
    table["date"] = pd.to_datetime(table.epoch, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    records = []
    for date, group in table.groupby("date"):
        day = epoch(date)
        coverage = clock_coverage(group.epoch.to_numpy(), day, day+86400, step)
        records.append(dict(date=date, model=group.temperature_c.mean(), native_records=len(group),
                            min_wet_area_fraction=group.wet_area_fraction.min(), **coverage))
    require(records, "no native temperatures inside requested window")
    return pd.DataFrame(records).merge(lake, on="date", how="left", validate="one_to_one")


def figure(out, paired, stats):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for site, rows in paired[paired.variable != "temperature_degC"].groupby("station"):
        axes[0].plot(pd.to_datetime(rows.epoch, unit="s"), rows.model-rows.observed, label=site, linewidth=.65)
    axes[0].set(ylabel="Model − observed (m)", title="Water-level residuals")
    axes[0].legend(ncol=5, fontsize=8)
    temps = stats[stats.variable == "water_temperature_degC"].dropna(subset="depth_m")
    for site, rows in temps.groupby("station"):
        axes[1].plot(rows.rmse, rows.depth_m, ".-", label=site, linewidth=1)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Temperature RMSE (°C)", ylabel="Sensor depth (m)")
    axes[1].legend(ncol=6, fontsize=8)
    fig.savefig(out/"observed_skill.png", dpi=160)
    plt.close(fig)


def validate(args, out):
    project = args.project
    request = load(project/"request.json")
    require(request["profile"] == "lake_superior_summer", "expected a lake_superior_summer project")
    project_start, project_end = (epoch(request["period"][key]) for key in ("analysis_start", "analysis_end"))
    start = epoch(args.analysis_start) if args.analysis_start else project_start
    end = epoch(args.analysis_end) if args.analysis_end else project_end
    subwindow = (start,end)!=(project_start,project_end)
    require(project_start<=start<end<=project_end and (not subwindow or args.partial),
            "a shorter analysis window must lie inside the project dates and use --partial")
    require(start % 86400 == end % 86400 == 0 and end > start, "expected a UTC-day analysis window")
    mapping_path = args.mapping or project/"agents/01_grid/stations/station_mapping.json"
    mapping = load(mapping_path)
    require(mapping["status"] == "ready" and mapping["station_location"] == "cell", "invalid station mapping")
    metrics = geometry(project, mapping)
    obs_dir = args.observations or project/"analysis/observations"
    sources = args.reference
    if sources is None:
        found = list((project/"agents/02_forcing/reference").glob("*_stations_*.nc"))
        require(len(found) == 1, "specify the archived station file with --reference")
        sources = found[0]
    require(digest(sources) == mapping["reference_sha256"], "NOAA reference differs from station spatial binding")
    ref = read_stations([sources], start, end)
    archived = args.reference_only
    cadence = args.station_seconds or (int(np.median(np.diff(ref["time"]))) if archived else request["output"]["station_seconds"])
    model = ref if archived else read_stations(args.station, start, end, cadence)
    if not archived:
        stations = mapping["stations"]
        require(model["ids"] == [row["station_id"] for row in stations], "native station order differs from mapping")
        for key, field, tolerance in [("lon", "cell_lon", 1e-4), ("lat", "cell_lat", 1e-4),
                                      ("depth", "model_depth_m", 1e-3)]:
            require(np.allclose(model[key], [row[field] for row in stations], rtol=0, atol=tolerance),
                    f"station output {key} differs from the mapped native elements")
    station_coverage = clock_coverage(model["time"], start, end, cadence)
    require(station_coverage["complete"] or args.partial, "incomplete station clock; use --partial for a diagnostic")
    datum = mapping.get("datum", {})
    offset = None
    if archived or args.model_datum == "lwd":
        require(datum.get("source") and np.isfinite(datum.get("offset_m_from_lwd_to_igld85", np.nan)), "missing LWD datum evidence")
        offset = float(datum["offset_m_from_lwd_to_igld85"])
    gauges, temperatures, lake = observations(obs_dir)
    scores, pairs, omitted = station_metrics(model, "" if archived else "O", gauges, temperatures,
                                             start, end, cadence, offset)
    require(archived or not omitted, f"independent sites missing from native station mapping: {omitted}")
    profiles = profile_metrics(pairs, scores)
    if profiles is not None:
        profiles.to_csv(out/"thermocline_pairs.csv", index=False)
    if not archived:
        reference_metrics(model, ref, start, end, cadence).to_csv(out/"model_reference_metrics.csv", index=False)
    grid_step = args.grid_seconds or (3600 if archived else request["output"]["grid_seconds"])
    surface_input = {}
    if args.fields:
        table = compact(args.fields, metrics, start, end)
        surface_input["fields"] = [file_record(path) for path in args.fields]
    else:
        receipt = load(args.lake_series.with_suffix(".json"))
        require(receipt["schema"] == "fvcom_lake_surface_series_v1" and
                receipt["metrics_sha256"] == digest(metrics) and receipt["series_sha256"] == digest(args.lake_series)
                and receipt["sources"], "compact series provenance or grid differs")
        table = pd.read_csv(args.lake_series)
        surface_input["compact_receipt"] = file_record(args.lake_series.with_suffix(".json"))
    daily = daily_surface(table, lake, start, end, grid_step)
    supported = daily[daily.complete & daily.observed.notna()]
    surface_coverage = {"expected_days": (end-start)//86400, "represented_days": len(daily),
                        "complete_observed_days": len(supported), "cadence_seconds": grid_step}
    complete = len(supported) == surface_coverage["expected_days"]
    require(complete or args.partial, "incomplete daily lake coverage; use --partial for a diagnostic")
    scores.append(dict(station="lake", variable="GLSEA_surface_temperature_degC", **score(supported.model, supported.observed)))
    daily.to_csv(out/"glsea_daily.csv", index=False)
    table.to_csv(out/"lake_surface_native.csv", index=False)
    stats = pd.DataFrame(scores)
    stats.to_csv(out/"observation_metrics.csv", index=False)
    pairs.to_csv(out/"observation_pairs.csv.gz", index=False, float_format="%.6f")
    figure(out, pairs, stats)
    receipt = {"schema": "fvcom_lake_validation_v1", "status": "complete" if complete and station_coverage["complete"] and not subwindow else "partial",
               "mode": "archived_noaa_reference" if archived else "hindcast", "analysis_start_epoch": start, "analysis_end_epoch": end,
               "project_start_epoch":project_start,"project_end_epoch":project_end,"station_time_methods":model["source_clocks"],
               "station_coverage": station_coverage, "lake_surface_coverage": surface_coverage,
               "datum_offset_m": offset, "datum_evidence": datum if offset is not None else None,
               "station_sources": [file_record(path) for path in ([sources] if archived else args.station)],
               "reference": file_record(sources), "mapping": file_record(mapping_path), "area": file_record(metrics),
               "observations": file_record(obs_dir/"observations.json"), "supplemental": file_record(obs_dir/"supplemental_temperatures.json"),
               "surface": surface_input, "unmapped_reference_observation_sites": omitted,
               "outputs": {p.name: digest(p) for p in out.iterdir()}}
    save(out/"validation_manifest.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reducer = commands.add_parser("compact", help="area-weight native wet-node upper-layer temperature")
    reducer.add_argument("--metrics", type=Path, required=True)
    reducer.add_argument("--fields", nargs="+", type=Path, required=True)
    reducer.add_argument("--start", required=True)
    reducer.add_argument("--end", required=True)
    reducer.add_argument("--out", type=Path, required=True)
    validate_cli = commands.add_parser("validate", help="score independent observations and compare the NOAA reference")
    validate_cli.add_argument("--project", type=Path, required=True)
    source = validate_cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--station", nargs="+", type=Path)
    source.add_argument("--reference-only", action="store_true")
    surface = validate_cli.add_mutually_exclusive_group(required=True)
    surface.add_argument("--fields", nargs="+", type=Path)
    surface.add_argument("--lake-series", type=Path)
    validate_cli.add_argument("--reference", type=Path)
    validate_cli.add_argument("--mapping", type=Path)
    validate_cli.add_argument("--observations", type=Path)
    validate_cli.add_argument("--model-datum", choices=["lwd", "unknown"], default="unknown")
    validate_cli.add_argument("--station-seconds", type=int)
    validate_cli.add_argument("--grid-seconds", type=int)
    validate_cli.add_argument("--partial", action="store_true")
    validate_cli.add_argument("--analysis-start", help="UTC midnight for a diagnostic subwindow inside the project")
    validate_cli.add_argument("--analysis-end", help="exclusive UTC midnight for a diagnostic subwindow")
    validate_cli.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "output exists; choose a fresh path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as work:
        staging = Path(work)/"result"
        if args.command == "compact":
            companion = args.out.with_suffix(".json")
            require(args.out.suffix == ".csv" and not companion.exists(), "choose fresh CSV and JSON paths")
            table = compact(args.fields, args.metrics, epoch(args.start), epoch(args.end))
            table.to_csv(staging, index=False)
            receipt = {"schema": "fvcom_lake_surface_series_v1", "metrics_sha256": digest(args.metrics),
                       "series_sha256": digest(staging), "sources": [file_record(p) for p in args.fields],
                       "node_weights": "native spherical nodal control areas; wet_nodes == 1"}
            save(Path(work)/"receipt.json", receipt)
            publish(staging, args.out)
            try:
                publish(Path(work)/"receipt.json", companion)
            except OSError:
                args.out.unlink()
                raise
        else:
            staging.mkdir()
            validate(args, staging)
            publish(staging, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
