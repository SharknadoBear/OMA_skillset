"""Compare measured practical salinity with native, fixed-in-time sensor elevations."""

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from estuary_sections import native_times, numeric, sample_vertical
from netCDF4 import Dataset


def salinity_metrics(trial, reference):
    model = np.asarray(trial, dtype=float)
    obs = np.asarray(reference, dtype=float)
    use = np.isfinite(model) & np.isfinite(obs)
    model, obs = model[use], obs[use]
    if not len(obs):
        return {"count": 0}
    residual = model - obs
    correlation = None
    if len(obs) > 1 and np.ptp(model) > 0 and np.ptp(obs) > 0:
        correlation = float(np.corrcoef(model, obs)[0, 1])
    return {
        "count": len(obs),
        "bias": float(residual.mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "centered_rmse": float(np.std(residual)),
        "correlation": correlation,
        "observed_range": [float(obs.min()), float(obs.max())],
        "modeled_range": [float(model.min()), float(model.max())],
    }


def interp_time(t, data, q, maximum_gap):
    """Only interpolate between finite adjacent time records with adequate coverage."""
    t = np.asarray(t, dtype="datetime64[ms]").astype("int64")
    q = np.asarray(q, dtype="datetime64[ms]").astype("int64")
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Need two or more strictly increasing timestamps")
    j = np.clip(np.searchsorted(t, q), 0, len(t) - 1)
    i = np.maximum(j - 1, 0)
    delta = t[j] - t[i]
    data = np.asarray(data, dtype=float)
    exact = t[j] == q
    result = np.full(len(q), np.nan)
    result[exact] = data[j[exact]]
    use = ~exact & (t[i] < q) & (q < t[j]) & (delta <= maximum_gap * 1000)
    frac = (q[use] - t[i[use]]) / delta[use]
    result[use] = data[i[use]] + frac * (data[j[use]] - data[i[use]])
    return result


def station_times(dataset, startup, startup_iint, step_seconds):
    if "Times" in dataset.variables or {"Itime", "Itime2"} <= set(dataset.variables):
        return native_times(dataset)
    if startup is None or step_seconds is None or step_seconds <= 0:
        raise ValueError(
            "Native station output needs the verified startup UTC and internal timestep"
        )
    var = dataset["time"]
    if not var.units.startswith("days since 1858-11-17"):
        raise ValueError("Unrecognized station time epoch")
    ticks = np.asarray(dataset["iint"][:], dtype="int64")
    if np.any(np.diff(ticks) <= 0):
        raise ValueError("Station iteration numbers are not strictly increasing")
    elapsed = np.rint((ticks - startup_iint) * step_seconds * 1000).astype(
        "timedelta64[ms]"
    )
    clock = np.datetime64(startup.rstrip("Z"), "ms") + elapsed
    native = np.asarray(var[:])
    mjd = (clock - np.datetime64("1858-11-17", "ms")) / np.timedelta64(1, "D")
    if not np.isfinite(native).all() or np.any(
        np.abs(mjd - native) > np.spacing(native) / 2 + 1.0e-8
    ):
        raise ValueError(
            "Reconstructed station clock disagrees with stored MJD precision"
        )
    return clock


def crossings(times, salinity, level, max_gap):
    dates = np.asarray(times, dtype="datetime64[ms]")
    values = np.asarray(salinity, dtype=float)
    result = []
    for i, (left, right) in enumerate(pairwise(values)):
        step = (dates[i + 1] - dates[i]) / np.timedelta64(1, "s")
        if (
            not np.isfinite(left + right)
            or step <= 0
            or step > max_gap
            or (left >= level) == (right >= level)
        ):
            continue
        fraction = (level - left) / (right - left)
        when = dates[i] + np.timedelta64(round(step * 1000 * fraction), "ms")
        result.append(
            {
                "utc": str(when) + "Z",
                "direction": "increasing" if right > left else "decreasing",
            }
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", type=Path, required=True)
    p.add_argument(
        "--sensors",
        type=Path,
        required=True,
        help="JSON array of sensor specifications",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--max-gap-seconds",
        type=int,
        default=540,
        help="Allow clock jitter within a six-minute cadence, but exclude a missing observation interval",
    )
    p.add_argument(
        "--startup-utc",
        help="Frozen run startup UTC for iteration-based station clocks",
    )
    p.add_argument("--startup-iint", type=int, default=0)
    p.add_argument("--internal-step-seconds", type=float)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    sensors = json.loads(args.sensors.read_text())
    pairs, metrics, aligned = {}, {}, {}
    with Dataset(args.stations) as d:
        times = station_times(
            d, args.startup_utc, args.startup_iint, args.internal_step_seconds
        )
        names = [
            v.tobytes().decode("utf8").replace("\x00", "").strip()
            for v in d["name_station"][:]
        ]
        fig, axes = plt.subplots(
            len(sensors),
            1,
            sharex=True,
            figsize=(8, 2.7 * len(sensors)),
            squeeze=False,
            layout="constrained",
        )
        for spec, ax in zip(sensors, axes.flatten()):
            if not spec.get("datum_evidence") or not spec.get("qc_evidence"):
                raise ValueError("Specify the sensor datum and provider QC evidence")
            with (args.sensors.parent / spec["csv"]).open(encoding="utf8") as stream:
                rows = csv.DictReader(stream)
                next(rows)  # ERDDAP units row
                data = [
                    row for row in rows if row["salinity_qc"] in spec["accepted_flags"]
                ]
            when = np.array(
                [r["time"].rstrip("Z") for r in data], dtype="datetime64[ms]"
            )
            obs = np.array([float(r["salinity"]) for r in data])
            col = names.index(spec["station"])
            eta = numeric(d["zeta"], (slice(None), col))
            h = float(d["h"][col])
            sg = numeric(d["siglay"], (slice(None), col))
            fields = numeric(d["salinity"], (slice(None), slice(None), col))
            target = spec["elevation_in_model_datum_m"]
            z = eta[:, None] + (eta + h)[:, None] * sg
            calc = np.array(
                [
                    sample_vertical(levels, row, target, -h, top)
                    for levels, row, top in zip(z, fields, eta)
                ]
            )
            if "wet_cells" in d.variables:
                calc[numeric(d["wet_cells"], (slice(None), col)) != 1] = np.nan
            paired = interp_time(times, calc, when, args.max_gap_seconds)
            key = spec["id"]
            pairs[key] = [
                (str(t), float(o), float(v)) for t, o, v in zip(when, obs, paired)
            ]
            metrics[key] = salinity_metrics(paired, obs)
            metrics[key].update(
                source_samples=len(when), station=spec["station"], elevation_m=target
            )
            metrics[key]["local_1_psu_passages"] = {
                "observed": crossings(when, obs, 1.0, args.max_gap_seconds),
                "model": crossings(times, calc, 1.0, args.max_gap_seconds),
            }
            aligned[key] = (interp_time(when, obs, times, args.max_gap_seconds), calc)
            ax.plot(times, calc, color="#176e86", label="FVCOM")
            ax.scatter(when, obs, color="#903a21", s=6, label="observations")
            ax.set(title=key, ylabel="Practical salinity")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("UTC")
        fig.savefig(args.output / "gauge_salinity.png", dpi=180)
        plt.close(fig)
    if len(sensors) == 2 and sensors[0]["station"] == sensors[1]["station"]:
        shallow, deep = sorted(
            sensors, key=lambda q: q["elevation_in_model_datum_m"], reverse=True
        )
        obs0, mod0 = aligned[shallow["id"]]
        obs1, mod1 = aligned[deep["id"]]
        metrics["deep_minus_shallow"] = salinity_metrics(mod1 - mod0, obs1 - obs0)
    with (args.output / "gauge_pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["instrument", "time", "observed", "modeled"])
        for key, values in pairs.items():
            writer.writerows((key, *row) for row in values)
    (args.output / "gauge_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
