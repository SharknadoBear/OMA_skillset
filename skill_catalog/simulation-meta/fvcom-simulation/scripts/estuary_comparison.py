"""Compare post-start hourly FVCOM and independent OFS nowcasts on fixed channel transects."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from estuary_gauges import salinity_metrics


def load_fronts(path):
    with path.open() as file:
        records = {}
        for row in csv.DictReader(file):
            key = (
                np.datetime64(row["time"], "ms"),
                row["channel"],
                float(row["threshold"]),
            )
            if key in records:
                raise ValueError("Duplicate front record")
            records[key] = (float(row["distance_km"]), row["status"])
    return records


def time_str(value):
    return np.datetime_as_string(value, unit="m") + "Z"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--start", type=np.datetime64, required=True)
    p.add_argument("--end", type=np.datetime64, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    model = np.load(a.model / "sections.npz")
    ref = np.load(a.reference / "sections.npz")
    for key in ("channel", "distance_km", "lonlat", "height_m"):
        if not np.array_equal(model[key], ref[key]):
            raise ValueError(f"Section mismatch in {key}")
    common = np.intersect1d(model["time"], ref["time"])
    common = common[(common > a.start) & (common <= a.end)]
    expected = np.arange(
        a.start + np.timedelta64(1, "h"),
        a.end + np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    )
    if not np.array_equal(common, expected):
        raise ValueError("Missing post-initial hourly comparison clocks")
    model_fronts, ref_fronts = (
        load_fronts(q / "fronts.csv") for q in (a.model, a.reference)
    )
    channels = list(dict.fromkeys(model["channel"]))
    metrics = {}
    rows = []
    fig, axes = plt.subplots(
        len(channels),
        1,
        sharex=True,
        figsize=(8.5, 2.8 * len(channels)),
        layout="constrained",
        squeeze=False,
    )
    for i, name in enumerate(channels):
        use = model["channel"] == name
        candidate_thresholds = sorted({k[2] for k in model_fronts if k[1] == name})
        for threshold in candidate_thresholds:
            m = []
            r = []
            for t in common:
                vm, sm = model_fronts[t, name, threshold]
                vr, sr = ref_fronts[t, name, threshold]
                rows.append(
                    [
                        time_str(t),
                        name,
                        threshold,
                        vm,
                        sm,
                        vr,
                        sr,
                        vm - vr if sm == sr == "crossing" else "",
                    ]
                )
                m.append(vm if sm == "crossing" else np.nan)
                r.append(vr if sr == "crossing" else np.nan)
            metric = salinity_metrics(m, r)
            if "observed_range" in metric:
                metric["reference_range"] = metric.pop("observed_range")
            metric["expected_clocks"] = len(common)
            for label, values in [("model", m), ("reference", r)]:
                if np.isfinite(values).any():
                    imin, imax = np.nanargmin(values), np.nanargmax(values)
                    metric[label + "_min_utc"] = time_str(common[imin])
                    metric[label + "_max_utc"] = time_str(common[imax])
                    metric[label + "_excursion_km"] = float(
                        np.nanmax(values) - np.nanmin(values)
                    )
            metrics[f"{name} {threshold:g} PSU front km"] = metric
            if threshold == candidate_thresholds[0]:
                ax = axes[i, 0]
                ax.plot(common, m, ".-", color="#176e86", label="FVCOM")
                ax.plot(common, r, ".--", color="#903a21", label="SSCOFS")
                ax.set(
                    ylabel="km from mouth", title=f"{name} Channel, {threshold:g} PSU"
                )
                ax.grid(alpha=0.25)
                ax.legend()
        index0 = np.searchsorted(model["time"], common)
        index1 = np.searchsorted(ref["time"], common)
        metrics[name + " near-bed salinity"] = salinity_metrics(
            model["near"][index0][:, use], ref["near"][index1][:, use]
        )
        # Compare the same two physical clocks at opposite ends of the reference excursion.
        primary = candidate_thresholds[0]
        exact = [t for t in common if ref_fronts[t, name, primary][1] == "crossing"]
        if exact:
            chosen = [
                min(exact, key=lambda t: ref_fronts[t, name, primary][0]),
                max(exact, key=lambda t: ref_fronts[t, name, primary][0]),
            ]
            art = None
            section, panels = plt.subplots(
                2, 2, sharex=True, sharey=True, figsize=(11, 5.6), layout="constrained"
            )
            for col, t in enumerate(chosen):
                for row, (label, dataset) in enumerate(
                    (("FVCOM", model), ("SSCOFS", ref))
                ):
                    ti = np.searchsorted(dataset["time"], t)
                    z = dataset["z"][ti][:, use]
                    salt = dataset["salt"][ti][:, use]
                    km = dataset["distance_km"][use]
                    ax = panels[row, col]
                    valid = np.all(np.isfinite(z), axis=0) & np.any(
                        np.isfinite(salt), axis=0
                    )
                    contiguous = np.split(
                        np.flatnonzero(valid),
                        np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1,
                    )
                    for block in contiguous:
                        if len(block) < 2:
                            continue
                        x = np.broadcast_to(km[block], z[:, block].shape)
                        art = ax.contourf(
                            x,
                            z[:, block],
                            salt[:, block],
                            levels=np.arange(0, 37, 2),
                            cmap="viridis",
                            extend="both",
                        )
                    ax.plot(km, -dataset["h"][use], lw=0.7, color="k")
                    ax.set(
                        title=f"{label} {time_str(t)}",
                        ylabel="Elevation (m)",
                        xlabel="Channel distance (km)",
                    )
            if art is not None:
                section.colorbar(
                    art, ax=panels.ravel().tolist(), label="Practical salinity"
                )
            section.savefig(a.output / (name.lower() + "_sections.png"), dpi=165)
            plt.close(section)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1, 0].set_xlabel("UTC")
    fig.savefig(a.output / "salt_fronts.png", dpi=180)
    plt.close(fig)
    with (a.output / "front_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time",
                "channel",
                "threshold",
                "model_km",
                "model_status",
                "ofs_km",
                "ofs_status",
                "difference_km",
            ]
        )
        w.writerows(rows)
    (a.output / "ofs_comparison.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
