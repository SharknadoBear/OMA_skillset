"""Form conditional wind/river effects on native FVCOM nodes and elements."""

from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import netCDF4 as nc
import numpy as np
import pandas as pd

from run_case_validation import parse_namelist, read, sha256

MEMBERS = ("tide", "wind", "river", "combined")
CONTRASTS = {
    "river_without_wind": ("R − T", (-1, 0, 1, 0)),
    "river_with_wind": ("WR − W", (0, -1, 0, 1)),
    "wind_without_river": ("W − T", (-1, 1, 0, 0)),
    "wind_with_river": ("WR − R", (0, 0, -1, 1)),
    "interaction": ("WR − W − R + T", (1, -1, -1, 1)),
}
QUANTITIES = {"zeta": ("node", "m"), "ua": ("nele", "m s-1"), "va": ("nele", "m s-1")}
FILL = np.float32(nc.default_fillvals["f4"])


def effects(values):
    t, w, r, both = (np.asarray(v, dtype="f8") for v in values)
    return (r - t, both - w, w - t, both - r, (both - w) - (r - t))


def common_wet(values):
    masks = [np.ma.asarray(v) for v in values]
    if any(np.ma.getmaskarray(m).any() or not np.isin(m, (0, 1)).all() for m in masks):
        raise ValueError("Native wet masks must contain only explicit 0 and 1")
    return np.logical_and.reduce(masks)


def grid_records(stack, case, geometry):
    records = {}
    nml = parse_namelist(case["namelist"])
    step = float(nml["EXTSTEP_SECONDS"]) * int(nml["ISPLIT"])
    for row in read(case["audit_path"])["full_grid_netcdf"]:
        path = Path(row["path"])
        if not path.is_file():
            path = case["attempt"] / Path(row["path"].replace("\\", "/")).name
        if sha256(path) != row["sha256"]:
            raise ValueError("Full-grid bytes differ from the native production audit")
        ds = stack.enter_context(nc.Dataset(path))
        grids = {k: np.asarray(ds[k][:]) for k in ("x", "y", "h", "nv")}
        if not geometry:
            geometry.update(grids)
        elif any(not np.array_equal(geometry[k], a) for k, a in grids.items()):
            raise ValueError("Member output geometry, depths or native connectivity differ")
        for k, (space, _) in QUANTITIES.items():
            if ds[k].dimensions != ("time", space):
                raise ValueError(f"Unexpected native dimensions for {k}")
        for name, space in (("wet_nodes", "node"), ("wet_cells", "nele")):
            if ds[name].dimensions != ("time", space):
                raise ValueError(f"Missing native wet support for {space}")
        for i, iint in enumerate(ds["iint"][:]):
            time = pd.Timestamp(case["run_start"]) + pd.Timedelta(seconds=int(iint) * step)
            if not case["start"] <= time < case["end"]:
                continue
            if time in records:
                raise ValueError("Duplicate native analysis timestamp")
            records[time] = (ds, i)
    return records


def maps(geometry, means, output):
    xy = [geometry[k].astype(float) / 1000 for k in ("x", "y")]
    cells = np.asarray(geometry["nv"], dtype=int).T - 1
    triang = mtri.Triangulation(*xy, triangles=cells)
    centers = [coord[cells].mean(axis=1) for coord in xy]
    for j, (name, (formula, _)) in enumerate(CONTRASTS.items()):
        level, u, v = (means[k][j] for k in QUANTITIES)
        speed = np.hypot(u, v)
        lim = np.nanmax(np.abs(level)) or 1e-6
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), constrained_layout=True)
        for ax, field, title, unit in zip(
            axes,
            (level, speed),
            ("Mean elevation difference", "Magnitude of mean velocity difference"),
            ("m", "m/s"),
        ):
            if field is level:
                triang.set_mask(~np.isfinite(level[cells]).all(axis=1))
                artist = ax.tripcolor(
                    triang,
                    np.nan_to_num(level),
                    shading="gouraud",
                    cmap="RdBu_r",
                    vmin=-lim,
                    vmax=lim,
                    rasterized=True,
                )
            else:
                triang.set_mask(~np.isfinite(speed))
                artist = ax.tripcolor(
                    triang, facecolors=np.nan_to_num(speed), cmap="viridis", vmin=0, rasterized=True
                )
                selected = np.flatnonzero(np.isfinite(speed))
                selected = selected[:: max(1, len(selected) // 120)]
                if np.any(speed[selected] > 0):
                    ax.quiver(
                        centers[0][selected],
                        centers[1][selected],
                        u[selected],
                        v[selected],
                        color="black",
                        width=0.002,
                        angles="xy",
                    )
            fig.colorbar(artist, ax=ax, label=unit, shrink=0.78)
            ax.set(
                title=f"{title}  ({formula})",
                xlabel="Model x (km)",
                ylabel="Model y (km)",
                aspect="equal",
            )
        fig.savefig(output / f"{name}_mean_fields.png", dpi=150)
        plt.close(fig)


def spatial(cases, output):
    geometry = {}
    path = output / "native_contrasts.nc"
    gauges = {
        str(s["station_id"]): int(s["cell_id"]) - 1
        for s in read(cases["tide"]["mapping_path"])["stations"]
        if s["role"] == "water_level"
    }
    gauge_wetness = []
    with ExitStack() as stack:
        records = [grid_records(stack, cases[k], geometry) for k in MEMBERS]
        times = sorted(records[0])
        start, end = (pd.Timestamp(cases["tide"][k]) for k in ("start", "end"))
        if times != list(pd.date_range(start, end, freq="3h", inclusive="left")) or any(
            sorted(other) != times for other in records[1:]
        ):
            raise ValueError("Missing or unmatched three-hour native analysis records")
        with nc.Dataset(path, "w", format="NETCDF4") as target:
            sizes = dict(
                node=len(geometry["x"]),
                nele=geometry["nv"].shape[1],
                three=3,
                time=len(times),
                contrast=len(CONTRASTS),
            )
            for dim, size in sizes.items():
                target.createDimension(dim, size)
            target.setncatts(
                dict(
                    title="Coincident four-member FVCOM factorial effects",
                    vector_frame="native Cartesian model x/y",
                    analysis_end_exclusive=end.isoformat(),
                    wet_support="All four native wet_nodes for zeta; all four native wet_cells for ua and va.",
                )
            )
            clock = target.createVariable("time", "i8", ("time",))
            clock.units = "seconds since 1970-01-01 00:00:00 UTC"
            clock[:] = [round(t.timestamp()) for t in times]
            strings = target.createVariable("contrast", str, ("contrast",))
            strings[:] = np.array(list(CONTRASTS), dtype=object)
            for key, dims, dtype, unit in (
                ("nv", ("three", "nele"), "i4", "1"),
                ("x", ("node",), "f8", "m"),
                ("y", ("node",), "f8", "m"),
                ("h", ("node",), "f8", "m"),
            ):
                name = target.createVariable(key, dtype, dims)
                name.units = unit
                name[:] = geometry[key]
            target["nv"].start_index = 1
            masks, counts, totals, products = {}, {}, {}, {}
            for space in ("node", "nele"):
                masks[space] = target.createVariable(
                    "common_wet_" + space, "u1", ("time", space), zlib=True, complevel=1
                )
                counts[space] = np.zeros(sizes[space], dtype="i4")
            for key, (space, unit) in QUANTITIES.items():
                var = target.createVariable(
                    key,
                    "f4",
                    ("time", "contrast", space),
                    fill_value=FILL,
                    chunksizes=(1, 1, sizes[space]),
                    zlib=True,
                    complevel=1,
                )
                var.units = unit
                products[key] = var
                totals[key] = np.zeros((len(CONTRASTS), sizes[space]))
            summaries = []
            for i, time in enumerate(times):
                samples = [member[time] for member in records]
                wet = {}
                member_masks = {}
                for space, flag in (("node", "wet_nodes"), ("nele", "wet_cells")):
                    member_masks[space] = [ds[flag][idx] for ds, idx in samples]
                    wet[space] = common_wet(member_masks[space])
                    if not wet[space].any():
                        raise ValueError(f"No common wet {space} at {time}")
                    counts[space] += wet[space]
                    masks[space][i] = wet[space]
                for m, member in enumerate(MEMBERS):
                    for sid, cell in gauges.items():
                        nodes = geometry["nv"][:, cell] - 1
                        gauge_wetness.append(
                            dict(
                                time_utc=time.isoformat(),
                                member=member,
                                station=sid,
                                cell_id=cell + 1,
                                wet_cell=int(member_masks["nele"][m][cell]),
                                wet_vertex_count=int(np.sum(member_masks["node"][m][nodes])),
                            )
                        )
                entries = {}
                for key, (space, _) in QUANTITIES.items():
                    arrays = [np.ma.filled(ds[key][idx], np.nan) for ds, idx in samples]
                    diffs = np.stack(effects(arrays))
                    if not np.isfinite(diffs[:, wet[space]]).all():
                        raise ValueError(f"Invalid wet {key} at {time}")
                    entries[key] = diffs
                    products[key][i] = np.where(wet[space], diffs, FILL).astype("f4")
                    totals[key] += np.where(wet[space], diffs, 0)
                for j, name in enumerate(CONTRASTS):
                    level = entries["zeta"][j, wet["node"]]
                    speed = np.hypot(entries["ua"][j, wet["nele"]], entries["va"][j, wet["nele"]])
                    summaries.append(
                        dict(
                            time_utc=time.isoformat(),
                            contrast=name,
                            common_wet_nodes=int(wet["node"].sum()),
                            common_wet_elements=int(wet["nele"].sum()),
                            min_level_m=level.min(),
                            max_level_m=level.max(),
                            max_vector_difference_m_s=speed.max(),
                        )
                    )
            means = {}
            for space, val in counts.items():
                target.createVariable("wet_samples_" + space, "i4", (space,))[:] = val
            for key, (space, unit) in QUANTITIES.items():
                means[key] = np.divide(
                    totals[key],
                    counts[space],
                    where=counts[space] > 0,
                    out=np.full_like(totals[key], np.nan),
                )
                var = target.createVariable("mean_" + key, "f4", ("contrast", space), fill_value=FILL)
                var.units = unit
                var[:] = np.ma.masked_invalid(means[key])
    pd.DataFrame(summaries).to_csv(output / "native_timeseries_extrema.csv", index=False)
    gauge_support = pd.DataFrame(gauge_wetness)
    gauge_support.to_csv(output / "gauge_wetness.csv", index=False)
    maps(geometry, means, output)
    return dict(
        path=path.name,
        sha256=sha256(path),
        timestamps=len(times),
        analysis_start=start.isoformat(),
        analysis_end_exclusive=end.isoformat(),
        coincident_wet_nodes=int((counts["node"] > 0).sum()),
        coincident_wet_cells=int((counts["nele"] > 0).sum()),
        gauge_dry_cell_samples=int((gauge_support.wet_cell == 0).sum()),
        gauge_samples_with_any_dry_vertex=int((gauge_support.wet_vertex_count < 3).sum()),
    )
