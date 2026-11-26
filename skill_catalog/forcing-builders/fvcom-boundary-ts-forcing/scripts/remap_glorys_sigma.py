"""Map checked GLORYS daily profiles or HYCOM ESPC salinity onto a geographic FVCOM boundary."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from ts_core import decode_netcdf_times, parse_iso_time_ms, read_boundary_2dm, sha256_file
from hycom_salinity import load_hycom


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arr(x):
    return np.ma.filled(np.ma.asarray(x).astype("f8"), np.nan)


def xyz(lon, lat):
    l, p = np.radians(lon), np.radians(lat)
    return np.stack((np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)), axis=-1)


def indices(axis, target):
    require(
        len(axis) >= 2
        and np.all(np.diff(axis) > 0)
        and np.all(target >= axis[0])
        and np.all(target <= axis[-1]),
        "Requested axis is outside source coverage",
    )
    lo = np.clip(np.searchsorted(axis, target, side="right") - 1, 0, len(axis) - 2)
    return lo, (target - axis[lo]) / (axis[lo + 1] - axis[lo])


def check_startup_sigma(dataset, ids, target_layers, target_interfaces):
    for key, expected in (("siglay", target_layers), ("siglev", target_interfaces)):
        variable = dataset[key]
        if variable.dimensions == (key,):
            observed = arr(variable[:])[:, None]
        elif variable.dimensions == (key, "node"):
            observed = arr(variable[:, ids - 1])
        else:
            raise ValueError(f"Unexpected startup {key} dimensions")
        require(
            observed.shape[0] == len(expected)
            and np.allclose(observed, expected[:, None], rtol=0, atol=2.0e-7),
            f"Frozen startup {key} differs from target sigma order or values",
        )


def donor_bank(lon, lat, valid, targets, seed, needed, max_km):
    """Common finite upper/lower donors must be four-connected to the offshore seed."""
    ny, nx = len(lat), len(lon)
    lon2, lat2 = np.meshgrid(lon, lat)
    src_xyz = xyz(lon2.ravel(), lat2.ravel())
    dst_xyz = xyz(targets[:, 0], targets[:, 1])
    require(
        lon[0] <= seed[0] <= lon[-1] and lat[0] <= seed[1] <= lat[-1],
        "Offshore seed outside the source grid",
    )
    sy, sx = np.unravel_index(np.argmin(np.linalg.norm(src_xyz - xyz(*seed), axis=1)), (ny, nx))
    i, x = indices(lon, targets[:, 0])
    j, y = indices(lat, targets[:, 1])
    corners = np.array([j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1]).T
    base = np.array([(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y]).T
    nnode = len(targets)
    nz = len(valid)
    donors = np.full((nz, nnode, 4), -1, dtype="i4")
    weights = np.zeros((nz, nnode, 4))
    distance = np.zeros((nz, nnode, 4))
    codes = np.full((nz, nnode), -1, dtype="i1")
    basin_mask = np.zeros((nz, ny, nx), bool)
    for k in np.flatnonzero(needed.any(axis=1)):
        common = valid[k] & valid[max(k - 1, 0)]
        cc, _ = label(common)
        require(common[sy, sx], f"Offshore seed dry at an actually used depth level {k}")
        basin = cc == cc[sy, sx]
        basin_mask[k] = basin
        ocean = np.flatnonzero(basin)
        tree = cKDTree(src_xyz[ocean])
        ns = np.flatnonzero(needed[k])
        wet = basin.ravel()[corners[ns]]
        w = base[ns] * wet
        total = w.sum(axis=1)
        idx = corners[ns].copy()
        supported = total > 0
        w[supported] /= total[supported, None]
        for rr in np.flatnonzero(~supported):
            _, ii = tree.query(dst_xyz[ns[rr]])
            idx[rr, 0] = ocean[ii]
            w[rr, 0] = 1.0
        chord = np.linalg.norm(src_xyz[idx] - dst_xyz[ns, None, :], axis=2)
        dist = 2.0 * 6371.0088 * np.arcsin(np.minimum(chord / 2.0, 1.0))
        used = w > 0
        require(
            np.all(dist[used] <= max_km) and np.all(basin.ravel()[idx[used]]),
            f"Disconnected or distant ocean donor at depth index {k}",
        )
        require(
            np.allclose(w.sum(axis=1), 1.0, rtol=0, atol=2e-15),
            "Nonconservative boundary interpolation weights",
        )
        donors[k, ns] = np.where(used, idx, -1)
        weights[k, ns] = w
        distance[k, ns] = np.where(used, dist, 0.0)
        codes[k, ns] = np.where(~supported, 2, np.where(np.any((base[ns] > 0) & ~wet, axis=1), 1, 0))
    return donors, weights, distance, codes, basin_mask


def load_glorys(directory):
    source = directory / "subset.nc"
    manifest = directory / "manifest.json"
    health = directory / "health_report.json"
    owner = json.loads(manifest.read_bytes())
    qa = json.loads(health.read_bytes())
    require(
        owner.get("complete")
        and qa.get("pass")
        and qa.get("complete")
        and qa["plan_hash"] == owner["plan_hash"]
        and owner["assembled"]["sha256"] == sha256_file(source),
        "Unqualified or modified GLORYS source",
    )
    with nc.Dataset(source) as ds:
        for k in ("so", "thetao"):
            require(
                ds[k].dimensions == ("time", "depth", "latitude", "longitude"),
                "Unexpected GLORYS field staggering",
            )
        require(
            ds["so"].units in ("1e-3", "PSU") and "practical" in ds["so"].unit_long.lower(),
            "Unrecognized salinity convention",
        )
        require(
            ds["thetao"].units in ("degrees_C", "degree_Celsius", "degrees_Celsius")
            and ds["depth"].positive.lower() == "down"
            and ds["depth"].units == "m",
            "GLORYS vertical or temperature unit mismatch",
        )
        lon, lat, z = [arr(ds[k][:]) for k in ("longitude", "latitude", "depth")]
        st, _, _ = decode_netcdf_times(ds)
        salt, temp = [arr(ds[k][:]) for k in ("so", "thetao")]
    return (
        lon,
        lat,
        z,
        st,
        salt,
        temp,
        owner["source"],
        dict(source=source, glorys_manifest=manifest, glorys_health=health),
    )


def run(a):
    require(
        np.isfinite(a.min_depth) and a.min_depth >= 0 and np.isfinite(a.max_donor_km) and a.max_donor_km > 0,
        "Invalid wet/dry depth or donor radius",
    )
    hycom = a.source_provider == "hycom-espc"
    if hycom:
        require(
            a.frozen_temperature_startup is not None,
            "HYCOM salinity-only fallback requires a frozen native temperature companion",
        )
        lon, lat, z, st, S, provider, source_files = load_hycom(a.source)
        T = None
    else:
        lon, lat, z, st, S, T, provider, source_files = load_glorys(a.source)
    b = read_boundary_2dm(a.mesh_2dm, a.open_ns)
    ids = b.node_ids
    obc = np.loadtxt(a.obc, skiprows=1, ndmin=2, dtype=int)
    require(np.array_equal(obc[:, 1], ids), "Geographic OBC differs from native ID order")
    depth = np.loadtxt(a.dep, skiprows=1, ndmin=2)[:, 2]
    h = depth[ids - 1]
    require(np.allclose(h, b.depth_m, rtol=0, atol=0.001), "Runtime and geographic bathymetry differ")
    text = a.sigma.read_text().upper()
    fields = dict(x.strip().split(" = ", 1) for x in text.splitlines() if x.strip())
    require(fields.get("SIGMA COORDINATE TYPE") == "UNIFORM", "Only native uniform sigma is supported")
    lev = np.linspace(0.0, -1.0, int(fields["NUMBER OF SIGMA LEVELS"]))
    lay = (lev[:-1] + lev[1:]) / 2
    start, end = [parse_iso_time_ms(t, False) for t in (a.start, a.end)]
    step = a.step_seconds * 1000
    guard = a.guard_hours * 3600000
    require(
        start < end and step > 0 and guard > 0 and (end - start) % step == 0 and guard % step == 0,
        "Invalid target clock or guards",
    )
    times = np.arange(start - guard, end + guard + 1, step, dtype="i8")
    with nc.Dataset(a.elevation) as ds:
        eta_clock, _, _ = decode_netcdf_times(ds)
        ev = arr(ds["elevation"][:])
        require(
            np.array_equal(ds["obc_nodes"][:], ids)
            and np.all(np.diff(eta_clock) == step)
            and start >= eta_clock[0]
            and end <= eta_clock[-1],
            "Elevation clock or IDs differ",
        )
        covered = (times >= eta_clock[0]) & (times <= eta_clock[-1])
        idx = np.searchsorted(eta_clock, times[covered])
        require(
            np.array_equal(eta_clock[idx], times[covered]),
            "Elevation must be on the requested exact UTC knots",
        )
        eta = np.zeros((len(times), len(ids)))
        eta[covered] = ev[idx]
    D = eta + h
    require(np.isfinite(D).all(), "Nonfinite native free surface/depth")
    dry = D <= a.min_depth
    # The donor selection is used by wet inflows only. Dry rows and unavailable pre-start elevation guards
    # receive a finite, explicitly flagged reference-bathymetry companion.
    target_z = -lay[None, :, None] * np.where(dry, h, D)[:, None, :]
    cadence = 10800 if hycom else 86400
    require(
        np.all(np.diff(st) == cadence * 1000) and st[0] < start and end < st[-1],
        "Native ocean clock must be regular and bracket both experiment ends",
    )
    ltime, ft = indices(st, times)
    require(
        z[0] >= 0 and np.all(np.diff(z) > 0) and target_z.max() <= z[-1],
        "Ocean levels must cover target bottom",
    )
    upper = np.searchsorted(z, target_z)
    lower = np.maximum(upper - 1, 0)
    fraction = np.divide(
        target_z - z[lower], z[upper] - z[lower], out=np.zeros_like(target_z), where=upper != lower
    )
    needed = np.zeros((len(z), len(ids)), bool)
    node = np.broadcast_to(np.arange(len(ids)), upper.shape)
    needed[upper, node] = True
    valid = np.isfinite(S)
    if T is not None:
        valid &= np.isfinite(T)
    donor, weights, dist, method, basins = donor_bank(
        lon, lat, np.all(valid, axis=0), np.column_stack((b.lon, b.lat)), a.seed, needed, a.max_donor_km
    )

    def mapped(field):
        # One donor stencil describes each vertical source bracket throughout its complete time coverage.
        vals = np.zeros((len(st), len(z), len(ids), 2))
        for kk, nn in zip(*np.where(needed)):
            live = weights[kk, nn] > 0
            ix = donor[kk, nn, live]
            ww = weights[kk, nn, live]
            for side, ll in enumerate((max(kk - 1, 0), kk)):
                vals[:, kk, nn, side] = field[:, ll].reshape(len(st), -1)[:, ix] @ ww

        def at(clock):
            k = clock[:, None, None]
            return (1 - fraction) * vals[k, upper, node, 0] + fraction * vals[k, upper, node, 1]

        out = (1 - ft[:, None, None]) * at(ltime) + ft[:, None, None] * at(ltime + 1)
        require(
            np.isfinite(out).all()
            and out.min() >= np.nanmin(field) - 1e-9
            and out.max() <= np.nanmax(field) + 1e-9,
            "Remap generated missing values or an extremum beyond source",
        )
        return out

    salt = mapped(S)
    ocean_t = mapped(T) if T is not None else None
    temp = ocean_t
    if a.frozen_temperature_startup:
        with nc.Dataset(a.frozen_temperature_startup) as ds:
            exact, _, _ = decode_netcdf_times(ds)
            require(exact.tolist() == [start], "Frozen temperature startup time mismatch")
            require(
                ds["temp"].dimensions == ("time", "siglay", "node")
                and np.allclose(ds["h"][:], depth, atol=0.001, rtol=0),
                "Frozen temperature layout or grid depth differs",
            )
            check_startup_sigma(ds, ids, lay, lev)
            frozen = arr(ds["temp"][0, :, ids - 1])
            require(
                frozen.shape == (len(lay), len(ids)) and np.isfinite(frozen).all(),
                "Missing fixed-temperature profile",
            )
            temp = np.broadcast_to(frozen, salt.shape)
    require(not a.output.exists(), "Choose a new output directory for the sigma-ready boundary")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.output.parent) as tmp:
        p = Path(tmp) / "ocean"
        p.mkdir()
        with nc.Dataset(p / "sigma_ready.nc", "w") as out:
            for k, n in (
                ("time", len(times)),
                ("siglay", len(lay)),
                ("siglev", len(lev)),
                ("node", len(ids)),
            ):
                out.createDimension(k, n)
            tm = out.createVariable("time", "f8", ("time",))
            tm.units = "seconds since 1970-01-01 00:00:00 +00:00"
            tm[:] = times / 1000.0
            for k, d, v, units in (
                ("node_id", "node", ids, "1"),
                ("siglay", "siglay", lay, "1"),
                ("siglev", "siglev", lev, "1"),
            ):
                vout = out.createVariable(k, "i4" if k == "node_id" else "f8", (d,))
                vout.units = units
                vout[:] = v
            variables = [("obc_temp", temp, "Celsius"), ("obc_salinity", salt, "PSU")]
            if ocean_t is not None:
                variables.append(("ocean_potential_temperature", ocean_t, "Celsius"))
            for k, value, unit in variables:
                vout = out.createVariable(k, "f8", ("time", "siglay", "node"), zlib=True)
                vout.units = unit
                vout[:] = value
        np.savez_compressed(
            p / "remap_support.npz",
            source_times_ms=st,
            source_z_m=z,
            source_lon=lon,
            source_lat=lat,
            target_times_ms=times,
            node_id=ids,
            lon=b.lon,
            lat=b.lat,
            reference_depth_m=h,
            free_surface_m=eta,
            elevation_covered=covered,
            dry_reference_depth=dry,
            physical_target_depth_m=target_z,
            time_lo=ltime,
            time_fraction=ft,
            z_lo=lower,
            z_hi=upper,
            z_fraction=fraction,
            donor_flat_indices=donor,
            donor_weights=weights,
            donor_distance_km=dist,
            donor_method=method,
            common_wet_component=basins,
        )
        files = {n: sha256_file(p / n) for n in ("sigma_ready.nc", "remap_support.npz")}
        provenance = {
            k: sha256_file(v)
            for k, v in dict(
                **source_files,
                mesh=a.mesh_2dm,
                obc=a.obc,
                depth=a.dep,
                sigma=a.sigma,
                elevation=a.elevation,
                **(
                    {"frozen_temperature_startup": a.frozen_temperature_startup}
                    if a.frozen_temperature_startup
                    else {}
                ),
            ).items()
        }
        report = dict(
            schema="fvcom_wet_hycom_boundary_remap_v1" if hycom else "fvcom_wet_glorys_boundary_remap_v1",
            status="verified",
            source=provider,
            files=files,
            inputs_sha256=provenance,
            utc=(
                "Three-hour instantaneous UTC analyses"
                if hycom
                else "Preserved stored UTC labels of daily means; linearly interpolate between these labels"
            ),
            start_utc=a.start,
            end_utc=a.end,
            guard_hours=a.guard_hours,
            step_seconds=a.step_seconds,
            records=len(times),
            nodes=len(ids),
            layers=len(lay),
            positive_down=True,
            water_depth="h + native prescribed elevation on exact available knots; reference h at dry nodes and outside available elevation guards",
            reference_depth_guard_records=int((~covered).sum()),
            dry_node_records=int(dry.sum()),
            above_first_source_level_targets=int((upper == 0).sum()),
            source_depth_m=[float(z[0]), float(z[-1])],
            target_depth_m=[float(target_z.min()), float(target_z.max())],
            offshore_seed_lon_lat=list(a.seed),
            donor_policy="four-neighbor ocean component spanning both depth levels and all source times; bilinear, renormalized wet corners, then nearest connected wet center",
            donor_methods_target_counts={
                k: int(np.sum(method[upper, node] == i))
                for i, k in enumerate(("bilinear", "renormalized", "nearest"))
            },
            maximum_used_donor_km=float(dist.max()),
            maximum_allowed_donor_km=a.max_donor_km,
            salinity_psu=[float(salt.min()), float(salt.max())],
            ocean_potential_temperature_C=(
                None if ocean_t is None else [float(ocean_t.min()), float(ocean_t.max())]
            ),
            temperature_source=(
                "frozen startup" if a.frozen_temperature_startup else "GLORYS potential temperature"
            ),
            regular_output_shape=list(salt.shape),
            unresolved_values=0,
        )
        (p / "remap_manifest.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        p.rename(a.output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-provider", choices=("glorys", "hycom-espc"), default="glorys")
    for key in ("source", "mesh-2dm", "obc", "dep", "sigma", "elevation", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--open-ns", nargs="+", type=int, required=True)
    p.add_argument("--seed", nargs=2, type=float, required=True, metavar=("LON", "LAT"))
    p.add_argument("--max-donor-km", type=float, required=True)
    p.add_argument("--min-depth", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--step-seconds", type=int, required=True)
    p.add_argument("--guard-hours", type=int, required=True)
    p.add_argument(
        "--frozen-temperature-startup",
        type=Path,
        help="Optional native startup temperature companion for fixed-temperature simulations.",
    )
    print(json.dumps(run(p.parse_args()), indent=2))
