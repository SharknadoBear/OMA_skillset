"""Read FVCOM station and native spherical grid output for lake validation."""
import hashlib
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    return {"name": str(path), "sha256": digest(path)}


def values(data):
    return np.ma.asarray(data, dtype=float).filled(np.nan)


def native_times(ds):
    var = ds["time"]
    times = netCDF4.num2date(var[:], var.units, only_use_cftime_datetimes=False,
                             only_use_python_datetimes=True)
    return np.asarray(times, dtype="datetime64[s]").astype("i8")


def check_units(ds):
    units = ds["temp"].units.lower()
    require(units in ("degrees_c", "degrees c", "degree_celsius", "degrees_celsius", "celsius"),
            f"unrecognized model temperature units {units!r}")
    for key in ["zeta", "h"]:
        require(ds[key].units.lower() in ("m", "meter", "meters", "metre", "metres"),
                f"{key} must use meters")


def station_clock(ds, cadence):
    times = native_times(ds)
    record = dict(method="source_time", adjusted_records=0, maximum_adjustment_seconds=0)
    t = ds["time"]
    if not cadence or t.dtype != np.dtype("f4") or "iint" not in ds.variables or len(times)<2:
        return times,record
    units = t.units.lower().split(" since ",1)[0]
    scale = {"days":86400., "hours":3600., "minutes":60., "seconds":1.}.get(units)
    require(scale is not None and cadence>0, "unsupported single-precision station clock")
    raw = np.asarray(t[:],dtype=np.float32)
    precision = np.abs(np.spacing(raw).astype(float))*scale/2.
    if np.max(precision)<1.:
        return times,record
    require(np.all(precision+1<cadence/2), "single-precision station clock cannot distinguish adjacent slots")
    nominal = np.rint(times/cadence).astype("i8")*cadence
    correction = nominal-times
    require(np.all(np.abs(correction)<=precision+1), "station time outside its native packing precision")
    steps = np.diff(np.asarray(ds["iint"][:],dtype=np.int64))
    seconds = np.diff(nominal)
    require(np.all(steps>0) and np.all(seconds>0) and np.all(steps*seconds[0]==steps[0]*seconds),
            "internal model steps disagree with nominal station clock")
    return nominal,dict(method="declared_UTC_schedule_verified_by_float32_precision_and_iint",
                        adjusted_records=int(np.count_nonzero(correction)),maximum_adjustment_seconds=int(np.max(abs(correction))),
                        maximum_half_ulp_seconds=float(np.max(precision)),seconds_per_internal_step=float(seconds[0]/steps[0]))


def read_stations(paths, start, end, cadence=None):
    blocks = []
    clocks = []
    static = None
    for path in paths:
        with netCDF4.Dataset(path) as ds:
            check_units(ds)
            ids = [row.tobytes().decode("ascii").strip(" \x00") for row in ds["name_station"][:]]
            require(len(set(ids)) == len(ids), f"duplicate station identifiers in {path}")
            sig, h = (values(ds[key][:]) for key in ("siglay", "h"))
            lon = (values(ds["lon"][:])+180) % 360-180
            lat = values(ds["lat"][:])
            require(sig.shape == (len(ds.dimensions["siglay"]), len(ids)) and
                    len(sig) >= 2 and np.all(np.diff(sig, axis=0) < 0) and
                    np.all((sig >= -1) & (sig <= 0)), "invalid native sigma midpoints")
            require(h.shape == (len(ids),) and np.all(np.isfinite(h) & (h > 0)), "invalid station depths")
            if static is None:
                static = (ids, sig, h, lon, lat)
            else:
                require(ids == static[0] and all(np.array_equal(a, b) for a, b in zip((sig, h, lon, lat), static[1:])),
                        "station mapping or geometry changes between output segments")
            times, clock_info = station_clock(ds,cadence)
            clocks.append(clock_info)
            selected = np.flatnonzero((times >= start - 86400) & (times <= end))
            if not len(selected):
                continue
            sl = slice(selected[0], selected[-1] + 1)
            require(np.all(np.diff(times[sl]) > 0), "native time must increase within each station file")
            require(ds["temp"].dimensions == ("time", "siglay", "station") and
                    ds["zeta"].dimensions == ("time", "station"), "unexpected station dimension order")
            blocks.append((times[sl], values(ds["zeta"][sl]), values(ds["temp"][sl])))
    require(bool(blocks), "no station records overlap the analysis window")
    time, zeta, temp = (np.concatenate(column) for column in zip(*blocks))
    order = np.argsort(time, kind="stable")
    time, zeta, temp = time[order], zeta[order], temp[order]
    for i in np.flatnonzero(np.diff(time) == 0):
        require(np.array_equal(zeta[i], zeta[i+1], equal_nan=True) and
                np.array_equal(temp[i], temp[i+1], equal_nan=True), "conflicting native records at segment overlap")
    keep = np.r_[True, np.diff(time) > 0]
    require(np.sum(keep) >= 2, "station validation requires two or more distinct native times")
    return dict(ids=static[0], sigma=static[1], depth=static[2], lon=static[3], lat=static[4], source_clocks=clocks,
                time=time[keep], zeta=zeta[keep], temp=temp[keep])


def on_clock(clock, field, target, cadence):
    right = np.searchsorted(clock, target).clip(0, len(clock)-1)
    left = np.where(target == clock[right], right, np.maximum(0, right-1))
    span = clock[right]-clock[left]
    support = (target >= clock[0]) & (target <= clock[-1]) & (span <= 1.5*cadence)
    alpha = (target-clock[left])/np.maximum(span, 1)
    if field.ndim == 2:
        alpha = alpha[:, None]
    result = field[left] + alpha*(field[right]-field[left])
    result[~support] = np.nan
    return result


def at_depth(model, index, target, sensor_depth, cadence):
    profile = on_clock(model["time"], model["temp"][:, :, index], target, cadence)
    water_depth = model["depth"][index] + on_clock(model["time"], model["zeta"][:, index], target, cadence)
    profile[~np.isfinite(water_depth) | (water_depth <= 0)] = np.nan
    if not np.isfinite(sensor_depth):
        return profile[:, 0], np.zeros(len(target), dtype=bool)
    require(sensor_depth >= 0, "sensor depth must be positive down")
    depths = -water_depth[:, None]*model["sigma"][:, index]
    above = np.clip((depths < sensor_depth).sum(axis=1)-1, 0, depths.shape[1]-2)
    i = np.arange(len(target))
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.clip((sensor_depth-depths[i, above])/(depths[i, above+1]-depths[i, above]), 0, 1)
    result = profile[i, above] + fraction*(profile[i, above+1]-profile[i, above])
    result[sensor_depth > water_depth] = np.nan
    outer = (sensor_depth < depths[:, 0]) | (sensor_depth > depths[:, -1])
    return result, outer & np.isfinite(result)


def thermocline(depths, profiles):
    """Midpoint of the sharpest stable adjacent measured temperature gradient."""
    require(np.all(np.diff(depths) > 0), "thermistor depths must increase")
    gradient = -np.diff(profiles, axis=1)/np.diff(depths)
    finite = np.isfinite(gradient)
    strongest = np.argmax(np.where(finite, gradient, -np.inf), axis=1)
    support = ((np.isfinite(profiles).sum(axis=1) >= 3) & (finite.sum(axis=1) >= 2) &
               (gradient[np.arange(len(profiles)), strongest] >= 0.15))
    return np.where(support, (depths[strongest] + depths[strongest+1])/2, np.nan)


def wet_mean(field, area, wet):
    mask = np.ma.asarray(wet)
    require(mask.shape == area.shape and not np.ma.getmaskarray(mask).any() and
            np.isin(mask, [0, 1]).all(), "invalid native wet-node mask")
    selected = np.asarray(mask) == 1
    require(selected.any() and np.isfinite(field[selected]).all(), "missing temperatures at native wet nodes")
    return np.average(field[selected], weights=area[selected]), area[selected].sum()/area.sum()


def compact(fields, metrics_path, start, end):
    records = []
    with np.load(metrics_path) as metrics:
        area = np.asarray(metrics["node_control_area_m2"], float)
        require(np.isfinite(area).all() and (area > 0).all() and
                np.array_equal(metrics["node_id"], np.arange(1, len(area)+1)), "invalid native nodal areas/order")
        for path in fields:
            with netCDF4.Dataset(path) as ds:
                check_units(ds)
                require(ds["temp"].dimensions == ("time", "siglay", "node") and
                        ds["wet_nodes"].dimensions == ("time", "node"), "native field dimensions differ")
                require(len(ds.dimensions["node"]) == len(area), "node count differs from native area geometry")
                lon = (values(ds["lon"][:])+180) % 360-180
                require(np.allclose(lon, metrics["lon"], rtol=0, atol=2.1e-5) and
                        np.allclose(values(ds["lat"][:]), metrics["lat"], rtol=0, atol=1e-5),
                        "native field node geography/order differs from area geometry")
                for i, time in enumerate(native_times(ds)):
                    if start <= time < end:
                        mean, fraction = wet_mean(values(ds["temp"][i, 0]), area, ds["wet_nodes"][i])
                        records.append((time, mean, fraction))
    table = pd.DataFrame(records, columns=["epoch", "temperature_c", "wet_area_fraction"])
    require(not table.empty, "no grid fields inside analysis interval")
    require(not table.drop_duplicates().epoch.duplicated().any(), "conflicting grid records at segment overlap")
    return table.drop_duplicates("epoch").sort_values("epoch")
