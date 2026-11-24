"""Read independently documented and hash-bound HYCOM SSH and reference subsets."""
import json
from pathlib import Path
from urllib.parse import urlparse

import netCDF4 as nc
import numpy as np

from glorys_obc import monthly_days, sample_common_stencil
from waterlevel_core import DAY_MS, decode_netcdf_times, sha256_file

TERMS = ("tide", "inverse_barometer")


def validate_policy(source):
    if source.get("provider") != "HYCOM" or source.get("schema") != "hycom_offshore_reference_v1":
        raise ValueError("Invalid HYCOM fallback policy")
    for name in ("run", "reference"):
        item = source[name]
        if any(not isinstance(item.get(key),str) or not item[key].strip() for key in
               ("experiment","vertical_reference","ssh_variable","clock","evidence_summary")):
            raise ValueError(f"Incomplete HYCOM {name} experiment, datum, clock, or evidence")
        if urlparse(item["evidence_url"]).scheme != "https":
            raise ValueError("Require an HTTPS source for the exact HYCOM product conventions")
        if set(item["physics"]) != set(TERMS):
            raise ValueError("HYCOM source must declare both tide and inverse-barometer conventions")
        variables = {item["ssh_variable"]}
        for term in TERMS:
            rule = item["physics"][term]
            if rule == {"convention":"absent"}:
                continue
            variable = rule.get("variable")
            if (set(rule) != {"convention","variable"} or rule["convention"] != "present_subtract"
                    or not isinstance(variable,str) or not variable or variable in variables):
                raise ValueError(f"Ambiguous or duplicate {term} correction for {name}")
            variables.add(variable)
    if source["run"]["vertical_reference"] != source["reference"]["vertical_reference"]:
        raise ValueError("HYCOM SSH and historical reference must share a documented vertical datum")
    if (source["run"]["clock"] != "native_utc_sample" or source["reference"]["clock"] != "calendar_month_mean"
            or source["run"].get("cadence_seconds",0) <= 0
            or source["run"]["cadence_seconds"] > DAY_MS/1000):
        raise ValueError("Missing documented HYCOM sample clock, cadence, or calendar-month means")


def artifact(root, ref):
    path = Path(ref["path"])
    path = path if path.is_absolute() else root/path
    if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != ref["sha256"]:
        raise ValueError(f"Changed or missing HYCOM source/evidence: {path.name}")
    return path


def read_record(root, item):
    path = artifact(root, item["file"])
    artifact(root, item["evidence_file"])
    coord = item["coordinates"]
    with nc.Dataset(path) as ds:
        lon = np.asarray(ds[coord["longitude"]][:],dtype=float)
        lat = np.asarray(ds[coord["latitude"]][:],dtype=float)
        times, _, td = decode_netcdf_times(ds, coord["time"], True)
        if (lon.ndim != 1 or lat.ndim != 1 or len(lon)<2 or len(lat)<2
                or ds[coord["longitude"]].units not in ("degrees_east","degree_east")
                or ds[coord["latitude"]].units not in ("degrees_north","degree_north")):
            raise ValueError("Require declared regular geographic HYCOM coordinates")
        def field(var):
            v = ds[var]
            if (v.dimensions != (td,ds[coord["latitude"]].dimensions[0],ds[coord["longitude"]].dimensions[0])
                    or v.units not in ("m","meter","meters","metres")):
                raise ValueError("HYCOM SSH and corrections must share source dimensions and metres")
            return np.ma.asarray(v[:],dtype="f8").filled(np.nan)
        fields = {"raw":field(item["ssh_variable"])}
        for term, rule in item["physics"].items():
            if rule["convention"] == "present_subtract":
                fields[term] = field(rule["variable"])
    # Normalize conventional 0..360 longitude, retaining source values and field alignment.
    lon = (lon+180)%360-180
    order_x, order_y = np.argsort(lon), np.argsort(lat)
    lon, lat = lon[order_x], lat[order_y]
    if np.any(np.diff(lon)<=0) or np.any(np.diff(lat)<=0) or np.ptp(lon)>180:
        raise ValueError("Duplicate or dateline-crossing HYCOM axes")
    return times, lon, lat, {key:values[:,order_y][:,:,order_x] for key,values in fields.items()}


def corrected(fields, rule):
    expected = {"raw", *(term for term in TERMS if rule["physics"][term]["convention"] == "present_subtract")}
    if set(fields) != expected:
        raise ValueError("Missing or unrequested HYCOM correction in component evidence")
    value = fields["raw"].copy()
    for term in TERMS:
        if term in fields:
            value -= fields[term]
    return value


def prepare_hycom(package, points, nearest_limit_km):
    package = Path(package)
    source = json.loads(package.read_text(encoding="utf-8-sig"))
    validate_policy(source)
    clocks = {}
    axes = {}
    groups = []
    for name in ("run","reference"):
        t, x, y, fields = read_record(package.parent,source[name])
        clocks[name], axes[name] = t,(x,y)
        groups.extend((name,key,array) for key,array in fields.items())
    if any(not np.array_equal(a,b) for a,b in zip(axes["run"],axes["reference"])):
        raise ValueError("HYCOM run/reference must use compatible identical geographic source axes")
    monthly_days(clocks["reference"])
    if len(clocks["run"]) < 2 or np.any(np.diff(clocks["run"]) != 1000*source["run"]["cadence_seconds"]):
        raise ValueError("Missing or irregular HYCOM native run samples")
    samples, mapping = sample_common_stencil(*axes["run"], np.concatenate([v for _,_,v in groups]), points, nearest_limit_km)
    fields_by_period = {"run":{}, "reference":{}}
    offset = 0
    for name,key,array in groups:
        fields_by_period[name][key] = samples[offset:offset+len(array)]
        offset += len(array)
    source.update(mapping=mapping, nearest_limit_km=nearest_limit_km, package_sha256=sha256_file(package),
                  time_method="linear on native UTC samples, with no temporal extrapolation",
                  reference_method="1993–2012 day-weighted independent monthly reference")
    return (clocks["run"], corrected(fields_by_period["run"],source["run"]), clocks["reference"],
            corrected(fields_by_period["reference"],source["reference"]),source,fields_by_period)
