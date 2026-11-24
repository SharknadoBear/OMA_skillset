"""Sample verified GLORYS12 daily and monthly SSH on a common offshore stencil."""
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

from waterlevel_core import DAY_MS, decode_netcdf_times, haversine_km, sha256_file

PRODUCT = "GLOBAL_MULTIYEAR_PHY_001_030"
DATASET = "cmems_mod_glo_phy_my_0.083deg_"
PUM = "https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-030.pdf"
TIDES_AND_IB = "https://doi.org/10.3390/jmse13122379"


def read_source(folder, frequency):
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    health_path = folder / "health_report.json"
    health = json.loads(health_path.read_text(encoding="utf-8-sig"))
    source = manifest["source"]
    if (not manifest["complete"] or not health["pass"] or health["plan_hash"] != manifest["plan_hash"]
            or source["product_id"] != PRODUCT or source["dataset_id"] != DATASET + frequency + "-m"):
        raise ValueError("Require verified GLORYS12 acquisition of the specified daily/monthly product")
    path = folder / manifest["assembled"]["file"]
    if sha256_file(path) != manifest["assembled"]["sha256"]:
        raise ValueError("GLORYS subset differs from verified acquisition")
    with nc.Dataset(path) as dataset:
        z = dataset["zos"]
        if (z.dimensions != ("time", "latitude", "longitude") or z.units != "m"
                or z.standard_name != "sea_surface_height_above_geoid"):
            raise ValueError("Require geographic GLORYS zos in metres above its geoid")
        clocks, _, _ = decode_netcdf_times(dataset, "time", True)
        lon = np.asarray(dataset["longitude"][:])
        lat = np.asarray(dataset["latitude"][:])
        values = np.ma.filled(z[:].astype("f8"), np.nan)
    if not np.all(np.diff(lon) > 0) or not np.all(np.diff(lat) > 0):
        raise ValueError("GLORYS coordinates must increase on both axes")
    identity = dict(source, subset_sha256=manifest["assembled"]["sha256"],
                    manifest_sha256=sha256_file(manifest_path), health_sha256=sha256_file(health_path))
    return clocks, lon, lat, values, identity


def monthly_days(clocks):
    months = np.asarray(clocks, dtype="datetime64[ms]").astype("datetime64[M]")
    expected = np.arange("1993-01", "2013-01", dtype="datetime64[M]")
    if not np.array_equal(months, expected):
        raise ValueError("The 1993–2012 reference requires all 240 unique consecutive monthly means")
    return ((months + np.timedelta64(1,"M")).astype("datetime64[D]") - months.astype("datetime64[D]")).astype(int)


def daily_midpoints(clocks):
    if len(clocks) < 2 or np.any(np.diff(clocks) != DAY_MS):
        raise ValueError("GLORYS daily source has missing, repeated, or irregular dates")
    phase = np.unique(clocks % DAY_MS)
    if len(phase) != 1 or int(phase[0]) not in (0, DAY_MS // 2):
        raise ValueError("Daily UTC clock is inconsistent with GLORYS noon-centered means")
    return clocks - phase[0] + DAY_MS // 2


def sample_common_stencil(x, y, series, points, nearest_limit_km):
    """Use bilinear wet cells or bounded nearest all-period wet cell; never switch through time."""
    if not np.isfinite(nearest_limit_km) or nearest_limit_km <= 0:
        raise ValueError("Set a positive nearest-wet cutoff in kilometres")
    if (len(points) == 0 or not np.all(np.isfinite(points))
            or np.any(points[:, 0] < x[0]) or np.any(points[:, 0] > x[-1])
            or np.any(points[:, 1] < y[0]) or np.any(points[:, 1] > y[-1])):
        raise ValueError("Every boundary point must be enclosed by the padded source axes")
    wet = np.all(np.isfinite(series), axis=0)
    wet_indices = np.argwhere(wet)
    if not len(wet_indices):
        raise ValueError("No common wet ocean cell through both acquisitions")
    result = np.empty((len(series), len(points)), dtype=float)
    mapping = []
    for target, (lon, lat) in enumerate(points):
        i = np.clip(np.searchsorted(x, lon) - 1, 0, len(x) - 2)
        j = np.clip(np.searchsorted(y, lat) - 1, 0, len(y) - 2)
        u, v = (lon - x[i]) / (x[i+1] - x[i]), (lat - y[j]) / (y[j+1] - y[j])
        nodes = np.asarray([(j,i), (j,i+1), (j+1,i), (j+1,i+1)])
        weights = np.asarray([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v])
        keep = weights > 0
        nodes, weights = nodes[keep], weights[keep]
        method = "bilinear"
        if not np.all(wet[nodes[:, 0], nodes[:, 1]]):
            distances = np.array([haversine_km(lon, lat, x[col], y[row]) for row, col in wet_indices])
            closest = int(np.argmin(distances))
            if distances[closest] > nearest_limit_km:
                raise ValueError(f"No shared offshore wet cell within {nearest_limit_km:g} km of boundary position {target+1}")
            nodes, weights = wet_indices[closest:closest+1], np.ones(1)
            method = "nearest_common_wet"
        coords = np.column_stack((x[nodes[:, 1]], y[nodes[:, 0]]))
        distances = [haversine_km(lon, lat, sx, sy) for sx, sy in coords]
        result[:, target] = series[:, nodes[:, 0], nodes[:, 1]] @ weights
        mapping.append(dict(method=method, longitude_latitude=coords.tolist(), weights=weights.tolist(),
                            distance_km=distances))
    return result, mapping


def prepare(daily_dir, monthly_dir, points, nearest_limit_km):
    daily_t, dx, dy, daily, daily_source = read_source(daily_dir, "P1D")
    month_t, mx, my, monthly, month_source = read_source(monthly_dir, "P1M")
    daily_midpoints(daily_t)
    monthly_days(month_t)
    if (not np.array_equal(dx, mx) or not np.array_equal(dy, my)
            or daily_source["dataset_version"] != month_source["dataset_version"]):
        raise ValueError("Daily and reference SSH need matching GLORYS grid and product version")
    sampled, mapping = sample_common_stencil(dx, dy, np.concatenate((daily, monthly)), points, nearest_limit_km)
    provenance = dict(provider="GLORYS", product=PRODUCT, daily=daily_source, monthly=month_source, mapping=mapping,
                      nearest_limit_km=nearest_limit_km, tide_and_inverse_barometer="both_absent",
                      tide_and_inverse_barometer_evidence=TIDES_AND_IB, daily_center_evidence=PUM,
                      time_method="linear between documented daily noon midpoints; native labels retained",
                      reference="one 1993–2012 day-weighted time mean per ordered boundary node")
    return daily_t, sampled[:len(daily)], month_t, sampled[len(daily):], provenance
