---
name: fvcom-hycom-obc-codex
description: Use when interactively planning, testing, debugging, or extending HYCOM OPeNDAP downloads for FVCOM open-boundary conditions, including surf_el, water_temp, salinity, water_u, and water_v fields, chunked date ranges, point or station extraction, and OBC interpolation workflows.
---

# FVCOM HYCOM OBC Codex

Use this skill for human-interactive development and testing of the HYCOM
external-data connector for FVCOM open-boundary workflows.

## Source Code

- Packaged connector: `scripts/hycom_fetcher.py`
- Packaged common helper: `scripts/grid_utils.py`
- Package initializer: `scripts/__init__.py`
- Remote source: HYCOM THREDDS/OPeNDAP, `https://tds.hycom.org/thredds/dodsC`

## Workflow

1. Inspect the packaged connector before making changes.
2. Use `HycomDownloadRequest` for new driver code instead of hard-coding
   monthly loops.
3. Run examples from the skill `scripts/` directory or add that directory to
   `PYTHONPATH` before importing `hycom_fetcher`; for package-style use, add
   the skill folder parent to `PYTHONPATH` and import through `scripts`.
4. Call `plan_hycom_chunks(request)` before live downloads to review the
   experiment/month split, variables, bounding box, and chunk settings.
5. For live tests, start with tiny time windows and bounding boxes.
6. Use `fetch_hycom(request)` for gridded fields and `fetch_hycom_points`
   for stations or named points.
7. Keep existing functions such as `fetch_hycom_ssh_month`,
   `fetch_hycom_ts_month`, `interp_ssh_to_obc`, and `interp_ts_to_obc`
   compatible unless the user explicitly asks for a breaking refactor.
8. Do not store tokens, credentials, personal accounts, or unsupported approval
   claims in scripts, logs, examples, or generated outputs.

## Variables

- 2D HYCOM field: `surf_el`
- 3D HYCOM fields: `water_temp`, `salinity`, `water_u`, `water_v`
- Accepted aliases: `ssh -> surf_el`, `temp -> water_temp`,
  `salt -> salinity`, `u -> water_u`, `v -> water_v`

## Examples

Dry-run a mixed SSH and current request:

```python
from hycom_fetcher import HycomDownloadRequest, plan_hycom_chunks

req = HycomDownloadRequest(
    start="2019-06-01",
    end="2019-06-03",
    variables=["surf_el", "u", "v"],
    lon_range=(283.0, 288.0),
    lat_range=(36.0, 41.0),
    max_depth=200.0,
    chunk_t=8,
)
for chunk in plan_hycom_chunks(req):
    print(chunk)
```

Fetch temperature and salinity for OBC interpolation:

```python
from hycom_fetcher import HycomDownloadRequest, fetch_hycom, interp_ts_to_obc

req = HycomDownloadRequest(
    start="2019-07-01",
    end="2019-07-02",
    variables=["water_temp", "salinity"],
    lon_range=(283.0, 288.0),
    lat_range=(36.0, 41.0),
    max_depth=200.0,
    cache_dir="cache/hycom",
)
ds = fetch_hycom(req)
obc = interp_ts_to_obc(ds, obc_lon, obc_lat)
```

Fetch named stations or points from the same bounded-grid path:

```python
from hycom_fetcher import HycomDownloadRequest, fetch_hycom_points

req = HycomDownloadRequest(
    start="2019-08-01",
    end="2019-08-01T12:00:00",
    variables=["ssh", "water_u", "water_v"],
    lon_range=(-77.5, -72.0),
    lat_range=(36.0, 41.0),
    points={
        "mouth": {"lon": -75.1, "lat": 38.8},
        "shelf": {"lon": -74.2, "lat": 38.1},
    },
)
station_ds = fetch_hycom_points(req)
```

Compose a driver path for FVCOM boundary products:

```python
from hycom_fetcher import (
    HycomDownloadRequest,
    fetch_hycom,
    interp_ts_to_obc,
    remap_hycom_z_to_sigma,
)

req = HycomDownloadRequest(
    start="2019-09-01",
    end="2019-09-10",
    variables=["temp", "salt", "u", "v"],
    lon_range=(283.0, 288.0),
    lat_range=(36.0, 41.0),
    max_depth=300.0,
)
ds = fetch_hycom(req)
obc = interp_ts_to_obc(ds, obc_lon, obc_lat)
temp_sigma = remap_hycom_z_to_sigma(obc["temp"], obc["depth"], node_depths, siglay)
u_sigma = remap_hycom_z_to_sigma(obc["u"], obc["depth"], node_depths, siglay)
```

## Validation

- Run Python syntax checks after edits.
- Exercise `plan_hycom_chunks` and alias normalization without network access.
- Mock `_fetch_coords`, `_fetch_depth_values`, `_fetch_ssh_block`, and
  `_fetch_ts_block` for unit tests.
- Run live smoke tests only when `pydap` is installed and HYCOM access is
  allowed in the current environment.
