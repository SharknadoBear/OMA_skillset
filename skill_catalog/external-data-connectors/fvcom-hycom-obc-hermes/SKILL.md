---
name: fvcom-hycom-obc-hermes
description: Use for unattended Hermes Agent execution of HYCOM OPeNDAP downloads for FVCOM open-boundary workflows, including explicit request-driven surf_el, water_temp, salinity, water_u, water_v, chunked date ranges, point extraction, and OBC interpolation without interactive prompts.
---

# FVCOM HYCOM OBC Hermes

Use this skill for noninteractive Hermes Agent execution of the HYCOM
external-data connector for FVCOM open-boundary workflows.

## Source Code

- Packaged connector: `scripts/hycom_fetcher.py`
- Packaged common helper: `scripts/grid_utils.py`
- Package initializer: `scripts/__init__.py`
- Remote source: HYCOM THREDDS/OPeNDAP, `https://tds.hycom.org/thredds/dodsC`

## Runtime Contract

1. All inputs must be supplied by the driver, manifest, or calling workflow.
   Do not ask the user questions at runtime.
2. Use `HycomDownloadRequest` as the execution interface.
3. Run from the skill `scripts/` directory or add that directory to
   `PYTHONPATH` before importing `hycom_fetcher`; for package-style use, add
   the skill folder parent to `PYTHONPATH` and import through `scripts`.
4. Call `plan_hycom_chunks(request)` before download and persist or return the
   plan with the run evidence.
5. Use `fetch_hycom(request)` for gridded fields and `fetch_hycom_points`
   for station or named-point products.
6. Fail loudly on invalid variables, invalid date ranges, missing points, empty
   HYCOM index selections, missing dependencies, or failed downloads.
7. Write logs to the supplied `cache_dir` when available.
8. Do not modify source code, install packages, prompt interactively, use
   credentials, or access unrelated URLs during unattended execution.

## Variables

- 2D HYCOM field: `surf_el`
- 3D HYCOM fields: `water_temp`, `salinity`, `water_u`, `water_v`
- Accepted aliases: `ssh -> surf_el`, `temp -> water_temp`,
  `salt -> salinity`, `u -> water_u`, `v -> water_v`

## Manifest Shape

The driver should map its manifest into `HycomDownloadRequest` fields:

```yaml
start: "2019-06-01"
end: "2019-06-03"
variables: ["surf_el", "water_u", "water_v"]
lon_range: [283.0, 288.0]
lat_range: [36.0, 41.0]
max_depth: 200.0
chunk_t: 8
ssh_chunk_t: 24
cache_dir: "cache/hycom"
points:
  mouth:
    lon: -75.1
    lat: 38.8
```

## Examples

Unattended gridded current and SSH download:

```python
from hycom_fetcher import HycomDownloadRequest, plan_hycom_chunks, fetch_hycom

req = HycomDownloadRequest(
    start=manifest["start"],
    end=manifest["end"],
    variables=manifest["variables"],
    lon_range=tuple(manifest["lon_range"]),
    lat_range=tuple(manifest["lat_range"]),
    max_depth=manifest.get("max_depth"),
    cache_dir=manifest.get("cache_dir"),
    chunk_t=manifest.get("chunk_t", 20),
    ssh_chunk_t=manifest.get("ssh_chunk_t", 50),
)
plan = plan_hycom_chunks(req)
ds = fetch_hycom(req)
```

Unattended station extraction:

```python
from hycom_fetcher import HycomDownloadRequest, fetch_hycom_points

req = HycomDownloadRequest(
    start=manifest["start"],
    end=manifest["end"],
    variables=["ssh", "u", "v"],
    lon_range=tuple(manifest["lon_range"]),
    lat_range=tuple(manifest["lat_range"]),
    points=manifest["points"],
    cache_dir=manifest.get("cache_dir"),
)
station_ds = fetch_hycom_points(req)
```

Unattended FVCOM OBC composition:

```python
from hycom_fetcher import (
    HycomDownloadRequest,
    fetch_hycom,
    interp_ts_to_obc,
    remap_hycom_z_to_sigma,
)

req = HycomDownloadRequest(
    start=manifest["start"],
    end=manifest["end"],
    variables=["water_temp", "salinity", "water_u", "water_v"],
    lon_range=tuple(manifest["lon_range"]),
    lat_range=tuple(manifest["lat_range"]),
    max_depth=manifest.get("max_depth", 300.0),
    cache_dir=manifest.get("cache_dir"),
)
ds = fetch_hycom(req)
obc = interp_ts_to_obc(ds, obc_lon, obc_lat)
temp_sigma = remap_hycom_z_to_sigma(obc["temp"], obc["depth"], node_depths, siglay)
v_sigma = remap_hycom_z_to_sigma(obc["v"], obc["depth"], node_depths, siglay)
```

## Validation

- Validate the manifest before network access.
- Test chunk planning and variable aliases without network access.
- Use mocked network helpers in CI or offline validation.
- Run live HYCOM smoke tests only in an environment with `pydap` installed and
  permitted HYCOM network access.
