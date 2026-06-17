---
name: fvcom-cudem-bathy
description: Fetch NOAA CUDEM bathymetry/topobathymetry for FVCOM preprocessing. Use when Codex needs to build a CUDEM tile index, download or subset CUDEM THREDDS/OPeNDAP NetCDF or Digital Coast GeoTIFF tiles for a bbox, export FVCOM-ready positive-down bathymetry NetCDF/PNG products, or interpolate CUDEM bathymetry to FVCOM grid nodes.
---

# FVCOM CUDEM Bathy

Use this skill to fetch NOAA CUDEM bathymetry/topobathymetry and convert it into products that downstream FVCOM grid-generation skills can consume.

## Workflow

1. Work in `Workspace/Preprocessing/fvcom-cudem-bathy` unless the user gives another run directory.
2. Build or reuse a local `cudem_tile_index.json` before live data fetches.
3. Use CUDEM-only coverage for v1. If no CUDEM tile covers the bbox, report no coverage instead of silently switching to another product.
4. Prefer `resolution=auto`, which selects one resolution tier only: `tiled_19as`, then `tiled_13as`, then `tiled_1as`, then `tiled_3as`.
5. Prefer OPeNDAP NetCDF tiles when available; use Digital Coast HTTPS GeoTIFF tiles for regions such as SE Alaska and Puget Sound that are not advertised in the root THREDDS tile catalog.
6. Export both `elevation_m` and `depth_m`; define `depth_m = max(-elevation_m, 0)`.
7. Preserve metadata: bbox, selected tiles, source URLs, source mode, datum notes, finite coverage, and NOAA citation.
8. Make a PNG diagnostic map for every bbox fetch.
9. Interpolate to FVCOM nodes only as a separate CSV product. Do not rewrite `.2dm`, `*_grd.dat`, or depth files unless the user explicitly asks for that later.
10. For large sparse meshes, prefer mesh-driven tile sampling over a full rectangular native-resolution mosaic.

## Commands

Build a tile index:

```powershell
python scripts\build_cudem_index.py --output Workspace\Preprocessing\fvcom-cudem-bathy\cache\cudem_tile_index.json
```

Fetch one bbox:

```powershell
python scripts\fetch_cudem_bathy.py --bbox -75.35 38.75 -74.95 39.10 --run-dir Workspace\Preprocessing\fvcom-cudem-bathy\runs\delaware_bay --name delaware_bay --index Workspace\Preprocessing\fvcom-cudem-bathy\cache\cudem_tile_index.json
```

Run required smoke tests:

```powershell
python scripts\smoke_cudem_estuaries.py --run-dir Workspace\Preprocessing\fvcom-cudem-bathy\runs\smoke --index Workspace\Preprocessing\fvcom-cudem-bathy\cache\cudem_tile_index.json
```

Interpolate a bathymetry product to a mesh:

```powershell
python scripts\interpolate_cudem_to_fvcom.py --bathy-netcdf case_cudem_bathy.nc --mesh-2dm mesh.2dm --output-csv node_depths.csv
```

Compare CUDEM directly against the large SE-AK mesh:

```powershell
python scripts\compare_seak_cudem.py --mesh-2dm Resources\SE_AK_merged_complete_v5_latlon.2dm --run-dir Workspace\Preprocessing\fvcom-cudem-bathy\runs\SE_AK --rebuild-index
```

## References

- Read `references/noaa_cudem_sources.md` before changing source URLs, source priority, datum assumptions, tile parsing, or smoke-test regions.

## Guardrails

- Do not download whole CUDEM bulk folders. Fetch only intersecting tiles.
- Keep `target_spacing_arcsec` coarse enough for smoke tests when using large HTTPS GeoTIFF tiles.
- For large meshes, stream/read selected GeoTIFF windows and sample mesh nodes directly.
- Treat CUDEM products as modeling/planning data, not navigation data.
- Use Matplotlib-only plotting; do not require Cartopy.
