---
name: fvcom-cudem-bathy
description: Fetch CUDEM-first NOAA bathymetry/topobathymetry for FVCOM preprocessing, with NOAA NBS BlueTopo candidates plus explicit NOAA Coastal Relief Model and ETOPO 2022 fallback. Use when Codex needs to build CUDEM or combined bathymetry indexes, index/sample NOAA NBS BlueTopo GeoTIFF tiles, download or subset CUDEM THREDDS/OPeNDAP NetCDF or Digital Coast GeoTIFF tiles, fill gaps from CRM/ETOPO, export FVCOM-ready positive-down bathymetry NetCDF/PNG products, or interpolate/sample bathymetry to FVCOM grid nodes.
---

# FVCOM CUDEM-First Bathy

Use this skill to fetch NOAA CUDEM bathymetry/topobathymetry and convert it into products that downstream FVCOM grid-generation skills can consume. When requested, use a provenance-preserving fallback stack:

```text
CUDEM + NOAA NBS BlueTopo candidates -> NOAA Coastal Relief Model -> ETOPO 2022
```

## Workflow

1. Work in `Workspace/Preprocessing/fvcom-cudem-bathy` unless the user gives another run directory.
2. Build or reuse a local `cudem_tile_index.json` for CUDEM-only commands, or `bathy_source_index.json` for CUDEM/NBS/CRM/ETOPO fallback commands.
3. Use CUDEM as the preferred conservative source in `--resolution-policy source-priority`. Use `--resolution-policy finest` when Bear wants whichever local source has the finest usable native resolution to win, even if that lets ETOPO 15 arc-second outrank a coarser regional CRM in gaps.
4. For CUDEM-only commands, prefer `resolution=auto`, which selects one resolution tier only: `tiled_19as`, then `tiled_13as`, then `tiled_1as`, then `tiled_3as`.
5. Prefer OPeNDAP NetCDF tiles when available; use Digital Coast HTTPS GeoTIFF tiles for regions such as SE Alaska and Puget Sound that are not advertised in the root THREDDS tile catalog.
6. Export both `elevation_m` and `depth_m`; define `depth_m = max(-elevation_m, 0)`.
7. Preserve metadata: bbox, selected tiles/sources, source URLs, source mode, datum notes, finite coverage, source coverage fractions, NOAA citations, and BlueTopo uncertainty/contributor fields where sampled.
8. Make a PNG diagnostic map for every bbox fetch; fallback-enabled bbox fetches must also write a source-ID PNG.
9. Interpolate to FVCOM nodes only as a separate CSV product. Do not rewrite `.2dm`, `*_grd.dat`, or depth files unless the user explicitly asks for that later.
10. For large sparse meshes, prefer mesh-driven source sampling over a full rectangular native-resolution mosaic.
11. Treat mixed-source products as modeling/preprocessing data. CUDEM, NBS BlueTopo, CRM, and ETOPO can use different vertical datums; record this clearly and do not claim vertical-datum harmonization.
12. For ETOPO 2022, index both 15 arc-second `bed` tiles and the global 15 arc-second `surface` tiles. The `bed` catalog only covers ice-sheet bedrock tiles; the `surface` catalog provides the global fallback coverage needed for ordinary coastal domains.

## Commands

Build a tile index:

```powershell
python scripts\build_cudem_index.py --output Workspace\Preprocessing\fvcom-cudem-bathy\cache\cudem_tile_index.json
```

Build a combined CUDEM/NBS/CRM/ETOPO source index:

```powershell
python scripts\build_bathy_source_index.py --output Workspace\Preprocessing\fvcom-cudem-bathy\cache\bathy_source_index.json
```

Fetch one bbox:

```powershell
python scripts\fetch_cudem_bathy.py --bbox -75.35 38.75 -74.95 39.10 --run-dir Workspace\Preprocessing\fvcom-cudem-bathy\runs\delaware_bay --name delaware_bay --index Workspace\Preprocessing\fvcom-cudem-bathy\cache\cudem_tile_index.json
```

Fetch one bbox with NBS/CRM/ETOPO fallback:

```powershell
python scripts\fetch_bathy_sources.py --bbox -75.8150778 37.6409854 -73.5018316 40.2205458 --run-dir Workspace\Preprocessing\fvcom-cudem-bathy\runs\base_cd_bathy_fallback_smoke --name base_cd_bathy_fallback --index Workspace\Preprocessing\fvcom-cudem-bathy\cache\bathy_source_index.json --fallback-policy cudem-nbs-crm-etopo --resolution-policy source-priority --target-spacing-arcsec 3
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

Sample CUDEM/NBS/CRM/ETOPO fallback sources directly to a large mesh:

```powershell
python scripts\compare_bathy_sources_to_mesh.py --mesh-2dm Resources\SE_AK_merged_complete_v5_latlon.2dm --run-dir Workspace\Preprocessing\fvcom-cudem-bathy\runs\seak_bathy_fallback_smoke --name seak_bathy_fallback --index Workspace\Preprocessing\fvcom-cudem-bathy\cache\bathy_source_index.json --fallback-policy cudem-nbs-crm-etopo --resolution-policy source-priority
```

Run a regional source-discovery review without injecting unvalidated products into the fetch stack:

```powershell
python scripts\research_bathy_sources.py --bbox -139.9796460 50.7216567 -128.5859190 59.4927575 --run-dir Workspace\Preprocessing\fvcom-cudem-bathy\runs\seak_source_research --name seak_source_research
```

## References

- Read `references/noaa_cudem_sources.md` before changing source URLs, source priority, datum assumptions, tile parsing, NBS handling, or smoke-test regions.

## Guardrails

- Do not download whole CUDEM bulk folders. Fetch only intersecting tiles.
- Do not download whole CRM or ETOPO files for ordinary runs; use OPeNDAP subsetting/windowed reads.
- Do not download whole NBS BlueTopo collections. Use the BlueTopo tile scheme to select intersecting tiles, and for large meshes sample only mesh nodes inside selected tile windows.
- Keep `target_spacing_arcsec` coarse enough for smoke tests when using large HTTPS GeoTIFF tiles.
- For large meshes, stream/read selected GeoTIFF windows and sample mesh nodes directly.
- Use `--resolution-policy source-priority` for conservative CUDEM-first products and `--resolution-policy finest` when Bear wants the finest usable local source to win per node/cell across CUDEM, NBS, CRM, and ETOPO.
- Treat CUDEM products as modeling/planning data, not navigation data.
- Use Matplotlib-only plotting; do not require Cartopy.
