---
name: topobathy-flownet
description: Build reusable DHSVM-compatible channel, drainage, or bathymetric-thalweg networks from a local GeoTIFF or NetCDF elevation surface and an explicit polygon mask. Use for DEM flow-network extraction, GRASS D8/SFD routing, physical source-area stream initiation, longest-upstream-path SegOrder generation, channel/thalweg QA, or preparation of vector constraints for FVCOM and other spatial workflows.
---

# Topobathy Flownet

Create a fresh, run-local flow network with one fixed method: GRASS GIS 8
`r.watershed -s -a`, followed by `r.stream.extract` with `mexp=0` and
`stream_length=0`.

## Run

1. Confirm that the surface and polygon mask are local, explicit inputs. Never
   discover or reuse a previous run implicitly.
2. State the input vertical sign. The runner always writes a positive-up
   projected surface.
3. Choose a physical source area appropriate to the scientific scale. The
   runner derives the GRASS cell threshold from the actual projected affine
   cell area.
4. Create a new output directory under the task workspace and run:

```powershell
python scripts/run_topobathy_flownet.py `
  --surface <surface.tif-or-nc> `
  --surface-positive up `
  --mask <analysis-mask.gpkg> `
  --mask-layer <optional-layer> `
  --source-area-km2 <area> `
  --out-dir <new-run-folder>
```

For a NetCDF, pass `--surface-variable` explicitly. If GDAL lacks
georeferencing, the runner accepts only a two-dimensional variable backed by
regular, monotonic one-dimensional lon/lat coordinates; it normalizes those
pixel-center axes and records the xarray adapter in the manifest. It never
assigns a CRS to an ungeoreferenced GeoTIFF or non-lon/lat dataset. The default
projected CRS is local WGS84 UTM selected from the mask centroid; pass
`--target-crs` when UTM is unsuitable. Pass `--target-resolution-m` only when
the caller intentionally needs a different square working-cell size;
otherwise the source-derived resolution is preserved. The default GIS launcher is
`C:/OSGeo4W/OSGeo4W.bat`, with `grass85` inside that environment.
On Windows, the runner loads the trusted OSGeo4W environment and invokes GDAL
executables or the GRASS Python launcher directly; run and input paths never
pass through nested batch `%*` expansion.

## Review

- Require `run_manifest.json` schema `topobathy_flownet_v1`, status
  `complete`, and structural status `pass`.
- Require `health_check.json` status `pass`; inspect every failed check if it
  does not pass.
- Review both hillshade-backed QA maps and confirm that high SegOrder arcs and
  accumulation paths follow the intended terrain or topobathymetry.
- Review `topology_qa.json.raw_stream_reader`. Mixed GRASS point primitives
  and degenerate line exports are skipped per primitive with reason counts;
  valid line parts remain the only topology inputs.
- Review `topology_qa.json.orientation`. Increasing accumulation is the
  primary routing direction; elevation is only a tolerance-aware tie fallback,
  and unresolved ties retain GRASS vector direction.
- Treat multiple terminals or sinks as a diagnostic, not a failure. This keeps
  closed analysis domains valid.
- Use `topobathy_flownet.gpkg`, layer `topobathy_flownet`, as the stable
  downstream vector contract. Use `stream.network.dat` only where the DHSVM
  text layout is required.

Read [references/dhsvm_segorder.md](references/dhsvm_segorder.md) when
interpreting topology fields, provenance, or the exact ordering rule.

## Validate the Skill

Run the tests that do not require GRASS:

```powershell
python scripts/selftest_topobathy_flownet.py
python -m py_compile scripts/topobathy_flownet_core.py scripts/run_topobathy_flownet.py
```

Do not download source surfaces, infer a watershed, substitute another routing
method, or silently change the source-area threshold.
