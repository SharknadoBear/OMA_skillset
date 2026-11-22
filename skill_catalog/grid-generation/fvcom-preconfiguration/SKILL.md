---
name: fvcom-preconfiguration
description: Generate hash-bound FVCOM ASCII preconfiguration packages from complete grid-delivery contracts or legacy SMS 2DM meshes. Use when Codex must preserve plural OBC chains and exact node order, write projected-metric grid/depth files, derive per-node latitude, bind attempt-specific sponge controls, or create ten-level sigma inputs before forcing and execution.
---

# fvcom-preconfiguration

Use this skill after an FVCOM-ready SMS `.2dm` mesh exists and before forcing,
namelist finalization, or model execution.

## Core Rules

- Prefer a `fvcom_grid_delivery_contract_v1` input. Verify its mesh hash,
  counts, projected CRS/units, and every OBC node-order hash before writing.
- Keep `_dep.dat` and `_grd.dat` depths positive down. For 2DM meshes with
  negative bed elevations, use the default `--depth-mode auto`.
- Use the installed `pyproj` engine with `always_xy=True` for coordinate
  transforms and record its version in the DAT manifest. Identical CRSs retain
  the serialized coordinates. When `pyproj` is unavailable, the legacy UTM
  series fallback is disclosed and still requires the independent delivery
  coordinate audit; transform errors never activate a fallback.
- Treat `_cor.dat` values as latitude degrees for geophysical meter grids;
  FVCOM converts them to physical Coriolis internally. Contract mode derives
  every latitude from the declared CRS. EPSG:326xx/327xx works without an
  optional GIS package; other CRSs require `pyproj`. Use zero only for a
  laboratory or non-geophysical setup.
- Preserve each OBC chain independently in the manifest and flatten chains
  into `_obc.dat` only in contract order. Never sort, reverse, concatenate
  away chain identity, or allow nodes to overlap between OBCs.
- Preserve an explicitly declared boolean `cyclic` in each OBC manifest row.
  A cyclic chain has at least three unique nodes, omits a repeated first node,
  and includes its last-to-first edge exactly once in exterior topology and
  median-edge sponge audits. An absent `cyclic` retains legacy open-chain behavior.
- Audit boundary-cell topology before writing DAT files using FVCOM 4.3.1's
  exact `TGE.F` node-flag sequence: mark exterior nodes `ISONB=1`, overwrite
  OBC nodes with `ISONB=2`, accept an open cell when its three-node sum equals
  four, and reject sums greater than four. For a production build, pass the
  frozen executable's exact source through `--tge-source`; verify the defining
  expressions and publish its SHA-256. Route a failure back to gridding so the
  OBC endpoint or exterior topology is repaired and every dependent hash and
  forcing file is regenerated.
- Do not put river/inflow nodestrings into `_obc.dat` or `_spg.dat` when river
  forcing owns them.
- Set each baseline sponge radius to three times that OBC's local median edge
  length and coefficient to `0.0025`. These are stability seeds, not final
  scientific parameters.
- For a stability attempt, consume a hash-bound `fvcom_sponge_profile_v1`
  carrying an immutable `attempt_id`. Apply only explicit per-OBC values or
  multipliers and retain both the baseline and override in the manifest.
- Generate `_sig.dat` with the horizontal package. Default to ten uniform
  sigma levels unless the project contract explicitly selects another
  FVCOM-supported sigma type.

Read [the input contract](references/grid_delivery_contract.md) before
authoring or consuming a new contract.

## Primary Commands

Generate the complete package:

```powershell
python scripts/fvcom_2dm_to_dat.py --grid-contract GRID_CONTRACT.json --out-dir OUT --prefix galveston --tge-source TGE.F
```

Apply an immutable stability-attempt sponge profile:

```powershell
python scripts/fvcom_2dm_to_dat.py --grid-contract GRID_CONTRACT.json --out-dir ATTEMPT_OUT --prefix galveston --attempt-id attempt_0002 --sponge-profile attempt_0002_sponge.json
```

The legacy route remains available for bounded compatibility work:

```powershell
python scripts/fvcom_2dm_to_dat.py --mesh MESH.2dm --out-dir OUT --prefix case --open-ns 1 --open-ns 2 --source-crs EPSG:32615 --coriolis-mode crs-latitude
```

Estimate one baseline sponge:

```powershell
python scripts/estimate_sponge.py --mesh MESH.2dm --nodestring 1 --default-coeff 0.0025 --radius-scale 3
```

## Output Conventions

`fvcom_2dm_to_dat.py` writes:

- `<prefix>_grd.dat`: node/cell counts, element connectivity, and metric
  node-id/x/y/positive-depth rows.
- `<prefix>_dep.dat`: node count and metric x/y/positive-depth rows.
- `<prefix>_cor.dat`: node count and metric x/y/geodetic-latitude rows.
- `<prefix>_obc.dat`: sequential flattened index, node ID, and FVCOM type.
- `<prefix>_spg.dat`: node ID, per-OBC radius, and coefficient.
- `<prefix>_sig.dat`: FVCOM sigma-coordinate configuration.
- `<prefix>_fvcom_dat_manifest.json`: ready status, input/output hashes, CRS
  latitude range, exact per-OBC and flattened order hashes, sigma selection,
  and baseline/attempt sponge controls.

The standalone `fvcom_sig.py` supports `UNIFORM`, `GEOMETRIC`, `TANH`,
`GENERALIZED`, and `USER`. `USER` also requires FVCOM's
`sigma_level_user.inp` in `INPUT_DIR`.

## River intake and downstream volume budget

For `barotropic-tide-river`, run the geometry builder after producing the metric
grid and OBC. Use a reviewed, bank-to-bank upstream geometric boundary segment.
Draw the downstream section across the whole wetted reach, extending both ends
onto land. Supply WGS84 longitude/latitude pairs in `cutoff_line_lonlat` and
`section_line_lonlat` (two endpoints each), `gauge_lonlat`, `horizontal_crs`
(the runtime grid's projected metre CRS), `cutoff_tolerance_m`,
`wet_depth_floor_m`, and `cutoff_description` in the request JSON. Select the
cutoff matching tolerance from actual mesh/bank displacement. Select the depth
floor from the low-water level and model wetting threshold.

```powershell
python scripts/prepare_river_geometry.py --grid case_grd.dat --obc case_obc.dat --request section_request.json --out-dir river_geometry
```

`river_mapping.json` records receiving node IDs, width times reference-depth
weights, wet connections to the ocean, and gauge-to-cutoff distance. The distance
is a straight-line geodesic. `river_budget_geometry.json` records the upstream
triangle IDs, downstream cut edges, upstream-side sampling cells, and seaward
unit normals in the model's projected frame. Each output binds the grid, OBC
and geometry request by hash. Verify receiving-node wetness from the run and
disclose any excluded shallow flow paths and the resulting discharge allocation.

## Validation

From the skill folder:

```powershell
python scripts/selftest_contract_preconfiguration.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```

For a project mesh, also run `selftest_fvcom_preconfig.py` with known node,
element, OBC, and sigma counts.

For diagnosed noncyclic endpoint-only TGE failures, return the exact delivery and sidecars to the Grid-owned `repair_fvcom_tge_endpoints.py` command. Require its passing source-bound result, then rebuild this package and all tidal forcing against the returned OBC order. Keep the repair implementation in Grid.
