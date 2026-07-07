---
name: fvcom-preconfiguration
description: Generate FVCOM ASCII preconfiguration files from SMS 2DM meshes. Use when Codex needs to convert boundary-defined .2dm meshes into FVCOM _grd.dat, _dep.dat, _cor.dat, _obc.dat, _spg.dat, or _sig.dat files, estimate initial sponge-layer parameters, validate 2DM nodestring roles, or package mesh pre-run inputs before FVCOM forcing and model execution.
---

# fvcom-preconfiguration

Use this skill after a FVCOM-ready SMS `.2dm` mesh exists and before forcing, namelist finalization, or model execution.

## Core Rules

- Keep `_dep.dat` and `_grd.dat` depths positive down. For 2DM meshes with negative bed elevations, use the default `--depth-mode auto`.
- Treat `_cor.dat` values as latitude degrees for meter-grid geophysical cases; FVCOM converts them to physical Coriolis internally. Use `--coriolis-mode zero` for laboratory/flume/non-geophysical setups.
- Do not put river/inflow nodestrings into `_obc.dat` or `_spg.dat` when they will be handled by river forcing.
- Treat sponge radius and coefficient as initial smoke-run calibration seeds, not final scientific parameters.
- Generate `_sig.dat` separately from horizontal mesh files. Default to `41` sigma levels and `UNIFORM` unless the project explicitly selects another FVCOM-supported sigma type.

## Primary Commands

Generate horizontal DAT files:

```powershell
python scripts/fvcom_2dm_to_dat.py --mesh MESH.2dm --out-dir OUT --prefix waterPACT --open-ns 1 --river-ns 2 --obc-type prescribed --coriolis-mode zero --sponge-mode estimate
```

Generate sigma configuration:

```powershell
python scripts/fvcom_sig.py --out OUT/waterPACT_sig.dat --levels 41 --type UNIFORM
```

Estimate sponge parameters only:

```powershell
python scripts/estimate_sponge.py --mesh MESH.2dm --nodestring 1 --default-coeff 0.0025
```

Validate generated files:

```powershell
python scripts/selftest_fvcom_preconfig.py --mesh MESH.2dm --out-dir OUT --prefix waterPACT
```

## Output Conventions

`fvcom_2dm_to_dat.py` writes:

- `<prefix>_grd.dat`: node/cell counts, element connectivity, node-id/x/y/depth rows.
- `<prefix>_dep.dat`: node count, x/y/positive-depth rows.
- `<prefix>_cor.dat`: node count, x/y/COR rows.
- `<prefix>_obc.dat`: sequential OBC index, node id, FVCOM OBC type code.
- `<prefix>_spg.dat`: node id, sponge radius, sponge coefficient.
- `<prefix>_fvcom_dat_manifest.json`: provenance, boundary roles, depth/coriolis/sponge choices, and generated files.

`fvcom_sig.py` supports `UNIFORM`, `GEOMETRIC`, `TANH`, `GENERALIZED`, and `USER` syntax accepted by the local FVCOM source. For `USER`, also provide FVCOM's required `sigma_level_user.inp` in `INPUT_DIR`.

## Validation

From the skill folder:

```powershell
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```

For a project-specific mesh, run the selftest with expected node counts, element counts, depth, Coriolis value, OBC nodes, and sigma levels when those expectations are known.
