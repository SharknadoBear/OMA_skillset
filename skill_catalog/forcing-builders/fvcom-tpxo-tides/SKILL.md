---
name: fvcom-tpxo-tides
description: Build TPXO-based FVCOM open-boundary tidal elevation forcing. Use when Codex needs a self-contained toolbox for loading TPXO9 harmonic elevation data, interpolating amplitude/phase to FVCOM OBC nodes, reconstructing tide time series, or writing FVCOM elevation boundary forcing with bundled helper scripts.
---

# FVCOM TPXO Tides

Use this skill as an initial toolbox for FVCOM tidal elevation boundary forcing. It is packaged for completeness and code reuse; do not treat its outputs as scientifically accepted until a case-specific tide validation is performed.

## Core Rules

- Use the existing scripts as library functions. Do not add a new driver unless a project case needs one.
- Require a local TPXO9 elevation file, usually `h_tpxo9.v5a.nc`; do not attempt anonymous TPXO download because TPXO access requires registration.
- Keep longitude handling consistent with `tpxo_tides.py`: negative longitudes are converted to the internal `[0, 360]` convention.
- Interpolate phase through complex phasors, not raw degrees.
- For model-ready NetCDF, combine `tpxo_tides.py` with bundled `fvcom_writer.py` and `grid_utils.py`.

## Bundled Scripts

- `scripts/tpxo_tides.py`: TPXO load, harmonic interpolation, and tide reconstruction helpers.
- `scripts/grid_utils.py`: FVCOM OBC node reading and time conversion helpers.
- `scripts/fvcom_writer.py`: FVCOM NetCDF writer helpers including `write_elevation_obc`.

## Typical Use

Use the scripts from Python:

```python
from tpxo_tides import load_tpxo9, interp_tpxo_to_nodes, reconstruct_tidal_all_nodes
from fvcom_writer import write_elevation_obc
from grid_utils import read_obc_nodes_dat, datetime64_to_mjd
```

Recommended workflow:

1. Read FVCOM OBC node ids and node lon/lat from project mesh/preconfiguration outputs.
2. Load TPXO9 harmonic elevation data from the local TPXO directory.
3. Interpolate amplitudes and phases to OBC nodes.
4. Reconstruct tidal elevation on the FVCOM time axis.
5. Write the FVCOM elevation forcing NetCDF with `write_elevation_obc`.

## Validation

For packaging checks only:

```powershell
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```

For scientific use, compare reconstructed tides with NOAA CO-OPS or another local water-level reference before accepting the forcing.
