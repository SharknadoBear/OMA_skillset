---
name: fvcom-tpxo-tides
description: Build scientifically gated FVCOM open-boundary tidal elevation forcing from model-neutral TPXO9v5 harmonic products. Use when Codex needs exact ordered OBC-node mapping, complex-coefficient interpolation, astronomical and nodal reconstruction, FVCOM elevation NetCDF, or interpolation diagnostics.
---

# FVCOM TPXO Tides

Use this skill to convert a validated `tpxo9v5_harmonics_v1` point or native-subset
product into FVCOM elevation forcing. This forcing gate proves source coverage,
node order, phase convention, time coverage, and file readability. Scientific
acceptance still requires case-specific tide validation.

## Core Rules

- Consume only a local model-neutral product produced by `tpxo9v5-data-fetcher`.
  Never download TPXO here and never bypass its registered-data access gate.
- Default to every constituent discovered in the product. The Galveston workflow
  expects 22; a different count blocks until the caller supplies a documented
  scientific explanation.
- Use an ordered CSV with `node_id,longitude,latitude`. Point products must carry
  the same IDs in `target_id(point)` and the exact same coordinates and order.
- For native subsets, interpolate `A*exp(-i*g)` complex coefficients, never phase
  angles. Use bounded nearest-wet fallback and block any unresolved value.
- Interpret `g` as Greenwich phase lag. Reconstruct
  `Re{f exp[i(V+u)] A exp(-ig)}` using UTide's astronomical argument `V` and
  time-varying nodal amplitude/phase corrections `f,u`, evaluated at each OBC
  node latitude. Do not silently fall back to an uncorrected harmonic sum.
- Use UTide flags `[False, False, False, False]` for exact-time nodal and
  astronomical evaluation. The first flag set to true freezes nodal `f,u` at
  the record midpoint. Require `utide_exact_time_nodal_v1`, the flags, UTide
  version and builder SHA-256 in the manifest and matching NetCDF attributes.
  Preserve older products as historical evidence; a corrected forcing method
  requires new spin-up/restart, benchmarks, production and validation.
- Build a strictly monotonic, inclusive UTC axis. Write FVCOM MJD time as float64;
  float32 cannot reliably distinguish six-minute steps at present-day dates.
- Write atomically and publish source, point-order, forcing, and diagnostics hashes.
- Runtime dependencies are Python 3.10+, NumPy, SciPy, netCDF4, and UTide. UTide
  is mandatory for production reconstruction; a missing installation is a blocker.

## Bundled Scripts

- `scripts/prepare_obc_points.py`: convert a hash-bound grid-delivery contract
  into the exact ordered geographic OBC CSV required by the forcing builder.
- `scripts/build_fvcom_tides.py`: production CLI, scientific gates, exact-order
  mapping, UTide reconstruction, atomic FVCOM output, diagnostics, and manifest.
- `scripts/selftest_fvcom_tpxo_tides.py`: offline 22-constituent, phase-wrap,
  time/order, unit-conversion, and dry-source fallback tests.
- `scripts/selftest_nodal_time.py`: full-window, 22-constituent regression against
  an independent cosine expansion, plus invariance across reconstruction windows.
- `scripts/tpxo_tides.py`: legacy direct-source loading and reconstruction helpers;
  do not use it to bypass the model-neutral connector in production.
- `scripts/grid_utils.py`: FVCOM OBC node reading and time conversion helpers.
- `scripts/fvcom_writer.py`: FVCOM NetCDF writer helpers including `write_elevation_obc`.

## Typical Use

For an exact-point product created from the ordered OBC CSV:

```python
python scripts/build_fvcom_tides.py \
  --harmonics forcing/shared_source/galveston_obc_harmonics.nc \
  --obc-points input/accepted_t6v6/base/obc_points.csv \
  --obc-dat input/accepted_t6v6/base/galveston_obc.dat \
  --start 2025-03-25T00:00:00Z --end 2025-05-01T00:00:00Z \
  --interval-minutes 6 --expected-constituents 22 \
  --case-name galveston_tide_2025 \
  --output forcing/accepted_t6v6/galveston_elevtide.nc \
  --report forcing/accepted_t6v6/forcing_manifest.json
```

Required gates:

1. Confirm the source product schema, source hash, constituent names/count, and
   elevation units.
2. Prove exact OBC ID, longitude, latitude, and row-order agreement for point
   products, or interpolate the native subset in the supplied OBC order.
3. Block unresolved/dry-out-of-range/NaN coefficients; record nearest-wet use.
4. Reconstruct the inclusive UTC axis with documented astronomical/nodal handling.
5. Validate exact OBC order, float64 monotonic time, endpoints, interval, shape,
   and finite elevation before replacing the requested output.
6. Return `status`, artifact root, hashes, provenance, warnings, blocking reasons,
   and resume token in the forcing report.

## Validation

For packaging checks only:

```powershell
python -m compileall scripts
python scripts/selftest_fvcom_tpxo_tides.py
python scripts/selftest_nodal_time.py
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```

For scientific use, compare reconstructed tides with NOAA CO-OPS or another local water-level reference before accepting the forcing.
