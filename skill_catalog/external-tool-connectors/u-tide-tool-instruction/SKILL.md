---
name: u-tide-tool-instruction
description: Install and use the public Python UTide solve and reconstruct interfaces for scalar water-level or vector-current harmonic analysis, tidal prediction, constituent diagnostics, confidence filtering, and residual checks. Use when Codex needs UTide guidance without cloning or vendoring its repository.
---

# UTide Tool Instruction

Use the official [wesleybowman/UTide](https://github.com/wesleybowman/UTide)
Python package. Do not clone or vendor the repository into this skill.

## Install And Record

Install in the active project environment:

```bash
python -m pip install utide
# or
conda install --channel conda-forge utide
```

Record `python -c "import utide; print(utide.__version__)"`, the time convention,
latitude, options, and retained constituents with every reproducible result. Treat
the package interface as version-sensitive; its official repository states that
the software remains under active development.

## Prepare Inputs

- Use a monotonically increasing UTC time array. Remove duplicate timestamps and
  document gaps, sampling changes, and timezone conversion.
- Use finite observations in consistent physical units and provide representative
  latitude because nodal corrections depend on latitude.
- Detrend or retain a trend deliberately. Do not use harmonic prediction to hide
  datum shifts, sensor drift, non-tidal surge, or unresolved gaps.
- Ensure the record length and sampling support the requested constituent
  separation; use `Rayleigh_min` and an explicit constituent list when necessary.

## Scalar Water-Level Analysis

```python
import numpy as np
from utide import reconstruct, solve

coef = solve(
    time_utc,
    water_level,
    lat=latitude,
    nodal=True,
    trend=True,
    method="ols",
    conf_int="linear",
    Rayleigh_min=0.95,
    verbose=False,
)
prediction = reconstruct(
    time_utc, coef, min_SNR=2, min_PE=0, verbose=False
).h
residual = np.asarray(water_level) - np.asarray(prediction)
```

Inspect constituent names, amplitudes, phases, confidence intervals, signal-to-noise
ratio, percent energy, fitted mean/trend, predicted series, and residuals. Repeat the
analysis with reviewed filtering thresholds rather than silently discarding weak
constituents.

## Vector-Current Analysis

Pass both earth-relative velocity components to `solve`:

```python
from utide import reconstruct, solve

coef = solve(
    time_utc,
    eastward_velocity,
    northward_velocity,
    lat=latitude,
    nodal=True,
    trend=True,
    method="ols",
    conf_int="linear",
    Rayleigh_min=0.95,
    verbose=False,
)
prediction = reconstruct(time_utc, coef, min_SNR=2, verbose=False)
u_tide = prediction.u
v_tide = prediction.v
u_residual = eastward_velocity - u_tide
v_residual = northward_velocity - v_tide
```

Preserve the coordinate convention used for east/north components and examine the
ellipse parameters and uncertainties returned by the installed version.

## Validation

- Plot observations, reconstruction, and residual together and over representative
  spring-neap intervals.
- Check residual bias, variance, autocorrelation, missing intervals, extrema, and
  constituent stability under modest changes to record window and options.
- Compare major constituent amplitude and phase against an independent gauge,
  published harmonic station, or model product when available.
- Treat confidence intervals and automated SNR/energy thresholds as diagnostics,
  not proof of physical validity.

UTide analyzes or reconstructs time series. It does not read TPXO files, interpolate
spatial harmonic grids, or create model forcing. Use `$tpxo9v5-data-fetcher` first
when the source is TPXO. Avoid private UTide modules and hand-built internal
coefficient objects in basic workflows; those interfaces can change between
versions.
