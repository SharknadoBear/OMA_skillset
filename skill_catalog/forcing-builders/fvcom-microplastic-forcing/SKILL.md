---
name: fvcom-microplastic-forcing
description: Prepare microplastic concentration and tracer helper products for FVCOM river forcing. Use when Codex needs a self-contained toolbox for DRBC microfiber workbook parsing, SR size-range extrapolation, particle number-to-mass conversion, FVCOM-MP particle config parsing, or river microplastic tracer arrays.
---

# FVCOM Microplastic Forcing

Use this skill as an initial microplastic tracer toolbox for FVCOM river-forcing setup. It is packaged for completeness and reuse; do not treat computed concentrations as accepted forcing without project-specific data and particle-parameter review.

## Core Rules

- Use `scripts/microplastic_forcing.py` as the source of truth for microplastic concentration calculations.
- Keep DRBC workbook parsing, size-spectrum extrapolation, particle geometry, and mass conversion in this skill rather than rewriting formulas in river-forcing code.
- Do not assume the DRBC workbook or FVCOM-MP `generic_plastic.inp` files are bundled with the skill; pass explicit paths when project layout differs.
- Preserve units: particle number concentration is converted through particle mass to `g/L` numerically equivalent to `kg/m3`.

## Bundled Scripts

- `scripts/microplastic_forcing.py`: DRBC workbook parsing, Schuylkill River supplementary extrapolation, FVCOM-MP particle config parsing, boundary concentration summaries, and river `mp1` array helpers.

## Typical Use

Use from Python:

```python
from microplastic_forcing import (
    build_case_microfiber_summary,
    boundary_mp_concentrations_for_case,
    river_mp_array_from_boundary_values,
)
```

Recommended workflow:

1. Parse or provide the FVCOM-MP particle configuration for the case.
2. Load the DRBC microfiber workbook or another compatible observation table.
3. Compute boundary concentration summaries for Trenton and Schuylkill-style sources.
4. Pass the resulting concentrations to `fvcom-river-forcing` when writing river tracer arrays.

## Validation

For packaging checks only:

```powershell
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```

For scientific use, review workbook rows, particle settings, extrapolation ranges, and source-to-river mapping before accepting the forcing.
