# Sigma-ready boundary T/S input contract

## Required NetCDF content

Provide one NetCDF file with:

- Temperature and salinity on `(time, siglay, node)`. Dimension order may differ, but each dimension must be identifiable.
- A one-dimensional integer boundary-node-ID variable.
- Sigma-layer midpoints and sigma-level interfaces. These may be shared one-dimensional arrays or node-dependent two-dimensional arrays.
- At least two strictly increasing UTC-decodable timestamps.
- Temperature units of Celsius or Kelvin and recognized practical-salinity units.

Default aliases include `obc_temp`, `temperature`, `temp`, or `water_temp`; `obc_salinity`, `salinity`, or `salt`; `obc_nodes` or `node_id`; and `siglay`/`siglev`. Use CLI overrides when discovery is ambiguous.

The input may contain additional node IDs. Every requested FVCOM OBC node ID must occur exactly once; missing or duplicate IDs are fatal. Extra IDs are ignored after being recorded in provenance.

## Sigma requirements

- Use nondimensional sigma coordinates within `[-1,0]`.
- Supply one more `siglev` interface than `siglay` midpoint.
- Include the bed (`-1`) and surface (`0`) interfaces.
- Place each `siglay` strictly between its adjacent interfaces.
- Use one monotonic orientation consistently at every node.

The builder preserves either bottom-to-surface or surface-to-bottom ordering and keeps T/S values aligned with that ordering. It does not remap depth coordinates to sigma coordinates.

## Time behavior

CF numeric time, FVCOM `Times`, and paired `Itime`/`Itime2` are supported. The reader prefers `Times`, then integer MJD plus milliseconds, then CF numeric `time`. Timezone-free text requires explicit `--assume-utc` confirmation.

Regular source time is preserved unless all of `--start`, `--end`, and `--dt-seconds` are supplied. Target times must remain inside source coverage. Gaps larger than `--max-gap-factor` times the median source cadence are rejected.

## Units and physical gates

Recognized Kelvin values are converted to Celsius. Practical salinity is retained numerically. Ambiguous or absent units require explicit overrides; the builder never guesses.

Default accepted ranges are `-5–45 Celsius` and `0–50 PSU`. Override these bounds only with a documented scientific reason. Values outside the reviewed range fail rather than being clipped.

## Missing-value repair

Repair temperature and salinity independently while preserving every original finite value:

1. Interpolate or endpoint-extend each node/layer series in time.
2. Interpolate or surface/bed-extend each time/node profile in sigma space.
3. Interpolate or endpoint-extend along cumulative distance within the same disconnected boundary arc.
4. Repeat until no additional values can be filled.

A single valid value may populate its otherwise-missing line during a repair stage. A line with no valid value remains unavailable for that stage. Repair never crosses disconnected arcs and never inserts an arbitrary climatology or zero. The build fails atomically if any NaN remains.

The report records original and final missing counts, method counts, repaired fraction, iteration history, maximum missing runs, and confirmation that original finite values were unchanged.

## Output contract

The builder writes `NETCDF3_CLASSIC` with:

- Dimensions `nobc`, `siglay`, `siglev`, unlimited `time`, and `DateStrLen=26`.
- `obc_nodes(nobc)` and positive-down `obc_h(nobc)` in metres.
- `siglay(siglay,nobc)` and `siglev(siglev,nobc)`.
- `obc_temp(time,siglay,nobc)` in `Celsius`.
- `obc_salinity(time,siglay,nobc)` in `PSU`.
- One-based `iint`, float32 MJD `time`, integer `Itime`/`Itime2`, and `Times`.
- Literal global `type = "FVCOM TIME SERIES OBC TS FILE"`.

All time variables originate from one integer UTC millisecond axis. The output is promoted to its requested path only after read-back validation and QA creation succeed.
