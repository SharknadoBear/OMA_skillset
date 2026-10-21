---
name: glorys-fetcher
description: Inventory, plan, download, resume, and independently validate bounded daily or monthly GLORYS12 published ocean fields through the Copernicus Marine Toolbox. Use for regional water level, potential temperature, salinity, depth-resolved horizontal currents, or other catalogue-listed GLORYS12 fields.
---

# GLORYS Fetcher

Acquire model-neutral source subsets with `scripts/glorys_fetcher.py`. Keep requests, downloaded data, and run evidence outside this skill. Read [the request contract](references/request_contract.md) when constructing requests and [source interpretation](references/source_interpretation.md) when using fields downstream.

Use Python 3.11 in an isolated `.venv-glorys` environment. Install [the frozen requirements](references/requirements-lock.txt), which pin Copernicus Marine Toolbox 2.5.0. This skill has no dependency on another OMA skill. Its CLI module exports the corresponding Python helpers; pass a backend only for explicitly offline testing.

## Acquisition

1. Run `check-runtime --access`. Credentials stay in the local Toolbox configuration or its documented environment variables. If missing, have the user run `copernicusmarine login` locally; no API key or password in chat. A CDSE browser session alone does not prove Toolbox authentication.
2. Run `inventory --output inventory.json`, then `plan --request request.json --run-dir RUN --toolbox-estimate`. Planning reads actual source coordinates without field-data retrieval, resolves the version, and freezes it. Review selected clocks, depths, halo, decoded volume and disk headroom. Toolbox estimates describe output size; ARCO network overhead is unknown.
3. Run `fetch --run-dir RUN`. Default calendar-month chunks may be subdivided by the decoded memory budget. For a small regional multi-year series, compare Toolbox estimates and use request `service: "arco-time-series"` with `chunk_mode: "bounded_batch"` when it avoids repeated source-block reads. Batches cross month boundaries but retain the same decoded chunk budget, atomic validation and identity rules. The plan freezes the service and batch selection. `--max-chunks 1` allows a controlled pause. Repeat the same command to resume or verify a completed run. Verified chunks are reused; corrupt committed chunks fail visibly. A file without a commit receipt is reacquired. Three attempts apply only to transient transport failures.
4. Run `health --run-dir RUN` and `inspect --file RUN/subset.nc`. Only a complete independently verified run passes health. Validate masks and offshore coverage before interpolation. Use `scripts/render_glorys_report.py` for a compact visual review when the five ocean pilot fields are present.

Example invocation, with `PY` pointing to the environment's Python executable and `ENTRY` to this skill's `scripts/glorys_fetcher.py`:

```powershell
& $PY $ENTRY plan --request request.json --run-dir pilot --toolbox-estimate
& $PY $ENTRY fetch --run-dir pilot --max-chunks 1
& $PY $ENTRY fetch --run-dir pilot
& $PY $ENTRY health --run-dir pilot
```

`snapshot --request request.json --run-dir SNAPSHOT` acquires the first actual timestamp into a separate run. Callable helpers are `check_runtime`, `inventory`, `plan`, `snapshot`, `fetch`, `inspect`, and `health`.

## Scientific invariants

Select source timestamps with start inclusive and end exclusive. Do not shift or synthesize clocks. Daily GLORYS fields are daily means described as centered at noon by the producer; check stored timestamps against that convention and report discrepancies. Select existing positive-down depth levels inclusively. The rectangular subset preserves land and below-seabed masks and the published collocated regular grid, not the native NEMO C grid.

Preserve source names, decoded values, units, attributes and masks. `thetao` is potential temperature; retain source salinity conventions. `zos` is height above the source geoid. Datum conversion, tidal combination, ADCIRC interpolation and forcing generation belong downstream. Plausibility excursions are reported without clipping or filling. Completely masked ice-specific fields may be valid for a temperate region; the main ocean fields must contain wet coverage in every record.

Run `scripts/selftest_glorys.py` for offline behavior tests and the standard skill validator after modifications. Offline fixtures cannot establish authenticated live readiness. Keep a completion checkpoint distinguishing offline results from actual live acquisition, resume, repeat and visual review.

If a process is killed, its `writer.lock` may remain. Confirm the recorded PID/host is no longer writing before removing that one lock. Never remove a live writer's lock. Retry the request; incomplete temporary files do not count as committed chunks.
