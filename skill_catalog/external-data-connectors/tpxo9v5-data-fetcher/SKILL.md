---
name: tpxo9v5-data-fetcher
description: Estimate, inventory, subset, interpolate, and health-check registered TPXO9v5 NetCDF harmonic elevation and transport data. Use when Codex needs model-neutral tidal constituents for a regional native-grid subset, arbitrary points, or a target grid, including safe Google Drive staging and removal of successful temporary raw downloads.
---

# TPXO9v5 Data Fetcher

Use this skill to turn authorized TPXO9v5a NetCDF source files into compact,
model-neutral harmonic products. Keep model-specific forcing and tidal time-series
reconstruction downstream.

## Source Rules

- Treat TPXO as registered, non-public source data. See the official
  [TPXO global-model page](https://www.tpxo.net/global) for access and use terms.
- Accept either an authorized Google Drive folder URL at runtime or a local source
  directory. Do not put private Drive URLs, file IDs, credentials, rclone
  configuration, or raw TPXO data in this skill or in Git.
- Use rclone for automated Drive inventory and staging because large registered
  NetCDF files can exceed app-connector transfer limits. Keep the rclone executable
  and its user-profile configuration outside this repository.
- Always run `scripts/check_rclone.py` before Drive work. If it reports
  `install_required`, stop and notify the user that rclone must be installed on
  this equipment. Never install software without the user's permission.
- If preflight reports `authorization_required`, stop and request one-time Google
  Drive authorization in a visible user-controlled browser. Configure only an
  account that is authorized for the registered data and prefer the
  `drive.readonly` scope. Never request, write, echo, or commit credentials or
  OAuth tokens.
- Require the grid file for every extraction. Require the elevation file for
  `elevation`, the transport file for `transport`, and all three for both fields.
- Never delete a Drive original or a caller-owned local source. Cleanup is permitted
  only for files inside an explicitly supplied `--staging-dir`.

## Required Workflow

1. For Drive data, preflight rclone and the configured read-only remote:

```bash
python scripts/check_rclone.py --remote gdrive-readonly
```

If rclone is absent, notify the user that installation is required. On Windows,
the official package can be installed after approval with:

```powershell
winget install --id Rclone.Rclone --exact --source winget --scope user
```

After installation, the user performs the one-time OAuth approval through
`rclone config`; choose Google Drive and `drive.readonly`. The default configuration
belongs in the user's profile, not the repository. See the official
[rclone Drive guide](https://rclone.org/drive/) and
[installation guide](https://rclone.org/install/).

For caller-owned local sources, skip rclone and begin at source inventory below.

2. Inventory only the exact remote files needed for the requested fields. Supply
the authorized folder URL or ID at runtime; do not save it in Git:

```bash
python scripts/stage_tpxo9v5_rclone.py inventory \
  --remote gdrive-readonly --drive-folder "<authorized-folder-url-or-id>" \
  --fields elevation,transport --output runs/case/source_manifest.json
```

The manifest contains basenames, sizes, available MD5 hashes, and a one-way folder
fingerprint, but not the Drive folder ID. Estimate source, staging, output, and
working-space requirements before downloading:

```bash
python scripts/estimate_data_request.py --manifest runs/case/source_manifest.json \
  --fields elevation,transport --run-dir runs/case \
  --output runs/case/download_estimate.json
```

Do not download unless free space exceeds four times the required raw bytes.
Install the script runtime in an isolated project environment when needed:

```bash
python -m pip install numpy scipy netCDF4
```

3. After the estimate passes, stage those same reviewed files. The command
re-inventories the folder, refuses changed metadata or mismatched existing files,
and verifies downloaded size plus Drive MD5 when available:

```bash
python scripts/stage_tpxo9v5_rclone.py download \
  --remote gdrive-readonly --drive-folder "<authorized-folder-url-or-id>" \
  --fields elevation,transport --manifest runs/case/source_manifest.json \
  --staging-dir runs/case/raw --output runs/case/staging_report.json
```

Inspect the real schema and native spatial coverage:

```bash
python scripts/inventory_tpxo9v5.py --source-dir runs/case/raw \
  --output runs/case/source_inventory.json
```

4. Produce a native-grid regional subset:

```bash
python scripts/extract_tpxo9v5.py subset --source-dir runs/case/raw \
  --staging-dir runs/case/raw --bbox -76 35 -70 42 \
  --fields elevation,transport --constituents M2,S2,K1,O1 \
  --output runs/case/tpxo_subset.nc --report runs/case/extract_report.json
```

Or interpolate to comma-separated point coordinates:

```bash
python scripts/extract_tpxo9v5.py interpolate --source-dir runs/case/raw \
  --staging-dir runs/case/raw --points points.csv \
  --fields elevation --output runs/case/tpxo_points.nc \
  --report runs/case/extract_report.json
```

Point CSV files must contain `longitude` and `latitude` columns. A target NetCDF may
instead be supplied with `--target-grid`, `--target-lon-var`, and
`--target-lat-var`; its coordinates are flattened without assuming a model layout.

5. Run or repeat the finishing gate:

```bash
python scripts/check_download_health.py --input runs/case/tpxo_subset.nc \
  --output runs/case/health_check.json
```

## Scientific And Cleanup Rules

- Discover constituents from `con`; match requested names case-insensitively and
  fail when any requested constituent is absent.
- Preserve separate elevation, U, and V staggered grids. Record their native spans
  and the actual extracted span.
- Interpolate native complex coefficients, never phase angles directly. Export
  real/imaginary coefficients plus amplitude and Greenwich phase lag.
- Normalize longitude internally, support dateline-crossing regions, and retain the
  original target longitude values in point products.
- Use linear interpolation first. Apply nearest-wet fallback only within the
  configured distance and export per-value method flags.
- Convert transport to velocity only when units are recognized and positive wet
  depth is available on the matching U or V grid.
- Write to a temporary output and replace the destination only after structural
  validation. The default cleanup policy removes staged raw NetCDF files after that
  successful validation. Pass `--keep-raw` for a reviewed exception.
- On extraction or validation failure, retain staged files and report the error.

## Output Contract

Write NetCDF with a `constituent` dimension, source-grid or target-point
coordinates, complex coefficients, amplitude, phase lag, masks/depth where
available, interpolation flags, requested/actual spans, source basenames and
SHA-256 hashes, and cleanup policy. Write a companion JSON report containing the
same provenance plus exact staged files removed.

Do not write FVCOM forcing files or reconstruct a time series in this skill. Use a
downstream forcing builder or `$u-tide-tool-instruction` for those tasks.

## Bundled Scripts

- `scripts/check_rclone.py`: distinguish missing installation, missing one-time
  Drive authorization, and a ready configured remote without reading tokens.
- `scripts/stage_tpxo9v5_rclone.py`: inventory and atomically stage only reviewed
  Drive files with size and available MD5 verification.
- `scripts/estimate_data_request.py`: select required file roles and enforce the
  four-times-free-space gate.
- `scripts/inventory_tpxo9v5.py`: report files, variables, constituents, units,
  staggered grids, and spatial spans without loading full harmonic arrays.
- `scripts/extract_tpxo9v5.py`: subset or interpolate and safely clean staged raw
  files after success.
- `scripts/check_download_health.py`: independently validate a generated product.

## Validation

Run the skill validator, compile all Python files, exercise every CLI help path,
and test with small synthetic staggered files before a registered real-data smoke
test. Confirm cleanup removes only explicitly staged raw files.
