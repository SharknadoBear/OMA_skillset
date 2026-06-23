---
name: nhm-river-fetcher
description: Discover, size-check, stage, download, extract, and inspect USGS Alaska NHM/NHM-PRMS river discharge data, especially ScienceBase NHM-PRMS stream-segment output such as seg_outflow.nc. Use when Codex needs NHM-PRMS manifests, estimate-first storage routing, Kestrel-hosted data staging, selected ZIP extraction, flowline discharge mapping preparation, or regional river-forcing products from public NHM stream-segment data.
---

# NHM River Fetcher

## Core Rule

Always create a ScienceBase manifest and run storage preflight before downloading NHM-PRMS data. Do not download a large archive locally unless available storage is greater than `4 * requested_bytes`; use Kestrel staging through the `kestrel-hpc` skill for large workflows.

For Kestrel, follow `kestrel-hpc` credential rules exactly. Never store passwords, OTPs, or private connection details in this skill, scripts, manifests, logs, or command files.

## Workflow

1. Read `references/task_profiles.md` to choose the task profile:
   - `metadata-smoke` for tiny validation.
   - `geofabric-only` for geometry-only setup.
   - `byPOIobs-seg-outflow` for `AK_byPOIobs_netcdf.zip` plus geofabric.
   - `molly-all-listed` for all listed Alaska NHM-PRMS release attachments.
2. Generate a manifest with `scripts/nhm_sciencebase_manifest.py`; save item JSON under `metadata/`.
3. Run an estimate-only gate with `scripts/estimate_data_request.py`, then run `scripts/nhm_storage_preflight.py` against the intended local or remote staging path using the selected manifest rows.
4. If local preflight fails, stage on Kestrel. For Molly-scale work, require an explicit project remote root; do not use `/scratch/yhuang168/skill_test` except for smoke tests.
5. Download only selected files with `scripts/nhm_download_files.py`.
6. Extract only required ZIP members with `scripts/nhm_extract_zip_members.py`; do not blindly unzip 20-30 GB archives.
7. Inspect `seg_outflow.nc` with `scripts/inspect_seg_outflow.py` after extraction.
8. Prepare flowline mapping locally with `scripts/map_haines_flowline_discharge.py` after the NetCDF and NHM geometry are available.
9. Run `scripts/check_download_health.py` after download or extraction to inspect data coverage, finite-value fraction, NaN/gap summary, and simple plots. Write the report every time, but only surface it to the user when it has important caveats.

## Standard Connector Gates

Estimate the requested data size and routing recommendation from the selected manifest rows:

```bash
python scripts/estimate_data_request.py \
  --request outputs/nhm_prms_ak/tables/manifest.json \
  --run-dir data/nhm_prms_ak \
  --output outputs/nhm_prms_ak/tables/estimate.json \
  --skill-name nhm-river-fetcher \
  --run-id nhm_prms_ak
```

Run the NHM-specific storage preflight with the same strict 4x local-space rule:

```bash
python scripts/nhm_storage_preflight.py \
  --path data/nhm_prms_ak \
  --manifest outputs/nhm_prms_ak/tables/manifest.csv \
  --out-json outputs/nhm_prms_ak/tables/storage_preflight.json
```

After downloading or extracting products, run the finishing health gate:

```bash
python scripts/check_download_health.py \
  --request outputs/nhm_prms_ak/tables/manifest.json \
  --run-dir data/nhm_prms_ak/extracted \
  --output outputs/nhm_prms_ak/tables/health_check.json \
  --plots-dir outputs/nhm_prms_ak/plots/health_check
```

Report health-check caveats only when they matter: missing requested coverage, empty or entirely non-finite variables, finite coverage below 95% for dense data products, temporal or spatial gaps that conflict with the request, or plot-generation failure.

## Typical Commands

Generate a manifest:

```bash
python scripts/nhm_sciencebase_manifest.py \
  --profile byPOIobs-seg-outflow \
  --metadata-dir data/nhm_prms_ak/metadata \
  --out-csv outputs/nhm_prms_ak/tables/manifest.csv \
  --out-json outputs/nhm_prms_ak/tables/manifest.json
```

Check storage with the default 4x multiplier:

```bash
python scripts/nhm_storage_preflight.py \
  --path data/nhm_prms_ak \
  --manifest outputs/nhm_prms_ak/tables/manifest.csv
```

Download selected manifest rows:

```bash
python scripts/nhm_download_files.py \
  --manifest outputs/nhm_prms_ak/tables/manifest.csv \
  --dest-dir data/nhm_prms_ak/raw
```

List ZIP members before extraction:

```bash
python scripts/nhm_extract_zip_members.py \
  --zip data/nhm_prms_ak/raw/AK_byPOIobs_netcdf.zip \
  --list
```

Extract only `seg_outflow.nc`:

```bash
python scripts/nhm_extract_zip_members.py \
  --zip data/nhm_prms_ak/raw/AK_byPOIobs_netcdf.zip \
  --member-regex '(^|/)seg_outflow\.nc$' \
  --out-dir data/nhm_prms_ak/extracted
```

## Kestrel Smoke Test Pattern

Use `/scratch/yhuang168/skill_test` only for small smoke tests. A smoke test should:

- make a folder under `/scratch/yhuang168/skill_test`;
- check `df -PB1` for the folder;
- query ScienceBase JSON or upload a tiny generated manifest;
- download only one small metadata/dictionary file;
- avoid `AK_byPOIobs_netcdf.zip`, `param.zip`, and all Molly production data.

Do not run Python plotting or install Python analysis environments on Kestrel by default. Use Kestrel for storage, download, extraction, checks, and compact products; run richer Python inspection and mapping locally unless Huan explicitly overrides.

If a bundled Python script must run on Kestrel for metadata, download, or ZIP extraction, load the available Python module first:

```bash
module load python/3.12.5
```

Do not install Python analysis packages on Kestrel for this skill unless Huan explicitly asks.

## References

- Read `references/sciencebase_items.md` for item IDs, roles, and size facts.
- Read `references/task_profiles.md` before selecting downloads.
