---
name: noaa-ww3-fetcher
description: Inventory, fetch, resume, extract and validate NOAA historical multi_1 WAVEWATCH III monthly gridded wave fields and selected-station directional spectra. Use for bounded historical wave requests, not live forecasts or model execution.
---

# NOAA WW3 fetcher

Use `scripts/noaa_ww3_fetcher.py` with Python 3.10+, requests, numpy, rasterio/GDAL (GRIB driver), and netCDF4. Outputs belong in a project run, outside this skill.

Read [request.md](references/request.md) to construct a bounded `ww3_request_v1` JSON request. Read [sources.md](references/sources.md) for archive semantics and spectral integration. One request selects either fields on one grid or spectra at explicit stations; months are handled independently.

```text
python scripts/noaa_ww3_fetcher.py inventory --request request.json --output run
python scripts/noaa_ww3_fetcher.py plan --request request.json --output run
python scripts/noaa_ww3_fetcher.py fetch --request request.json --output run
python scripts/noaa_ww3_fetcher.py extract --request request.json --output run
python scripts/noaa_ww3_fetcher.py inspect --output run
python scripts/noaa_ww3_fetcher.py health --output run
```

Inventory exact publisher sizes/checksums before transfer, then inspect the plan against the user's bounded request and storage budget. Existing authorization to acquire that request suffices; do not add an approval round. `fetch` requires a request-bound plan. Failed checks must be resolved before treating a product as complete.

Downloads preserve compressed archive bytes, use atomic completion, bounded retries and identity-bound range resumption. Do not manually decompress the HTTP stream. Publisher MD5 is verified when available; SHA-256 is always recorded. Cache reuse revalidates both source identity and local bytes. Never silently substitute a grid, station or archive family.

Extraction crops native grid centers and preserves native times, masks, GRIB labels and CRS. Equal duplicate records are removed; conflicting records fail. Interpolation is downstream. Spectra retain native frequency/direction order and their own cadence, separate from point-bulk time. Archive members are selected safely without `extractall`. The spectrum height check compares the same model point products, not observations.

For `delete_after_health`, run `health --cleanup` only after compact outputs pass independent hash, structure and scientific checks. This removes only source files listed in this run's manifest and keeps checksum/provenance evidence. A health failure leaves raw inputs intact.

Run `scripts/test_fetcher.py` when changing transfer, duplicate or parser behavior, and the bundled skill-creator validator after editing this skill.
