# FVCOM build contracts

## Supported scenario

The initial contract is FVCOM 4.3.1 on Kestrel for `three_dimensional_barotropic` simulations. The mesh coordinates are UTM metric, so `SPHERICAL` and `TWO_D_MODEL` are forbidden. Three-dimensional arrays remain present while the runtime namelist activates `BAROTROPIC=T` and inactive constant T/S fields.

## Pinned environment

Load these modules from a clean module environment, in order:

```text
gcc/12.1.0
intel/2023.2.0
intel-oneapi-mpi/2021.11.0-intel
netcdf-c/4.9.2-cray-mpich-intel
netcdf-fortran/4.6.1-intel
```

Use `mpiifort` and the established compiler options `-O3 -fpp -qopenmp-stubs`. Resolve NetCDF flags through `nf-config`; preserve the checkout's proven METIS and Julian configuration.

Required preprocessor definitions:

```text
DOUBLE_PRECISION
SINGLE_OUTPUT
MULTIPROCESSOR
WET_DRY
LIMITED_NO
GCN
```

Forbidden definition families include `SPHERICAL`, `TWO_D_MODEL`, `SEMI_IMPLICIT`, sediment, particles, rivers, surface forcing, atmospheric pressure, waves, ice, biology/water quality, nesting, and assimilation.

## `simulation_request_v1` fields used here

```yaml
mode: barotropic_tide | benchmark
case_id: string
project_dir: optional path
physics:
  formulation: three_dimensional_barotropic
kestrel:
  source_dir: /home/yhuang168/FVCOM_source_agent_test
  account: hindcastra
  partition: auto
```

Additional schema-valid fields are retained in provenance but do not broaden build authority.

## Mandatory evidence

`finalize` expects files with these stable names under one evidence directory:

```text
source_inventory_before.sha256
make.inc.before
make.inc.after
make.inc.diff
make_inc_audit.json
modules.txt
compiler.txt
nf-config.txt
build_commands.txt
build.log
executable.sha256
executable.file.txt
executable.ldd.txt
blank_namelist.nml
smoke.log
freeze.json
```

`smoke.log` must explicitly show a successful input-reading smoke. `blank_namelist.nml` must be produced by this executable's `--create_namelist=<basename>` mode on one MPI compute rank and becomes the authoritative template for `$fvcom-namelist-configuration`.

Every compatibility change additionally has a pre-change copy, a `.diff`, and a machine-readable reason record. `freeze.json` contains the executable SHA-256 and UTC freeze time. Any later executable hash mismatch invalidates the lineage.

## Worker manifest

```json
{
  "schema": "fvcom_build_worker_manifest_v1",
  "status": "ready | blocked_authentication | blocked_source | blocked_compile | blocked_physics_change_required",
  "artifact_root": "absolute-or-project-relative path",
  "hashes": {"executable_sha256": "...", "lineage_sha256": "..."},
  "provenance": {"source_dir": "...", "fvcom_version": "4.3.1", "modules": []},
  "warnings": [],
  "blocking_reasons": [],
  "resume_token": "build:<case_id>:<checkpoint>"
}
```

## paper_barotropic override

For `profile: paper_barotropic`, replace the pinned environment above with the
discovered native RH9 environment (document any RH8 fallback). Require
`kestrel.project_dir` and a source copy beneath it; record `kestrel.build_environment`
with `os`, `compiler`, `compiler_options`, `modules` and any `fallback_reason`.
Retain actual `os-release.txt`, module and build-command evidence, and bind the
identical environment object in `freeze.json`. All scientific feature, linking,
input-reading and executable-lineage checks still apply. These checks must not
advance model time outside the profile's counted smoke/production attempts.

## Paper source lineage

The paper profile pins `https://github.com/SharknadoBear/FVCOM_MP.git` at
`f8e4b50d4fd51409febcefc763cbc5dca11870e2`. This is a modified FVCOM source family,
not interchangeable with upstream release-v4.3.1. Run Control resolves the pin
into `kestrel.source_repository` and `kestrel.source_commit`.

Use a fresh trial-owned checkout of source/build files (root Fortran/C sources,
make/dependency files and required library source archives). A sparse checkout
avoids the repository's bundled cases, results, memories and binaries. Do not
read or reuse those artifacts. Compile cleanly for the selected native environment;
the repository's supplied `make.inc` enables plastic and is not a tide-only preset.
The numerical feature audit rejects plastic and sediment flags. The paper
namelist explicitly sets `PLASTIC_MODEL=F` and `FLUME_BOTTOM=F`.

Before build edits, record the observed `git remote get-url origin` (canonical
HTTPS URL above) and `git rev-parse HEAD` as `source_repository` and `source_commit`
in `source_revision.json`. Include these same two fields in `freeze.json`, alongside
the executable hash and native build environment. Preserve diffs for permitted
portability edits; the commit identifies the base, not an assertion of an unmodified
working tree.

At this pin, two plastic mass-diagnostic routines in `mod_plast.F` sit outside
the `PLASTIC` preprocessor guard and prevent a plastic-off build. Before compilation,
apply [fvcom_mp_no_plastic_guard.patch](../assets/fvcom_mp_no_plastic_guard.patch)
to the trial-owned checkout with `git apply --check <patch>` followed by
`git apply <patch>`. Do not enable plastic to work around the error. Preserve the
original file, applied patch and reason alongside build evidence. This patch
adds only the guard around those routines; it changes no numerical statements.
For this base file, the SHA-256 after normalizing CRLF to LF is
`4d8a53609fc3edab7c7849d8b99134aeb022f7a9083f586aadc7390158633cbe`
before and `66a07e67f161305668fe537aadfc8e10d30a5d1be4028a7386e22bdec2ee9657`
after. If neither matches, inspect the source discrepancy before proceeding;
if the after-hash already matches, record it rather than applying twice.
Retain the post-patch source inventory with the frozen executable. This is a
build-copy compatibility patch, not a change to the upstream repository.

The native RH9 probe also required renaming bundled METIS's private `__log2`
symbol in `libs/metis/rename.h` to `__fvcom_metis_log2` to avoid a glibc symbol
conflict. Apply this only when that bundled version still defines `__log2`,
recording its before-copy, diff and reason. Resolve matching NetCDF C/Fortran
libraries from the loaded modules; preserve their paths in link evidence.

Before compilation, check `mod_main.F` for `NML_PHYSICS.FLUME_BOTTOM` and
`FLUME_DRAG_COEF`, and `mod_station_timeseries.F` for
`NML_STATION_TIMESERIES.READ_GRID_EDGE_FROM_FILE` and `GRID_EDGE_FILE_NAME`.
Source declarations are an early compatibility check only. Finalization requires
these fields and `NML_ADDITIONAL_MODELS.PLASTIC_MODEL` in the exact executable's
generated blank template, then normal full rendering and input-reader validation.
Do not synthesize missing keys or weaken the template-lineage gate.
