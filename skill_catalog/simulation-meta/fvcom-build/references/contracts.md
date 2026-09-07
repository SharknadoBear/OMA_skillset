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
