---
name: fvcom-build
description: Audit, configure, compile, probe, and freeze reproducible native FVCOM executables on Kestrel or Perlmutter CPU. Use for source lineage, compiler portability, MPI and NetCDF verification, and native namelist reader checks.
---

# FVCOM Build

Freeze the native executable together with its source, compiler, runtime libraries, and input-reader evidence.


## paper_barotropic native build

Use the profile's pinned **FVCOM_MP** repository and commit in
`kestrel.source_repository` / `source_commit`; upstream release-v4.3.1 lacks
the custom flume and grid-edge namelist fields used by this workflow.
Obtain source/build files independently in the trial root; exclude bundled
binaries, case directories, results and memories. Audit the source declarations
before expensive preparation, then verify the exact executable's generated
template. Keep plastic/sediment compile flags disabled and `PLASTIC_MODEL=F`.
Before compiling this pin, apply the supplied plastic-off guard patch described
in the source contract; it excludes two plastic diagnostics from a plastic-off build.
Record observed Git origin/HEAD in `source_revision.json` and bind both in
`freeze.json`; see [the source contract](references/contracts.md#paper-source-lineage).

For the paper and Columbia paired profiles, replace the legacy module pins below with the actual native
environment discovered through Kestrel HPC. Prefer RH9; RH8 needs a documented
reason. Submit RH9 jobs from an RH9 login node and let Kestrel assign its OS
feature; do not force an `rh8`/`rh9` constraint. Verify the allocated node's OS.
Verify the scheduler accepts submissions from the selected native login host.
If a migration policy rejects that host, retain the rejection, reconcile the
original submission receipt, and retry from an eligible native login only after
confirming the rejected request created no job.
Set `kestrel.project_dir` and use a trial-owned source copy. When a login shell
exposes only Cray modules, verify the site CPU-stack profile at
`/nopt/nlr/apps/kestrel-cpu/init/profile`. Source it after `module purge` and
load the selected native modules. In scripts using nounset, bracket this sequence
with `set +u` and `set -u`. Use the same initialization in the compute job.
Record `kestrel.build_environment` (`os`, `compiler`, `compiler_options`, `modules`,
and RH8 `fallback_reason`). Capture `os-release.txt`, actual loaded modules and
compiler/build transcripts, and bind that same environment in `freeze.json`.
The planner/finalizer require this evidence. Match the runtime compute environment
to the build; do not silently reuse an RH8 binary on RH9. Blank-namelist and
input-reader checks must not advance model time; any integration is a counted
smoke attempt. The build skill never adds an uncounted stability or speed test.

## Paired tide and river executable

`columbia_river_pair` uses one project-owned source tree and one frozen binary for both members. Set `kestrel.pair_root` to their shared scratch parent, `project_dir` to the member directory and `source_dir` to the shared source directory beneath the pair root. Apply the same native build and pin checks as the paper profile. The default source already implements variable node discharge at runtime. Keep `RIVER_FLOAT` off so the native reader accepts the named `uniform` distribution. Confirm the compiled blank contains all five `NML_RIVER_TYPE` settings.

## Inputs and outputs

Accept a `simulation_request_v1` object and a build artifact root. For the Galveston workflow the source is `/home/yhuang168/FVCOM_source_agent_test`; the immutable evidence root belongs under the case's `run/build/` tree locally and its Kestrel mirror under `/scratch/yhuang168/FVCOM_Simulation/<case_id>/run/build/`.

Read [references/contracts.md](references/contracts.md) before creating or assessing build evidence. Use `scripts/fvcom_build.py` for request validation, `make.inc` auditing, plan generation, and final evidence collation.

Return a typed worker manifest with `status`, `artifact_root`, hashes, provenance, warnings, blocking reasons, and a resume token. Never report `ready` unless the executable has been frozen and all mandatory evidence passes.

## Kestrel boundary

Use `$kestrel-hpc` as the sole SSH, upload/download, and remote-command bridge. Bear enters Password+OTP only in its secure visible prompt. Never put credentials in chat, command files, logs, or artifacts. This skill must not submit production or stability jobs.

## Build workflow

1. Verify `pwd`, host, user, source path, and remote evidence root.
2. Before changing the source tree, record a sorted SHA-256 inventory, timestamps, version strings, current `make.inc`, current binary metadata when present, `module list`, and filesystem identity. Preserve a compressed pre-change source snapshot outside the source tree when storage permits; otherwise preserve every file that will change and record why the full snapshot was omitted.
3. Audit the active preprocessor definitions with `fvcom_build.py audit-make-inc`. The accepted Galveston build defines exactly the requested numerical feature set: `DOUBLE_PRECISION`, `SINGLE_OUTPUT`, `MULTIPROCESSOR`, `WET_DRY`, `LIMITED_NO`, and `GCN`. It must not define `SPHERICAL`, `TWO_D_MODEL`, sediment, particle, river, wave, ice, biology, assimilation, or semi-implicit features.
4. Correct `TOPDIR` to the resolved source root. Record the pre-change copy, unified diff, reason, and resulting checksum.
5. Load the pinned Kestrel modules in [references/contracts.md](references/contracts.md), use `mpiifort`, and retain NetCDF, METIS, and Julian linkage already proven for this source family. Capture `nf-config --all`, compiler version, and module list.
6. Run dependency generation if required by this checkout, then clobber/clean and compile. Preserve the complete command transcript and build log.
7. If compilation or the minimal input reader fails, a compatibility patch is allowed only when it changes compilation portability or input reading. Before each patch, copy the file, write a unified diff and reason, rebuild cleanly, and repeat the regression smoke. Do not change equations, constants, parameterizations, defaults that affect physics, or numerical algorithms. A requested physics change is a terminal blocker for this skill.
8. Verify the resulting executable with SHA-256, `file`, `ldd`, size/mtime, module list, and build log. Because the Kestrel binary initializes MPI before option parsing, run its blank-template probe on one compute rank as `srun ... <executable> --create_namelist=<basename>`. Then run a minimal input-reading smoke using the exact binary. A banner-only launch is not an input-reading smoke.
9. Write `executable_lineage.json`, make the freeze explicit, and record its hash. After freeze, refuse all source or `make.inc` edits; numerical stability attempts must reuse the frozen executable.
10. Download compact evidence and run `fvcom_build.py finalize`. Keep the bridge available only if the active parent workflow still needs it.

## Terminal states

- `ready`: build, link audit, generated blank namelist, input-reader smoke, executable hash, and freeze record all pass.
- `blocked_authentication`: secure bridge was not authenticated; no remote assumptions were made.
- `blocked_source`: source/version/layout is missing or incompatible.
- `blocked_compile`: clean build remains unsuccessful after permitted compatibility work.
- `blocked_physics_change_required`: success would require a forbidden physics/source change.

Do not blur a blocker into a warning. Set a resume token identifying the last immutable evidence checkpoint.

## Shared pressure-capable build

For a wind scenario or its matched control, run the make audit with `--air-pressure`. Both members require the same `AIR_PRESSURE` executable. Capture the compiler's expanded preprocessing commands for `mod_obcs.F`, `advave_edge_gcn.F` and `external_step.F`, the exact edited make.inc, binary hash, and native blank namelist. Run both native namelist parser probes and then exercise the actual variable field readers in the wind smoke run. Bind the shared executable to both member requests.

## Perlmutter CPU pilot

For `execution.site=perlmutter`, use the `perlmutter-hpc` bridge to inspect the selected CPU allocation and native Cray Programming Environment. Build the pinned source anew for the destination. Use the local payload builder and submission commands in [the Perlmutter build contract](references/contracts.md#perlmutter-cpu-build). The build job tests MPI and netCDF with the selected `ftn` wrapper, generates the native blank, and exercises the control and combined input parsers. Pass the actual selected module versions to Run Control for the smoke launch.

## Matched wind and discharge study

Use the same pressure-capable executable for every member of `columbia_wind_river_factorial`. Rebind an archived build with `fvcom_build.py reuse --request ... --binding ... --evidence ... --archive ... --binary ... --template ... --artifact-root MEMBER/run/build`. The command verifies the original evidence, executable, flags, template, build environment and source pin, then records the destination executable path and original binding hash.

## Freshwater lake

For the Lake Superior freshwater baroclinic study, use the [freshwater build contract](references/contracts.md#spherical-freshwater-lake). The build retains a source-specific thermal and signed-outflow audit.

For an explicitly requested freshwater COARE26Z experiment using prescribed hydrodynamic stress, prepare a separate diagnostic build from the frozen direct-heat baseline. Verify its calculated surface fields against a registered parallel reader run before comparing science windows. [Freshwater COARE26Z diagnostic build](references/freshwater_coare.md)

For a diagnosed vertical temperature transport excursion, the same builder accepts `prepare --temperature-donor`. Its separate freeze includes the source-locked temperature change and native moving-layer conservation and boundedness test. Qualify this derivative with its own coupled reader before a matched science window. [Temperature-only vertical donor experiment](references/freshwater_coare.md#temperature-only-vertical-donor-experiment)
