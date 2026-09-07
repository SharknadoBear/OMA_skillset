---
name: fvcom-build
description: Audit, configure, compile, smoke-test, and freeze a reproducible FVCOM executable on Kestrel. Use for FVCOM source/build lineage and compatibility fixes; do not use it to submit model simulations or change model physics.
---

# FVCOM Build

Produce an evidence-backed executable lineage, not merely a successful `make` exit.

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
