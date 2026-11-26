# Lake Superior freshwater COARE26Z diagnostics on Perlmutter CPU

Use this recipe to compare a 24- or 48-hour coupled heat window with a direct-heat run from the same native startup. The diagnostic uses the lake mesh, 128 MPI ranks, 2.5-second external and 5-second internal steps, hourly stations, six-hour history and daily restarts. The direct campaign continues to use its scheduled `lake-attempt` and `lake-audit` commands.

Prepare the original direct `executable_reuse_binding.json`, the diagnostic `coare_build_freeze.json` from `fvcom-build`, and the surface skill’s `coare_diagnostic_manifest.json`. Use the namelist skill’s `lake --coare-bulk` renderer with the derivative’s native blank, the original atmospheric stress/pressure and raw precipitation. Use `--input-root` with the staged input directory as documented by the renderer. Supply `--inputs` as a flat JSON object mapping each actual input filename to its SHA-256. Place those bytes under the remote project’s `data/<sha256>`. A value may also be `{"path":"/absolute/project/checkpoint.nc","sha256":"…"}` for a previously verified checkpoint or forcing file already on project scratch. Include the river NetCDF and companion, startup, grid, depth, sigma, station file, bulk, rain, hydrodynamic file and both byte-identical Coriolis names.

```bash
cli="$SKILLS/fvcom-run-control/scripts/fvcom_run_control.py"
python "$cli" lake-attempt --project "$PROJECT" --name coare_cold --revision 0 \
  --namelist superior_run.nml --inputs inputs.json --binding executable_reuse_binding.json \
  --coare-build coare_build_freeze.json --coare-reader reader_check.json \
  --coare-forcing coare_diagnostic_manifest.json --qos debug --walltime 00:30:00
```

For a first build, render `coare_reader` for 600 seconds, with history at startup and 600 seconds and the first restart at 600 seconds. Omit `--coare-reader` for that attempt. Use the same transfer and submission sequence below, then run the public `fvcom-build/scripts/freshwater_coare.py verify-reader` against its `case.json`, `launch_identity.json`, 600-second history, bulk, rain and stress/pressure inputs. Save its JSON as `reader_check.json`. It qualifies the derivative binary across windows. Each cold, warm or continuation attempt independently checks its own applied forcing at every saved post-startup grid record. A continuation reads the SHA-bound previous daily restart as its startup.

Copy the complete control directory `$PROJECT/run/native_lsofs/attempts/attempt_coare_cold_0000/` into the exact `remote_run` of its `attempt_manifest.json` using the Perlmutter skill. Keep the per-file SHA-256s identical. The job materializes the hardlinks from the recorded scratch inputs, verifies the input, namelist, build, reader and forcing receipts, records the allocation and invokes `srun`. Submit the frozen script once with the site skill and retain its receipt.

```bash
python "$SKILLS/perlmutter-hpc/scripts/perlmutter.py" --user "$NERSC_USER" submit \
  --account "$ACCOUNT" --workdir "$RUN" --script "$RUN/job.sbatch" --sha256 "$JOB_SHA256" --receipt slurm.json
```

After completion, stage that model submission receipt on scratch. Capture the parent job and the native MPI step with headered `sacct -P -j "$JOB_ID" -o JobID,Account,State,ExitCode,NNodes,NTasks,ElapsedRaw`. Choose a fresh QA directory and an existing project content-addressed directory on the same filesystem. Run the public scripts with netCDF4 and NumPy in the Perlmutter CPU Python environment.

```bash
python "$cli" lake-coare-audit --attempt "$RUN" --case-name superior \
  --submission "$QA/slurm.json" --accounting "$QA/sacct.psv" \
  --geometry "$GRID/lsofs_fvcom_dat_manifest.json" --river "$RIVERS/river_manifest.json" \
  --cas "$PROJECT_DATA" --output "$QA"
python "$SKILLS/fvcom-simulation/scripts/lake_coare_ledger.py" --run "$RUN" \
  --native-audit "$QA/native_audit.json" --output "$QA/hourly.json"
```

`native_audit.json` binds the completed allocation and MPI step, binary, model inputs, startup/IINT, output SHA-256s, the history, station and restart clocks, direct stress/pressure, applied sensible/latent/radiative heat and signed vapor/rain fields, hourly ledgers on every rank and the three native point traces. `phase_screen_passed` reports the pressure-dependent freshwater phase test separately from successful numerical verification. Review wet thermal errors and the heat/volume reductions before interpreting the diagnostic.

For the optional vertical temperature donor experiment, pass the build freeze and 600-second reader receipt from the same donor executable to `lake-attempt`. Use the forward explicit tracer configuration with `BACKWARD_ADVECTION=F`. The attempt and audit record `temperature_transport=vertical_donor_temperature_only`. Compare a 24-hour restart with the same mesh, input hashes and namelist using the standard vertical operator. Inspect temperature at depth, upper-layer gradients, heat storage and local freshwater freezing margins. The donor option uses first-order upstream temperatures at vertical faces and can smooth a thermocline. [Build and operator probe](../../fvcom-build/references/freshwater_coare.md)

**Archived real invocation.** Job 62020855 replayed original 24-hour model 62018619 without a model submission. Here `RUN` ended in `diagnostics/coare26z_cases/cold/v_5a2ccd443773a289` and `ORIGINAL_QA="$RUN/qa_3efcde5a369a6ebc"`. The original frozen case predates the copied public receipts, so that invocation supplied the matching evidence paths explicitly. The 24 hourly reductions covered 17,280 internal and 34,560 external steps.

```bash
python fvcom-run-control/scripts/fvcom_run_control.py lake-coare-audit \
  --attempt "$RUN" --case-name superior --submission "$ORIGINAL_QA/slurm.json" \
  --accounting "$ORIGINAL_QA/sacct.psv" --river "$ORIGINAL_QA/river.json" \
  --cas cas --output output --geometry geometry.json --coare-build coare_build_freeze.json \
  --coare-reader reader_check.json --coare-forcing forcing.json
python fvcom-simulation/scripts/lake_coare_ledger.py --run "$RUN" \
  --native-audit output/native_audit.json --output output/hourly.json
```
