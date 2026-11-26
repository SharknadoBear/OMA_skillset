# Freshwater COARE26Z diagnostic build

Use this opt-in recipe for an independently qualified Lake Superior heat-flux experiment. Its baseline is a frozen GNU Perlmutter CPU build of FVCOM_MP at `f8e4b50d4fd51409febcefc763cbc5dca11870e2` with the spherical freshwater EOS, pressure, wetting and drying, and signed river-edge support. The native baseline, its compute source tree and its evidence directory must be available. Each derivative gets a distinct scratch work directory and build receipt. [Freshwater baseline](contracts.md#spherical-freshwater-lake)

## Prepare and compile

```text
python scripts/freshwater_coare.py prepare --binding executable_reuse_binding.json --baseline-source /scratch/study/source/build_JOB_gnu --baseline-evidence /scratch/study/run/build/attempt_JOB/gnu/evidence --remote-workdir /scratch/study/diagnostics/coare/v001 --account PROJECT --output coare_payload.tgz
```

Use the Perlmutter skill to transfer the archive. Verify its SHA-256 at the destination, extract it in the declared work directory and submit `build.sbatch` using the same CPU account. The compute job checks the baseline executable, freeze, source revision, compiler inputs and native EOS and MPI/netCDF evidence. It creates a fresh object tree, applies the source-locked transformations and adds `HEATING_CALCULATED` and `LAKE_COARE_DIAG` to the original defines. It regenerates the active dependencies, links the executable, verifies the actual preprocessed source units, and captures its generated blank namelist. The resulting `evidence_JOB.tgz` contains `coare_build_freeze.json`. This freeze requires a separate native field-reader test.

## Qualify the field readers

Use the exact derivative executable for a short registered multi-rank case. Give the heat reader model SST, HRRR 2 m temperature (°C) and RH (%), local thermodynamic pressure (Pa), downwelling longwave and incident net shortwave (W/m²). Supply 10 m element winds (m/s) separately from the original element stress and absolute sea-level pressure used by hydrodynamics. Set `HEATING_FRESHWATER=T`, `COARE_VERSION='COARE26Z'` and `ZUU/ZTT/ZQQ=10/2/2`. Use raw liquid rain and zero prescribed evaporation. At every internal step the model uses the same latent heat and SST-dependent latent heat of vaporization for the signed vapor mass in the tracer and external water-volume terms.

The tested source computes the vapor field on distributed nodes. Its budget weights use the current wet dual areas and the native partition multiplicity. The external continuity routine applies its signed water-source arrays before its wet/dry surface update. The point ledger stores the source routine's post-call air temperature in kelvin using its 273.16 offset and its Coriolis-file latitude in degrees. The pinned instrumentation records Lake Superior shoreline nodes 694 and 39652 and deep node 69552. The underlying native COARE fluxes use all primary nodes. [Source pin](../assets/freshwater_coare/source_pin.json) · [Native ledgers](../assets/freshwater_coare/ledger.F)

```text
python scripts/freshwater_coare.py verify-reader --freeze coare_build_freeze.json --launch RUN/launch_identity.json --case RUN/case.json --history RUN/superior_0001.nc --bulk RUN/input/superior_bulk.nc --rain RUN/input/superior_rain.nc --hydrodynamic RUN/input/superior_atmosphere.nc
```

Run this command in an environment with NumPy and netCDF4. The registration binds `mpi_ranks`, `start_utc`, `end_utc`, `executable.sha256`, and each input basename and SHA-256 in `inputs`. The native launch receipt binds the parallel job ID, the case file and the launched executable. The reader stack must cover the full registered diagnostic interval. Pair the resulting JSON with terminal Slurm accounting and the model's completion proof. For every post-initial clock, the check reconstructs wind speed from the native triangle graph and the original element U/V brackets, checks the heat-component sum and latent-energy/mass conversion, and verifies rain, evaporation, incoming solar, the hydrodynamic stress and sea-level pressure.

For the verified derivative built on Perlmutter as 62018436, the parallel native reader 62018531 reconstructed all 90,964 wet-node wind speeds to within 4.77 × 10⁻⁷ m/s. Its heat-component and latent-energy residuals were below 3.5 × 10⁻⁵ W/m². An earlier diagnostic omitted the nodal wind halo exchange and wrote 4,818 spurious zero speeds. Test this regression with `python scripts/selftest_freshwater_coare.py`. Supplying `--baseline-source` replays the transformations against an actual clean freshwater source tree and compares all five output-source hashes to the compiled reference.
