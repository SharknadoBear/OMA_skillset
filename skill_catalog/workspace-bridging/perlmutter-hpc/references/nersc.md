# NERSC operating notes

Operating references checked 2026-10-16; endpoint and host fingerprint reverified 2026-10-20. Measure paths, versions, quotas and allocation access in the active session.

## Authentication

[Connections and fingerprints](https://docs.nersc.gov/connect/): target `perlmutter.nersc.gov`; published RSA SHA-256 is `SHA256:Db9s2Fa4J3qx7An5oIMgUqUAdK7UWJGTPGoIKD44+Gs`. The bridge pins this key with RSA SHA-2 signatures and changes no global SSH settings.

Use the official Windows SSHProxy client to renew the daily certificate. See the skill’s native `status` and `renew` operations. The signing key and adjacent certificate stay in the user's SSH directory; new connections require a valid principal, validity interval, and pinned host. [NERSC authentication and renewal](https://docs.nersc.gov/connect/mfa/#sshproxy-on-windows-systems)

## Modules and compilers

[Lmod](https://docs.nersc.gov/environment/lmod/): `module avail` shows the reachable hierarchy; `module spider` also finds packages behind prerequisites. Inspect `module spider cray-netcdf-hdf5parallel/<version>` to find compiler/HDF5 requirements. `module show` explains settings; `module -t list` records the loaded stack.

```bash
module -t list
module avail PrgEnv
module spider cmake
module spider cray-netcdf-hdf5parallel
```

[Compiler wrappers](https://docs.nersc.gov/development/compilers/wrappers/): `cc`, `CC`, `ftn` invoke C, C++, Fortran using the selected environment. Check actual compiler versions and CPU/GPU targeting. A module load does not establish working MPI/GPU execution. Avoid editing `.bashrc` or saved module collections during inspection.

## Storage

- [Global home](https://docs.nersc.gov/filesystems/global-home/): use `$HOME` for small sources/configuration and builds, not large streaming I/O. Resolve with `readlink -f "$HOME"`.
- [Scratch](https://docs.nersc.gov/filesystems/perlmutter-scratch/): use `$SCRATCH` or `$PSCRATCH`. Documented pattern `/pscratch/sd/<first-letter>/<username>`; report the actual account expansion. High-performance temporary Lustre; files not accessed for eight weeks are subject to purge. Retain important results elsewhere.
- [Community storage](https://docs.nersc.gov/filesystems/community/): verify project membership and intended destination before writing.
- [Quotas](https://docs.nersc.gov/filesystems/quotas/): use `showquota` when available. `df` reports filesystem capacity, not remaining user quota.

## Jobs and transfers

[Perlmutter jobs](https://docs.nersc.gov/systems/perlmutter/running-jobs/) and [examples](https://docs.nersc.gov/jobs/examples/) describe constraints, QoS and allocations. Obtain allocation details from task evidence, not the username. Use bounded `sinfo`, `squeue -u "$USER"` or `sacct` when relevant. Substantial simulations run on compute nodes through Slurm.

For bulk data, consult [DTNs](https://docs.nersc.gov/systems/dtn/) and [Globus](https://docs.nersc.gov/services/globus/). The bridge supports command orchestration and modest single-file transfers.
