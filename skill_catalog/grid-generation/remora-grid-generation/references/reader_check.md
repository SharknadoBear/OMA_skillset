# Executable reader verification

Local CDF5 readback proves serialization and numerical checks only. A REMORA executable check additionally proves that a specific compiled reader consumes the package. Use the authorized HPC connector and keep its private session/runtime data out of this skill.

1. Record REMORA and AMReX revisions, executable hash, compiler/MPI/PnetCDF modules, and the grid SHA256.
2. Use `scripts/reader_fixture.py` to create a minimal constant IC and initialization-only inputs. It is explicitly synthetic reader-test data, not scientific forcing. Use the exact delivered grid; do not substitute a smaller or altered one without reporting a separate test.
3. Launch a bounded single-rank initialization with `remora.max_step=0` under the scheduler. Enable curvilinear geometry and NetCDF grid/IC/masks. Closed test boundaries avoid production forcing requirements; they do not select physical case BCs.
4. Require successful accounting/exit code and evidence of actual grid ingestion. Retain initialized output and compare bathymetry, masks and dimensions where the executable exposes them. Successful exit alone must not conceal a configuration that selected analytic initialization instead.
5. Run `python scripts/check_reader_output.py --grid remora_grid.nc --returned-dir returned --job-id JOB --output reader_check.json`. It verifies staged hashes, actual ingestion, successful accounting and 22 initialized fields, then binds source revision, executable hash and evidence. Place its receipt beside the delivery. `not_run` or `failed` remains distinct from a local pass. Returned-dir layout follows the explicit file names used by the checker; retain original receipts and log alongside the history file.

Do not modify REMORA physics to make a grid pass. A compatible build may require compiling REMORA on the model execution host; local grid generation has no compiler requirement. This check does not certify time integration, nesting or scientific accuracy.
