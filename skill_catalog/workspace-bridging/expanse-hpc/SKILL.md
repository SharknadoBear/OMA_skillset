---
name: expanse-hpc
description: Use for authorized SSH/SFTP access, file transfer, compilation, Slurm submission or monitoring, storage selection, environment inspection, and compact-result retrieval on SDSC Expanse. Use the public example target user@login.expanse.sdsc.edu while obtaining real account, allocation, path, password, key, and TOTP details privately at runtime.
---

# Expanse HPC

## Runtime and credential rules

- Use `user@login.expanse.sdsc.edu` as the public example. Replace `user` only from approved private runtime context.
- Keep real usernames, project names, paths, passwords, keys, TOTP values, and task descriptions out of the versioned package.
- Confirm that the intended data and operations are authorized for the Expanse allocation.
- Authenticate with an ACCESS-wide password followed by TOTP, or an SSH-agent key followed by TOTP. Enter secrets only in the visible bridge or SSH terminal; never paste them into chat.
- Use `login.expanse.sdsc.edu` normally. It distributes sessions across the login nodes; use a specific login node only for diagnosed login-endpoint problems.
- Use login nodes for editing, transfer, submission, and light inspection. Request compute resources for compilation and computational work.

Direct access patterns:

```bash
ssh user@login.expanse.sdsc.edu
scp <local-file> user@login.expanse.sdsc.edu:<remote-path>
rsync -av --progress <local-dir>/ user@login.expanse.sdsc.edu:<remote-path>/
```

## Local bridge workflow

Use the persistent Paramiko bridge for multi-command sessions:

```text
Agent_skill_dev\skill_catalog\workspace-bridging\expanse-hpc\hpc_bridge
```

Create a purpose-bound session:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\expanse-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "user@login.expanse.sdsc.edu"
```

Start the printed session directory in a visible window. The bridge first tries approved SSH-agent keys and then follows the server's hidden password/TOTP prompts:

```powershell
Set-Location "<session-dir>"
.\start_bridge_window.ps1
```

Verify identity before every operation:

```powershell
.\.venv\Scripts\python.exe .\send_expanse_command.py identity
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> stop
```

Reuse a session only while its bridge name, purpose, project root, and remote target still match the active work. Keep `.venv/`, `bridge_sessions/`, `commands/`, `results/`, identity files, and status files local and uncommitted.

## Nodes and login behavior

Treat these figures as a planning snapshot and confirm current partitions and limits with `sinfo`, `scontrol show partition`, and the official guide before a costly submission.

- Standard CPU nodes: two AMD EPYC 7742 processors, 128 cores, about 256 GB RAM, and 1 TB local NVMe.
- V100 GPU nodes: four NVIDIA V100 GPUs, 40 CPU cores, about 384 GB RAM, and 1.6 TB local NVMe.
- Large-memory nodes: 128 CPU cores, 2 TB RAM, and 3.2 TB local NVMe.
- Expanse AI/NAIRR H100 nodes are a separate resource with special GPU syntax and storage behavior; follow the current guide rather than assuming V100-node rules apply.

CPU and GPU work uses the same login endpoints, but software paths and module environments differ. Compile inside an interactive allocation matching the target resource rather than building production executables on a login node.

## Partitions, charging, and scheduling

- `compute`: exclusive standard CPU nodes, normally up to 48 hours and 32 nodes per job. A job is charged for all 128 cores on each allocated node.
- `shared`: partial standard CPU nodes, normally one node and up to 48 hours. Charges use the larger of the CPU fraction or memory fraction.
- `gpu` and `gpu-shared`: exclusive or partial V100 GPU nodes, normally up to 48 hours. Explicitly request GPUs, CPUs, and system memory.
- `large-shared`: one large-memory node, normally up to 48 hours; request at least 256 GB and explicit memory.
- `debug` and `gpu-debug`: short tests, normally up to 30 minutes.
- `preempt` and `gpu-preempt`: discounted resources that may be preempted without a refundable charge; checkpoint appropriately.
- Industry, NAIRR, and other allocation-specific partitions may appear. Use only partitions authorized for the selected project.

Expanse limits queued and running jobs by allocation and partition. Bundle very large collections of short jobs or consult SDSC rather than overwhelming the scheduler. Discover valid projects before submission:

```bash
expanse-client user -r expanse
expanse-client resource
expanse-client project <project> -p
```

Monitor jobs and resources with:

```bash
sinfo
squeue -u "$USER"
scontrol show job <job-id>
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel <job-id>
```

## Interactive and batch templates

Request a small CPU debug shell for compilation or testing:

```bash
srun --partition=debug --pty --account=<project> --nodes=1 \
  --ntasks-per-node=4 --mem=8G --time=00:30:00 \
  --wait=0 --export=ALL /bin/bash
```

Canonical MPI job:

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --output=<job-name>.%j.%N.out
#SBATCH --error=<job-name>.%j.%N.err
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --mem=0
#SBATCH --account=<project>
#SBATCH --export=ALL
#SBATCH --time=01:30:00

module purge
module load cpu
module load gcc
module load mvapich2
module load slurm
module list
srun --mpi=pmi2 -n 256 ./<executable>
```

Partial-node shared job:

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --account=<project>
#SBATCH --output=<job-name>.%j.%N.out
#SBATCH --error=<job-name>.%j.%N.err
#SBATCH --export=ALL

module purge
module load cpu
module load gcc
module load mvapich2
module load slurm
module list
srun -n 8 ./<executable>
```

Always specify memory. On `compute`, `--mem=0` requests the full node memory; shared jobs should request only what they need. Add `#SBATCH --constraint="lustre"` whenever the job accesses Expanse Lustre, or the job can land on a node without that mount and fail.

## Filesystems and data movement

- `$HOME`: 100 GB, rolling backup currently retained for eight weeks; use for source and small files, never high-throughput jobs.
- `/expanse/lustre/scratch/$USER/temp_project`: high-performance scratch, not backed up; files are purged 90 days after creation.
- `/expanse/lustre/projects/`: allocation project space, not archival or backed up; project data is purged after allocation expiration according to policy.
- `/scratch/$USER/job_$SLURM_JOB_ID`: node-local job storage, erased after the job; capacity is about 1 TB on CPU/shared, 1.6 TB on GPU, and 3.2 TB on large-memory nodes.
- Expanse Lustre scratch has a two-million-file per-user limit. Archive small-file collections and consult support for metadata-intensive workflows.

Copy durable results out of node-local scratch before job exit and back up important Lustre data to approved storage. Expanse AI/NAIRR H100 nodes do not mount the regular Expanse Lustre filesystem; use their documented node-local and Ceph/S3 workflow.

## Modules and build environments

Expanse uses Lmod and does not expose every application until the correct CPU or GPU path is active:

```bash
module list
module avail
module spider <application>
module display <module>
```

- Start with `module purge`, then load exactly one of `cpu` or `gpu`; do not mix their paths.
- Load the compiler, MPI implementation, libraries, and `slurm` module explicitly, then record `module list` in the job log.
- Use `module spider` to discover packages and dependency chains rather than assuming a module name is immediately loadable.
- If `module` is unavailable in a non-login shell, initialize the site module setup before continuing.
- Use existing build scripts and test the chosen GNU, Intel, or AOCC compiler/MPI combination in an interactive allocation.
- Prefer compact remote summaries and download results for local analysis unless remote Python analysis is explicitly authorized.

## Operational safeguards

1. Inspect local changes and the exact remote command scope before connecting.
2. Confirm identity and location with `hostname`, `whoami`, and `pwd`.
3. Select and verify the correct allocation with `expanse-client`.
4. Preserve unrelated remote work and upload only intended files.
5. Inspect partition, memory, project, Lustre constraint, modules, paths, and logs before `sbatch`.
6. Monitor scheduler state, accounting records, and logs before drawing conclusions.
7. Copy durable results out of node-local and purgeable storage.
8. Stop the bridge when the active workplan no longer needs it.

Keep processed commands outside the watched queue, read JSON as `utf-8-sig`, prefer a single POSIX-path ZIP for unreliable recursive transfers, and never store credentials in identity, command, result, log, or memory files.

## Official reference

- [SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html)
