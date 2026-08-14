---
name: kestrel-hpc
description: Use for authorized SSH/SFTP access, compilation, Slurm submission or monitoring, storage selection, environment inspection, and compact-result retrieval on NLR Kestrel. Use the public example target user@kestrel.nlr.gov while obtaining real account, allocation, path, and credential details privately at runtime. Never store passwords or OTPs.
---

# Kestrel HPC

## Runtime and credential rules

- Use `user@kestrel.nlr.gov` as the public example. Replace `user` only from approved private runtime context.
- Keep real usernames, allocation handles, project and scratch paths, credentials, and task descriptions out of the versioned package.
- Confirm that the intended data and operations are permitted on Kestrel.
- Enter Password+Token only in a secure interactive prompt. Never store, echo, log, or request credentials in chat.
- Use `hmac-sha2-256` for direct SSH, SCP, and rsync when required by the endpoint.
- Compile and submit from a login node matching the target CPU or GPU compute architecture. Do not run compute-intensive work on login nodes.

Direct connection and transfer patterns:

```bash
ssh -m hmac-sha2-256 user@kestrel.nlr.gov
scp -O -o "MACs hmac-sha2-256" <local-file> user@kestrel.nlr.gov:<remote-path>
rsync -av --progress -e "ssh -m hmac-sha2-256" <local-dir>/ user@kestrel.nlr.gov:<remote-path>/
```

For WinSCP, create an SFTP site with host `kestrel.nlr.gov`, username `user`, and the interactive Password+Token flow. Transfer individual files by drag-and-drop or use directory synchronization only after reviewing its direction and deletion settings.

## Local bridge workflow

Use the persistent Paramiko bridge for multi-command sessions:

```text
Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge
```

Create a purpose-bound session with the target supplied at runtime:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "user@kestrel.nlr.gov"
```

Start the printed session directory and enter Password+Token only in its visible window:

```powershell
Set-Location "<session-dir>"
.\start_bridge_window.ps1
```

Verify identity before every operation:

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py identity
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> stop
```

Reuse a session only while its bridge name, purpose, project root, and remote target still match the active work. Keep `.venv/`, `bridge_sessions/`, `commands/`, `results/`, identity files, and status files local and uncommitted.

## Nodes and resource selection

Treat these figures as a planning snapshot and confirm current availability with `sinfo`, `scontrol show partition`, and the official documentation before a costly submission.

- Standard CPU nodes: 104 cores and about 240 GB usable RAM.
- `medmem`: 1 TB CPU nodes.
- `bigmem`: 2 TB RAM and about 5.6 TB NVMe; `bigmeml` serves longer jobs.
- `hbw`: dual-NIC CPU nodes for communication-heavy multi-node workloads; single-node jobs are not eligible.
- `nvme`: standard CPU nodes with about 1.7 TB node-local NVMe.
- H100 GPU nodes: four 80 GB H100 GPUs and 128 CPU cores, with several system-memory and NVMe capacities.

Most standard CPU nodes do not have physical local disk. On a node without local disk, `$TMPDIR` can consume RAM. Request `nvme` or an appropriate `--tmp` value when the job requires physical node-local storage. GPU and big-memory nodes have local disk, but GPU nodes may be shared unless exclusivity is requested.

## Partitions, priority, and scheduling

- Omit `--partition` for ordinary exclusive CPU work when automatic routing is appropriate; Slurm can route from requested node count, wall time, memory, and local disk. Specify `shared`, `debug`, `hbw`, or `nvme` when that behavior or hardware is required.
- Use `debug` only for short development and troubleshooting. Use `shared` for work that does not need a full 104-core node.
- Current routing families include `short` (up to 4 hours), `standard` (up to 2 days), and `long` (over 2 days, up to 10 days), plus memory, network, NVMe, shared, and GPU variants.
- Request realistic wall time. Backfill can start a lower-priority job only when it will not delay a higher-priority job, so an inflated limit can reduce opportunities.
- Priority combines age, job size, partition, QoS, and allocation fair-share. Inspect it with `sprio -j <job-id>` rather than inferring priority from queue position.
- `--qos=high` gives a small priority boost but charges the allocation at twice the normal rate. Use it only with explicit authorization.
- Check allocation usage with `aus_report` before large or high-priority submissions.

Useful inspection commands:

```bash
sinfo
scontrol show partition
squeue -u "$USER"
squeue --start -j <job-id>
sprio -j <job-id>
scontrol show job <job-id>
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel <job-id>
```

## Batch templates

Inspect and replace every placeholder before submission. Keep all `#SBATCH` directives before executable shell statements.

Canonical full CPU-node job:

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --account=<allocation-handle>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --time=01:00:00
#SBATCH --output=<job-name>.%j.out
#SBATCH --error=<job-name>.%j.err

module list
cd <approved-run-directory>
srun ./<executable>
```

Apply only the variation required by the workload:

```bash
# Partial CPU node
#SBATCH --partition=shared
#SBATCH --ntasks=26
#SBATCH --mem-per-cpu=2G

# Physical node-local NVMe
#SBATCH --partition=nvme
#SBATCH --tmp=1600000
# Use $TMPDIR during the job and copy durable results out before exit.

# GPU
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=50G

# Authorized high priority; charges 2x
#SBATCH --qos=high
```

For a new workflow, test a reduced case in `debug`, verify logs and resource use, and then submit the production request.

## Filesystems and data movement

- `/home/$USER`: 50 GB quota; use for shell configuration, source, scripts, and small files.
- `/projects/<allocation>`: longer-term shared project data and applications; quota depends on the allocation.
- `/scratch/$USER`: high-performance temporary work; files inactive for 28 days are subject to deletion.
- `/kfs2/shared-projects`: optional cross-allocation shared storage arranged through HPC Help.
- `/kfs2/datasets`: read-only shared datasets.
- `$TMPDIR`: job-local temporary space. It is erased after the job and may be RAM-backed unless physical disk was requested.

ProjectFS and ScratchFS are not backed up. Copy critical inputs and results to approved durable storage. Use `lfs quota` for home/project checks and avoid large collections of tiny files on Lustre; archive bundles when that reduces metadata pressure.

## Modules and build environments

Inspect the live environment before changing it:

```bash
module list
module avail
module spider <application>
module show <module>
which cc CC ftn gcc g++ gfortran mpicc mpifort
```

- Kestrel provides Cray `PrgEnv-*` environments and NLR-built toolchains. `PrgEnv-gnu` normally supplies GCC, Cray MPICH, and the `cc`, `CC`, and `ftn` wrappers.
- Use `module swap <current> <required>` when changing programming environments, and record the exact compiler/MPI/module set used for both build and runtime.
- Prefer Cray MPICH for multi-node work. Official guidance warns that OpenMPI may be unstable or underperform on Kestrel.
- Use an existing project build script or Makefile when available. Do not silently replace the toolchain or install an analysis environment.
- Prefer compiled or existing remote tools for compact summaries. Download compact outputs for local Python analysis unless remote Python work is explicitly authorized.

## Operational safeguards

1. Inspect local changes and the exact remote command scope before connecting.
2. Confirm identity and location with `hostname`, `whoami`, and `pwd`.
3. Preserve unrelated remote work and upload only intended files.
4. Inspect allocation, resource, module, path, and log directives before `sbatch`.
5. Monitor scheduler state, accounting records, and logs before drawing conclusions.
6. Copy durable outputs out of `$TMPDIR` and purgeable scratch before they are lost.
7. Stop the bridge when the active workplan no longer needs it.

Keep processed commands outside the watched queue, read JSON as `utf-8-sig`, prefer a single POSIX-path ZIP for unreliable recursive transfers, and never store credentials in identity, command, result, log, or memory files.

## Official references

- [Kestrel nodes, partitions, and scheduling](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Running/)
- [Job priority and fair-share](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Running/kestrel_job_priorities/)
- [Example sbatch scripts](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Running/example_sbatch/)
- [Filesystems](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/)
- [Programming environments](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Environments/)
- [WinSCP transfers](https://natlabrockies.github.io/HPC/Documentation/Managing_Data/Transferring_Files/winscp/)
