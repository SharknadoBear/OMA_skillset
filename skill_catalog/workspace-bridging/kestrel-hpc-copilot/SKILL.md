---
name: kestrel-hpc-copilot
description: GitHub Copilot instructions for authorized Kestrel SSH/SFTP access, Slurm work, storage and module inspection, and compact-result retrieval through the sibling Paramiko bridge. Use user@kestrel.nlr.gov as the public example and obtain real account, allocation, path, and credential details privately at runtime.
---

# Kestrel HPC - GitHub Copilot

## Privacy and access

- Use `user@kestrel.nlr.gov` in public examples; replace `user` only from approved private runtime context.
- Never store real account identifiers, allocation handles, project paths, passwords, tokens, or task descriptions in the repository.
- Ask the user to type Password+Token directly into the visible bridge terminal. Never send credentials through terminal automation or chat.
- Use only authorized data and operations. Keep compute-intensive work off login nodes.

Direct transfer examples:

```powershell
ssh -m hmac-sha2-256 user@kestrel.nlr.gov
scp -O -o "MACs hmac-sha2-256" <local-file> user@kestrel.nlr.gov:<remote-path>
```

WinSCP may use SFTP with host `kestrel.nlr.gov`, username `user`, and interactive Password+Token entry.

## Bridge workflow

Use the scripts in the sibling package:

```text
Agent_skill_dev/skill_catalog/workspace-bridging/kestrel-hpc/hpc_bridge/
```

Create the session synchronously:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "user@kestrel.nlr.gov"
```

Start `start_bridge_window.ps1` asynchronously from the printed session directory. After the credential prompt is visible, tell the user to enter Password+Token in that window.

Verify the identity and submit commands synchronously:

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py identity
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> stop
```

Reuse a session only when its bridge name, purpose, project root, and remote target still match. Keep all session identities, commands, results, and task summaries local and uncommitted.

## Nodes and scheduling

- Standard CPU nodes have 104 cores and about 240 GB usable RAM.
- Special CPU resources include 1 TB `medmem`, 2 TB `bigmem`, dual-NIC `hbw`, and 1.7 TB node-local `nvme` nodes.
- GPU nodes have four 80 GB H100 GPUs and 128 CPU cores.
- Omit the partition for ordinary exclusive CPU work when Slurm automatic routing is appropriate. Request `shared`, `debug`, `hbw`, or `nvme` explicitly when required.
- Use realistic wall time so backfill can schedule the job. Priority reflects age, job size, partition, QoS, and allocation fair-share.
- Inspect priority with `sprio`. Use `--qos=high` only with explicit authorization because it charges twice the normal allocation units.

```bash
sinfo
squeue -u "$USER"
squeue --start -j <job-id>
sprio -j <job-id>
scontrol show job <job-id>
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel <job-id>
```

## Batch template

Replace and inspect every placeholder before submission:

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

Use `--partition=shared` with explicit CPU and memory requests for partial-node work, `--partition=nvme` or `--tmp` for physical local disk, and `--gpus=<count>` plus explicit system RAM for GPU work. Test reduced workflows in `debug` before production runs.

## Filesystems and environment

- `/home/$USER`: 50 GB, for source, scripts, and small files.
- `/projects/<allocation>`: quota-controlled shared project storage.
- `/scratch/$USER`: high-performance temporary storage; files inactive for 28 days are purgeable.
- `$TMPDIR`: job-local and erased after the job; it may consume RAM unless physical disk was requested.
- ProjectFS and ScratchFS are not backed up. Preserve critical data elsewhere.

Inspect rather than guess the build environment:

```bash
module list
module avail
module spider <application>
module show <module>
which cc CC ftn mpicc mpifort
```

Compile on a login node matching the target compute architecture. Prefer the existing project toolchain and Cray MPICH for multi-node work; official guidance warns that OpenMPI may be unstable or slow. Download compact outputs for local analysis rather than installing Python analysis environments on Kestrel by default.

## References

Use the sibling `kestrel-hpc/SKILL.md` for the full operational workflow and official NLR links for [nodes and partitions](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Running/), [priority](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Running/kestrel_job_priorities/), [batch scripts](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Running/example_sbatch/), [filesystems](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/), [environments](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Environments/), and [WinSCP](https://natlabrockies.github.io/HPC/Documentation/Managing_Data/Transferring_Files/winscp/).
