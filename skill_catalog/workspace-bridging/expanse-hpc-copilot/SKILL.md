---
name: expanse-hpc-copilot
description: GitHub Copilot instructions for authorized SDSC Expanse SSH/SFTP access, Slurm work, storage and module inspection, and compact-result retrieval through the sibling Paramiko bridge. Use user@login.expanse.sdsc.edu as the public example and obtain real account, project, path, password, key, and TOTP details privately at runtime.
---

# Expanse HPC - GitHub Copilot

## Privacy and access

- Use `user@login.expanse.sdsc.edu` in public examples; replace `user` only from approved private runtime context.
- Never store real account identifiers, project names, paths, passwords, private keys, TOTP values, or task descriptions in the repository.
- Start the bridge in a visible terminal and ask the user to answer its hidden authentication prompts there. Never route credentials through chat or command JSON.
- Use login nodes only for transfer, submission, editing, and light inspection. Request compute resources for compilation and computational work.

## Bridge workflow

Use the scripts in:

```text
Agent_skill_dev/skill_catalog/workspace-bridging/expanse-hpc/hpc_bridge/
```

Create the session synchronously:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\expanse-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "user@login.expanse.sdsc.edu"
```

Start `start_bridge_window.ps1` asynchronously from the printed session directory. The bridge tries approved SSH-agent keys first and then follows Expanse's password/TOTP challenge in hidden prompts.

Verify the identity and submit commands synchronously:

```powershell
.\.venv\Scripts\python.exe .\send_expanse_command.py identity
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_expanse_command.py --bridge-name <name> stop
```

Reuse a session only when its bridge name, purpose, project root, and remote target still match. Keep runtime identities, commands, results, credentials, and task summaries local and uncommitted.

## Resources and scheduling

- Standard CPU nodes have 128 AMD EPYC cores, about 256 GB RAM, and 1 TB local NVMe.
- V100 GPU nodes have four GPUs, 40 CPU cores, about 384 GB RAM, and 1.6 TB local NVMe.
- Large-memory nodes have 128 CPU cores, 2 TB RAM, and 3.2 TB local NVMe.
- `compute` and `gpu` are node-exclusive; `shared`, `gpu-shared`, and `large-shared` support fractional use. `debug` variants are for short tests, and preemptible queues can terminate jobs.
- Always specify the allocation and memory. Discover valid projects with `expanse-client user -r expanse`.
- Use `#SBATCH --constraint="lustre"` for any job accessing Expanse Lustre.

```bash
sinfo
squeue -u "$USER"
scontrol show job <job-id>
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel <job-id>
```

## Batch template

Replace and inspect every placeholder before submission:

```bash
#!/bin/bash
#SBATCH --job-name=<job-name>
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --mem=0
#SBATCH --account=<project>
#SBATCH --export=ALL
#SBATCH --time=01:30:00
#SBATCH --output=<job-name>.%j.%N.out
#SBATCH --error=<job-name>.%j.%N.err
#SBATCH --constraint="lustre"

module purge
module load cpu
module load gcc
module load mvapich2
module load slurm
module list
srun --mpi=pmi2 -n 256 ./<executable>
```

Use `shared` with explicit tasks and memory for partial CPU nodes. Test reduced workloads with an interactive `debug` allocation before production submission.

## Filesystems and environment

- `$HOME`: 100 GB, for source and small files; do not run high-throughput jobs there.
- `/expanse/lustre/scratch/$USER/temp_project`: purgeable, unbacked scratch.
- `/expanse/lustre/projects/`: unbacked allocation project storage.
- `/scratch/$USER/job_$SLURM_JOB_ID`: job-local NVMe, erased after job exit.

Expanse uses Lmod. Inspect `module list`, `module avail`, `module spider`, and `module display`; after `module purge`, load exactly one of `cpu` or `gpu`, then the chosen compiler, MPI, libraries, and `slurm`. Compile and test inside a matching interactive allocation.

Use the sibling `expanse-hpc/SKILL.md` for the full workflow and the official [SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html) for current limits. Prefer compact remote summaries and local analysis of downloaded results.
