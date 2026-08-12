---
name: kestrel-hpc
description: Use when working with Bear's NLR Kestrel HPC account for SSH/SCP/rsync access, uploading model source files, compiling on Kestrel, submitting or monitoring Slurm jobs, retrieving logs/results, or generating compact model-output artifacts such as .mat files. Do not run Python analysis, plotting, or install Python analysis environments on the HPC server unless Bear explicitly overrides this; download compact outputs and do Python analysis locally. Includes the required SSH MAC setting and credential handling rules, but never stores passwords or OTPs.
---

# Kestrel HPC

## Core Context

- Account username: `yhuang168`
- Host: `kestrel.nlr.gov` (formerly `kestrel.nrel.gov`)
- Required SSH form: `ssh -m hmac-sha2-256 yhuang168@kestrel.nlr.gov`
- Typical login node after connection: `kl4`
- Scheduler: Slurm
- Research/data constraint: Treat Kestrel as a DOE/NLR HPC system; keep work to low/non-sensitive scientific research data.

Plain SSH/SCP may fail with a "Corrupted MAC" error. Always include the MAC setting for SSH, SCP, and rsync-style transport.

## Credential Rules

- Never store, write, commit, echo, or repeat Bear's Kestrel password.
- Never put the password, OTP, or combined password+OTP in this skill, shell history, scripts, config files, logs, or command-line arguments.
- Ask Bear for the current 6-digit OTP for each new authentication session; OTPs rotate about every 30 seconds.
- Wait for the `Password+OTP` prompt to appear before asking Bear for the OTP to avoid timeout.
- If a password is needed and no key/agent/session is already available, ask for the password at runtime as a secret, not as durable context.
- Bear prefers a direct secure prompt over a separate terminal window. When possible, use a local secure prompt or askpass-style flow for Password+OTP entry; do not ask Bear to paste password, OTP, or password+OTP into chat.
- The Kestrel prompt is normally `(yhuang168@kestrel.nlr.gov) Password+OTP:`. Authentication expects password immediately followed by the 6-digit OTP, with no space or separator.
- If the available terminal tool cannot securely answer an interactive password prompt, explain the limitation and use an SSH key, existing authenticated session, user-run command, or other secure workflow instead.

## Connection Commands

Use:

```bash
ssh -m hmac-sha2-256 yhuang168@kestrel.nlr.gov
```

For file copy:

```bash
scp -O -o "MACs hmac-sha2-256" "local\path\file" yhuang168@kestrel.nlr.gov:~/
scp -O -o "MACs hmac-sha2-256" yhuang168@kestrel.nlr.gov:~/remote_file "local\path\newname.txt"
```

SCP requires `-O` plus `-o "MACs hmac-sha2-256"` and the same Password+OTP flow as SSH.

For directory sync:

```bash
rsync -av --progress -e "ssh -m hmac-sha2-256" local_dir/ yhuang168@kestrel.nlr.gov:/remote/path/
rsync -av --progress -e "ssh -m hmac-sha2-256" yhuang168@kestrel.nlr.gov:/remote/path/ local_dir/
```

If an SSH config alias exists, prefer the alias only after verifying it includes `MACs hmac-sha2-256`.

## Typical Workflow

1. Confirm the local files to upload and inspect local changes before transfer.
2. Ask Bear for the current OTP only when ready to connect.
3. Connect with the required MAC setting.
4. Verify the remote working directory with `pwd`, `hostname`, and `ls`.
5. Upload only the intended changed files, preserving unrelated remote work.
6. Compile using the project's existing Kestrel build scripts or Makefiles.
7. For analysis workflows, prefer MATLAB/compiled tools or existing project tools on Kestrel to generate compact artifacts such as `.mat`, NetCDF summaries, logs, or tables.
8. Do not install Python analysis dependencies or run Python plotting/analysis on Kestrel by default; download compact outputs and run Python analysis locally in the user's workspace.
9. Submit jobs with `sbatch` only after checking the Slurm script, account/partition/time settings, input paths, and output paths.
10. Monitor with `squeue -u yhuang168`, inspect output/error logs, and summarize relevant failures or results.
11. Retrieve only the needed logs/results back to the local workspace.

## Local Bridge Workflow

Use this workflow when Codex needs several rounds of Kestrel navigation but the
available terminal tool cannot securely answer an interactive Password+OTP
prompt.

The reusable bridge implementation is kept inside this skill directory:

```text
kestrel-hpc\hpc_bridge
```

For the active local Codex skill install, that resolves to:

```text
C:\Users\huan111\.codex\skills\kestrel-hpc\hpc_bridge
```

The earlier WaterPACT-local bridge folder is only historical fallback context,
not the primary reusable bridge location.

The bridge uses a local Python virtual environment with Paramiko. Bear enters
Password+OTP only in a visible PowerShell window; the bridge keeps one SSH
session open and watches local JSON command files. Codex writes commands to a
named session folder, the bridge executes them on Kestrel, and results are
written to that session's `results/` folder. The bridge supports `exec`,
`upload`, `download`, and `stop` actions.

Before starting or reusing a bridge, inspect the named bridge identity. Reuse
an existing bridge only when its Japanese `bridge_name` is intended and its
purpose/project root match the current task. If the purpose or project root is
different, create a new bridge session.

Bridge lifecycle protocol:

- Close the bridge when the work is finished and no more Kestrel commands are
  needed for the active workplan.
- Close the bridge when the same task still needs Kestrel access but the bridge
  is malfunctioning, stale, or stuck, and a replacement bridge must be built.
  Clear stale lock/queued command state before starting the replacement.
- Do not close a healthy bridge merely because time has passed. If it is
  reusable for future work within the same active `plan.md`/workplan, keep it
  available and reuse it after rechecking the bridge identity.

Create a named bridge session from the reusable helper folder:

```powershell
Set-Location "C:\Users\huan111\.codex\skills\kestrel-hpc\hpc_bridge"
python .\make_bridge_session.py --purpose "short purpose" --work-summary "1-3 sentence work summary" --project-root "C:\path\to\project"
```

The command prints a `session_dir`. Start the bridge from that session
directory using the existing launcher:

```powershell
Set-Location "<session_dir>"
.\start_bridge_window.ps1
```

The first launch creates a local `.venv` and installs the bridge dependency
from `requirements.txt`. Runtime folders such as `.venv/`, `commands/`, and
`results/` are disposable session state and must not be committed. Legacy
bridge folders without `bridge_identity.json` are not reusable for new work.

The checked-in staging copy lives at:

```text
Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge
```

Use that repo copy as the source of truth when updating and publishing the
skill, then sync the active `.codex` copy.

When launching the bridge programmatically, prefer `start_bridge_window.ps1`,
`run_bridge.cmd`, or `Start-Process -WorkingDirectory <bridge_dir> -FilePath
powershell.exe -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass',
'-File', '.\start_bridge_window.ps1'`. Avoid `powershell -Command
"Set-Location <path>; ..."` for OneDrive paths because `OneDrive - PNNL` can
be parsed incorrectly if quotes are stripped.

Inspect identity and queue commands from Codex/local shell:

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py identity
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name Akatsuki exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_kestrel_command.py --purpose "short purpose" --project-root "C:\path\to\project" upload local_file /scratch/yhuang168/path/file
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name Akatsuki download /scratch/yhuang168/path/file local_file
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name Akatsuki stop
```

Important bridge lessons from the 2026-05-26 setup:

- Windows OpenSSH `ControlMaster` failed with `getsockname failed: Not a socket`;
  prefer the Paramiko bridge on this workstation.
- Completed bridge command files must be moved out of the watched `commands/`
  directory, e.g. to `commands/processed/`, so `*.done.json` files are not
  reprocessed.
- Read JSON commands with `utf-8-sig` because PowerShell can write a UTF-8 BOM.
- Use a single-file ZIP upload for bundles when recursive SFTP hits directory
  permission/path issues.
- When creating ZIP bundles on Windows for Linux, build them with Python
  `zipfile` and `Path.as_posix()` paths rather than PowerShell
  `Compress-Archive`, which can preserve backslashes in member names.
- Do not store credentials in bridge files, command JSON, logs, or results.

Kestrel environment observations from the bridge test:

```text
login node observed: kl4
scratch root: /scratch/yhuang168
test scratch: /scratch/yhuang168/test_postprocessing
default python: /usr/bin/python3 = Python 3.6.8
usable module: python/3.12.5
```

Historical note only: a past postprocessing smoke test briefly used scratch-local
Python dependencies. Do not repeat that pattern for new Kestrel work. Generate
compact outputs on Kestrel with MATLAB/compiled tools when needed, download them,
and run Python analysis, plotting, and reporting in the local workspace.

## Slurm Commands

Common commands:

```bash
squeue -u yhuang168
sbatch job_script.sh
sacct -u yhuang168 --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel JOBID
```

Before submitting a job, inspect the script for `#SBATCH` settings, working directory assumptions, module loads, executable paths, and output/error log paths.

## Project Notes

The known local working area is:

```text
c:\Users\huan111\OneDrive - PNNL\Desktop\WaterPACT_Local\1_Model_Build\Model_develop
```

Known active source context includes:

```text
FVCOM_source_repo_github/mod_sed.F
```

Remote folder structure is not yet documented. When discovered, record non-sensitive path conventions in this skill or in a separate reference file, but do not record credentials.
