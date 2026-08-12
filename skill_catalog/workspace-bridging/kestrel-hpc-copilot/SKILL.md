---
name: kestrel-hpc-copilot
harness: github-copilot
description: GitHub Copilot variant for Kestrel HPC access. Use run_in_terminal for bridge orchestration, vscode_askQuestions for OTP collection, and read_file for result parsing. Scripts are shared with sibling kestrel-hpc/ folder.
---

# Kestrel HPC — GitHub Copilot Harness

## Core Context

- Account username: `yhuang168`
- Host: `kestrel.nlr.gov` (formerly `kestrel.nrel.gov`)
- Required SSH MAC: `ssh -m hmac-sha2-256 yhuang168@kestrel.nlr.gov`
- Login node: `kl4`
- Scheduler: Slurm
- Scratch: `/scratch/yhuang168`
- Constraint: DOE/NLR HPC — low/non-sensitive scientific research data only.

## Credential Rules

- NEVER store, echo, log, or repeat password/OTP in any tool output, memory, memo, or file.
- NEVER send password or OTP via `send_to_terminal` — tell the user to type it directly.
- OTP rotates every ~30 seconds. Ask only after the `Password+OTP:` prompt appears.
- Authentication: password immediately followed by 6-digit OTP, no space.

## Copilot-Specific Bridge Workflow

Scripts live in the sibling folder:

```
Agent_skill_dev/skill_catalog/workspace-bridging/kestrel-hpc/hpc_bridge/
```

### Step 1: Create Bridge Session

```powershell
# Run via run_in_terminal (mode=sync)
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge"
python .\make_bridge_session.py --purpose "short purpose" --work-summary "summary" --project-root "C:\path\to\project"
```

Capture the printed `session_dir` path from output.

### Step 2: Start Bridge (User Interactive)

```powershell
# Run via run_in_terminal (mode=async) — this stays running
Set-Location "<session_dir>"
.\start_bridge_window.ps1
```

**CRITICAL**: When the terminal shows `Password:` or `Password+OTP:`, do NOT send credentials. Instead tell the user:
> "The bridge is waiting for your Password+OTP. Please type it directly in the terminal panel."

Wait for the user to confirm authentication succeeded.

### Step 3: Send Commands

```powershell
# Run via run_in_terminal (mode=sync)
Set-Location "<session_dir>"
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <NAME> exec "hostname; whoami; pwd"
```

### Step 4: Read Results

Use `read_file` to read JSON results from `<session_dir>/results/`.

### Step 5: File Transfer

```powershell
# Upload
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <NAME> upload local_file /scratch/yhuang168/path/file

# Download
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <NAME> download /scratch/yhuang168/path/file local_file
```

### Step 6: Stop Bridge

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <NAME> stop
```

Then use `kill_terminal` on the async terminal ID.

## SCP Alternative (No Bridge)

For quick one-off transfers without the bridge:

```powershell
# Run via run_in_terminal — will prompt for Password+OTP
scp -O -o "MACs hmac-sha2-256" "local\file" yhuang168@kestrel.nlr.gov:~/
```

Tell user to type Password+OTP directly when prompted.

## Slurm Commands (via bridge exec)

```bash
squeue -u yhuang168
sbatch job_script.sh
sacct -u yhuang168 --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel JOBID
```

## Copilot Tool Mapping

| Codex Pattern | Copilot Equivalent |
|---------------|-------------------|
| exec script directly | `run_in_terminal` (mode=sync) |
| start bridge server | `run_in_terminal` (mode=async) |
| read result JSON | `read_file` on results/ folder |
| ask for OTP | Tell user to type in terminal |
| view bridge status | `read_file` on bridge_status.txt |

## Key Lessons

- Windows OpenSSH ControlMaster fails — use Paramiko bridge.
- ZIP bundles for recursive transfers (avoid SFTP directory issues).
- Read JSON with `utf-8-sig` (PowerShell UTF-8 BOM).
- Kestrel Python: 3.6 default, 3.12.5 via `module load python/3.12.5`.
- Do NOT run Python analysis on Kestrel — download outputs, analyze locally.
