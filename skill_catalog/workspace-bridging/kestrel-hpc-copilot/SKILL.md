---
name: kestrel-hpc-copilot
description: GitHub Copilot instructions for an authorized Kestrel HPC Paramiko bridge. Use terminal execution for bridge orchestration and file reading for result JSON; obtain the SSH target and credentials privately at runtime. Scripts are shared with the sibling kestrel-hpc package.
---

# Kestrel HPC — GitHub Copilot

## Privacy and credentials

- Obtain `<username>@<hostname>`, approved remote paths, and the current authentication material only at runtime.
- Never store account identifiers, hosts, project paths, passwords, or OTPs in this skill or repository.
- Never send credentials through terminal automation. Ask the user to type them directly into the interactive bridge terminal.
- Use only approved data and operations on the remote system.

## Bridge workflow

Use the scripts in:

```text
Agent_skill_dev/skill_catalog/workspace-bridging/kestrel-hpc/hpc_bridge/
```

Create a session synchronously:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "<username>@<hostname>"
```

Start `start_bridge_window.ps1` in an asynchronous terminal from the printed `session_dir`. When the credential prompt appears, tell the user to type the required password/OTP directly in that terminal.

Submit commands synchronously and read result JSON from `results/`:

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> stop
```

Verify `bridge_identity.json` before reuse. Keep runtime identities, commands, results, and task summaries local and uncommitted.

## Direct transfer and Slurm patterns

```powershell
scp -O -o "MACs hmac-sha2-256" <local-file> <username>@<hostname>:<remote-path>
```

```bash
squeue -u "$USER"
sbatch <job-script>
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel <job-id>
```

Use local analysis by default: retrieve compact outputs rather than installing or running Python analysis environments on the HPC system.
