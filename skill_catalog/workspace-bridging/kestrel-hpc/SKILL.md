---
name: kestrel-hpc
description: Use for SSH and file transfer, compilation, Slurm submission or monitoring, and compact-result retrieval on an authorized Kestrel HPC account. Obtain the username, hostname, approved paths, and credentials privately at runtime; never store account identifiers, passwords, or OTPs in the skill or repository.
---

# Kestrel HPC

## Runtime configuration

- Obtain the SSH target as `<username>@<hostname>` from the user or an approved private configuration at runtime.
- Do not place the target, account name, project paths, scratch paths, credentials, or task descriptions in the versioned skill package.
- Confirm that the intended data and operations are permitted on the remote system.
- Use the required `hmac-sha2-256` MAC setting for direct SSH, SCP, and rsync connections when the configured Kestrel endpoint requires it.

Direct connection patterns:

```bash
ssh -m hmac-sha2-256 <username>@<hostname>
scp -O -o "MACs hmac-sha2-256" <local-file> <username>@<hostname>:<remote-path>
rsync -av --progress -e "ssh -m hmac-sha2-256" <local-dir>/ <username>@<hostname>:<remote-path>/
```

## Credential rules

- Never store, write, commit, echo, log, or repeat passwords or OTPs.
- Enter the combined password and current OTP only in a secure interactive prompt, using the authentication format required by the configured endpoint.
- Wait until the credential prompt is visible before requesting user interaction.
- Do not ask the user to paste credentials into chat.
- If secure interactive entry is unavailable, use an approved key, agent, authenticated session, or user-run command.

## Local bridge workflow

Use the Paramiko bridge for multi-command sessions. The versioned helper is in:

```text
Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge
```

Create a purpose-bound session and supply the private SSH target only at runtime:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\kestrel-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "<username>@<hostname>"
```

Start the bridge from the printed `session_dir`:

```powershell
Set-Location "<session-dir>"
.\start_bridge_window.ps1
```

Inspect identity and submit operations from that session directory:

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py identity
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name <name> stop
```

Before reusing a session, verify that `bridge_identity.json` has the intended bridge name, purpose, project root, and remote target. Replace stale or malfunctioning sessions; retain a healthy session only while it still serves the same active workplan.

Runtime folders such as `.venv/`, `bridge_sessions/`, `commands/`, `results/`, identity files, and status files are local state and must not be committed.

## Operational workflow

1. Inspect the local files and remote command scope before transfer or execution.
2. Confirm the remote working directory and identity with `hostname`, `whoami`, and `pwd`.
3. Preserve unrelated remote work and upload only intended files.
4. Use existing build scripts, modules, and Makefiles rather than inventing a new environment.
5. Inspect every Slurm script before `sbatch`, including allocation, partition, time, modules, paths, and logs.
6. Monitor with `squeue -u "$USER"` and inspect job records and logs before drawing conclusions.
7. Prefer compiled or existing remote tools for compact summaries. Download compact outputs for local Python analysis unless the user explicitly authorizes remote Python analysis.
8. Stop the bridge after the workplan no longer needs the authenticated session.

## Bridge safeguards

- Match the named bridge identity before queueing every command.
- Keep processed command files outside the watched queue.
- Read JSON as `utf-8-sig` to tolerate a PowerShell UTF-8 BOM.
- Prefer a single archive for bundle transfers when recursive SFTP is unreliable.
- Build cross-platform ZIP archives with POSIX member paths.
- Never store credentials in bridge files, command JSON, logs, results, or repository memory.

## Common Slurm commands

```bash
squeue -u "$USER"
sbatch <job-script>
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -S today
scancel <job-id>
```
