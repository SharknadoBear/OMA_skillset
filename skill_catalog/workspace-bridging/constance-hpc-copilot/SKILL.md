---
name: constance-hpc-copilot
description: GitHub Copilot instructions for secure, read-only Constance HPC access through the sibling Paramiko bridge. Obtain the SSH target, approved host-key fingerprint, project paths, and credentials privately at runtime; use terminal execution for orchestration and file reading for result JSON.
---

# Constance HPC — GitHub Copilot

## Privacy and access rules

- Obtain `<username>@<hostname>`, the approved SHA256 host-key fingerprint, and remote paths only at runtime.
- Never commit account identifiers, infrastructure addresses, fingerprints, project paths, or task details.
- Never send credentials through terminal automation. Ask the user to type them directly in the interactive terminal.
- Keep operations read-only. The bridge supports `exec`, small `download`, and `stop`; it does not support upload.

## Bridge workflow

Use the scripts in:

```text
Agent_skill_dev/skill_catalog/workspace-bridging/constance-hpc/hpc_bridge/
```

Create a session synchronously:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\constance-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "<username>@<hostname>" `
  --host-key-fingerprint "SHA256:<approved-fingerprint>"
```

Start `start_bridge_window.ps1` asynchronously from the printed `session_dir`. Ask the user to type credentials directly into that terminal.

Submit operations synchronously and read `results/*.json`:

```powershell
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <name> download <remote-small-file> <local-path>
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <name> stop
```

Use task-supplied paths for directory and NetCDF metadata inventories. Keep session identities, commands, results, and work summaries local and uncommitted.
