---
name: cloudvm-bridge-copilot
description: GitHub Copilot instructions for authorized cloud VM access through the sibling Paramiko bridge. Obtain the SSH target, approved paths, and credentials privately at runtime; use terminal execution for bridge orchestration and file reading for result JSON.
---

# Cloud VM Bridge — GitHub Copilot

## Privacy and credentials

- Obtain `<username>@<hostname>` and approved remote paths only at runtime.
- Never commit account identifiers, hosts, project paths, credentials, or task details.
- Never send a password through terminal automation. Ask the user to type it directly into the interactive bridge terminal.
- Keep session identities, commands, results, and work summaries local and uncommitted.

## Bridge workflow

Use the scripts in:

```text
Agent_skill_dev/skill_catalog/workspace-bridging/cloudvm-bridge/scripts/
```

Create a session synchronously:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\cloudvm-bridge\scripts"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "<username>@<hostname>"
```

Start `start_bridge_window.ps1` asynchronously from the printed `session_dir`. Ask the user to enter the password directly when the terminal prompt appears.

Submit operations synchronously and read `results/*.json`:

```powershell
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> stop
```

Verify the bridge identity before reuse, inspect commands before queueing, keep work inside approved paths, and stop the session when finished. Accept a smoke test only when the returned user and hostname match the privately configured target.
