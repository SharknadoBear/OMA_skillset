---
name: cloudvm-bridge
description: Use when working with Huan's PNNL cloud VM account at huan111@automodeldev01.pnl.gov for secure SSH command execution, file upload/download, Hermes/FVCOM setup checks, VM evidence collection, and controlled project-workspace operations through a local Paramiko bridge. Never store passwords or secrets.
---

# Cloud VM Bridge

## Core Context

- Account username: `huan111`
- Host: `automodeldev01.pnl.gov`
- SSH target: `huan111@automodeldev01.pnl.gov`
- Bridge scripts: `scripts/`
- Research/data constraint: use only public or non-sensitive OMI/FVCOM project material unless a separate approved pathway exists.

## Credential Rules

- Never store, write, commit, echo, log, or repeat the cloud VM password.
- Never put the password in this skill, command JSON, result JSON, shell history, prompts, memory, scripts, or generated outputs.
- Ask Huan to enter the password only in the visible local bridge window when `Password:` appears.
- Do not ask Huan to paste the password into chat.
- If the available terminal cannot securely answer an interactive password prompt, use the local bridge window, an existing authenticated session, or another approved secure workflow.

## Local Bridge Workflow

Use the bridge when Codex needs several rounds of VM navigation or file transfer. The bridge keeps one authenticated Paramiko SSH session open. Huan enters the password only in the visible PowerShell window; Codex queues JSON command files locally, the bridge executes them on the VM, and results are written locally.

The active installed skill path is:

```text
C:\Users\huan111\.codex\skills\cloudvm-bridge
```

The versioned source-of-truth path is:

```text
Agent_skill_dev\skill_catalog\model-execution-hpc\cloudvm-bridge
```

Start the bridge from the installed skill:

```powershell
Set-Location "C:\Users\huan111\.codex\skills\cloudvm-bridge\scripts"
.\start_bridge_window.ps1
```

The first launch creates a local `.venv` and installs `paramiko` from `requirements.txt`. Runtime folders such as `.venv/`, `commands/`, and `results/` are disposable session state and must not be committed.

Queue commands from Codex or a local shell:

```powershell
.\.venv\Scripts\python.exe .\send_cloudvm_command.py exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_cloudvm_command.py upload local_file ~/remote_file
.\.venv\Scripts\python.exe .\send_cloudvm_command.py download ~/remote_file local_file
.\.venv\Scripts\python.exe .\send_cloudvm_command.py stop
```

Supported bridge actions are `exec`, `upload`, `download`, and `stop`. Result JSON files contain only `id`, `status`, `exit_status`, `stdout`, and `stderr`.

## Operating Rules

1. Inspect the intended remote command before queueing it.
2. Keep remote work inside approved project/workspace paths.
3. Prefer read-only inspection and small smoke tests before setup changes.
4. Upload only intended files and avoid broad recursive transfers unless the user explicitly approves them.
5. Use single-file archives for bundles if directory transfers become fragile.
6. Do not place secrets, credentials, tokens, SSH keys, or sensitive files in uploaded paths, command output, logs, or evidence packages.
7. Stop the bridge when finished so the authenticated SSH session closes.

## Validation Pattern

After bridge changes, validate locally before live VM use:

```powershell
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py Agent_skill_dev\skill_catalog\model-execution-hpc\cloudvm-bridge
python -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in [Path('Agent_skill_dev/skill_catalog/model-execution-hpc/cloudvm-bridge/scripts/bridge_server.py'), Path('Agent_skill_dev/skill_catalog/model-execution-hpc/cloudvm-bridge/scripts/send_cloudvm_command.py')]]"
```

For a live smoke test, start the installed bridge, have Huan enter the password in the bridge window, then run:

```powershell
.\.venv\Scripts\python.exe .\send_cloudvm_command.py exec "hostname; whoami; pwd; uname -a"
```

Accept the smoke test only if `whoami` is `huan111` and the hostname matches the approved cloud VM.
