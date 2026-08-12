---
name: cloudvm-bridge-copilot
harness: github-copilot
description: GitHub Copilot variant for PNNL Cloud VM access via Paramiko bridge. Use run_in_terminal for bridge orchestration, read_file for results. Scripts shared with sibling cloudvm-bridge/ folder.
---

# Cloud VM Bridge — GitHub Copilot Harness

## Core Context

- Account username: `huan111`
- Host: `automodeldev01.pnl.gov`
- SSH target: `huan111@automodeldev01.pnl.gov`
- Constraint: Public or non-sensitive OMI/FVCOM project material only.

## Credential Rules

- NEVER store, echo, log, or repeat the cloud VM password.
- NEVER send password via `send_to_terminal` — tell user to type directly in the bridge terminal.
- Do not place credentials in command JSON, result files, or memory.

## Copilot-Specific Bridge Workflow

Scripts live in the sibling folder:

```
Agent_skill_dev/skill_catalog/workspace-bridging/cloudvm-bridge/scripts/
```

### Step 1: Create Bridge Session

```powershell
# run_in_terminal (mode=sync)
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\cloudvm-bridge\scripts"
python .\make_bridge_session.py --purpose "short purpose" --work-summary "summary" --project-root "C:\path\to\project"
```

### Step 2: Start Bridge (User Interactive)

```powershell
# run_in_terminal (mode=async)
Set-Location "<session_dir>"
.\start_bridge_window.ps1
```

Tell user: "Please type your password in the terminal when `Password:` appears."

### Step 3: Send Commands

```powershell
# run_in_terminal (mode=sync)
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <NAME> exec "hostname; whoami; pwd"
```

### Step 4: File Transfer

```powershell
# Upload
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <NAME> upload local_file ~/remote_file

# Download
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <NAME> download ~/remote_file local_file
```

### Step 5: Read Results

Use `read_file` on `<session_dir>/results/*.json`. Result JSON contains `id`, `status`, `exit_status`, `stdout`, `stderr`.

### Step 6: Stop Bridge

```powershell
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <NAME> stop
```

Then `kill_terminal` on the async terminal.

## Operating Rules

1. Inspect the intended remote command before queueing.
2. Keep work inside approved project/workspace paths.
3. Prefer read-only inspection and small smoke tests before setup changes.
4. Upload only intended files; avoid broad recursive transfers.
5. Use single-file archives for bundles if directory transfers are fragile.
6. Stop the bridge when finished.

## Smoke Test

After bridge starts and user authenticates:

```powershell
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <NAME> exec "hostname; whoami; pwd; uname -a"
```

Accept only if `whoami` returns `huan111` and hostname matches the cloud VM.

## Copilot Tool Mapping

| Codex Pattern | Copilot Equivalent |
|---------------|-------------------|
| exec command | `run_in_terminal` → `send_cloudvm_command.py exec` |
| read result JSON | `read_file` on results/ folder |
| password entry | Tell user to type in async terminal |
| start bridge | `run_in_terminal` (mode=async) |
| stop bridge | `run_in_terminal` (mode=sync) + `kill_terminal` |
