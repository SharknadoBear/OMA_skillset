---
name: cloudvm-bridge
description: Use for secure SSH command execution and controlled file transfer on an authorized cloud VM through a local Paramiko bridge. Obtain the SSH target, approved workspace paths, and credentials privately at runtime; never store account identifiers, infrastructure addresses, secrets, or task details in the skill repository.
---

# Cloud VM Bridge

## Runtime configuration

- Obtain the SSH target as `<username>@<hostname>` at runtime.
- Obtain approved remote workspace paths from the current task.
- Keep the target, account, paths, credentials, and task details out of the versioned package.
- Use only data and operations permitted for the configured VM.

## Credential rules

- Never store, write, commit, echo, log, or repeat the VM password.
- Enter the password only in the visible local bridge window.
- Do not place credentials in chat, commands, JSON, results, prompts, memory, scripts, or generated evidence.
- If secure interactive entry is unavailable, use an approved key, authenticated session, or user-run workflow.

## Local bridge workflow

The versioned helper is in:

```text
Agent_skill_dev\skill_catalog\workspace-bridging\cloudvm-bridge\scripts
```

Create a purpose-bound session with the private target supplied at runtime:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\cloudvm-bridge\scripts"
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

Inspect identity and submit operations:

```powershell
.\.venv\Scripts\python.exe .\send_cloudvm_command.py identity
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_cloudvm_command.py --bridge-name <name> stop
```

Supported actions are `exec`, `upload`, `download`, and `stop`. Result JSON contains only the command ID, status, exit status, standard output, and standard error.

Runtime folders and files—including sessions, identities, project roots, purposes, commands, results, and status—must remain local and uncommitted.

## Operating rules

1. Inspect each command before queueing it.
2. Confirm the runtime identity with `hostname`, `whoami`, and `pwd`.
3. Keep remote work inside approved paths.
4. Prefer read-only inspection and small smoke tests before setup changes.
5. Upload only intended files and avoid broad recursive transfers.
6. Use a single archive for bundles when directory transfer is unreliable.
7. Stop the bridge when the workplan no longer needs the authenticated session.

## Validation

Validate the catalog package with the installed skill validator, then compile the Python helpers without creating repository bytecode:

```powershell
python "<skill-creator-root>\scripts\quick_validate.py" "Agent_skill_dev\skill_catalog\workspace-bridging\cloudvm-bridge"
python -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in Path('Agent_skill_dev/skill_catalog/workspace-bridging/cloudvm-bridge/scripts').glob('*.py')]"
```

For a live smoke test, accept only when `whoami` and `hostname` match the privately configured and approved target.
