---
name: constance-hpc
description: Use for secure, read-only SSH inspection of an authorized Constance HPC account, including small text downloads, module discovery, and compact NetCDF metadata inventory. Obtain the SSH target, approved host-key fingerprint, and project paths privately at runtime; never store account identifiers or task details in the skill repository.
---

# Constance HPC

## Runtime configuration

- Obtain the SSH target as `<username>@<hostname>` at runtime.
- Obtain the approved SHA256 host-key fingerprint through a trusted channel at runtime.
- Obtain each project or data path from the current task; do not embed it in this package.
- Keep access read-only unless the user explicitly authorizes a separately reviewed write workflow.
- Use remote tools only for inspection or compact metadata generation. Download small artifacts for local analysis.

## Credential rules

- Never store, write, commit, echo, log, or repeat passwords or MFA values.
- Enter credentials only in the visible bridge window.
- Do not place credentials in commands, JSON, shell history, logs, memos, or results.
- Stop if the available interface cannot securely handle the authentication challenge.

## Local bridge workflow

The versioned helper is in:

```text
Agent_skill_dev\skill_catalog\workspace-bridging\constance-hpc\hpc_bridge
```

Create a purpose-bound session with private connection values supplied at runtime:

```powershell
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\constance-hpc\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "<username>@<hostname>" `
  --host-key-fingerprint "SHA256:<approved-fingerprint>"
```

Start the bridge from the printed `session_dir`, then inspect its identity before sending commands:

```powershell
Set-Location "<session-dir>"
.\start_bridge_window.ps1
.\.venv\Scripts\python.exe .\send_constance_command.py identity
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <name> download <remote-small-file> <local-path>
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <name> stop
```

Supported actions are `exec`, `download`, and `stop`. Upload is intentionally unavailable. Downloads are size-limited by the helper.

Runtime folders and files—including sessions, identities, project roots, purposes, commands, results, and status—must remain local and uncommitted.

## Read-only inventory pattern

Use a task-supplied approved root:

```bash
ROOT=<approved-remote-root>
hostname; whoami; pwd
ls -ld "$ROOT"
find "$ROOT" -maxdepth 2 -type d | sort
find "$ROOT" -maxdepth 4 -type f \( -name '*.nc' -o -name '*.nc4' -o -name '*.cdf' \) -printf '%TY-%Tm-%Td %TH:%TM %12s %p\n' | sort
```

Discover metadata tools with:

```bash
module avail 2>&1 | grep -Ei 'netcdf|nco|cdo|ncl|hdf' || true
for tool in ncdump ncks cdo; do command -v "$tool" || true; done
```

Inspect representative headers with `ncdump -h`, `ncks -m`, or `cdo sinfo` when available. Avoid reading full arrays remotely.
