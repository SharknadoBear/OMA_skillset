---
name: constance-hpc-copilot
harness: github-copilot
description: GitHub Copilot variant for PNNL Constance HPC read-only access. Use run_in_terminal for bridge orchestration, read_file for result parsing. Scripts shared with sibling constance-hpc/ folder.
---

# Constance HPC — GitHub Copilot Harness

## Core Context

- Account username: `huan111`
- Host: `constance.pnl.gov`
- SSH: `ssh huan111@constance.pnl.gov`
- Host fingerprint: `SHA256:tb23nJucub3vtSE3254A7U1AVajet/ITL3eiTE6zUtE`
- Project data: `/rcfs/projects/mhk_modeling/pic/waveResource/salishSea`
- Constraint: Read-only inspection. No Python analysis/plotting on Constance.

## Credential Rules

- NEVER store, echo, log, or repeat password/MFA in any tool output, memory, or file.
- NEVER send credentials via `send_to_terminal` — tell user to type directly.
- Use the bridge for multi-command sessions. User enters credentials in the bridge window only.

## Copilot-Specific Bridge Workflow

Scripts live in the sibling folder:

```
Agent_skill_dev/skill_catalog/workspace-bridging/constance-hpc/hpc_bridge/
```

### Step 1: Create Bridge Session

```powershell
# run_in_terminal (mode=sync)
Set-Location "Agent_skill_dev\skill_catalog\workspace-bridging\constance-hpc\hpc_bridge"
python .\make_bridge_session.py --purpose "short purpose" --work-summary "summary" --project-root "C:\path\to\project"
```

### Step 2: Start Bridge (User Interactive)

```powershell
# run_in_terminal (mode=async)
Set-Location "<session_dir>"
.\start_bridge_window.ps1
```

Tell user: "Please type your password in the terminal when prompted."

### Step 3: Send Commands

```powershell
# run_in_terminal (mode=sync)
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <NAME> exec "hostname; whoami; pwd"
```

### Step 4: Read Results

Use `read_file` on `<session_dir>/results/*.json`.

### Step 5: Download (Small Files Only)

```powershell
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <NAME> download /remote/small.txt local_small.txt
```

No upload action in v1.

### Step 6: Stop Bridge

```powershell
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name <NAME> stop
```

## NetCDF Inventory Pattern

Via bridge exec:

```bash
ROOT=/rcfs/projects/mhk_modeling/pic/waveResource/salishSea
hostname; whoami; pwd
ls -ld "$ROOT"
find "$ROOT" -maxdepth 2 -type d | sort
find "$ROOT" -maxdepth 4 -type f \( -name '*.nc' -o -name '*.nc4' \) -printf '%TY-%Tm-%Td %TH:%TM %12s %p\n' | sort
```

Tool discovery:

```bash
module avail 2>&1 | grep -Ei 'netcdf|nco|cdo|ncl|hdf' || true
for t in ncdump ncks cdo; do command -v "$t" || true; done
```

## Project Reference Schema

SWAN/UnSWAN COMPGRID (NetCDF4):
- Dimensions: `time=744`, `npnt=120073`, `nele=217388`, `three=3`
- Coordinates: `Xp`, `Yp` (degrees)
- Connectivity: zero-based triangular `elements`
- Fields: `Depth`, `Hsig`, `Dir`, `RTpeak`, `Period`, `X-Windv`, `Y-Windv`
- Reference node: #25434 ≈ (-123.016, 48.5359) (Friday Harbor)

## Copilot Tool Mapping

| Codex Pattern | Copilot Equivalent |
|---------------|-------------------|
| exec command | `run_in_terminal` → `send_constance_command.py exec` |
| read result | `read_file` on results/ JSON |
| password entry | Tell user to type in terminal panel |
| view metadata | `read_file` on bridge_identity.json |
