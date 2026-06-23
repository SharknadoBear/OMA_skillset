---
name: constance-hpc
description: Use when working with Bear's PNNL Constance HPC account at huan111@constance.pnl.gov for secure SSH command execution, small text downloads, read-only project data inspection, module/tool discovery, and NetCDF metadata inventory. Do not run Python analysis, plotting, or install Python analysis environments on the HPC server unless Bear explicitly overrides this; download compact outputs and do Python analysis locally. Never store passwords, MFA codes, or other secrets; keep project data folders read-only unless the user explicitly asks for a separate write workflow.
---

# Constance HPC

## Core Context

- Account username: `huan111`
- Host: `constance.pnl.gov`
- SSH form: `ssh huan111@constance.pnl.gov`
- Known ED25519 host-key fingerprint observed on 2026-06-22:
  `SHA256:tb23nJucub3vtSE3254A7U1AVajet/ITL3eiTE6zUtE`
- Current project data path:
  `/rcfs/projects/mhk_modeling/pic/waveResource/salishSea`

Treat Constance as a PNNL HPC environment. For resource-assessment discovery,
prefer read-only commands such as `ls`, `find`, `stat`, `module avail`,
`ncdump -h`, and `ncks -m`.

Do not use Constance for Python analysis, plotting, or dependency installs by
default. Use it for read-only inspection and compact metadata/output generation,
then download compact artifacts for local Python analysis.

## Credential Rules

- Never store, write, commit, echo, log, or repeat Bear's Constance password,
  MFA code, or combined credential.
- Never put credentials in command lines, bridge command JSON, shell history,
  scripts, logs, memos, or result files.
- Use the local bridge when Codex needs several remote commands. Bear enters
  credentials only in the visible PowerShell bridge window.
- If authentication requires an unsupported interactive challenge, stop and ask
  Bear to run the provided SSH commands manually or configure key/GSSAPI access.

## Local Bridge Workflow

The reusable bridge helper lives inside this skill:

```text
C:\Users\huan111\.codex\skills\constance-hpc\hpc_bridge
```

Before starting or reusing a bridge, inspect the named bridge identity. Reuse
an existing bridge only when its Japanese `bridge_name` is intended and its
purpose/project root match the current task. If the purpose or project root is
different, create a new bridge session.

Create a named bridge session from the reusable helper folder:

```powershell
Set-Location "C:\Users\huan111\.codex\skills\constance-hpc\hpc_bridge"
python .\make_bridge_session.py --purpose "short purpose" --work-summary "1-3 sentence work summary" --project-root "C:\path\to\project"
```

The command prints a `session_dir`. Start the bridge from that session
directory using the existing launcher:

```powershell
Set-Location "<session_dir>"
.\start_bridge_window.ps1
```

The first launch creates a local `.venv` and installs Paramiko from
`requirements.txt`. Runtime folders such as `.venv/`, `commands/`, and
`results/` are disposable session state. Legacy bridge folders without
`bridge_identity.json` are not reusable for new work.

Inspect identity and queue commands from Codex or a local shell:

```powershell
.\.venv\Scripts\python.exe .\send_constance_command.py identity
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name Akatsuki exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_constance_command.py --purpose "short purpose" --project-root "C:\path\to\project" download /remote/small.txt local_small.txt
.\.venv\Scripts\python.exe .\send_constance_command.py --bridge-name Akatsuki stop
```

Bridge actions are intentionally narrow:

- `exec`: run shell commands on Constance.
- `download`: retrieve small text artifacts only; the bridge rejects large
  downloads.
- `stop`: close the persistent SSH session.

There is no upload action in v1.

## Read-Only NetCDF Inventory Pattern

For the Salish Sea wave-resource folder, begin with:

```bash
set -e
ROOT=/rcfs/projects/mhk_modeling/pic/waveResource/salishSea
hostname; whoami; pwd
ls -ld "$ROOT"
find "$ROOT" -maxdepth 2 -type d | sort
find "$ROOT" -maxdepth 4 -type f \( -name '*.nc' -o -name '*.nc4' -o -name '*.cdf' \) -printf '%TY-%Tm-%Td %TH:%TM %12s %p\n' | sort
```

Discover available tools with:

```bash
module avail 2>&1 | grep -Ei 'netcdf|nco|cdo|ncl|hdf' || true
for t in ncdump ncks cdo; do command -v "$t" || true; done
```

Inspect representative NetCDF files with `ncdump -h` when available, then
`ncks -m` or `cdo sinfo` when present. Keep metadata output compact and avoid
reading full arrays remotely. If these tools are unavailable, transfer a small
sample or metadata task back to the local workspace rather than using Python on
Constance.

## Project Reference Schema

The local sample
`Resource/Stations_CG_201101.nc` is a NetCDF4 SWAN/UnSWAN `COMPGRID` file:

- Dimensions: `time=744`, `npnt=120073`, `nele=217388`, `three=3`.
- Coordinates: `Xp`, `Yp` in degrees.
- Connectivity: zero-based triangular `elements`.
- Wave and wind fields: `Depth`, `Hsig`, `Dir`, `RTpeak`, `Period`,
  `X-Windv`, `Y-Windv`.
- Friday Harbor reference node: zero-based node `25434`, approximately
  `(-123.016, 48.5359)`.

Use this schema to plan winter/summer wave maps, wave roses, Hsig-period
frequency plots, wind composites, wave age, and wave steepness diagnostics.
