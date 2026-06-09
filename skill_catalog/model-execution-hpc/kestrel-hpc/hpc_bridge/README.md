# Kestrel JSON Bridge

This bridge keeps one authenticated Paramiko SSH session open to Kestrel and
lets Codex submit JSON command files locally. Huan enters `Password+OTP` only
inside the visible bridge window. Credentials are never written to disk.

## Start

```powershell
.\start_bridge_window.ps1
```

The first launch creates `.venv` and installs `paramiko`.
Runtime folders such as `.venv/`, `commands/`, and `results/` are local
session state and should not be committed.

## Send Commands

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py exec "hostname; pwd"
.\.venv\Scripts\python.exe .\send_kestrel_command.py upload local_file /scratch/yhuang168/path/file
.\.venv\Scripts\python.exe .\send_kestrel_command.py download /scratch/yhuang168/path/file local_file
.\.venv\Scripts\python.exe .\send_kestrel_command.py stop
```

Command JSON files are written under `commands/`; result JSON files are written
under `results/`. Processed command files move under `commands/processed/`.
