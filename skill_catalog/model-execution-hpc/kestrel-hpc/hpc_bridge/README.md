# Kestrel JSON Bridge

This bridge keeps one authenticated Paramiko SSH session open to Kestrel and
lets Codex submit JSON command files locally. Bear enters `Password+OTP` only
inside the visible bridge window. Credentials are never written to disk.

## Start

Create a named bridge session first:

```powershell
python .\make_bridge_session.py --purpose "short purpose" --work-summary "1-3 sentence summary" --project-root "C:\path\to\project"
```

Then start the bridge from the printed `session_dir`:

```powershell
.\start_bridge_window.ps1
```

The first launch creates `.venv` and installs `paramiko`.
Runtime folders such as `.venv/`, `commands/`, and `results/` are local
session state and should not be committed.

## Send Commands

```powershell
.\.venv\Scripts\python.exe .\send_kestrel_command.py identity
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name Akatsuki exec "hostname; pwd"
.\.venv\Scripts\python.exe .\send_kestrel_command.py --purpose "short purpose" --project-root "C:\path\to\project" upload local_file /scratch/yhuang168/path/file
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name Akatsuki download /scratch/yhuang168/path/file local_file
.\.venv\Scripts\python.exe .\send_kestrel_command.py --bridge-name Akatsuki stop
```

Command JSON files are written under `commands/`; result JSON files are written
under `results/`. Processed command files move under `commands/processed/`.
