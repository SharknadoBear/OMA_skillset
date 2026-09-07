# Kestrel JSON Bridge

This helper keeps one authenticated Paramiko SSH session open and processes local JSON command files. The SSH target is supplied only when creating a local session; credentials are entered only in the visible bridge window and are never written to disk.

Create a named session:

```powershell
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "<username>@<hostname>"
```

Start the printed session directory with `start_bridge_window.ps1`. When launching from automation, use `Start-Process -WorkingDirectory <session-dir> -FilePath powershell.exe` rather than `wt.exe`; the latter can misparse the working directory and fail with `0x80070002`. The authenticated SSH transport sends a keepalive every 60 seconds. Use `send_kestrel_command.py` for `identity`, `exec`, `upload`, `download`, and `stop` actions.

Runtime `.venv`, `bridge_sessions`, `commands`, `results`, identity, status, account, path, and task information must remain local and uncommitted.
