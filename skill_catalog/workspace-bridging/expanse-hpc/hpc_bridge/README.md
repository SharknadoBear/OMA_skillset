# Expanse JSON Bridge

This helper keeps one authenticated Paramiko SSH session open and processes local JSON command files. The SSH target is supplied only when creating a local session; credentials are entered only in the visible bridge window and are never written to disk.

Create a named session:

```powershell
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "user@login.expanse.sdsc.edu"
```

Start the printed session directory with `start_bridge_window.ps1`. The bridge tries approved SSH-agent keys, then follows the server's hidden password and TOTP prompts. Use `send_expanse_command.py` for `identity`, `exec`, `upload`, `download`, and `stop` actions.

Runtime `.venv`, `bridge_sessions`, `commands`, `results`, identity, status, account, path, and task information must remain local and uncommitted.
