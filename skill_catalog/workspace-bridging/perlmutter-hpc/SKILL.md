---
name: perlmutter-hpc
description: Use for authorized SSH/SFTP access, file transfers, compilation, Slurm workflows and compact-result retrieval on NERSC Perlmutter through the same named-session bridge interface as Kestrel. Obtain NERSC account and allocation details privately; credentials stay in the visible terminal.
---

# Perlmutter HPC

Use `user@perlmutter.nersc.gov` as the public example. Resolve the actual NERSC username from approved private context, never from the local Windows username. Keep real usernames, allocations, project paths and session records outside the versioned package.

Before asking for account details, inspect `%LOCALAPPDATA%/Codex/perlmutter-hpc/account.local.json` when present. Reuse user-confirmed details for the matching workspace, distinguishing a confirmed project from live-verified CPU/GPU charging-account access.

The bridge verifies NERSC's published RSA host fingerprint before requesting authentication, with RSA SHA-2 signatures and no Kestrel MAC override. Enter password immediately followed by OTP only in its visible terminal. Never collect credentials through chat, files, arguments, environment variables, screenshots or logs. The credential-free host preflight closes before you type. After Enter, the bridge opens and verifies a fresh connection, then makes one authentication attempt; this prevents the server's pre-login timeout from expiring during human input. The terminal shows DNS/TCP, SSH verification and authentication progress plus a sanitized failure phase. DNS/TCP share a 20-second total deadline (DNS at most 5 seconds), with at most four resolved endpoint attempts in flight; the limit does not multiply by the number of DNS addresses. Investigate failures before explicit reconnect. See [NERSC authentication](references/nersc.md#authentication).

## Local bridge workflow

Use the persistent Paramiko bridge for multi-command sessions:

```text
<perlmutter-hpc-skill>\hpc_bridge
```

Create a purpose-bound session with the target supplied at runtime:

```powershell
Set-Location "<perlmutter-hpc-skill>\hpc_bridge"
python .\make_bridge_session.py `
  --purpose "short purpose" `
  --work-summary "brief work summary" `
  --project-root "<local-project-root>" `
  --remote-target "user@perlmutter.nersc.gov"
```

Start the printed session directory and enter Password+Token only in its visible window:

```powershell
Set-Location "<session-dir>"
.\start_bridge_window.ps1
```

When launching the interactive bridge from automation on Windows, start PowerShell directly with the session as its working directory. Do not route the command through `wt.exe`, a Windows Terminal profile/action string, or a constructed tab title; those forms can reinterpret `-d` and OneDrive paths and report `0x80070002` even when the script exists.

```powershell
Start-Process -WorkingDirectory "<session-dir>" -FilePath "powershell.exe" `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ".\start_bridge_window.ps1") `
  -WindowStyle Normal
```

The bridge keeps commands and transfers in separate lanes, reports local health every second, and uses a 60-second SSH keepalive. Reuse a healthy purpose-matched session. For transfer recovery, cancellation, or remote waits, read [bridge reliability and continuity](references/bridge-reliability.md).

Use an explicit `--timeout` for the operation; the caller returns a request ID after a short wait. A pending response is not failure: inspect `status <request-id>` and `result <request-id>` before retrying. Reconcile uncertain remote effects before any repeat submission. Transfers use verified partial files; `--resume` checks the prefix, and `--overwrite` explicitly permits replacing a different destination. Await an upload's successful receipt before running a command that consumes it.

Commands must be noninteractive; remote stdin is closed. Protect shell metacharacters across PowerShell and the remote shell. After a lost connection, request `reconnect` and authenticate in the visible window; failed commands are not replayed. Cancelling a bridge request or stopping the bridge does not establish that a Slurm job has stopped.

Before yielding an authorized workflow that is waiting on remote work, record its request/job identities and next check, and create or update a same-chat heartbeat through the app's automation interface. Follow the continuity procedure above: inspect original receipts, stay quiet on unchanged state, preserve human timing and stop authority, and pause the follow-up at completion or stop. If scheduling is unavailable, report that automatic continuation is not established. Do not create global hooks or a campaign controller.

Verify identity before every operation:

```powershell
.\.venv\Scripts\python.exe .\send_perlmutter_command.py identity
.\.venv\Scripts\python.exe .\send_perlmutter_command.py --bridge-name <name> exec "hostname; whoami; pwd"
.\.venv\Scripts\python.exe .\send_perlmutter_command.py --bridge-name <name> upload <local-file> <remote-path>
.\.venv\Scripts\python.exe .\send_perlmutter_command.py --bridge-name <name> download <remote-file> <local-path>
.\.venv\Scripts\python.exe .\send_perlmutter_command.py --bridge-name <name> stop
```

Reuse a session only while its bridge name, purpose, project root, and remote target still match the active work. Keep environments, session folders, commands, inflight claims, results, progress, identity, and status files local and uncommitted. New helpers apply to newly created sessions; leave unrelated active sessions untouched.

## NERSC environment and work scope

Read [NERSC operating notes](references/nersc.md) when choosing modules, storage or resources. Inspect current modules and actual home/scratch paths in the authenticated session. The read-only [probe](references/probe.sh) supplies a bounded environment inventory. Commands use independent remote shells: put `cd`, module setup and dependent work in the same command; explicitly use `bash -lc` when a login environment is needed.

Use home for small durable sources and configuration, scratch for active high-I/O work, and approved durable storage for retained outputs. Verify quota and purge policy. Do not copy Kestrel partitions, CPU counts, paths or accounting rules.

Task authorization determines scope. For authorized builds/jobs, inspect the exact script, allocation, constraint, QoS, resources and output paths before submission. Substantial computation belongs in Slurm allocations. Preserve exact job IDs and inspect accounting, logs and outputs before claiming completion. Bridge cancellation does not cancel an owned scheduler job.

Optional existing NERSC sshproxy credentials use `make_bridge_session.py --key-file <private-key>` alongside the common session arguments. The adjacent `-cert.pub` certificate is required. Certificate failure does not fall back to password authentication or another identity.

## Migration and validation

The old `scripts/bridge.py --session` interface is retired. Create a new named session for this implementation; retain old sessions and their original request receipts as evidence. Do not replay interrupted old requests. The download order is now `remote-file local-file`, matching Kestrel. Automatic idle disconnection is replaced by the shared explicit `stop` lifecycle. Existing unrelated bridges are not patched or restarted.

`hpc_bridge/bridge_runtime.py`, `bridge_transfer.py`, the setup launcher, naming pool and output-completion tests are copied unchanged from the latest installed Kestrel bridge. The sender differs only in its site name. The server's connection function delegates NERSC authentication to `site_auth.py`; session creation additionally validates the NERSC target and supports sshproxy. Preserve this separation when updating the common transport.

Run `test_bridge_output_completion.py`, `test_bridge_reliability.py`, `test_site_auth.py` and `test_site_network.py` in the Paramiko 5 environment, then the skill-creator validator. These local tests require no credentials. A separate authenticated check verifies live identity and bounded file transfers. Keep environments, private accounts, test reports and session records uncommitted.
