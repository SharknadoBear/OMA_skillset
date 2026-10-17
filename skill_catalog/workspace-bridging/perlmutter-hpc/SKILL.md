---
name: perlmutter-hpc
description: Connect to NERSC Perlmutter through a purpose-scoped SSH/SFTP bridge; inspect modules and storage, transfer files, build software, and run authorized Slurm workflows. Use for Perlmutter or NERSC work, including requests calling the system permuter. Credentials are entered only in a local terminal.
---

# Perlmutter HPC

Use `perlmutter.nersc.gov` with the NERSC username supplied by the user or matching private session identity. Keep usernames, allocations, inventories and session records outside the versioned package. Never infer the remote username from the local Windows account.

## Connect

Resolve paths from this skill directory. Run `scripts/Setup.ps1` once on Windows to create `.venv` with locked dependencies. The CLI is `scripts/bridge.py`; below, `$python` means the skill's `.venv/Scripts/python.exe`, and `$bridge` means that CLI.

Inspect existing sessions under `%LOCALAPPDATA%/Codex/perlmutter-hpc/sessions`. Reuse only a healthy connection whose user, purpose and project root match the current task. An old heartbeat is not evidence of a live connection.

```powershell
& $python $bridge init --user <nersc-user> --purpose "Inspect modules and storage" --project-root <local-project-root>
& $python $bridge hostcheck
```

Start the returned session in a visible terminal for the user's password plus OTP. Start PowerShell directly rather than a `wt.exe` action string, which can mishandle paths containing spaces:

```powershell
Start-Process -FilePath powershell.exe -WorkingDirectory "$skillRoot/scripts" -WindowStyle Normal -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','.\Start-Bridge.ps1','-Session',('"'+$session+'"'))
```

Credentials belong only in that terminal's hidden input prompt, never chat, arguments, files, environment variables, screenshots or logs. Do not capture the authentication window. The helper verifies NERSC's published host fingerprint before asking for credentials. One authentication attempt is made per launch; investigate mismatches or failures instead of automatically retrying. See [authentication notes](references/nersc.md#authentication).

```powershell
& $python $bridge status --session $session
& $python $bridge probe --session $session --timeout 180 --wait 1
& $python $bridge result --session $session <request-id> --wait 30
```

Wait for `ready` before queueing work. `login.json` records authenticated user, host and initial directory. The probe reads home/scratch paths, module defaults and relevant software availability. Distinguish observed paths from documented examples. Module tools often write informational text to stderr; inspect it with the exit status.

## Commands and transfers

```powershell
& $python $bridge exec --session $session --command-file <utf8-shell-script> --timeout 120 --wait 1
& $python $bridge upload --session $session <local-file> <absolute-remote-file> --timeout 300
& $python $bridge download --session $session <local-file> <absolute-remote-file> --timeout 300
& $python $bridge stop --session $session --wait 5
```

Both transfers use **local path first, remote path second**. Existing destinations are refused unless `--overwrite` is supplied for an authorized replacement. SFTP does not expand variables or `~`; use verified absolute paths. Receipts contain a stream SHA-256, not an independent destination checksum. Failed transfers can leave partial destinations; inspect before retry or cleanup. Use DTNs/Globus for bulk data; this bridge targets Perlmutter and individual files.

Each command runs in a fresh `bash -l` shell on the persistent connection. `cd` and module changes do not persist between requests; put setup and dependent work in the same script. Prefer command files to nested PowerShell/POSIX quoting. Remote stdin closes immediately, so commands must be noninteractive.

Requests are claimed before execution and have immutable IDs. Reconcile pending or uncertain results by the original ID, never automatic duplicate submissions. Timeout closes a channel but does not prove the remote process stopped. Startup refuses unresolved claimed requests. Results expose nonzero exit codes and truncation; missing output is not success. `--wait` controls local waiting (maximum 60 seconds); `--timeout` controls the remote operation.

## Environment and work scope

Read [NERSC operating notes](references/nersc.md) when choosing modules, storage or compute resources. Discover current versions using `module -t list`, `module avail`, `module spider` and `module show`. Inspect HDF5/compiler prerequisites before loading NetCDF and verify CPU/GPU targeting against the intended compute environment. Do not copy Kestrel partitions, paths, CPU counts or MAC overrides.

Use `$HOME` for small durable sources/configuration and `$SCRATCH` for active high-I/O runs. Retain important outputs in authorized durable storage. Verify paths, quotas and purge policy before staging data.

Task authorization determines scope. Inspection permits inspection, not submissions or edits to startup files. For authorized builds/jobs, review the exact script, allocation, constraint, QoS, node count, wall time and output paths. Substantial computation belongs in Slurm allocations. Keep job IDs and inspect accounting/logs before claiming completion; scheduler success is not scientific validation.

Stop the bridge when its purpose is complete. It also disconnects after 30 idle minutes by default. For an existing sshproxy credential, initialize with `--key-file <private-key>`; the adjacent `-cert.pub` certificate must exist. A rejected certificate never silently falls back to another identity.

## Maintenance

Run `scripts/test_bridge.py` with the skill's Python for local behavioral tests without credentials, and run the skill-creator validator after edits. Exclude `.venv`, local runtime records and private test reports from commits. Update the host fingerprint pin only after independent verification against current NERSC documentation.
