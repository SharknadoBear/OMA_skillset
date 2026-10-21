# Bridge reliability and workflow continuity

## Design and implementation plan

1. Separate one command lane from one transfer lane. Handle cancellation, bridge stop, and explicit reauthentication outside those lanes. Keep SSH keepalive and bridge-health reporting separate from agent wakeups.
2. Claim requests on disk before execution. Return request IDs promptly and retain original receipts. After an uncertain execution or process restart, reconcile effects before retrying; never replay a submission automatically.
3. Stage transfers in deterministic partial files. Resume only on explicit request after verifying the entire existing prefix. Verify final SHA-256 before atomic publication. Preserve partials on interruption and preserve a different destination unless replacement was explicitly requested.
4. Bound SFTP concurrency, transfer buffers, captured command output, and bridge bookkeeping. Enforce operation deadlines and transfer inactivity limits. Require fresh interactive authentication after a lost session; never save or reuse Password+OTP.
5. Require an app-scheduled follow-up before yielding a workflow that is waiting on remote work. Keep its scope and durable state within the current project. Preserve human timing and stop authority.
6. Validate with a loopback SSH/SFTP server and injected failures, then perform a separate, purpose-bound Kestrel acceptance check with human authentication. Sync the tested skill and publish the repository correction.

## Request lifecycle

The client now waits at most 30 seconds by default. `--timeout` is the total request deadline, including queue time; `--wait` changes only how long the caller waits. A pending response returns exit code 2 and the original request ID. `--no-wait` returns that ID immediately. Otherwise the submission ID is printed to stderr before waiting, with the eventual result on stdout. Neither a pending response nor caller interruption means remote failure or authorizes resubmission.

Use the named session's client:

```powershell
python .\send_kestrel_command.py --bridge-name <name> --timeout 3600 --no-wait upload <local-file> <absolute-remote-path>
python .\send_kestrel_command.py --bridge-name <name> status <request-id>
python .\send_kestrel_command.py --bridge-name <name> result <request-id>
python .\send_kestrel_command.py --bridge-name <name> cancel <request-id>
python .\send_kestrel_command.py --bridge-name <name> reconnect
```

`status` reads local health and progress without queueing another remote operation. Health is refreshed once a second while the bridge loop is alive; older than ten seconds is stale. A fresh local heartbeat does not prove that SSH, SFTP, Slurm, or the simulation is progressing. Inspect those states separately.

`cancel` requests closure of that operation's channels. Closing an exec channel does not prove that its remote process or submitted job stopped. `stop` shuts down this bridge and cancels its queued requests; it does **not** cancel Slurm jobs. For a human stop instruction, identify and cancel only the workflow's owned jobs, verify terminal state, then stop its bridge and pause its follow-up.

Remote commands remain ordered within their lane; transfers remain ordered within theirs. They can overlap, so wait for an upload's successful receipt before submitting any command that consumes that file. A bridge restart marks unfinished claims uncertain. Inspect their original evidence and remote effects before proceeding.

## Transfer recovery

These partial-file and atomic-publication rules describe the SSH bridge. For large files or repeated bridge transfer failures, use [Globus transfers and recovery](globus-transfers.md), which has a separate task lifecycle. First reconcile or cancel the original writer and verify it has stopped; do not overlap tools on the same destination.

- Both paths must identify individual files; remote paths must be absolute. File bundles are useful for many small files.
- The partial filename is derived from the session directory and source/destination pair. Reconnect or restart that same session to use `--resume`, which verifies the prefix against the current source before appending. A mismatch is preserved and rejected, never silently restarted over the partial. A different session uses its own partial file.
- An identical existing destination is recognized by size and SHA-256. A different destination requires `--overwrite`; publication uses atomic replacement. No remove-then-rename fallback is permitted.
- Downloads use four-MiB read windows with at most 32 prefetch requests. Uploads close four-MiB pipelined windows to collect acknowledgements. These bound buffering; they do not guarantee network bandwidth.
- `--stall-timeout` defaults to 120 seconds. Channel I/O also has a maximum 30-second wait. Hashing and publication remain subject to the overall operation deadline; their phase is visible, but a remote checksum is not reported as byte-by-byte progress.
- Interrupted transfers retain partials. Reauthenticate in the visible session window with `reconnect` if needed, inspect the original result, then explicitly reissue that transfer with `--resume`. No command is replayed by reconnecting.
- Never resume against files being actively rewritten. Transfer closed outputs or stable bundles; hashes and metadata checks reject detected changes.

One MiB per stdout/stderr stream is retained. Larger output is drained but returns `output_limit` and `capture_complete: false`; obtain a compact remote summary or deliberately retrieve the log. Progress files are overwritten, not appended. At 64 MiB of session bookkeeping, new remote operations are held and health reports `storage_limit`; status, cancellation, and stop remain available. Preserve required receipts before explicitly pruning session bookkeeping. Scientific files and transfer partials are outside this bookkeeping cap.

## Mandatory follow-up for a remote wait

For an authorized workflow that must continue after an asynchronous transfer, remote command, or submitted job, establish the follow-up **before ending the turn**:

1. Preserve one small project-local wait record: workflow/chat identity, transport, bridge session/request ID or Globus collection/submission/task IDs and exact paths, owned job IDs with submission receipts, current phase, deadline, expected next check, last successful observation, and permitted next action. Record the automation identity after creation. Never store credentials or full repeated snapshots.
2. Use the app's automation interface to create or update a heartbeat for this same chat. Reuse the workflow's existing heartbeat instead of creating duplicates. Choose a check interval suited to the expected wait and state the next check time. An SSH keepalive or a running bridge is not a substitute for a scheduled chat wakeup.
3. On wakeup, read the wait record and original request receipt first. For SSH work, inspect bridge freshness and exact owned job state. For Globus, inspect the original task and its events; an SSH bridge need not be running for a Globus transfer. Continue only the already-authorized workflow; do not launch a replacement trial, reset a human clock, change scoring, or infer success from an empty/failed queue response or an incomplete destination file.
4. Stay quiet while state is unchanged. Notify only for completion, failure, authentication, missed/stale monitoring that needs action, or another required human decision. Request SSH Password+OTP only through the visible bridge window; request Globus authentication/consent only through its official browser flow. While authentication is pending, preserve the wait and avoid repeated prompts or reconnect loops.
5. Pause the heartbeat when the workflow is complete, stopped, or explicitly handed back to the user. Record the final disposition. On resume after a missed wakeup or app outage, report the observation gap honestly and reconcile current state; do not invent observations or alter the user's timing record.

If scheduling is unavailable, record that automatic continuation is not established and provide the concrete next manual check. Never claim a wakeup is scheduled without a successful automation receipt. Do not create a campaign controller, global hooks, another polling daemon, or an automation when no active workflow needs one.

## Validation and rollout

Run `test_bridge_output_completion.py` and `test_bridge_reliability.py` using the pinned Paramiko environment. The latter uses a loopback-only SSH/SFTP server with temporary files and synthetic authentication; it never contacts Kestrel.

The pre-trial Kestrel acceptance check uses a new purpose-bound session and a small disposable directory outside scientific trials. Confirm identity, upload/download a bounded fixture, compare hashes, interrupt and resume a transfer, and check that a small read-only command and cancellation remain responsive. Exercise one explicit reauthentication and a scheduled follow-up. Do not submit a simulation as part of bridge validation. This live check requires human authentication and is separate from the local test result.

Existing sessions contain copies of their helpers. Updating the installed skill affects newly created sessions; it must not hot-patch or restart unrelated active bridges.
