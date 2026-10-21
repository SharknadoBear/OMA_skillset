# Globus transfers and recovery

Use Globus for large uploads/downloads or repeated SSH transfer failures. Continue commands and Slurm work through the bridge. Before changing transport, reconcile the original receipt and stop its writer. Filesystem outages can affect both transports; preserve errors and use a bounded deadline.

## Establish access and paths

- Globus Connect Personal (GCP), the browser login, and CLI authorization are separate. Reuse an authorized CLI; an existing browser session may still require CLI consent. Use `globus whoami` and a scoped `ls` to check access. When needed, run `globus login` or the CLI's reported `globus session consent` command and let the user authenticate on the official page. Never read, copy, print, or commit OAuth tokens, passwords, OTPs, or authorization codes.
- Find an existing CLI before installing another. A Windows isolated installation can use `%LOCALAPPDATA%\GlobusCLI\venv\Scripts\globus.exe`; call its full path if it is not on PATH. For a new installation, use `python -m venv <cli-venv>` and that environment's `python -m pip install globus-cli`. Record `globus version` in the private receipt.
- GCP must be running and the workstation awake/connected for data movement. Do not stop another user's transfers or restart GCP merely to check health. Use `globus endpoint local-id`, or discover and verify the user's local collection, then confirm its accessible folders.
- Discover the current official Kestrel collection and record its UUID privately. Public names include `nrel#kglobus_scratch` and `nrel#kglobus_projects`; verify against the [NLR guidance](https://natlabrockies.github.io/HPC/Documentation/Managing_Data/Transferring_Files/globus/). Grant only the required data-access consent. Collection-management scopes are not needed to transfer files.
- Collection paths differ from SSH paths. Inspect `globus ls <collection-id>:/ --format json` and `globus stat <collection-id>:<approved-parent> --format json`. Verify the intended owner and directory. A scratch collection can expose `/<user>/` rather than `/scratch/<user>/`; a Windows `/~/` can map to Documents rather than the Windows profile root. Verify the mapping with a known file and size. Do not assume home is exposed or silently change the requested destination.
- Keep source/destination roots within the approved project. Do not broaden GCP folder permissions or stage data elsewhere without authorization. Reject unexpected symlinks/junctions at selected roots and files; explicitly selected source symlinks can be followed by Globus. For many small files, a stable archive and hash manifest can reduce metadata work.

## Submit once and retain identity

Transfer only closed, stable inputs/outputs. Record source size and SHA-256, collection UUIDs, exact paths, intended overwrite behavior, and a task deadline in one small private receipt. Use a new destination or verify that replacing it is within the authorized scope.

The commands below use `globus` as shorthand for the verified CLI executable. Substitute runtime values; do not put private identities in this skill. Check each exit code before using JSON output.

```text
globus task generate-submission-id --format json
```

Save the returned `value` **before** submitting. Then submit an individual file:

```text
globus transfer <source-collection>:<source-file> <destination-collection>:<destination-file> --no-recursive --verify-checksum --encrypt-data --submission-id <saved-id> --deadline <UTC-YYYY-MM-DDTHH:MM:SS> --label <short-private-label> --format json
```

Use an explicit UTC deadline; this CLI format omits the trailing `Z`. Retain the complete small submission response and `task_id`. Download by swapping source and destination. For a directory, review the manifest and scope before using `--recursive`; do not add destination deletion or recursive symlink following.

An accepted request means queued, not completed. If the submission response is lost, preserve the submission ID and exact arguments. Reconcile task history; if necessary resubmit the **same request with the same submission ID** to recover its task identity. Never allocate a fresh ID while the original outcome is uncertain.

## Observe and recover the original task

```text
globus task show <task-id> --format json
globus task event-list <task-id> --limit 20 --format json
```

- Bind every observation to the saved task's collections, paths/submission receipt, and label. Keep only compact progress and meaningful errors. Use bounded checks; retry backoff can leave a task active without immediate byte growth. A missing or failed query is an observation gap, not proof of failure or completion.
- An active Globus task owns retry/recovery after an endpoint or network interruption. Restore the connection or GCP as appropriate and keep observing that same task. Do not cancel it or launch SFTP/SCP merely because the CLI returned or progress paused. Retain faults even when it eventually succeeds; do not infer exact partial-byte reuse from file size alone.
- If consent or authentication expires, report it once and complete the indicated official browser flow. Reconcile the existing task afterward. Browser login does not prove every collection's consent is current.
- Require terminal `SUCCEEDED`, checksum verification enabled, no skipped-error files, and the intended file count/size before releasing files to downstream work. A destination can exist, or even have its final size, before success. Globus does not inherit the bridge's atomic-publication guarantee. For downloads, compare local SHA-256 with the retained source hash when available. For a round-trip test, require exact source/returned-file hash equality.
- A terminal failed/canceled task may leave files. After preserving the failure and confirming no writer remains, an authorized new task can use `--sync-level checksum --verify-checksum` to reconcile the same manifest; existence or size alone cannot certify partial files. Retain the old and new task IDs and their relationship. Do not claim a new task resumed the old one.
- A user stop applies to the exact owned task: `globus task cancel <task-id>`, followed by terminal reconciliation. Cancellation is separate from stopping a bridge or a Slurm job and may leave partial data. Delete only approved disposable paths after all writers are terminal; never use `--delete-destination-extra` as cleanup.

Before ending a turn while an authorized workflow needs to continue after transfer, use the [same-chat follow-up procedure](bridge-reliability.md#mandatory-follow-up-for-a-remote-wait). Save Globus identities, deadline, last observation, and permitted next step. No SSH session, new polling daemon, or global controller is required. Pause the follow-up at completion or user stop; preserve human timing and launch authority.

## Bounded live acceptance

Use fresh disposable roots outside scientific trials. Generate a roughly 128 MiB fixture, retain its SHA-256, upload, download, and compare hashes. For authorized interruption testing, first verify no unrelated transfers or shared-collection work would be affected. Bind GCP processes by executable, ancestry, and creation time before stopping only those processes; always restore GCP afterward.

Record partial progress before interruption, actual stop/restart times, a transfer error/offline observation, subsequent retry under the original task ID, and final checksum/hash results. If the file finishes before interruption, mark that recovery check untested and make at most one bounded repeat with earlier interruption or a modestly larger fixture. A process stop alone does not prove the data transfer was interrupted. Retain test-harness failures separately from service failures. After all tasks terminate, remove only test data and retain a compact receipt outside the removed roots.

## References

- [Globus CLI transfer options and checksum synchronization](https://docs.globus.org/cli/reference/transfer/)
- [CLI setup and authorization](https://docs.globus.org/cli/quickstart/)
- [GCP on Windows](https://docs.globus.org/globus-connect-personal/install/windows/)
