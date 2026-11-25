# Globus transfers and recovery

Use Globus for large files or repeated SSH transfer failures. Continue commands and Slurm work through the bridge. Before changing transport, reconcile the original receipt and stop its writer. Filesystem outages can affect both transports; preserve errors and use a bounded deadline.

## Establish access and paths

- Reuse an authorized CLI when available. Check `globus version`, `globus whoami`, and a scoped `globus ls`. A Windows installation may be at `%LOCALAPPDATA%\GlobusCLI\venv\Scripts\globus.exe`; call its full path when needed. If installation is necessary, use an isolated environment and the official `globus-cli` package.
- CLI authorization, browser login, and collection consent are separate. Run `globus login` or the CLI's reported `globus session consent` command when needed, and let the user authenticate on the official page. Never read, copy, print, or commit OAuth tokens, passwords, OTPs, or authorization codes. Collection-management scopes are not needed for a transfer.
- Discover and record the current collection UUIDs. NERSC recommends **NERSC DTN** for transfers into and out of NERSC, with **NERSC Perlmutter** as the fallback when DTN is unavailable. Check the [NERSC collection list](https://docs.nersc.gov/services/globus/#availability); do not substitute the NERSC SHARE or HPSS collection for ordinary CFS/scratch transfers.
- For Kestrel-to-NERSC transfers, discover the source collection and verify its mapping under the [Kestrel Globus guidance](https://natlabrockies.github.io/HPC/Documentation/Managing_Data/Transferring_Files/globus/). Kestrel scratch and project collections can expose paths relative to their own roots rather than the SSH filesystem root. Keep both endpoint identities and exact collection-relative paths in the receipt.
- Inspect only approved roots with `globus ls <collection-id>:<approved-root> --format json` and `globus stat <collection-id>:<approved-parent> --format json`. NERSC CFS normally uses `/global/cfs/cdirs/<project>/`; scratch on DTNs is mounted under `/global/pscratch`, while Perlmutter uses `/pscratch`. Verify the intended mapping with a known file and size before writing.
- Globus Connect Personal (GCP) is required only for transfers involving the workstation. For those transfers, verify the local collection, allowed folders, running GCP instance, and workstation connectivity. Direct transfers between Kestrel and NERSC use their managed collections; the data does not pass through the workstation and no local GCP instance is required.
- Keep roots within the authorized project. Do not broaden GCP permissions, follow unexpected symlinks, or stage data at another destination without authorization. For many small files, a stable archive and a hash manifest can reduce metadata work. Check destination quota before a large transfer; scratch remains purgeable.

## Submit once and retain identity

Transfer closed, stable outputs. Record source sizes and available SHA-256 hashes, source/destination collection UUIDs, exact paths, intended overwrite behavior, and a UTC deadline in one small private receipt. Verify that replacing any existing destination is within scope. Check every CLI exit code before using its JSON output.

Generate and save a submission ID before submitting:

```text
globus task generate-submission-id --format json
```

Save the returned `value`. Then submit an individual file:

```text
globus transfer <source-collection>:<source-file> <destination-collection>:<destination-file> --no-recursive --verify-checksum --encrypt-data --submission-id <saved-id> --deadline <UTC-YYYY-MM-DDTHH:MM:SS> --label <short-private-label> --format json
```

The CLI deadline format above omits the trailing `Z`; supply an explicit UTC value. Retain the submission response and `task_id`. For a directory, review its manifest and scope before using `--recursive`; do not add destination deletion or recursive symlink following.

An accepted request is queued, not completed. If the submission response is lost, preserve the submission ID and exact arguments. Reconcile task history; if necessary, resubmit the same request with the same submission ID to recover its task identity. Never generate a fresh ID while the original outcome is uncertain.

## Observe and recover the original task

```text
globus task show <task-id> --format json
globus task event-list <task-id> --limit 20 --format json
```

- Bind observations to the saved collections, paths, label, and submission receipt. Retain compact progress and meaningful errors. Retry backoff can leave a task active without immediate byte growth. A failed query is an observation gap, not proof of completion or failure.
- An active Globus task owns its retry and recovery. Restore endpoint connectivity or consent as needed, then continue observing the same task. Do not start SFTP/SCP against its destination merely because progress pauses. Retain faults even if the task eventually succeeds.
- Require terminal `SUCCEEDED`, checksum verification enabled, no skipped-error files, and the intended file count/size before downstream work. Destination existence or final size alone is insufficient. Globus does not inherit the SSH bridge's atomic-publication guarantee. Compare retained source and destination hashes where available; require exact equality for a round-trip fixture.
- A failed or canceled task can leave files. Preserve its evidence and confirm no writer remains. A subsequent authorized task may use `--sync-level checksum --verify-checksum` against the same manifest. Retain both task IDs and their relationship; do not describe a new task as the original task resuming.
- A user stop applies to the exact owned task: `globus task cancel <task-id>`, followed by terminal reconciliation. Stopping a bridge or Slurm job does not cancel Globus. Delete only approved disposable paths after writers are terminal; do not use `--delete-destination-extra` as cleanup.

Preserve collection, submission and task IDs, exact paths, deadline and the last verified observation. Reconcile the original task through completion. Managed transfers continue independently of SSH.

## Validation and rollout

Keep offline bridge tests separate from Globus acceptance. A live test uses approved disposable roots, a bounded fixture and manifest, terminal task evidence, and source/destination checksum comparison. For a workstation round trip, upload and download a roughly 128 MiB fixture and compare SHA-256. For a Kestrel-to-NERSC check, use a stable, bounded source fixture and retain both endpoint observations.

Interruption testing requires its own authorized scope. Do not stop shared managed endpoints or unrelated GCP processes. If a transfer finishes before the intended interruption, record the recovery path as untested. Do not claim live acceptance from a documentation or offline-code check.

## References

- [NERSC Globus collections and usage](https://docs.nersc.gov/services/globus/)
- [NERSC data transfer nodes and mounts](https://docs.nersc.gov/systems/dtn/)
- [Globus CLI transfer options](https://docs.globus.org/cli/reference/transfer/)
- [Globus CLI setup and authorization](https://docs.globus.org/cli/quickstart/)
