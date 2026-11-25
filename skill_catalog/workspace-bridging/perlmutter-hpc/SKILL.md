---
name: perlmutter-hpc
description: Access NERSC Perlmutter using native Windows OpenSSH and a daily sshproxy certificate. Verify quotas and modules, transfer files with SHA-256, submit Slurm jobs and resume monitoring using the original job receipt.
---

# Perlmutter HPC

Use `scripts/perlmutter.py` from the installed skill for new Perlmutter work. It invokes Windows OpenSSH directly. Keep usernames, allocations, evidence and work directories in the private project. Resolve the NERSC user from approved context or the private account record in `%LOCALAPPDATA%/Codex/perlmutter-hpc/account.local.json`. Use the allocation the user selected for this run.

## Daily authentication

Run `py -3.13 scripts/perlmutter.py --user USER status`. It inspects the principal and local validity window of the public certificate next to the private key, normally `~/.ssh/nersc-cert.pub`. A valid credential works across chats until expiry. For a missing or expired credential, run `renew` once and ask the user to enter MFA in the official visible NERSC SSHProxy window. Check status again after completion. Keep credentials in that window. On a wrong principal or a user-selected custom key, verify the intended identity before renewal. [NERSC Windows SSHProxy](https://docs.nersc.gov/connect/mfa/#sshproxy-on-windows-systems)

Each connection verifies the published NERSC RSA host key and uses only certificate authentication in batch mode. On Windows, let OpenSSH find the adjacent certificate from the signing key given with `-i`. Supplying the certificate as an independent identity can cause OpenSSH to try to sign with the public file. Existing named Paramiko sessions retain their original records and can finish independently.

## Commands and transfers

Run `perlmutter.py --user USER [--timeout SECONDS] ACTION ...` with the Python interpreter. The timeout applies to each network operation. Every command opens its own shell.

```text
exec -- hostname -f
exec --script local_inventory.sh
upload local_input.tgz /absolute/remote/input.tgz
download /absolute/remote/results.tgz local_results.tgz
submit --account ACCOUNT --script /absolute/remote/job.sbatch --sha256 HASH --receipt local_job.json
monitor --receipt local_job.json
```

An exec command returns the remote exit status and separates standard output and error. For shell expressions or a sequence of module commands, put Bash in a local script and use `exec --script`. Verify identity, actual scratch paths, available quota and allocation before staging a new study. Use login nodes for lightweight inspection, and compute allocations for compiles and model tests. [Environment and storage](references/nersc.md)

SFTP stages uploads into unique sibling partial files, verifies SHA-256 and publishes a closed file. A matching existing destination is reusable. Different existing files are protected. Downloads verify remote and local SHA-256 before publication. After a transfer interruption, reconcile the destination and the original process first. For bulk data or SFTP failure, use Globus with the approved collections, transfer checksum verification and task receipts. [Globus transfers](references/globus-transfers.md)

## Slurm and continuity

Inspect the exact job file, account, resource request, and output locations. Save the local receipt before submission. Use `submit` with the same receipt to recover the original remote job ID after a lost reply. A remote claim with uncertain accounting requires reconciliation by the recorded Slurm comment and work directory before any retry. Monitor an accepted ID with `monitor`. Certificate expiry affects new SSH connections. Renew locally and continue with the same receipt. Collect terminal accounting, logs and scientific outputs for validation.

Run `scripts/test_perlmutter.py` for offline transport and receipt regressions. Verify live acceptance with the authorized inventory, a checksum round trip and original-job monitoring.
