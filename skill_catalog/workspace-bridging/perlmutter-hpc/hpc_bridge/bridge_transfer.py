"""Bounded SFTP transfers with verified partial files and atomic publication."""
from __future__ import annotations

import errno
import hashlib
import os
import posixpath
import shlex
import stat
import time
from pathlib import Path

import paramiko

BLOCK = 32768
WINDOW = 4 * 1024 * 1024
PREFETCH_REQUESTS = 32


def local_digest(path, op, length=None):
    digest = hashlib.sha256()
    done = 0
    with Path(path).open("rb") as source:
        while length is None or done < length:
            op.check()
            data = source.read(min(WINDOW, length - done) if length is not None else WINDOW)
            if not data:
                break
            digest.update(data)
            done += len(data)
            op.update(activity=True, verified_bytes=done)
    if length is not None and done != length:
        raise ValueError("File ended during prefix verification")
    return digest.hexdigest()


def remote_digest(client, path, op, execute, length=None):
    op.update(force=True, phase="remote_checksum")
    if length is None:
        command = "sha256sum -- " + shlex.quote(path)
    else:
        pipeline = f"head -c {int(length)} -- {shlex.quote(path)} | sha256sum"
        command = "bash -o pipefail -c " + shlex.quote(pipeline)
    result = execute(client, command, max(.01, op.deadline - time.monotonic()), op)
    if result["status"] != "ok" or result["exit_status"] != 0:
        raise RuntimeError("Remote checksum failed: " + result.get("stderr", "")[:1000])
    digest = result["stdout"].strip().split()[0]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid remote SHA-256 result")
    return digest


def remote_file_stat(sftp, path):
    try:
        info = sftp.lstat(path)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected regular file, not a symlink/directory: {path}")
    return info


def local_file_stat(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected regular local file, not a symlink/directory: {path}")
    return info


def signature(info):
    return info.st_size, getattr(info, "st_mtime_ns", info.st_mtime)


def transfer(client, payload, op, execute):
    upload = payload["action"] == "upload"
    # Preserve a final-component symlink for lstat rejection rather than resolving it.
    local = Path(payload["local_path"]).expanduser().absolute()
    remote = payload["remote_path"]
    if not posixpath.isabs(remote) or "\x00" in remote:
        raise ValueError("Remote transfer path must be an absolute POSIX path")
    key = hashlib.sha256((str(op.root) + "\n" + str(local) + "\n" + remote).encode()).hexdigest()[:20]
    local_part = local.with_name(local.name + ".bridge-" + key + ".part")
    remote_part = remote + ".bridge-" + key + ".part"
    part = remote_part if upload else str(local_part)
    op.update(force=True, phase="opening_sftp", partial_path=part)
    channel = client.get_transport().open_session(timeout=min(30, max(.01, op.deadline-time.monotonic())))
    op.bind(channel)
    channel.settimeout(min(op.stall_timeout, 30))
    channel.invoke_subsystem("sftp")
    sftp = paramiko.SFTPClient(channel)
    try:
        before = local_file_stat(local) if upload else remote_file_stat(sftp, remote)
        if before is None:
            raise FileNotFoundError("Transfer source does not exist")
        total = before.st_size
        op.update(force=True, phase="source_checksum", bytes_total=total)
        expected = (local_digest(local, op) if upload else
                    remote_digest(client, remote, op, execute))
        destination = remote_file_stat(sftp, remote) if upload else local_file_stat(local)
        if destination is not None:
            op.update(force=True, phase="destination_checksum")
            existing = (remote_digest(client, remote, op, execute) if upload else
                        local_digest(local, op))
            if destination.st_size == total and existing == expected:
                after = local_file_stat(local) if upload else remote_file_stat(sftp, remote)
                if after is None or signature(after) != signature(before):
                    raise ValueError("Source changed while checking the existing destination")
                return success(total, expected, total, already_present=True)
            if not payload.get("overwrite", False):
                raise FileExistsError("Destination differs; use --overwrite only when replacement is intended")
        partial = remote_file_stat(sftp, remote_part) if upload else local_file_stat(local_part)
        offset = partial.st_size if partial is not None else 0
        if partial is not None and not payload.get("resume", False):
            raise FileExistsError(f"Partial retained at {part}; use --resume to verify and continue it")
        if offset > total:
            raise ValueError("Partial is larger than source; preserved for inspection")
        if offset:
            op.update(force=True, phase="prefix_checksum")
            left = local_digest(local if upload else local_part, op, offset)
            right = remote_digest(client, remote_part if upload else remote, op, execute, offset)
            if left != right:
                raise ValueError("Partial prefix differs from source; preserved, not resumed")
        op.update(force=True, phase="transferring", bytes_done=offset, resumed_bytes=offset)
        if upload:
            with local.open("rb") as source:
                source.seek(offset)
                done = offset
                # Closing each bounded window drains pipelined write acknowledgements.
                # It also bounds outstanding requests instead of buffering an entire file.
                while done < total or total == 0 and partial is None:
                    op.check()
                    mode = "r+b" if partial is not None or done > 0 else "x+b"
                    with sftp.open(remote_part, mode) as target:
                        target.seek(done)
                        target.set_pipelined(True)
                        end = min(done + WINDOW, total)
                        while done < end:
                            op.check()
                            data = source.read(min(BLOCK, end - done))
                            if not data:
                                raise ValueError("Local source changed during transfer")
                            target.write(data)
                            done += len(data)
                            op.update(activity=True, bytes_done=done)
                    partial = remote_file_stat(sftp, remote_part)
                    if partial is None or partial.st_size != done:
                        raise IOError("Remote partial size does not match acknowledged bytes")
                    if total == 0:
                        break
            if signature(local_file_stat(local)) != signature(before):
                raise ValueError("Local source changed; partial retained")
            actual = remote_digest(client, remote_part, op, execute)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            with sftp.open(remote, "rb") as source, local_part.open("r+b" if partial else "xb") as target:
                target.seek(offset)
                done = offset
                while done < total:
                    op.check()
                    end = min(done + WINDOW, total)
                    chunks = [(at, min(BLOCK, end - at)) for at in range(done, end, BLOCK)]
                    for (at, count), data in zip(chunks, source.readv(chunks, max_concurrent_prefetch_requests=PREFETCH_REQUESTS)):
                        op.check()
                        if len(data) != count or at != done:
                            raise IOError("Incomplete SFTP read; partial retained")
                        target.write(data)
                        done += len(data)
                        op.update(activity=True, bytes_done=done)
                    if done != end:
                        raise IOError("Incomplete SFTP read window")
                target.flush()
                os.fsync(target.fileno())
            if signature(remote_file_stat(sftp, remote)) != signature(before):
                raise ValueError("Remote source changed; partial retained")
            op.update(force=True, phase="destination_checksum")
            actual = local_digest(local_part, op)
        if actual != expected:
            raise ValueError("SHA-256 mismatch; partial retained and destination not published")
        op.check()
        op.update(force=True, phase="publishing", sha256=actual)
        if upload:
            if payload.get("overwrite", False):
                # Do not fall back to remove-then-rename: that loses atomic replacement.
                sftp.posix_rename(remote_part, remote)
            else:
                sftp.rename(remote_part, remote)
        else:
            if payload.get("overwrite", False):
                local_part.replace(local)
            else:
                # Hard-link publication is atomic and fails if destination appeared.
                os.link(local_part, local)
                local_part.unlink()
        return success(total, actual, offset)
    finally:
        sftp.close()


def success(size, digest, resumed, already_present=False):
    return {"status": "ok", "exit_status": 0, "stdout": "Transfer verified.\n", "stderr": "",
            "bytes": size, "sha256": digest, "resumed_bytes": resumed,
            "already_present": already_present, "verification": "sha256"}
