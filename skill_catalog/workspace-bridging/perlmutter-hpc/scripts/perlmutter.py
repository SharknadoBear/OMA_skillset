"""Use a NERSC sshproxy certificate with native OpenSSH. Requires Python 3.10+."""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import uuid

import slurm_receipt

HOST = "perlmutter.nersc.gov"
RSA_PIN = "SHA256:Db9s2Fa4J3qx7An5oIMgUqUAdK7UWJGTPGoIKD44+Gs"


def local(argv, **options):
    return subprocess.run(argv, check=True, **options)


def certificate_info(description, user, now):
    principals = re.search(r"^\s*Principals:\s*\n((?:[ \t]+[^:\n]+\n)+)", description, re.M)
    valid = re.search(r"Valid: from ([\dT:-]+) to ([\dT:-]+)", description)
    if not valid or not principals or " user certificate" not in description:
        raise ValueError("Expected a time-limited OpenSSH user certificate")
    start, end = map(dt.datetime.fromisoformat, valid.groups())
    owners = [line.strip() for line in principals[1].splitlines()]
    if user not in owners:
        state = "wrong_principal"
    elif now < start:
        state = "not_yet_valid"
    else:
        state = "valid" if now < end else "expired"
    return dict(status=state, user=user, valid_from_local=start.isoformat(),
                expires_local=end.isoformat(), remaining_seconds=max(0, int((end-now).total_seconds())))


def check(args):
    cert = Path(str(args.key) + "-cert.pub")
    if not cert.is_file() or not args.key.is_file():
        return dict(status="missing", user=args.user)
    result = local(["ssh-keygen", "-L", "-f", str(cert)], capture_output=True, text=True)
    return certificate_info(result.stdout, args.user, dt.datetime.now())


def renew(args):
    state = check(args)
    if state["status"] == "valid":
        print(json.dumps(state))
        return
    if state["status"] in {"wrong_principal", "not_yet_valid"}:
        raise ValueError(f"Certificate status {state['status']}. Verify the account and local clock before renewing.")
    if args.key != Path.home() / ".ssh/nersc":
        raise ValueError("Renew the selected custom identity in its owning sshproxy workflow")
    if os.name != "nt":
        raise ValueError("Use sshproxy -u USER in your interactive terminal")
    proc = subprocess.Popen(["sshproxy.exe", "-u", args.user],
                            creationflags=subprocess.CREATE_NEW_CONSOLE)
    print(json.dumps(dict(status="renewal_started", pid=proc.pid, prior=state)))


def pinned_line(lines):
    for line in lines.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1] != "ssh-rsa":
            continue
        digest = hashlib.sha256(base64.b64decode(parts[2], validate=True)).digest()
        if "SHA256:" + base64.b64encode(digest).decode().rstrip("=") == RSA_PIN:
            return f"{HOST} ssh-rsa {parts[2]}\n"
    raise ValueError("NERSC's published RSA host key was not found; check the official fingerprint")


def options(args, directory):
    state = check(args)
    if state["status"] != "valid":
        raise ValueError(f"Certificate status {state['status']}. Use the renew command in a local terminal.")
    known = Path.home() / ".ssh/known_hosts"
    lines = subprocess.run(["ssh-keygen", "-F", HOST, "-f", str(known)],
                           capture_output=True, text=True).stdout if known.exists() else ""
    if not lines.strip():
        lines = local(["ssh-keyscan", "-T", "8", "-t", "rsa", HOST],
                      capture_output=True, text=True, timeout=25).stdout
    hosts = Path(directory) / "known_hosts"
    hosts.write_text(pinned_line(lines), encoding="ascii")
    config = ["BatchMode=yes", "IdentitiesOnly=yes", "IdentityAgent=none",
              "StrictHostKeyChecking=yes", "GlobalKnownHostsFile=none",
              f'UserKnownHostsFile="{hosts.as_posix()}"', "UpdateHostKeys=no",
              # Let OpenSSH associate the adjacent -cert.pub with the signing key.
              "ForwardAgent=no",
              "PreferredAuthentications=publickey", "PasswordAuthentication=no",
              "KbdInteractiveAuthentication=no",
              "PubkeyAcceptedAlgorithms=rsa-sha2-512-cert-v01@openssh.com,rsa-sha2-256-cert-v01@openssh.com,ssh-ed25519-cert-v01@openssh.com",
              "HostKeyAlgorithms=rsa-sha2-512,rsa-sha2-256", "ConnectTimeout=15",
              "ServerAliveInterval=60", "ServerAliveCountMax=3", "ClearAllForwardings=yes"]
    return ["-F", "none", "-i", str(args.key), *[s for opt in config for s in ("-o", opt)]]


def remote_path(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) == "/" or any(ord(c) < 32 for c in value):
        raise ValueError("Specify an absolute remote file path without control characters or traversal")
    return str(path)


def sftp_quote(value):
    if any(ord(c) < 32 for c in value):
        raise ValueError("Control character in SFTP path")
    escaped = "".join("\\" + c if c in '\\"*?[]' else c for c in value)
    return '"' + escaped + '"'


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def operate(args, opts):
    target = args.user + "@" + HOST
    ssh = ["ssh", "-T", *opts, target]
    if args.action in ("monitor", "submit"):
        return slurm_receipt.call(args, ssh, remote_path, HOST)
    if args.action == "exec":
        if not args.command and not args.script:
            raise ValueError("Supply --script FILE or a command with arguments")
        if args.script:
            if b"\0" in (body := args.script.read_bytes()):
                raise ValueError("Expected a shell script")
            proc = subprocess.run([*ssh, shlex.join(["bash", "-l", "-s", "--", *args.command])],
                                  input=body.replace(b"\r\n", b"\n"), timeout=args.timeout)
        else:
            proc = subprocess.run([*ssh, shlex.join(args.command)], stdin=subprocess.DEVNULL, timeout=args.timeout)
        return proc.returncode
    remote = remote_path(args.remote)
    path = args.local.resolve()
    def remote_sha():
        return local([*ssh, shlex.join(["sha256sum", "--", remote])],
                     capture_output=True, text=True, timeout=args.timeout).stdout.split()[0]
    def transfer(verb, src, dst):
        local(["sftp", "-q", "-b", "-", *opts, target],
              input=f"{verb} {sftp_quote(src)} {sftp_quote(dst)}\n".encode(),
              stdout=subprocess.DEVNULL, timeout=args.timeout)
    if args.action == "upload":
        expected = digest(path)
        partial = remote + ".partial-" + uuid.uuid4().hex
        transfer("put", path.as_posix(), partial)
        # Publish without clobber, accepting an already identical destination on retry.
        script = """set -eu
src=$1
dst=$2
expected=$3
trap 'rm -f -- "$src"' EXIT
verify() { test "$(sha256sum -- "$1" | cut -d' ' -f1)" = "$expected"; }
verify "$src"
if ! ln -T -- "$src" "$dst" 2>/dev/null; then
    test -f "$dst" && test ! -L "$dst" && verify "$dst" || exit 1
fi
"""
        local([*ssh, shlex.join(["bash", "-s", "--", partial, remote, expected])],
              input=script.encode(), timeout=args.timeout)
    else:
        expected = remote_sha()
        if path.exists():
            if digest(path) != expected:
                raise ValueError("A different local download destination already exists")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=path.parent) as temp:
                partial = Path(temp) / "download"
                transfer("get", remote, partial.as_posix())
                if digest(partial) != expected:
                    raise ValueError("Downloaded SHA-256 differs from remote")
                os.link(partial, path)
    print(json.dumps(dict(action=args.action, sha256=expected, bytes=path.stat().st_size, remote=remote)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh/nersc")
    parser.add_argument("--timeout", type=int, default=300)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("status")
    commands.add_parser("renew")
    submit = commands.add_parser("submit")
    submit.add_argument("--account", required=True)
    submit.add_argument("--script", required=True)
    submit.add_argument("--workdir", help="Remote Slurm submission directory (defaults to script parent)")
    submit.add_argument("--argument", action="append", default=[], help="Argument for the batch script")
    submit.add_argument("--sha256", required=True)
    submit.add_argument("--receipt", required=True, type=Path)
    commands.add_parser("monitor").add_argument("--receipt", required=True, type=Path)
    run = commands.add_parser("exec")
    run.add_argument("--script", type=Path, help="Local Bash file delivered on standard input")
    run.add_argument("command", nargs=argparse.REMAINDER)
    for verb in ("upload", "download"):
        op = commands.add_parser(verb)
        if verb == "upload":
            op.add_argument("local", type=Path)
            op.add_argument("remote")
        else:
            op.add_argument("remote")
            op.add_argument("local", type=Path)
    args = parser.parse_args()
    args.key = args.key.expanduser().resolve()
    if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*", args.user):
        parser.error("Invalid NERSC username")
    if args.action == "status":
        state = check(args)
        print(json.dumps(state))
        return 0 if state["status"] == "valid" else 2
    if args.action == "renew":
        renew(args)
        return 0
    if args.action == "exec" and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    with tempfile.TemporaryDirectory() as directory:
        return operate(args, options(args, directory))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
