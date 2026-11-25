"""Preserve a Slurm submission across SSH disconnects and certificate renewal."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import uuid


def write(path, record, *, create=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as out:
        tmp = Path(out.name)
        json.dump(record, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    try:
        if create:
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


SUBMIT = """set -eu
script=$1
account=$2
expected=$3
claim=.nersc-submit-$4
cd -- "$5"
shift 5
if mkdir -m 700 -- "$claim" 2>/dev/null; then
    test "$(sha256sum -- "$script" | cut -d' ' -f1)" = "$expected"
    sbatch --parsable --account="$account" --comment="$claim" "$script" "$@" > "$claim/submitted.tmp"
    mv -- "$claim/submitted.tmp" "$claim/job.id"
fi
if ! test -s "$claim/job.id"; then
    echo "Original submission unresolved; reconcile the job comment $claim in Slurm." >&2
    exit 3
fi
cat -- "$claim/job.id"
"""


def call(args, ssh, remote_path, host):
    path = args.receipt
    if args.action == "submit":
        script = remote_path(args.script)
        if not re.fullmatch(r"[a-f0-9]{64}", args.sha256) or not re.fullmatch(r"[\w.-]+", args.account):
            raise ValueError("Invalid expected script SHA-256 or Slurm account")
        workdir = remote_path(args.workdir) if args.workdir else script.rsplit("/", 1)[0] or "/"
        identity = dict(host=host, user=args.user, script=script, script_sha256=args.sha256,
                        account=args.account, workdir=workdir, arguments=args.argument)
        if not path.exists():
            write(path, dict(**identity, submission=uuid.uuid4().hex), create=True)
        record = json.loads(path.read_bytes())
        if any(record.get(k) != v for k, v in identity.items()):
            raise ValueError("This receipt belongs to a different submission")
        if "job_id" not in record:
            cmd = shlex.join(["bash", "-s", "--", script, args.account, args.sha256,
                              record["submission"], workdir, *args.argument])
            out = subprocess.run([*ssh, cmd], input=SUBMIT.encode(), stdout=subprocess.PIPE,
                                 timeout=args.timeout, check=True).stdout.decode().strip()
            if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", out):
                raise ValueError("Unrecognized sbatch response; reconcile original submission")
            record["job_id"] = out.split(";")[0]
            write(path, record)
        print(json.dumps(record))
        return 0
    record = json.loads(path.read_bytes())
    if record.get("host") != host or record.get("user") != args.user or not re.fullmatch(r"\d+", record.get("job_id", "")):
        raise ValueError("No matching accepted Slurm job in this receipt")
    command = shlex.join(["sacct", "-X", "-nP", "-j", record["job_id"],
                          "-o", "JobIDRaw,State,ExitCode,Elapsed,Account,AllocCPUS,NodeList"])
    return subprocess.run([*ssh, command], timeout=args.timeout, stdin=subprocess.DEVNULL).returncode
