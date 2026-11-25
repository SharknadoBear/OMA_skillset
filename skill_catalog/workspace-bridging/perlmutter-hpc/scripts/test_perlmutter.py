"""Offline evidence for certificate routing, transfer failure and Slurm continuity."""
import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import subprocess

import perlmutter as p
import slurm_receipt


class NativeTransport(unittest.TestCase):
    cert = ("Type: ssh-rsa-cert-v01@openssh.com user certificate\n"
            "Valid: from 2026-11-01T01:00:00 to 2026-11-02T01:00:00\n"
            "        Principals:\n                tester\n        Critical Options: (none)\n")

    def test_validity_principal_and_expiry(self):
        for date, name, state in [(1,"tester","valid"), (2,"tester","expired"), (1,"other","wrong_principal")]:
            self.assertEqual(state, p.certificate_info(self.cert,name,dt.datetime(2026,11,date,3))["status"])

    def test_strict_certificate_pair(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(p,"check",return_value={"status":"valid"}), \
                patch.object(p,"pinned_line",return_value="pinned key"), patch("subprocess.run") as invoked:
            invoked.return_value.stdout = "known public host record"
            opts=p.options(Namespace(key=Path(tmp)/"nersc",user="tester"),tmp)
            self.assertIn("StrictHostKeyChecking=yes",opts)
            self.assertIn("BatchMode=yes",opts)
            self.assertFalse(any("CertificateFile=" in o for o in opts))
            alg=next(o for o in opts if o.startswith("PubkeyAcceptedAlgorithms="))
            self.assertTrue(all(a.endswith("-cert-v01@openssh.com") for a in alg.split("=")[1].split(",")))
            self.assertEqual(opts[opts.index("-i")+1], str(Path(tmp)/"nersc"))
        with self.assertRaises(ValueError):
            p.pinned_line("perlmutter.nersc.gov ssh-rsa ZmFrZQ==")

    def test_expired_then_monitor_same_job(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args=Namespace(receipt=Path(tmp)/"job.json",user="tester",action="submit",
                           script="/work/job.sbatch",account="project",sha256="f"*64,timeout=20,
                           workdir="/work",argument=[])
            with patch("subprocess.run",return_value=subprocess.CompletedProcess([],0,stdout=b"419;perlmutter\n")) as runner:
                p.operate(args,[])
                p.operate(args,[])
                self.assertEqual(runner.call_count,1)
            args.action="monitor"
            with patch.object(p,"check",return_value={"status":"expired"}), self.assertRaises(ValueError):
                p.options(args,tmp)
            with patch("subprocess.run",return_value=subprocess.CompletedProcess([],0)) as runner:
                self.assertEqual(p.operate(args,[]),0)
                self.assertIn("-j 419",runner.call_args.args[0][-1])
                self.assertNotIn("sbatch",runner.call_args.args[0][-1])
            self.assertEqual(json.loads(args.receipt.read_bytes())["job_id"],"419")

    def test_uncertain_submission_reuses_claim(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args=Namespace(receipt=Path(tmp)/"r.json",user="tester",action="submit",script="/r/j",
                           account="project",sha256="0"*64,timeout=20,workdir=None,argument=[])
            with patch("subprocess.run",side_effect=subprocess.TimeoutExpired("ssh",20)), self.assertRaises(subprocess.TimeoutExpired):
                p.operate(args,[])
            saved=json.loads(args.receipt.read_bytes())["submission"]
            with patch("subprocess.run",return_value=subprocess.CompletedProcess([],0,stdout=b"555\n")) as runner:
                p.operate(args,[])
                self.assertIn(saved,runner.call_args.args[0][-1])
                self.assertIn(b'if mkdir -m 700',runner.call_args.kwargs["input"])

    def test_download_rejects_bad_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=Namespace(action="download",user="tester",remote="/scratch/a",local=Path(tmp)/"result",timeout=10)
            def simulate(argv, **kwargs):
                if argv[0]=="ssh":
                    return subprocess.CompletedProcess(argv,0,stdout="0"*64+"  /scratch/a\n")
                import re
                name=re.search(r'get "/scratch/a" "(.*)"',kwargs["input"].decode())[1]
                Path(name).write_bytes(b"wrong")
            with patch.object(p,"local",side_effect=simulate),self.assertRaises(ValueError):
                p.operate(args,[])
            self.assertFalse(args.local.exists())


if __name__ == "__main__":
    unittest.main()
