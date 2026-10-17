"""Local behavioral checks; no live credentials or remote changes."""
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bridge


class Channel:
    def __init__(self, frames, exit_status=0, stalled=False):
        self.frames=frames;self.n=0;self.code=exit_status;self.stalled=stalled
        self.closed=False;self.eof_received=False;self.input_closed=False
    def settimeout(self,*args):pass
    def exec_command(self,command):self.command=command
    def shutdown_write(self):self.input_closed=True
    def recv_ready(self):return self.n<len(self.frames) and bool(self.frames[self.n][0])
    def recv(self,n):
        out=self.frames[self.n][0];self.frames[self.n][0]=b'';return out
    def recv_stderr_ready(self):return self.n<len(self.frames) and bool(self.frames[self.n][1])
    def recv_stderr(self,n):
        out=self.frames[self.n][1];self.frames[self.n][1]=b'';return out
    def exit_status_ready(self):
        # Simulate exit status arriving before a second data frame and EOF.
        if not self.stalled:
            self.n+=1
            self.eof_received=self.n>=len(self.frames)
        return True
    def recv_exit_status(self):return self.code
    def close(self):self.closed=True


class Transport:
    def __init__(self,channel):self.channel=channel
    def open_session(self,**kwargs):return self.channel


class BridgeTests(unittest.TestCase):
    def test_complete_capture_waits_for_eof_and_preserves_failure(self):
        channel=Channel([[b'first\n',b'warning\n'],[b'last\n',b'']],exit_status=7)
        result=bridge.run_command(Transport(channel),'printf "x"',1)
        self.assertEqual(result['stdout'],'first\nlast\n')
        self.assertEqual(result['stderr'],'warning\n')
        self.assertEqual(result['exit_status'],7)
        self.assertEqual(result['state'],'completed')
        self.assertTrue(channel.input_closed and channel.closed)

    def test_timeout_does_not_claim_remote_cancellation(self):
        channel=Channel([[b'partial',b'']],stalled=True)
        result=bridge.run_command(Transport(channel),'sleep 10',.025)
        self.assertEqual(result['state'],'uncertain')
        self.assertIsNone(result['exit_status'])
        self.assertEqual(result['stdout'],'partial')
        self.assertIn('unknown',result['error'])

    def test_capture_limit_is_disclosed(self):
        channel=Channel([[b'abcdefghij',b'12345678']])
        with patch.object(bridge,'CAPTURE_BYTES',5):
            result=bridge.run_command(Transport(channel),'echo test',1)
        self.assertEqual(result['stdout'],'abcde')
        self.assertTrue(result['truncated'])
        self.assertEqual(result['stdout_bytes'],10)

    def test_wrong_host_key_is_rejected(self):
        key=type('Key',(),{'asbytes':lambda self:b'not the NERSC key'})()
        with self.assertRaisesRegex(RuntimeError,'Host key mismatch'):
            bridge.verify_host_key(key)

    def test_matching_host_key_uses_sha256(self):
        key=type('Key',(),{'asbytes':lambda self:b'fixture public key'})()
        pin=bridge.fingerprint(key)
        with patch.object(bridge,'HOST_SHA256',pin):
            self.assertEqual(bridge.verify_host_key(key),pin)

    def session(self,root):
        with patch.object(bridge,'RUNTIME',Path(root)):
            record=bridge.initialize(argparse.Namespace(user='testuser',purpose='local test',project_root=root,idle_seconds=100,key_file=None))
        folder=Path(record['session']);identity=record['identity']
        bridge.status(folder,identity,'ready')
        return folder,identity

    def test_queue_refuses_duplicate_work_and_result_recovers_original(self):
        with tempfile.TemporaryDirectory() as root:
            folder,identity=self.session(root)
            token=bridge.queue_request(folder,identity,{'action':'exec','command':'true','timeout':5})
            self.assertEqual(bridge.result(folder,token,0)['state'],'pending')
            with self.assertRaisesRegex(RuntimeError,'already pending'):
                bridge.queue_request(folder,identity,{'action':'exec'})
            bridge.write_json(folder/'results'/(token+'.json'),{'state':'completed','exit_status':0,'id':token})
            self.assertEqual(bridge.result(folder,token,0)['id'],token)
            self.assertEqual(len(list((folder/'queue').glob('*.json'))),1)

    def test_restart_refuses_unresolved_claim_before_connecting(self):
        with tempfile.TemporaryDirectory() as root:
            folder,identity=self.session(root)
            bridge.write_json(folder/'processing'/('a'*32+'.json'),{'action':'exec'})
            with patch.object(bridge,'verified_transport') as connect:
                with self.assertRaisesRegex(RuntimeError,'Unresolved requests'):
                    bridge.serve(folder)
                connect.assert_not_called()

    def test_second_server_cannot_hold_same_lock(self):
        with tempfile.TemporaryDirectory() as root:
            with bridge.session_lock(Path(root)):
                with self.assertRaises(OSError):
                    with bridge.session_lock(Path(root)):pass

    def test_changed_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            folder,identity=self.session(root)
            identity['host']='different.example'
            bridge.write_json(folder/'identity.json',identity)
            with self.assertRaises(ValueError):bridge.load_session(folder)

    def test_stale_bridge_does_not_accept_commands(self):
        with tempfile.TemporaryDirectory() as root:
            folder,identity=self.session(root)
            state=bridge.read_json(folder/'status.json');state['updated_utc']='2000-01-01T00:00:00+00:00'
            bridge.write_json(folder/'status.json',state)
            with self.assertRaisesRegex(RuntimeError,'not ready'):
                bridge.queue_request(folder,identity,{'action':'exec'})

    def test_sftp_download_refuses_local_overwrite_and_hashes_stream(self):
        class Sftp:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def get_channel(self):return self
            def settimeout(self,*args):pass
            def open(self,*args):return io.BytesIO(b'remote fixture')
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'download'
            payload={'action':'download','remote':'/tmp/fixture','local':str(path),'timeout':5}
            with patch('paramiko.SFTPClient.from_transport',return_value=Sftp()):
                result=bridge.transfer(None,payload)
                self.assertEqual(path.read_bytes(),b'remote fixture')
                self.assertEqual(result['stream_sha256'],bridge.hashlib.sha256(b'remote fixture').hexdigest())
                with self.assertRaises(FileExistsError):bridge.transfer(None,payload)

    def test_nonabsolute_transfer_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'absolute remote path'):
            bridge.transfer(None,{'action':'download','remote':'$HOME/file','local':'fixture','timeout':5})


if __name__=='__main__':unittest.main(verbosity=2)
