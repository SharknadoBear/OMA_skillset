"""Offline NERSC authentication and session-interface regressions."""
import argparse
import hashlib
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import paramiko
import site_auth as site
import make_bridge_session as sessions
import send_perlmutter_command as sender


class SiteTests(unittest.TestCase):
    def setup_transport(self):
        sock = Mock()
        transport = Mock()
        transport.is_authenticated.return_value = True
        return sock, transport

    def test_target_validation(self):
        self.assertEqual(site.validate_target('example@perlmutter.nersc.gov')[0], 'example')
        for target in ('example@kestrel.nlr.gov', 'example@perlmutter.nersc.gov.evil', 'bad name@perlmutter.nersc.gov', 'root@localhost'):
            with self.assertRaises(ValueError):
                sessions.normalize_remote_target(target)

    def test_mismatch_precedes_credentials_and_authentication(self):
        sock, t = self.setup_transport()
        t.get_remote_server_key.return_value.asbytes.return_value = b'wrong host'
        with patch.object(site,'connect_tcp',return_value=sock), patch.object(site.paramiko,'Transport',return_value=t), patch.object(site.sys.stdin,'isatty',return_value=True), patch.object(site.getpass,'getpass') as prompt:
            with self.assertRaises(RuntimeError):
                site.connect_verified({'remote_target':'example@perlmutter.nersc.gov'})
            prompt.assert_not_called()
            t.auth_password.assert_not_called()
            t.close.assert_called_once()

    def test_verified_mfa_once_and_keepalive(self):
        sock, t = self.setup_transport()
        events = []
        with patch.object(site,'connect_tcp',return_value=sock), patch.object(site.paramiko,'Transport',return_value=t), patch.object(site,'verify_host_key',side_effect=lambda key:events.append('verified')), patch.object(site.sys.stdin,'isatty',return_value=True), patch.object(site.getpass,'getpass',side_effect=lambda text:events.append('prompt') or 'synthetic-test-only') as prompt:
            client = site.connect_verified({'remote_target':'example@perlmutter.nersc.gov'})
        self.assertEqual(events, ['verified','prompt','verified'])
        prompt.assert_called_once()
        t.auth_password.assert_called_once_with('example','synthetic-test-only',fallback=True)
        t.set_keepalive.assert_called_once_with(60)
        self.assertIs(client.get_transport(),t)

    def test_auth_failure_closes_without_retry(self):
        sock,t = self.setup_transport()
        t.auth_password.side_effect = paramiko.AuthenticationException('synthetic rejection')
        with patch.object(site,'connect_tcp',return_value=sock), patch.object(site.paramiko,'Transport',return_value=t), patch.object(site,'verify_host_key'), patch.object(site.sys.stdin,'isatty',return_value=True), patch.object(site.getpass,'getpass',return_value='synthetic-test-only') as prompt:
            with self.assertRaises(paramiko.AuthenticationException):
                site.connect_verified({'remote_target':'example@perlmutter.nersc.gov'})
        prompt.assert_called_once()
        self.assertEqual(t.close.call_count,2)
        t.auth_password.assert_called_once()

    def test_noninteractive_refused(self):
        sock,t = self.setup_transport()
        with patch.object(site,'connect_tcp',return_value=sock) as network, patch.object(site.paramiko,'Transport',return_value=t), patch.object(site,'verify_host_key'), patch.object(site.sys.stdin,'isatty',return_value=False), patch.object(site.getpass,'getpass') as prompt:
            with self.assertRaises(RuntimeError):
                site.connect_verified({'remote_target':'example@perlmutter.nersc.gov'})
            prompt.assert_not_called()
            network.assert_not_called()

    def test_human_wait_cannot_expire_authentication_transport(self):
        preflight=Mock()
        fresh=Mock()
        fresh.is_authenticated.return_value=True
        def prompt(_):
            preflight.close.assert_called_once()
            preflight.is_active.return_value=False
            # The human may wait indefinitely; there is no live SSH transport.
            self.assertEqual(opening.call_count,1)
            return 'synthetic-test-only'
        with patch.object(site,'open_verified_transport',side_effect=[preflight,fresh]) as opening, patch.object(site.sys.stdin,'isatty',return_value=True), patch.object(site.getpass,'getpass',side_effect=prompt):
            client=site.connect_verified({'remote_target':'example@perlmutter.nersc.gov'})
        preflight.auth_password.assert_not_called()
        fresh.auth_password.assert_called_once_with('example','synthetic-test-only',fallback=True)
        self.assertIs(client.get_transport(),fresh)

    def test_second_host_verification_failure_sends_no_credentials(self):
        preflight=Mock()
        with patch.object(site,'open_verified_transport',side_effect=[preflight,RuntimeError('synthetic pin mismatch')]), patch.object(site.sys.stdin,'isatty',return_value=True), patch.object(site.getpass,'getpass',return_value='synthetic-test-only'):
            with self.assertRaises(RuntimeError):
                site.connect_verified({'remote_target':'example@perlmutter.nersc.gov'})
        preflight.auth_password.assert_not_called()

    def test_authentication_error_is_visible_without_exception_text(self):
        preflight,fresh=Mock(),Mock()
        fresh.auth_password.side_effect=paramiko.SSHException('synthetic-sensitive-error-content')
        log=io.StringIO()
        with patch.object(site,'open_verified_transport',side_effect=[preflight,fresh]), patch.object(site.sys.stdin,'isatty',return_value=True), patch.object(site.getpass,'getpass',return_value='synthetic-test-only'),redirect_stdout(log):
            with self.assertRaises(paramiko.SSHException):
                site.connect_verified({'remote_target':'example@perlmutter.nersc.gov'})
        self.assertIn('Connection failed during password/OTP authentication (SSHException)',log.getvalue())
        self.assertNotIn('synthetic-sensitive-error-content',log.getvalue())
        self.assertNotIn('synthetic-test-only',log.getvalue())

    def test_certificate_failure_has_no_password_fallback(self):
        sock,t = self.setup_transport()
        with tempfile.TemporaryDirectory() as d:
            keypath=Path(d)/'key'
            keypath.write_text('synthetic')
            Path(str(keypath)+'-cert.pub').write_text('synthetic certificate')
            t.auth_publickey.side_effect=paramiko.AuthenticationException('expired test certificate')
            with patch.object(site,'connect_tcp',return_value=sock), patch.object(site.paramiko,'Transport',return_value=t), patch.object(site,'verify_host_key'), patch.object(site.paramiko.PKey,'from_path',return_value=Mock()), patch.object(site.getpass,'getpass') as prompt:
                with self.assertRaises(paramiko.AuthenticationException):
                    site.connect_verified({'remote_target':'example@perlmutter.nersc.gov','key_file':str(keypath)})
                prompt.assert_not_called()
                t.auth_password.assert_not_called()

    def test_session_copies_runtime_without_private_state(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'session with spaces'
            dest.mkdir()
            sessions.copy_helper_files(dest)
            for name in ('site_auth.py','bridge_runtime.py','bridge_transfer.py','send_perlmutter_command.py','start_bridge_window.ps1'):
                self.assertTrue((dest/name).is_file())
            for name in sessions.EXCLUDED_NAMES | sessions.EXCLUDED_FILES:
                self.assertFalse((dest/name).exists())
            args=argparse.Namespace(purpose='fixture',project_root=d,work_summary='offline',remote_target='example@perlmutter.nersc.gov',key_file=None)
            identity=sessions.write_identity(dest/'bridge_identity.json',args,'Fixture',sessions.datetime.now())
            self.assertEqual(identity['remote_target'],args.remote_target)
            with patch.object(sender,'IDENTITY',dest/'bridge_identity.json'):
                sender.validate_identity(sender.parse_args(['--bridge-name','Fixture','status']))
                with self.assertRaises(SystemExit):
                    sender.validate_identity(sender.parse_args(['--bridge-name','Wrong','status']))


if __name__=='__main__':
    unittest.main(verbosity=2)
