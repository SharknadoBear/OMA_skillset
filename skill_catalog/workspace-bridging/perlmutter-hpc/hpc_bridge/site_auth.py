"""NERSC endpoint policy; queue, transfer and session mechanics are shared."""
from __future__ import annotations
import base64
import getpass
import hashlib
from pathlib import Path
import re
import socket
import sys
import paramiko
from site_network import connect_tcp

HOST = 'perlmutter.nersc.gov'
# Independently verified against https://docs.nersc.gov/connect/ on 2026-10-20.
HOST_SHA256 = 'SHA256:Db9s2Fa4J3qx7An5oIMgUqUAdK7UWJGTPGoIKD44+Gs'


def validate_target(target):
    if target.count('@') != 1:
        raise ValueError('Expected NERSC username@perlmutter.nersc.gov')
    user, host = target.split('@', 1)
    if host != HOST or not re.fullmatch(r'[a-z_][a-z0-9_-]*', user):
        raise ValueError('Use the approved NERSC username and perlmutter.nersc.gov endpoint')
    return user, host


def verify_host_key(key):
    observed = 'SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
    if observed != HOST_SHA256:
        raise RuntimeError('NERSC host key mismatch; verify current official documentation before changing the pin')
    return observed


def open_verified_transport(host):
    """Verify a fresh transport without collecting or sending credentials."""
    print('Resolving NERSC and opening TCP port 22 (20-second total limit)...', flush=True)
    sock = connect_tcp(host)
    transport = None
    try:
        transport = paramiko.Transport(sock)
        transport.banner_timeout = 30
        transport.auth_timeout = 60
        transport.get_security_options().key_types = ('rsa-sha2-512', 'rsa-sha2-256')
        print('Verifying SSH handshake and NERSC host key...', flush=True)
        transport.start_client(timeout=30)
        verify_host_key(transport.get_remote_server_key())
        return transport
    except BaseException:
        if transport is not None:
            transport.close()
        else:
            sock.close()
        raise


def connect_verified(identity):
    user, host = validate_target(str(identity.get('remote_target', '')))
    transport = None
    secret = None
    phase = 'host verification'
    try:
        if identity.get('key_file'):
            key_path = Path(identity['key_file']).expanduser()
            certificate = Path(str(key_path)+'-cert.pub')
            if not certificate.is_file():
                raise ValueError('Refresh sshproxy: expected adjacent -cert.pub certificate')
            key = paramiko.PKey.from_path(key_path)
            key.load_certificate(str(certificate))
            transport = open_verified_transport(host)
            phase = 'certificate authentication'
            transport.auth_publickey(user, key)
        else:
            if not sys.stdin.isatty():
                raise RuntimeError('Authenticate in the visible start_bridge_window.ps1 terminal only')
            # Human entry can outlast the server's unauthenticated SSH grace
            # period. Close the preflight, then verify a fresh connection after
            # input; never submit credentials over the waiting connection.
            preflight = open_verified_transport(host)
            preflight.close()
            print('NERSC host fingerprint verified. Ready for terminal input.', flush=True)
            print('Type your NERSC password followed immediately by the current OTP.', flush=True)
            phase = 'terminal input'
            secret = getpass.getpass('Password+OTP: ')
            phase = 'fresh host verification'
            print('Connecting and verifying a fresh SSH session...', flush=True)
            transport = open_verified_transport(host)
            phase = 'password/OTP authentication'
            print('Authenticating with NERSC (up to 60 seconds)...', flush=True)
            transport.auth_password(user, secret, fallback=True)
        if not transport.is_authenticated():
            raise paramiko.AuthenticationException('Authentication did not complete')
        transport.set_keepalive(60)
        client = paramiko.SSHClient()
        client._transport = transport
        print('Connected. Watching JSON command files.', flush=True)
        return client
    except BaseException as error:
        if transport is not None:
            transport.close()
        # Do not print exception text: the authentication callback may include
        # server-provided content. Phase and class safely explain a failed wait.
        print(f'Connection failed during {phase} ({type(error).__name__}).', flush=True)
        if isinstance(error, paramiko.AuthenticationException):
            print('NERSC did not accept authentication. Check the account and current OTP before requesting reconnect.', flush=True)
        else:
            print('No authenticated session remains. Inspect bridge status and request reconnect when ready.', flush=True)
        raise
    finally:
        secret = None
