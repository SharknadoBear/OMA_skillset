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


def connect_verified(identity):
    user, host = validate_target(str(identity.get('remote_target', '')))
    sock = socket.create_connection((host, 22), timeout=30)
    transport = None
    secret = None
    try:
        transport = paramiko.Transport(sock)
        transport.banner_timeout = 30
        transport.auth_timeout = 60
        transport.get_security_options().key_types = ('rsa-sha2-512', 'rsa-sha2-256')
        transport.start_client(timeout=30)
        verify_host_key(transport.get_remote_server_key())
        print('NERSC host fingerprint verified.', flush=True)
        if identity.get('key_file'):
            key_path = Path(identity['key_file']).expanduser()
            certificate = Path(str(key_path)+'-cert.pub')
            if not certificate.is_file():
                raise ValueError('Refresh sshproxy: expected adjacent -cert.pub certificate')
            key = paramiko.PKey.from_path(key_path)
            key.load_certificate(str(certificate))
            transport.auth_publickey(user, key)
        else:
            if not sys.stdin.isatty():
                raise RuntimeError('Authenticate in the visible start_bridge_window.ps1 terminal only')
            print('Type your NERSC password followed immediately by the current OTP.', flush=True)
            secret = getpass.getpass('Password+OTP: ')
            transport.auth_password(user, secret, fallback=True)
        if not transport.is_authenticated():
            raise paramiko.AuthenticationException('Authentication did not complete')
        transport.set_keepalive(60)
        client = paramiko.SSHClient()
        client._transport = transport
        print('Connected. Watching JSON command files.', flush=True)
        return client
    except BaseException:
        if transport is not None:
            transport.close()
        else:
            sock.close()
        raise
    finally:
        secret = None
