"""Purpose-scoped local queue for a verified Perlmutter SSH connection."""
from __future__ import annotations

import argparse
import base64
import contextlib
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('LOCALAPPDATA', Path.home()/'.local/share'))/'Codex'/'perlmutter-hpc'
HOST = 'perlmutter.nersc.gov'
# NERSC https://docs.nersc.gov/connect/#key-fingerprints, checked 2026-10-16.
HOST_SHA256 = 'SHA256:Db9s2Fa4J3qx7An5oIMgUqUAdK7UWJGTPGoIKD44+Gs'
CAPTURE_BYTES = 4*1024*1024


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def fingerprint(key):
    return 'SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')


def verify_host_key(key):
    observed = fingerprint(key)
    if observed != HOST_SHA256:
        raise RuntimeError(f'Host key mismatch: {observed}. Verify current NERSC documentation before updating the pin; credentials were not requested.')
    return observed


def verified_transport():
    import paramiko
    sock = socket.create_connection((HOST, 22), timeout=20)
    transport = paramiko.Transport(sock)
    try:
        # Match NERSC's published RSA key using SHA-2 signatures, never SHA-1.
        transport.get_security_options().key_types = ('rsa-sha2-512', 'rsa-sha2-256')
        transport.start_client(timeout=20)
        verify_host_key(transport.get_remote_server_key())
        transport.set_keepalive(60)
        return transport
    except BaseException:
        transport.close()
        raise


def authenticate(transport, identity):
    import paramiko
    if identity.get('key_file'):
        key_path = Path(identity['key_file']).expanduser()
        key = paramiko.PKey.from_path(key_path)
        cert = Path(str(key_path)+'-cert.pub')
        if not cert.is_file():
            raise ValueError('Expected an sshproxy certificate alongside the private key. Refresh sshproxy before retrying.')
        key.load_certificate(str(cert))
        transport.auth_publickey(identity['user'], key)
    else:
        if not sys.stdin.isatty():
            raise RuntimeError('MFA requires a visible local terminal. Use Start-Bridge.ps1; never pass credentials in chat, files or arguments.')
        print('Host fingerprint verified against NERSC documentation.', flush=True)
        print('Type your NERSC password immediately followed by the current OTP, with no spaces.', flush=True)
        # Collect once on the main console thread; never echo or persist it.
        secret = getpass.getpass('Password + OTP: ')
        try:
            # Paramiko falls back to keyboard-interactive only if the server
            # does not offer password authentication. It does not retry failure.
            transport.auth_password(identity['user'], secret, fallback=True)
        finally:
            secret = None
    if not transport.is_authenticated():
        raise RuntimeError('Authentication did not complete.')


def run_command(transport, command, timeout=120):
    if not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError('Timeout must be between 0 and 86400 seconds.')
    channel = transport.open_session(timeout=min(timeout,30))
    out, err = bytearray(), bytearray()
    out_size = err_size = 0
    state, error = 'completed', None
    deadline = time.monotonic()+timeout
    try:
        channel.settimeout(min(timeout,30))
        channel.exec_command('bash -l -c '+shlex.quote(command))
        channel.shutdown_write()
        while True:
            # Drain both streams fairly so a busy stream cannot hide timeout.
            if channel.recv_ready():
                chunk=channel.recv(65536); out_size+=len(chunk)
                out.extend(chunk[:max(0,CAPTURE_BYTES-len(out))])
            if channel.recv_stderr_ready():
                chunk=channel.recv_stderr(65536); err_size+=len(chunk)
                err.extend(chunk[:max(0,CAPTURE_BYTES-len(err))])
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                if channel.eof_received:
                    break
                if channel.closed:
                    raise EOFError('Remote channel closed before output EOF.')
            if time.monotonic()>=deadline:
                raise TimeoutError('Command timed out; remote completion is unknown. Do not resubmit automatically.')
            time.sleep(.01)
        code=channel.recv_exit_status()
        if code<0:
            raise EOFError('No remote exit status received.')
    except Exception as exc:
        state, code, error = 'uncertain', None, str(exc)
    finally:
        channel.close()
    return {'state':state, 'exit_status':code, 'stdout':out.decode('utf-8','replace'),
            'stderr':err.decode('utf-8','replace'), 'error':error,
            'stdout_bytes':out_size,'stderr_bytes':err_size,
            'truncated':out_size>len(out) or err_size>len(err)}


@contextlib.contextmanager
def session_lock(folder):
    with (folder/'server.lock').open('a+b') as lock:
        if os.name=='nt':
            import msvcrt
            lock.write(b'0');lock.flush();lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def load_session(folder):
    folder=Path(folder).expanduser().resolve()
    identity=read_json(folder/'identity.json')
    if identity.get('host')!=HOST or not re.fullmatch(r'[a-z_][a-z0-9_-]*',identity['user']):
        raise ValueError('Invalid Perlmutter session identity.')
    return folder,identity


def status(folder, identity, state, **extra):
    write_json(folder/'status.json', {'session_id':identity['session_id'],'target':identity['user']+'@'+HOST,
                                    'state':state,'updated_utc':now(),'pid':os.getpid(),**extra})


def transfer(transport, payload):
    import paramiko
    remote=payload['remote']
    if not PurePosixPath(remote).is_absolute() or '..' in PurePosixPath(remote).parts:
        raise ValueError('Use a verified absolute remote path with no parent traversal.')
    local=Path(payload['local']).expanduser().resolve()
    upload=payload['action']=='upload'
    overwrite=payload.get('overwrite',False)
    digest=hashlib.sha256();total=0
    deadline=time.monotonic()+payload['timeout']
    with paramiko.SFTPClient.from_transport(transport) as sftp:
        sftp.get_channel().settimeout(min(payload['timeout'],30))
        if upload:
            # x mode prevents an accidental overwrite, including a symlink.
            source=local.open('rb')
            try: destination=sftp.open(remote,'wb' if overwrite else 'wx')
            except BaseException: source.close();raise
        else:
            local.parent.mkdir(parents=True,exist_ok=True)
            source=sftp.open(remote,'rb')
            try: destination=local.open('wb' if overwrite else 'xb')
            except BaseException: source.close();raise
        with source,destination:
            while True:
                if time.monotonic()>deadline:raise TimeoutError('Transfer timed out; destination may be partial. No automatic retry.')
                block=source.read(1024*1024)
                if not block:break
                destination.write(block);digest.update(block);total+=len(block)
    return {'state':'completed','exit_status':0,'bytes':total,'stream_sha256':digest.hexdigest(),
            'local':str(local),'remote':remote,'verification':'stream hash; destination not independently rehashed'}


def serve(folder):
    folder,identity=load_session(folder)
    with session_lock(folder):
        # An interrupted request can have remote side effects. Never replay it.
        unresolved=[p.stem for p in (folder/'processing').glob('*.json') if not (folder/'results'/p.name).exists()]
        if unresolved:raise RuntimeError('Unresolved requests need reconciliation before reuse: '+', '.join(unresolved))
        transport=None
        try:
            status(folder,identity,'connecting')
            print(f"Perlmutter bridge | {identity['user']}@{HOST}\nPurpose: {identity['purpose']}\nSession: {folder}",flush=True)
            transport=verified_transport()
            status(folder,identity,'awaiting_authentication',host_fingerprint=HOST_SHA256)
            authenticate(transport,identity)
            check=run_command(transport,'id -un; hostname; pwd',30)
            if check['state']!='completed' or check['exit_status']!=0 or check['stdout'].splitlines()[0]!=identity['user']:
                raise RuntimeError('Authenticated identity check failed.')
            write_json(folder/'login.json',{**check,'observed_utc':now(),'host_fingerprint':HOST_SHA256})
            print('Connected. Credentials are not saved. Leave this window open during authorized work.',flush=True)
            last_work=time.monotonic();last_heartbeat=0
            while transport.is_active() and transport.is_authenticated():
                if time.monotonic()-last_work>identity['idle_seconds']:
                    status(folder,identity,'stopped',reason='idle timeout');return
                if time.monotonic()-last_heartbeat>5:
                    status(folder,identity,'ready',host_fingerprint=HOST_SHA256);last_heartbeat=time.monotonic()
                for pending in sorted((folder/'queue').glob('*.json')):
                    claimed=folder/'processing'/pending.name
                    if claimed.exists() or (folder/'results'/pending.name).exists():
                        raise RuntimeError('Duplicate request ID; refused to execute again.')
                    pending.rename(claimed)
                    request_id=claimed.stem;started=now();action=None
                    status(folder,identity,'executing',request_id=request_id)
                    try:
                        payload=read_json(claimed);action=payload['action']
                        if payload.get('session_id')!=identity['session_id']:raise ValueError('Request belongs to a different session.')
                        if action=='exec':result=run_command(transport,payload['command'],payload['timeout'])
                        elif action in ('upload','download'):result=transfer(transport,payload)
                        elif action=='stop':result={'state':'completed','exit_status':0}
                        else:raise ValueError('Unknown action.')
                    except Exception as exc:
                        result={'state':'uncertain','exit_status':None,'error':str(exc)}
                    write_json(folder/'results'/pending.name,{'id':request_id,'session_id':identity['session_id'],
                               'action':action,'started_utc':started,'finished_utc':now(),**result})
                    claimed.rename(folder/'processed'/pending.name)
                    last_work=time.monotonic();last_heartbeat=0
                    if action=='stop':status(folder,identity,'stopped',reason='requested');return
                time.sleep(.15)
            raise ConnectionError('SSH connection ended. Pending requests were not retried.')
        except BaseException as exc:
            status(folder,identity,'error',error=str(exc))
            raise
        finally:
            if transport:transport.close()


def initialize(args):
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*',args.user):raise ValueError('Invalid NERSC username.')
    session_id=uuid.uuid4().hex
    folder=RUNTIME/'sessions'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+session_id[:8])
    folder.mkdir(parents=True,mode=0o700)
    for name in ('queue','processing','processed','results'):(folder/name).mkdir(mode=0o700)
    identity={'session_id':session_id,'host':HOST,'user':args.user,'purpose':args.purpose,
              'project_root':str(Path(args.project_root).resolve()),'created_utc':now(),
              'idle_seconds':args.idle_seconds,'key_file':str(Path(args.key_file).expanduser().resolve()) if args.key_file else None}
    write_json(folder/'identity.json',identity)
    status(folder,identity,'created')
    return {'session':str(folder),'identity':identity}


def queue_request(folder, identity, payload):
    current=read_json(folder/'status.json')
    updated=datetime.fromisoformat(current['updated_utc'])
    age=(datetime.now(timezone.utc)-updated).total_seconds()
    if current['state']!='ready' or age>30:
        raise RuntimeError(f"Bridge is not ready: {current['state']} (status age {age:.0f}s). Inspect status and existing requests first.")
    if any((folder/'queue').glob('*.json')) or any((folder/'processing').glob('*.json')):
        raise RuntimeError('A request is already pending. Read its original result before submitting more work.')
    token=uuid.uuid4().hex
    write_json(folder/'queue'/(token+'.json'),{'id':token,'session_id':identity['session_id'],'created_utc':now(),**payload})
    return token


def result(folder, token, wait):
    if not re.fullmatch('[a-f0-9]{32}',token):raise ValueError('Invalid request ID.')
    deadline=time.monotonic()+wait
    while True:
        path=folder/'results'/(token+'.json')
        if path.is_file():return read_json(path)
        if time.monotonic()>=deadline:return {'id':token,'state':'pending','session':str(folder),'instruction':'Read this request using result; do not resubmit it.'}
        time.sleep(.2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('init');p.add_argument('--user',required=True);p.add_argument('--purpose',required=True);p.add_argument('--project-root',required=True)
    p.add_argument('--key-file');p.add_argument('--idle-seconds',type=int,default=1800)
    sub.add_parser('hostcheck')
    for name in ('serve','status','exec','probe','upload','download','result','stop'):
        p=sub.add_parser(name);p.add_argument('--session',required=True)
        if name in ('exec','probe','upload','download','stop'):
            p.add_argument('--wait',type=float,default=1);p.add_argument('--timeout',type=float,default=120)
        if name=='exec':
            g=p.add_mutually_exclusive_group(required=True);g.add_argument('--command');g.add_argument('--command-file')
        if name in ('upload','download'):
            p.add_argument('local');p.add_argument('remote');p.add_argument('--overwrite',action='store_true')
        if name=='result':p.add_argument('id');p.add_argument('--wait',type=float,default=0)
    args=parser.parse_args()
    if hasattr(args,'wait') and not 0<=args.wait<=60:parser.error('--wait must be from 0 to 60 seconds')
    if hasattr(args,'timeout') and (not math.isfinite(args.timeout) or not 0<args.timeout<=86400):parser.error('Invalid --timeout')
    if args.action=='init':value=initialize(args)
    elif args.action=='hostcheck':
        with verified_transport() as transport:value={'host':HOST,'fingerprint':fingerprint(transport.get_remote_server_key()),'verified':True,'authenticated':False}
    elif args.action=='serve':serve(args.session);return
    else:
        folder,identity=load_session(args.session)
        if args.action=='status':value={'identity':identity,'status':read_json(folder/'status.json'),'pending':[p.stem for d in ('queue','processing') for p in (folder/d).glob('*.json')]}
        elif args.action=='result':value=result(folder,args.id,args.wait)
        else:
            payload={'action':args.action,'timeout':args.timeout}
            if args.action in ('exec','probe'):
                payload['action']='exec'
                if args.action=='probe':payload['command']=(ROOT/'scripts'/'probe.sh').read_text(encoding='utf-8')
                elif args.command_file:payload['command']=Path(args.command_file).read_text(encoding='utf-8-sig')
                else:payload['command']=args.command
            elif args.action in ('upload','download'):payload.update(local=str(Path(args.local).expanduser().resolve()),remote=args.remote,overwrite=args.overwrite)
            token=queue_request(folder,identity,payload)
            value=result(folder,token,args.wait)
    print(json.dumps(value,indent=2,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'state':'error','error':str(exc)}),file=sys.stderr)
        sys.exit(1)
