"""Failure-injection tests using real Paramiko over loopback, never Perlmutter.

All server files live in a TemporaryDirectory. No credentials are requested.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import paramiko

import bridge_runtime as runtime
import bridge_server as server
import bridge_transfer as transfers
import send_perlmutter_command as sender


class Handle(paramiko.SFTPHandle):
    def __init__(self, fixture, flags, stream):
        super().__init__(flags)
        self.fixture = fixture
        self.readfile = self.writefile = stream

    def read(self, offset, length):
        f = self.fixture
        if f.stall_reads:
            f.read_started.set()
            f.release_read.wait(4)
        if f.fail_reads_after is not None and offset >= f.fail_reads_after:
            return paramiko.SFTP_FAILURE
        return super().read(offset, length)

    def write(self, offset, data):
        f = self.fixture
        if f.fail_writes_after is not None and offset >= f.fail_writes_after:
            return paramiko.SFTP_FAILURE
        if f.corrupt_write and data:
            data = bytes([data[0] ^ 1]) + data[1:]
        return super().write(offset, data)


class SFTP(paramiko.SFTPServerInterface):
    def __init__(self, remote_server, *args, **kwargs):
        super().__init__(remote_server, *args, **kwargs)
        self.fixture = remote_server.fixture

    def path(self, value):
        return self.fixture.path(value)

    def lstat(self, path):
        try:
            return paramiko.SFTPAttributes.from_stat(self.path(path).lstat())
        except OSError as exc:
            return paramiko.SFTPServer.convert_errno(exc.errno)

    stat = lstat

    def open(self, path, flags, attr):
        try:
            fd = os.open(self.path(path), flags | getattr(os, "O_BINARY", 0), 0o600)
            mode = "r+b" if flags & (os.O_RDWR | os.O_WRONLY) else "rb"
            stream = os.fdopen(fd, mode, buffering=0)
            self.fixture.streams.append(stream)
            return Handle(self.fixture, flags, stream)
        except OSError as exc:
            return paramiko.SFTPServer.convert_errno(exc.errno)

    def rename(self, oldpath, newpath):
        try:
            dest = self.path(newpath)
            if dest.exists():
                return paramiko.SFTP_FAILURE
            self.path(oldpath).rename(dest)
            return paramiko.SFTP_OK
        except OSError as exc:
            return paramiko.SFTPServer.convert_errno(exc.errno)

    def posix_rename(self, oldpath, newpath):
        try:
            self.path(oldpath).replace(self.path(newpath))
            return paramiko.SFTP_OK
        except OSError as exc:
            return paramiko.SFTPServer.convert_errno(exc.errno)


class SSHServer(paramiko.ServerInterface):
    def __init__(self, fixture):
        self.fixture = fixture

    def check_auth_none(self, username):
        return paramiko.AUTH_SUCCESSFUL if username == "fixture" else paramiko.AUTH_FAILED

    def check_channel_request(self, kind, channel_id):
        return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_exec_request(self, channel, command):
        self.fixture.commands.append(command.decode())

        def respond():
            try:
                args = shlex.split(command.decode())
                if args[:2] == ["sha256sum", "--"]:
                    data = self.fixture.path(args[2]).read_bytes()
                    output = hashlib.sha256(data).hexdigest() + "  fixture\n"
                elif args[:4] == ["bash", "-o", "pipefail", "-c"]:
                    words = shlex.split(args[4])
                    if words[0:2] != ["head", "-c"] or words[-2:] != ["|", "sha256sum"]:
                        raise ValueError("Unexpected fixture checksum command")
                    data = self.fixture.path(words[4]).read_bytes()[:int(words[2])]
                    output = hashlib.sha256(data).hexdigest() + "  -\n"
                elif args == ["fixture", "status"]:
                    output = "scheduler reachable\n"
                elif args == ["fixture", "large"]:
                    output = "x" * (runtime.MAX_OUTPUT + 1024)
                elif args == ["fixture", "hang"]:
                    self.fixture.exec_started.set()
                    self.fixture.release_exec.wait(4)
                    output = "done\n"
                else:
                    raise ValueError("Only fixture commands are allowed")
                channel.sendall(output.encode())
                channel.send_exit_status(0)
                channel.shutdown_write()
                channel.close()
            except (EOFError, OSError):
                pass
        threading.Thread(target=respond, daemon=True).start()
        return True


class Fixture:
    def __init__(self, root):
        self.root = root
        self.fail_reads_after = self.fail_writes_after = None
        self.corrupt_write = self.stall_reads = False
        self.read_started = threading.Event()
        self.exec_started = threading.Event()
        self.release_read = threading.Event()
        self.release_exec = threading.Event()
        self.commands = []
        self.streams = []
        self.channels = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.client_transport = paramiko.Transport(socket.create_connection(self.sock.getsockname()))
        accepted, _ = self.sock.accept()
        self.server_transport = paramiko.Transport(accepted)
        self.server_transport.add_server_key(paramiko.RSAKey.generate(2048))
        self.server_transport.set_subsystem_handler("sftp", paramiko.SFTPServer, SFTP)
        self.server_transport.start_server(event=threading.Event(), server=SSHServer(self))
        self.client_transport.start_client(timeout=5)
        self.client_transport.auth_none("fixture")
        self.client = paramiko.SSHClient()
        self.client._transport = self.client_transport

    def path(self, value):
        candidate = (self.root / value.lstrip("/")).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ValueError("Fixture path escapes its temporary root")
        return candidate

    def close(self):
        self.release_read.set()
        self.release_exec.set()
        self.client.close()
        self.server_transport.close()
        self.sock.close()
        self.client_transport.join(2)
        self.server_transport.join(2)
        for stream in self.streams:
            stream.close()


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.fixture = Fixture(self.remote)
        self.data = bytes(range(256)) * 2049
        self.local = self.root / "source.nc"
        self.local.write_bytes(self.data)
        self.number = 0

    def tearDown(self):
        self.fixture.close()
        self.temp.cleanup()

    def payload(self, action="upload", **extra):
        self.number += 1
        return {"id": f"request_{self.number}", "action": action, "timeout": 10,
                "local_path": str(self.local), "remote_path": "/output.nc", **extra}

    def transfer(self, payload):
        op = runtime.Operation(self.root, payload)
        return transfers.transfer(self.fixture.client, payload, op, server.exec_command)

    def new_runtime(self):
        rt = runtime.BridgeRuntime(self.root / "session", server.dispatch,
                                   lambda: self.fixture.client, {"bridge_name": "fixture"})
        rt.client = self.fixture.client
        return rt

    def submit(self, rt, payload):
        runtime.atomic_json(rt.root / "commands" / (payload["id"] + ".json"), payload)

    def receipt(self, rt, request_id, timeout=5):
        end = time.monotonic() + timeout
        path = rt.root / "results" / (request_id + ".json")
        while not path.exists() and time.monotonic() < end:
            rt.tick()
            time.sleep(.01)
        self.assertTrue(path.exists(), f"Missing receipt: {request_id}")
        result = runtime.read_json(path)
        while any(op.id == request_id and t.is_alive() for op, t in rt.active.values()):
            time.sleep(.01)
        rt.tick()
        return result

    def test_upload_download_roundtrip_and_existing_destination(self):
        result = self.transfer(self.payload())
        self.assertEqual(result["sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertEqual((self.remote / "output.nc").read_bytes(), self.data)
        dest = self.root / "download.nc"
        result = self.transfer(self.payload("download", local_path=str(dest)))
        self.assertEqual(dest.read_bytes(), self.data)
        self.assertEqual(result["verification"], "sha256")
        self.assertTrue(self.transfer(self.payload())["already_present"])

    def test_empty_file_roundtrip(self):
        self.local.write_bytes(b"")
        self.assertEqual(self.transfer(self.payload())["bytes"], 0)
        dest = self.root / "empty.nc"
        self.transfer(self.payload("download", local_path=str(dest)))
        self.assertEqual(dest.read_bytes(), b"")

    def test_multiple_transfer_windows_roundtrip(self):
        data = bytes(range(256)) * (transfers.WINDOW // 256 * 2 + 1)
        self.local.write_bytes(data)
        self.transfer(self.payload())
        dest = self.root / "multiple.nc"
        self.transfer(self.payload("download", local_path=str(dest)))
        self.assertEqual(dest.read_bytes(), data)

    def test_interrupted_upload_resumes_verified_prefix(self):
        self.fixture.fail_writes_after = 65536
        with self.assertRaises(OSError):
            self.transfer(self.payload())
        self.assertFalse((self.remote / "output.nc").exists())
        part = next(self.remote.glob("*.part"))
        offset = part.stat().st_size
        self.assertGreater(offset, 0)
        self.fixture.fail_writes_after = None
        result = self.transfer(self.payload(resume=True))
        self.assertEqual(result["resumed_bytes"], offset)
        self.assertEqual((self.remote / "output.nc").read_bytes(), self.data)

    def test_interrupted_download_resumes_verified_prefix(self):
        (self.remote / "output.nc").write_bytes(self.data)
        dest = self.root / "resumed.nc"
        self.fixture.fail_reads_after = 65536
        with self.assertRaises(OSError):
            self.transfer(self.payload("download", local_path=str(dest)))
        self.assertFalse(dest.exists())
        self.fixture.fail_reads_after = None
        result = self.transfer(self.payload("download", local_path=str(dest), resume=True))
        self.assertGreater(result["resumed_bytes"], 0)
        self.assertEqual(dest.read_bytes(), self.data)

    def test_bad_partial_prefix_is_preserved_and_rejected(self):
        self.fixture.fail_writes_after = 65536
        with self.assertRaises(OSError):
            self.transfer(self.payload())
        part = next(self.remote.glob("*.part"))
        part.write_bytes(b"invalid prefix")
        self.fixture.fail_writes_after = None
        with self.assertRaisesRegex(ValueError, "prefix differs"):
            self.transfer(self.payload(resume=True))
        self.assertEqual(part.read_bytes(), b"invalid prefix")
        self.assertFalse((self.remote / "output.nc").exists())

    def test_hash_failure_never_replaces_existing_destination(self):
        dest = self.remote / "output.nc"
        dest.write_bytes(b"original")
        self.fixture.corrupt_write = True
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.transfer(self.payload(overwrite=True))
        self.assertEqual(dest.read_bytes(), b"original")

    def test_overwrite_requires_flag(self):
        dest = self.remote / "output.nc"
        dest.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.transfer(self.payload())
        self.transfer(self.payload(overwrite=True))
        self.assertEqual(dest.read_bytes(), self.data)

    def test_stalled_transfer_does_not_block_command_or_cancel(self):
        (self.remote / "output.nc").write_bytes(self.data)
        self.fixture.stall_reads = True
        rt = self.new_runtime()
        p = self.payload("download", local_path=str(self.root / "stalled.nc"))
        self.submit(rt, p)
        rt.tick()
        self.assertTrue(self.fixture.read_started.wait(2))
        status = self.payload("exec", command="fixture status")
        self.submit(rt, status)
        started = time.monotonic()
        result = self.receipt(rt, status["id"])
        self.assertIn("scheduler reachable", result["stdout"])
        cancel = self.payload("cancel", request_id=p["id"])
        self.submit(rt, cancel)
        self.assertEqual(self.receipt(rt, cancel["id"])["status"], "ok")
        result = self.receipt(rt, p["id"])
        self.assertEqual(result["reason"], "cancel_requested")
        self.assertLess(time.monotonic() - started, 2)

    def test_stall_watchdog_closes_transfer_channel(self):
        (self.remote / "output.nc").write_bytes(self.data)
        self.fixture.stall_reads = True
        rt = self.new_runtime()
        p = self.payload("download", local_path=str(self.root / "timeout.nc"), stall_timeout=.2)
        self.submit(rt, p)
        result = self.receipt(rt, p["id"])
        self.assertEqual(result["status"], "interrupted")
        self.assertFalse((self.root / "timeout.nc").exists())

    def test_disconnect_is_uncertain_and_never_replays_exec(self):
        rt = self.new_runtime()
        p = self.payload("exec", command="fixture hang")
        self.submit(rt, p)
        rt.tick()
        self.assertTrue(self.fixture.exec_started.wait(2))
        self.fixture.client.close()
        result = self.receipt(rt, p["id"])
        self.assertEqual(result["status"], "uncertain")
        self.submit(rt, p)
        rt.tick()
        self.assertEqual(self.fixture.commands.count("fixture hang"), 1)

    def test_output_capture_is_bounded_and_not_reported_complete(self):
        result = server.exec_command(self.fixture.client, "fixture large", 5)
        self.assertEqual(result["status"], "output_limit")
        self.assertFalse(result["capture_complete"])
        self.assertEqual(len(result["stdout"]), runtime.MAX_OUTPUT)

    def test_restart_marks_claim_uncertain_without_dispatch(self):
        rt = self.new_runtime()
        p = self.payload("exec", command="fixture status")
        runtime.atomic_json(rt.root / "inflight" / (p["id"] + ".json"), p)
        rt.recover()
        self.assertEqual(runtime.read_json(rt.root / "results" / (p["id"] + ".json"))["status"], "uncertain")
        self.assertEqual(self.fixture.commands, [])

    def test_queued_cancellation_and_expiry_never_execute(self):
        rt = self.new_runtime()
        p = self.payload("exec", command="fixture status")
        self.submit(rt, p)
        cancel = self.payload("cancel", request_id=p["id"])
        self.submit(rt, cancel)
        self.assertEqual(self.receipt(rt, p["id"])["status"], "cancelled_before_start")
        expired = self.payload("exec", command="fixture status", expires_at=time.time()-1)
        self.submit(rt, expired)
        self.assertEqual(self.receipt(rt, expired["id"])["status"], "expired_before_start")
        self.assertEqual(self.fixture.commands, [])

    def test_storage_gate_preserves_receipts_and_allows_stop(self):
        rt = self.new_runtime()
        runtime.atomic_json(rt.root / "results" / "old.json", {"evidence": "preserve"})
        self.submit(rt, self.payload("exec", command="fixture status"))
        with patch.object(runtime, "MAX_LEDGER_BYTES", 1):
            rt.tick()
        self.assertFalse(rt.storage_ok)
        self.assertEqual(self.fixture.commands, [])
        stop = self.payload("stop")
        self.submit(rt, stop)
        self.receipt(rt, stop["id"])
        self.assertTrue(rt.stopping)
        self.assertTrue((rt.root / "results" / "old.json").exists())

    def test_authentication_failure_keeps_status_responsive_without_retry_loop(self):
        calls = []
        def fail():
            calls.append(1)
            raise paramiko.AuthenticationException("fixture expired authentication")
        rt = runtime.BridgeRuntime(self.root / "session", server.dispatch, fail, {})
        rt.authenticate()
        rt.auth_thread.join(1)
        for _ in range(5):
            rt.tick()
        health = runtime.read_json(rt.root / "bridge_status.json")
        self.assertEqual(health["state"], "authentication_required")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("fixture expired", str(health))

    def test_caller_wait_expiry_keeps_original_request(self):
        rt = self.new_runtime()
        with patch.object(sender, "COMMANDS", rt.root / "commands"), patch.object(sender, "RESULTS", rt.root / "results"):
            result_path = sender.write_command({"action": "exec", "command": "fixture status"})
            with self.assertRaises(TimeoutError):
                sender.wait_result(result_path, .01)
        self.assertEqual(len(list((rt.root / "commands").glob("*.json"))), 1)
        self.assertEqual(self.receipt(rt, result_path.stem)["status"], "ok")
        self.assertEqual(self.fixture.commands.count("fixture status"), 1)

    def test_session_lock_refuses_second_server(self):
        with runtime.SessionLock(self.root):
            with self.assertRaises(RuntimeError):
                with runtime.SessionLock(self.root):
                    pass

    def test_heartbeat_staleness_is_not_hidden(self):
        rt = self.new_runtime()
        runtime.atomic_json(rt.root / "bridge_status.json", {
            "state": "connected", "heartbeat_at": "2000-01-01T00:00:00+00:00"})
        with patch.object(sender, "ROOT", rt.root):
            self.assertTrue(sender.snapshot()["bridge"]["heartbeat_stale"])
            rt.tick()
            self.assertFalse(sender.snapshot()["bridge"]["heartbeat_stale"])

    def test_explicit_reconnect_after_auth_failure_does_not_replay(self):
        calls = []
        def connect():
            calls.append(1)
            if len(calls) == 1:
                raise paramiko.AuthenticationException("fixture")
            return self.fixture.client
        rt = runtime.BridgeRuntime(self.root / "session", server.dispatch, connect, {})
        rt.authenticate()
        rt.auth_thread.join(1)
        self.assertFalse(rt.connected())
        request = self.payload("reconnect")
        self.submit(rt, request)
        self.receipt(rt, request["id"])
        rt.auth_thread.join(1)
        self.assertTrue(rt.connected())
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.fixture.commands, [])

    def test_cancel_running_exec_reports_uncertain_remote_effects(self):
        rt = self.new_runtime()
        request = self.payload("exec", command="fixture hang")
        self.submit(rt, request)
        rt.tick()
        self.assertTrue(self.fixture.exec_started.wait(2))
        self.submit(rt, self.payload("cancel", request_id=request["id"]))
        result = self.receipt(rt, request["id"])
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["request"]["command"], "fixture hang")

    def test_run_stops_with_final_health_receipt(self):
        rt = runtime.BridgeRuntime(self.root / "session", server.dispatch,
                                   lambda: self.fixture.client, {})
        worker = threading.Thread(target=rt.run)
        worker.start()
        deadline = time.monotonic() + 2
        while not rt.connected() and time.monotonic() < deadline:
            time.sleep(.01)
        self.submit(rt, self.payload("stop"))
        worker.join(3)
        self.assertFalse(worker.is_alive())
        result = runtime.read_json(rt.root / "bridge_status.json")
        self.assertEqual(result["state"], "stopped")
        self.assertEqual(result["unresolved_requests"], [])
        self.assertFalse(result["remote_jobs_cancelled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
