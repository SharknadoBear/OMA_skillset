"""Portable offline regression tests for bridge_server.exec_command output capture.

Usage:
    python test_bridge_output_completion.py --bridge-server /path/to/bridge_server.py
    python test_bridge_output_completion.py  # uses adjacent bridge_server.py

The supplied module is imported without calling main/connect. Fake clients and
clocks drive exec_command; no credentials, live SSH transport or network is used.
If Paramiko is absent, an import-only annotation stub permits the fake-channel
tests. Only the direct Paramiko semantics test is skipped in that environment.
"""
from __future__ import annotations

import argparse
import copy
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import types
import unittest
from unittest.mock import patch

try:
    import paramiko as ACTUAL_PARAMIKO
except ModuleNotFoundError as error:
    if error.name != "paramiko":
        raise
    ACTUAL_PARAMIKO = None

BRIDGE = None


def no_live_io(*args, **kwargs):
    raise AssertionError("Credentials and live network I/O are forbidden in this suite")


def import_bridge(path: Path):
    """Import the selected source; never run its command-line entry point."""
    path = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("bridge_output_test_target", path)
    if spec is None or spec.loader is None:
        raise ValueError("Unable to load supplied bridge module")
    module = importlib.util.module_from_spec(spec)
    with patch.object(getpass, "getpass", no_live_io), patch.object(socket, "create_connection", no_live_io):
        if ACTUAL_PARAMIKO is None:
            stub = types.ModuleType("paramiko")
            stub.SSHClient = object
            with patch.dict(sys.modules, {"paramiko": stub}):
                spec.loader.exec_module(module)
        else:
            spec.loader.exec_module(module)
    if not callable(getattr(module, "exec_command", None)):
        raise ValueError("Supplied module does not define exec_command")
    return module


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if seconds <= 0:
            raise AssertionError("Polling sleep must advance fake time")
        self.now += seconds


class FakeChannel:
    """A deterministic channel where exit status, data and EOF are separate events."""

    def __init__(self, clock, events, exit_status=0):
        self.clock = clock
        self.events = copy.deepcopy(events)
        self.out_buffer = bytearray()
        self.err_buffer = bytearray()
        self.eof_received = False
        self.closed = False
        self.status_received = False
        self.exit_status = exit_status
        self.closed_by_test_target = False
        self.exit_status_reads = 0

    def advance(self):
        while self.events and self.events[0][0] <= self.clock.now + 1e-9:
            _, kind, value = self.events.pop(0)
            if kind == "stdout":
                self.out_buffer.extend(value)
            elif kind == "stderr":
                self.err_buffer.extend(value)
            elif kind == "status":
                self.status_received = True
            elif kind == "eof":
                self.eof_received = True
            elif kind == "close":
                self.closed = True
            else:
                raise AssertionError("Unknown fixture event: " + kind)

    def recv_ready(self):
        self.advance()
        return bool(self.out_buffer)

    def recv_stderr_ready(self):
        self.advance()
        return bool(self.err_buffer)

    def recv(self, nbytes):
        self.advance()
        value = bytes(self.out_buffer[:nbytes])
        del self.out_buffer[:nbytes]
        return value

    def recv_stderr(self, nbytes):
        self.advance()
        value = bytes(self.err_buffer[:nbytes])
        del self.err_buffer[:nbytes]
        return value

    def exit_status_ready(self):
        self.advance()
        return self.closed or self.status_received

    def recv_exit_status(self):
        self.exit_status_reads += 1
        if not self.exit_status_ready():
            raise AssertionError("Blocking exit-status read attempted")
        return self.exit_status

    def close(self):
        self.closed = True
        self.closed_by_test_target = True


class FakeStdin:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, channel):
        self.channel = channel
        self.stdin = FakeStdin()
        self.calls = []

    def exec_command(self, command, timeout):
        self.calls.append((command, timeout))
        if command != "offline fixture":
            raise AssertionError("Unexpected command")
        return self.stdin, types.SimpleNamespace(channel=self.channel), object()


class OutputCompletionTests(unittest.TestCase):
    def run_capture(self, events, exit_status=0, timeout=0.3):
        clock = FakeClock()
        channel = FakeChannel(clock, events, exit_status)
        client = FakeClient(channel)
        with patch.object(BRIDGE, "time", clock), patch.object(getpass, "getpass", no_live_io), patch.object(socket, "create_connection", no_live_io):
            result = BRIDGE.exec_command(client, "offline fixture", timeout)
        self.assertTrue(client.stdin.closed)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(channel.exit_status_reads, 1)
        return result, channel, clock

    def assert_capture(self, events, stdout, stderr="", exit_status=0):
        result, channel, clock = self.run_capture(events, exit_status)
        self.assertEqual(result, {"status": "ok", "exit_status": exit_status, "stdout": stdout, "stderr": stderr})
        self.assertTrue(channel.eof_received, "Returned before stream EOF")
        self.assertFalse(channel.out_buffer)
        self.assertFalse(channel.err_buffer)
        return channel, clock

    def test_late_stdout_after_exit_status(self):
        self.assert_capture([(0, "stdout", b"prefix"), (0, "status", None), (.05, "stdout", b"-suffix"), (.1, "eof", None)], "prefix-suffix")

    def test_late_stderr_after_exit_status(self):
        self.assert_capture([(0, "stdout", b"body"), (0, "status", None), (.05, "stderr", b"fatal detail"), (.1, "eof", None)], "body", "fatal detail")

    def test_status_before_any_output(self):
        self.assert_capture([(0, "status", None), (.05, "stdout", b"body"), (.1, "eof", None)], "body")

    def test_utf8_character_spans_delivery_gap(self):
        self.assert_capture([(0, "stdout", b"caf\xc3"), (0, "status", None), (.05, "stdout", b"\xa9"), (.1, "eof", None)], "caf\u00e9")

    def test_normal_buffered_output(self):
        self.assert_capture([(0, "stdout", b"all"), (0, "stderr", b"note"), (0, "status", None), (0, "eof", None)], "all", "note")

    def test_eof_before_exit_status(self):
        _, clock = self.assert_capture([(0, "stdout", b"all"), (0, "eof", None), (.1, "status", None)], "all")
        self.assertGreaterEqual(clock.now, .1)

    def test_nonzero_exit_preserves_late_diagnostics(self):
        self.assert_capture([(0, "stdout", b"p"), (0, "status", None), (.05, "stderr", b"failure"), (.1, "eof", None)], "p", "failure", 9)

    def test_empty_output_still_waits_for_eof(self):
        _, clock = self.assert_capture([(0, "status", None), (.1, "eof", None)], "")
        self.assertGreaterEqual(clock.now, .1)

    def test_missing_eof_times_out_and_closes_channel(self):
        clock = FakeClock()
        channel = FakeChannel(clock, [(0, "stdout", b"prefix"), (0, "status", None)])
        client = FakeClient(channel)
        with patch.object(BRIDGE, "time", clock), self.assertRaises(TimeoutError):
            BRIDGE.exec_command(client, "offline fixture", .2)
        self.assertTrue(channel.closed_by_test_target)
        self.assertEqual(channel.exit_status_reads, 0)
        self.assertEqual(len(client.calls), 1)

    def test_closed_without_eof_is_incomplete(self):
        with self.assertRaises(EOFError):
            self.run_capture([(0, "stdout", b"prefix"), (0, "status", None), (.05, "close", None)])

    def test_invalid_timeout_rejected_before_client_call(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout):
                client = FakeClient(FakeChannel(FakeClock(), []))
                with self.assertRaises(ValueError):
                    BRIDGE.exec_command(client, "offline fixture", timeout)
                self.assertEqual(client.calls, [])

    @unittest.skipIf(ACTUAL_PARAMIKO is None, "Paramiko is not installed; fake-channel regressions still run")
    def test_actual_paramiko_status_is_distinct_from_eof(self):
        # No SSHClient, socket or transport thread is created.
        channel = ACTUAL_PARAMIKO.Channel(0)
        channel.transport = types.SimpleNamespace(server_object=None)
        message = ACTUAL_PARAMIKO.Message()
        message.add_string("exit-status")
        message.add_boolean(False)
        message.add_int(0)
        channel._handle_request(ACTUAL_PARAMIKO.Message(message.asbytes()))
        self.assertTrue(channel.exit_status_ready())
        self.assertFalse(channel.recv_ready())
        self.assertFalse(channel.eof_received)
        channel._feed(b"late-data")
        self.assertTrue(channel.recv_ready())
        self.assertFalse(channel.eof_received)
        channel._handle_eof(None)
        self.assertTrue(channel.eof_received)
        self.assertEqual(channel.recv(1024), b"late-data")
        self.assertFalse(channel.recv_ready())


def main():
    global BRIDGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-server", type=Path, default=Path(__file__).resolve().with_name("bridge_server.py"))
    parser.add_argument("--json-output", type=Path, help="Optional new result file; existing files are never replaced")
    args = parser.parse_args()
    source = args.bridge_server.resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    BRIDGE = import_bridge(source)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OutputCompletionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise AssertionError("Supplied source changed during tests")
    report = {"status": "passed" if result.wasSuccessful() else "failed", "bridge_source_sha256": digest,
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
              "paramiko_version": getattr(ACTUAL_PARAMIKO, "__version__", None), "live_network_calls": 0,
              "source_modified": False, "case_lineage_permission": False}
    if args.json_output is not None:
        with args.json_output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
