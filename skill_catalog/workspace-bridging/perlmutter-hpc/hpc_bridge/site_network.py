"""Bound the NERSC multi-address pre-authentication connection wait."""
from __future__ import annotations
import errno
import queue
import selectors
import socket
import threading
import time


def resolve_endpoints(host, port, timeout=5):
    result = queue.Queue(maxsize=1)
    def resolve():
        try:
            value = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except Exception as error:
            value = error
        result.put(value)
    threading.Thread(target=resolve, daemon=True).start()
    try:
        value = result.get(timeout=timeout)
    except queue.Empty as error:
        raise TimeoutError('DNS lookup exceeded its connection deadline') from error
    if isinstance(value, Exception):
        raise value
    endpoints = []
    for family, kind, proto, _, address in value:
        entry = (family, kind, proto, address)
        if entry not in endpoints:
            endpoints.append(entry)
    if not endpoints:
        raise OSError('DNS returned no TCP endpoints')
    return endpoints


def connect_tcp(host, port=22, timeout=20, address_timeout=3):
    """Race at most four TCP endpoints; no SSH or authentication is attempted."""
    if timeout <= 0 or address_timeout <= 0:
        raise ValueError('Connection deadlines must be positive')
    deadline = time.monotonic() + timeout
    endpoints = iter(resolve_endpoints(host, port, min(5, timeout)))
    active = {}
    exhausted = False
    pending = {errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, 10035, 10036, 10037}
    with selectors.DefaultSelector() as selector:
        def discard(sock):
            selector.unregister(sock)
            active.pop(sock)
            sock.close()
        try:
            while time.monotonic() < deadline:
                while not exhausted and len(active) < 4:
                    try:
                        family, kind, proto, address = next(endpoints)
                    except StopIteration:
                        exhausted = True
                        break
                    sock = socket.socket(family, kind, proto)
                    try:
                        sock.setblocking(False)
                        code = sock.connect_ex(address)
                        if code in (0, errno.EISCONN):
                            sock.settimeout(30)
                            return sock
                        if code not in pending:
                            sock.close()
                            continue
                        selector.register(sock, selectors.EVENT_WRITE)
                        active[sock] = time.monotonic() + address_timeout
                    except BaseException:
                        sock.close()
                        raise
                if not active:
                    break
                now = time.monotonic()
                wait = max(0, min(deadline, min(active.values())) - now)
                for key, _ in selector.select(wait):
                    sock = key.fileobj
                    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0:
                        selector.unregister(sock)
                        active.pop(sock)
                        sock.settimeout(30)
                        return sock
                    discard(sock)
                now = time.monotonic()
                for sock, expires in list(active.items()):
                    if now >= expires:
                        discard(sock)
        finally:
            for sock in list(active):
                discard(sock)
    raise TimeoutError(f'No NERSC TCP connection within the {timeout:g}-second deadline; no credentials sent')
