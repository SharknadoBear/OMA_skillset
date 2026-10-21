"""Bounded DNS/TCP tests; real connections are loopback only."""
import errno
import socket
import threading
import unittest
from unittest.mock import Mock, patch
import site_network as network


def entry(port):
    return (socket.AF_INET,socket.SOCK_STREAM,0,('127.0.0.1',port))


class NetworkTests(unittest.TestCase):
    def test_dns_deduplication(self):
        answer=(socket.AF_INET,socket.SOCK_STREAM,0,'',('127.0.0.1',22))
        with patch.object(network.socket,'getaddrinfo',return_value=[answer,answer]):
            self.assertEqual(network.resolve_endpoints('offline.invalid',22),[entry(22)])

    def test_dns_wait_is_bounded(self):
        release=threading.Event()
        def stalled(*args,**kwargs):
            release.wait(1)
            return []
        try:
            with patch.object(network.socket,'getaddrinfo',side_effect=stalled):
                with self.assertRaisesRegex(TimeoutError,'DNS lookup'):
                    network.resolve_endpoints('offline.invalid',22,.02)
        finally:
            release.set()

    def test_loopback_finds_reachable_endpoint(self):
        with socket.socket() as listener, socket.socket() as refused:
            listener.bind(('127.0.0.1',0));listener.listen(1)
            refused.bind(('127.0.0.1',0))
            addresses=[entry(refused.getsockname()[1]),entry(listener.getsockname()[1])]
            with patch.object(network,'resolve_endpoints',return_value=addresses):
                connected=network.connect_tcp('offline.invalid',timeout=1)
            with connected:
                self.assertEqual(connected.getpeername()[1],listener.getsockname()[1])
                accepted,_=listener.accept();accepted.close()

    def test_immediate_refusal_fails_without_authentication(self):
        with socket.socket() as refused:
            refused.bind(('127.0.0.1',0))
            with patch.object(network,'resolve_endpoints',return_value=[entry(refused.getsockname()[1])]):
                with self.assertRaises(TimeoutError):
                    network.connect_tcp('offline.invalid',timeout=.2)

    def test_many_unreachable_endpoints_share_one_deadline(self):
        now=[0.0];made=[];peak=[0]
        class Selector:
            def __init__(self):self.sockets=set()
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def register(self,sock,event):
                self.sockets.add(sock);peak[0]=max(peak[0],len(self.sockets))
            def unregister(self,sock):self.sockets.remove(sock)
            def select(self,timeout):now[0]+=timeout;return []
        def make(*args):
            sock=Mock();sock.connect_ex.return_value=errno.EINPROGRESS;made.append(sock);return sock
        with patch.object(network,'resolve_endpoints',return_value=[entry(100+i) for i in range(15)]),patch.object(network.time,'monotonic',side_effect=lambda:now[0]),patch.object(network.socket,'socket',side_effect=make),patch.object(network.selectors,'DefaultSelector',Selector):
            with self.assertRaises(TimeoutError):network.connect_tcp('offline.invalid',timeout=5,address_timeout=3)
        self.assertEqual(now[0],5)
        self.assertEqual(peak[0],4)
        self.assertEqual(len(made),8)
        for sock in made:sock.close.assert_called_once()

    def test_winner_closes_other_pending_sockets(self):
        pending,winner=Mock(),Mock()
        pending.connect_ex.return_value=errno.EINPROGRESS
        winner.connect_ex.return_value=0
        selector=Mock();selector.__enter__=Mock(return_value=selector);selector.__exit__=Mock()
        with patch.object(network,'resolve_endpoints',return_value=[entry(100),entry(101)]),patch.object(network.socket,'socket',side_effect=[pending,winner]),patch.object(network.selectors,'DefaultSelector',return_value=selector):
            self.assertIs(network.connect_tcp('offline.invalid'),winner)
        pending.close.assert_called_once()
        winner.close.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
