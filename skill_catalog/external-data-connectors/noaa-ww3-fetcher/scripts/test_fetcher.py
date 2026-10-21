"""Observable transfer/parser invariants using an isolated local HTTP fixture."""
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
import noaa_ww3_fetcher as f

WIRE=gzip.compress(b'known spectrum bytes '*5000,mtime=0)
class Server(BaseHTTPRequestHandler):
    etag='fixture-v1';truncate=False;requests=0
    def log_message(self,*args):pass
    def headers(self,n):
        self.send_header('Content-Length',str(n));self.send_header('ETag',type(self).etag)
        self.send_header('Last-Modified','Tue, 01 Jun 2021 16:48:29 GMT');self.send_header('Content-Encoding','gzip')
    def do_HEAD(self):
        self.send_response(200);self.headers(len(WIRE));self.end_headers()
    def do_GET(self):
        type(self).requests+=1
        offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
        self.send_response(206 if offset else 200)
        if offset:self.send_header('Content-Range',f'bytes {offset}-{len(WIRE)-1}/{len(WIRE)}')
        self.headers_out(len(WIRE)-offset)
        payload=WIRE[offset:]
        if type(self).truncate:payload=payload[:len(payload)//2]
        self.wfile.write(payload);self.close_connection=True
    # Separate name from BaseHTTPRequestHandler.headers request attribute.
    def headers_out(self,n):
        Server.headers(self,n);self.end_headers()

# do_HEAD must also avoid the request-header attribute shadowing its method.
def head(self):
    self.send_response(200);Server.headers(self,len(WIRE));self.end_headers()
Server.do_HEAD=head

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Server)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}/test.tar.gz'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close()
    def setUp(self):
        Server.etag='fixture-v1';Server.truncate=False;Server.requests=0
        self.temp=tempfile.TemporaryDirectory();self.raw=Path(self.temp.name)
        self.record={'filename':'test.tar.gz','url':self.url,'identity':f.identity(self.url),'publisher_md5':hashlib.md5(WIRE).hexdigest()}
    def tearDown(self):self.temp.cleanup()
    def test_wire_cache_resume(self):
        part=self.raw/'test.tar.gz.part';part.write_bytes(WIRE[:31]);f.save(str(part)+'.json',self.record)
        result=f.download(self.record,self.raw)
        self.assertEqual((self.raw/'test.tar.gz').read_bytes(),WIRE)
        self.assertEqual(result['sha256'],hashlib.sha256(WIRE).hexdigest())
        count=Server.requests;self.assertTrue(f.download(self.record,self.raw)['cache_reused']);self.assertEqual(count,Server.requests)
    def test_checksum_corruption(self):
        self.record['publisher_md5']='0'*32
        with self.assertRaisesRegex(ValueError,'checksum'):f.download(self.record,self.raw)
        self.assertFalse((self.raw/'test.tar.gz').exists())
    def test_truncation_bounded(self):
        Server.truncate=True
        with patch.object(f.time,'sleep'):
            with self.assertRaises(Exception):f.download(self.record,self.raw,retries=2)
        self.assertEqual(Server.requests,2);self.assertFalse((self.raw/'test.tar.gz').exists())
    def test_changed_identity_and_cache_corruption(self):
        Server.etag='fixture-v2'
        with self.assertRaisesRegex(ValueError,'identity'):f.download(self.record,self.raw)
        Server.etag='fixture-v1';(self.raw/'test.tar.gz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'cached'):f.download(self.record,self.raw)
    def test_conflicting_partial_identity(self):
        (self.raw/'test.tar.gz.part').write_bytes(WIRE[:31]);f.save(self.raw/'test.tar.gz.part.json',dict(self.record,url='changed'))
        with self.assertRaisesRegex(ValueError,'partial'):f.download(self.record,self.raw)
    def test_duplicates_bounds(self):
        a=np.array([[1,np.nan],[1,np.nan],[2,3]],dtype=float)
        t,aa,info=f.unique_records([0,0,3],[a],0,3)
        self.assertEqual(t.tolist(),[0]);self.assertEqual(info['boundary_records_excluded'],1)
        a[1,0]=4
        with self.assertRaisesRegex(ValueError,'Conflicting'):f.unique_records([0,0,3],[a],0,3)
    def test_spectral_native_order(self):
        text="'WAVEWATCH III SPECTRA' 2 2 1 'test'\n0.1 0.2\n4.71238898038469 1.5707963267949\n20181101 000000\n'44014' 36 -75 20 1 2 3 4\n1 2 3 4\n"
        t,fr,d,env,e=f.parse_spec(text,'44014')
        self.assertEqual(e.tolist(),[[[1,3],[2,4]]]);self.assertGreater(d[0],d[1])
        hs,df,dw=f.integrate(fr,d,e);self.assertAlmostEqual(hs[0],4*np.sqrt(np.sum(e[0]*df[:,None]*np.pi)))
    def test_safe_members(self):
        path=self.raw/'unsafe.tar.gz'
        with tarfile.open(path,'w:gz') as archive:
            member=tarfile.TarInfo('../multi_1.44014.SPEC.201811');member.size=1;archive.addfile(member,io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError,'Unsafe'):f.selected_members(path,['44014'],'SPEC','201811')

if __name__=='__main__':unittest.main(verbosity=2)
