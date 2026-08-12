#!/usr/bin/env python3
"""Offline regression tests for the self-contained NOAA archive adapter."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ofs_archive_sources as sources


class Response:
    def __init__(self, content, status=200, headers=None):
        self.content, self.status_code, self.headers = content, status, dict(headers or {})
        self.closed = False
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)
    def close(self): self.closed = True


class Session:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs)); return self.responses.pop(0)


def page(key, truncated=False, token=None):
    next_token = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Contents><Key>{key}</Key>'
            '<LastModified>2020-01-02T00:00:00Z</LastModified><ETag>&quot;opaque-3&quot;</ETag><Size>123</Size>'
            f'</Contents><IsTruncated>{str(truncated).lower()}</IsTruncated>{next_token}</ListBucketResult>').encode()


class AdapterTests(unittest.TestCase):
    def test_sjrofs_roots_scope_and_cache(self):
        aws = sources.get_source_descriptor("aws_operational", "sjrofs")
        ncei = sources.get_source_descriptor("ncei_long_term", "sjrofs")
        self.assertEqual(aws["root_prefix"], "sjrofs/netcdf/")
        self.assertTrue(ncei["root_prefix"].endswith("st-johns-river-operational-forecast-system-sjrofs/"))
        key = ncei["root_prefix"] + "2020/01/nos.sjrofs.fields.nowcast.20200101.t05z.nc"
        item = {**ncei, "source_id": "ncei_long_term", "key": key,
                "url": sources.canonical_object_url("ncei_long_term", "sjrofs", key),
                "size": 1, "etag": "opaque", "last_modified": "now"}
        item["source_identity"] = sources.source_identity_digest(item)
        sources.validate_source_object("sjrofs", item)
        self.assertTrue(sources.cache_relpath(item).startswith("ncei_long_term/2020/01/"))
        bad = dict(item, key=key.replace("2020/01", "2020/02"))
        bad["url"] = sources.canonical_object_url("ncei_long_term", "sjrofs", bad["key"])
        bad["source_identity"] = sources.source_identity_digest(bad)
        with self.assertRaisesRegex(ValueError, "YYYY/MM matching"):
            sources.validate_source_object("sjrofs", bad)

    def test_xml_pagination(self):
        root = sources.get_source_descriptor("ncei_long_term", "sjrofs")["root_prefix"]
        a = root + "2020/01/nos.sjrofs.fields.nowcast.20200101.t05z.nc"
        b = root + "2020/01/nos.sjrofs.stations.nowcast.20200101.t05z.nc"
        responses = [Response(page(a, True, "next")), Response(page(b))]
        session = Session(responses)
        rows = sources.list_objects_v2("ncei_long_term", "sjrofs", root + "2020/01/", session=session)
        self.assertEqual([row["key"] for row in rows], [a, b])
        self.assertEqual(session.calls[1][1]["params"]["continuation-token"], "next")
        self.assertTrue(all(response.closed for response in responses))

    def test_range_validation_and_provider_identity(self):
        result = sources.validate_download_response(Response(b"", 206, {
            "ETag": '"opaque-8"', "Content-Length": "60", "Content-Range": "bytes 40-99/100"
        }), {"size": 100, "etag": "opaque-8"}, offset=40)
        self.assertEqual(result["remaining_bytes"], 60)
        common = {"container": "x", "endpoint": "https://x", "key": "k", "url": "https://x/k", "size": 1, "etag": "e", "last_modified": "now"}
        self.assertNotEqual(sources.source_identity_digest({**common, "source_id": "aws_operational"}),
                            sources.source_identity_digest({**common, "source_id": "ncei_long_term"}))


if __name__ == "__main__": unittest.main(verbosity=2)
