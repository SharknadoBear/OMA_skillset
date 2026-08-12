#!/usr/bin/env python3
"""Offline parity regression tests for the shared OFS archive adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ofs_archive_sources as sources


class FakeResponse:
    def __init__(self, content: bytes, *, status: int = 200, headers=None):
        self.content = content
        self.status_code = status
        self.headers = dict(headers or {})
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def xml_page(key: str, *, truncated=False, token=None) -> bytes:
    continuation = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<Contents><Key>{key}</Key><LastModified>2020-01-02T00:00:00Z</LastModified>"
        '<ETag>&quot;opaque-3&quot;</ETag><Size>123</Size></Contents>'
        f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{continuation}"
        "</ListBucketResult>"
    ).encode()


class DescriptorTests(unittest.TestCase):
    def test_exact_model_roots_and_urls(self):
        for model, slug in sources.MODEL_SLUGS.items():
            aws = sources.get_source_descriptor("aws_operational", model)
            ncei = sources.get_source_descriptor("ncei_long_term", model)
            self.assertEqual(aws["root_prefix"], f"{model}/netcdf/")
            self.assertEqual(ncei["root_prefix"], sources.NCEI_BASE + slug + "/")
            key = f"{model}/netcdf/2026/07/20/{model}.t00z.20260720.fields.n006.nc"
            self.assertEqual(
                sources.canonical_object_url("aws_operational", model, key),
                sources.AWS_ENDPOINT + "/" + key,
            )

    def test_strict_aws_and_ncei_scope(self):
        aws_key = "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc"
        aws = {
            **sources.get_source_descriptor("aws_operational", "cbofs"),
            "key": aws_key,
            "url": sources.canonical_object_url("aws_operational", "cbofs", aws_key),
            "size": 1,
            "etag": "e",
            "last_modified": "now",
        }
        aws["source_identity"] = sources.source_identity_digest(aws)
        sources.validate_source_object("cbofs", aws, require_metadata=True)
        bad = dict(aws, key=aws_key.replace("2026/07/20", "2026/07/19"))
        bad["url"] = sources.canonical_object_url("aws_operational", "cbofs", bad["key"])
        bad["source_identity"] = sources.source_identity_digest(bad)
        with self.assertRaisesRegex(ValueError, "path does not match"):
            sources.validate_source_object("cbofs", bad)

        root = sources.get_source_descriptor("ncei_long_term", "dbofs")["root_prefix"]
        ncei_key = root + "2020/01/nos.dbofs.fields.n006.20200101.t00z.nc"
        ncei = {
            **sources.get_source_descriptor("ncei_long_term", "dbofs"),
            "key": ncei_key,
            "url": sources.canonical_object_url("ncei_long_term", "dbofs", ncei_key),
            "size_bytes": 10,
            "etag": "opaque",
            "last_modified": "now",
        }
        ncei["source_identity"] = sources.source_identity_digest(ncei)
        sources.validate_source_object("dbofs", ncei, require_metadata=True)
        self.assertTrue(sources.cache_relpath(ncei).startswith("ncei_long_term/2020/01/"))
        for field in ("provider", "archive_role", "container", "endpoint", "listing_endpoint"):
            with self.subTest(field=field):
                tampered = dict(ncei)
                tampered.pop(field)
                tampered["source_identity"] = sources.source_identity_digest(tampered)
                with self.assertRaisesRegex(ValueError, field):
                    sources.validate_source_object("dbofs", tampered)
        tampered = dict(ncei, source_identity="0" * 64)
        with self.assertRaisesRegex(ValueError, "source_identity"):
            sources.validate_source_object("dbofs", tampered)

    def test_xml_pagination(self):
        root = sources.get_source_descriptor("ncei_long_term", "nyofs")["root_prefix"]
        key1 = root + "2020/01/nos.nyofs.fields.nowcast.20200101.t05z.nc"
        key2 = root + "2020/01/nos.nyofs.stations.nowcast.20200101.t05z.nc"
        responses = [
            FakeResponse(xml_page(key1, truncated=True, token="next")),
            FakeResponse(xml_page(key2)),
        ]
        session = FakeSession(responses)
        records = sources.list_objects_v2(
            "ncei_long_term", "nyofs", root + "2020/01/", session=session,
        )
        self.assertEqual([item["key"] for item in records], [key1, key2])
        self.assertEqual(session.calls[1][1]["params"]["continuation-token"], "next")
        self.assertTrue(all(response.closed for response in responses))


class TransferTests(unittest.TestCase):
    def test_range_and_etag_validation(self):
        record = {"size": 100, "etag": "opaque-8"}
        response = FakeResponse(
            b"", status=206,
            headers={
                "ETag": '"opaque-8"', "Content-Length": "60",
                "Content-Range": "bytes 40-99/100",
            },
        )
        result = sources.validate_download_response(response, record, offset=40)
        self.assertEqual(result["remaining_bytes"], 60)
        self.assertEqual(sources.parse_content_range("bytes 40-99/100"), (40, 99, 100))
        self.assertEqual(sources.build_resume_headers(40), {"Range": "bytes=40-"})
        with self.assertRaisesRegex(RuntimeError, "ETag"):
            sources.validate_download_response(
                FakeResponse(b"", status=200, headers={"ETag": "changed"}), record,
            )

    def test_identity_is_provider_local(self):
        common = {
            "container": "bucket", "endpoint": "https://example", "key": "k",
            "url": "https://example/k", "size": 10, "etag": "e",
            "last_modified": "now",
        }
        aws = sources.source_identity_digest({**common, "source_id": "aws_operational"})
        ncei = sources.source_identity_digest({**common, "source_id": "ncei_long_term"})
        self.assertNotEqual(aws, ncei)


if __name__ == "__main__":
    unittest.main(verbosity=2)
