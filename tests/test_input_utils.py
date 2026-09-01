import base64
import io
import unittest
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import input_utils


ENDPOINT = "https://account.r2.cloudflarestorage.com"


def signed_url(host="account.r2.cloudflarestorage.com", expires=900):
    query = urllib.parse.urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "key/20260901/auto/s3/aws4_request",
            "X-Amz-Date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "X-Amz-Expires": str(expires),
            "X-Amz-SignedHeaders": "host",
            "X-Amz-Signature": "a" * 64,
        }
    )
    return f"https://{host}/bucket/input.png?{query}"


class FakeSocket:
    def settimeout(self, timeout):
        self.timeout = timeout


class FakeResponse:
    def __init__(self, data=b"data", status=200, headers=None):
        self.data = data
        self.status = status
        self.headers = headers or {}
        self.offset = 0
        self.closed = False

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, size):
        result = self.data[self.offset : self.offset + size]
        self.offset += len(result)
        return result

    def close(self):
        self.closed = True


class FakeConnection:
    response = FakeResponse()
    requested_target = None

    def __init__(self, host, port, timeout, context):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.context = context
        self.sock = FakeSocket()

    def request(self, method, target, headers):
        self.method = method
        self.headers = headers
        type(self).requested_target = target

    def getresponse(self):
        return type(self).response

    def close(self):
        self.closed = True


class InputSelectionTests(unittest.TestCase):
    def test_base64_compatibility(self):
        source = input_utils.select_image_source(
            {"front_image_base64": "abc"}, "front", True
        )
        self.assertEqual(source, ("base64", "front_image_base64", "abc"))

    def test_url_source(self):
        source = input_utils.select_image_source(
            {"front_image_url": "https://example.test/a"}, "front", True
        )
        self.assertEqual(source[0], "url")

    def test_sources_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            input_utils.select_image_source(
                {
                    "front_image_base64": "abc",
                    "front_image_url": "https://example.test/a",
                },
                "front",
                True,
            )

    def test_required_and_optional_views(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            input_utils.select_image_source({}, "front", True)
        self.assertIsNone(input_utils.select_image_source({}, "right", False))


class ImageValidationTests(unittest.TestCase):
    def test_base64_round_trip_preserves_alpha(self):
        from PIL import Image

        image = Image.new("RGBA", (3, 3), (0, 0, 0, 0))
        image.putpixel((0, 0), (255, 0, 0, 64))
        image.putpixel((1, 0), (255, 0, 0, 128))
        image.putpixel((0, 1), (255, 0, 0, 192))
        image.putpixel((1, 1), (255, 0, 0, 255))
        source = io.BytesIO()
        image.save(source, format="PNG")
        encoded = base64.b64encode(source.getvalue()).decode("ascii")
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "image.png"
            input_utils.decode_base64_image(
                encoded,
                "front_image_base64",
                destination,
                1024 * 1024,
                100,
            )
            with Image.open(destination) as restored:
                restored_alpha = restored.getchannel("A").tobytes()
        self.assertEqual(restored_alpha, image.getchannel("A").tobytes())


class SignedUrlTests(unittest.TestCase):
    def test_accepts_configured_https_host_and_sigv4(self):
        host, port, target = input_utils.validate_signed_r2_url(
            signed_url(),
            ENDPOINT,
            "front_image_url",
            900,
        )
        self.assertEqual(host, "account.r2.cloudflarestorage.com")
        self.assertEqual(port, 443)
        self.assertTrue(target.startswith("/bucket/input.png?"))

    def test_rejects_http_wrong_host_and_long_ttl(self):
        cases = (
            signed_url().replace("https://", "http://", 1),
            signed_url("example.com"),
            signed_url(expires=901),
        )
        for value in cases:
            with self.subTest(value=value.split("?", 1)[0]):
                with self.assertRaises(ValueError):
                    input_utils.validate_signed_r2_url(
                        value,
                        ENDPOINT,
                        "front_image_url",
                        900,
                    )

    def test_rejects_unsigned_url(self):
        with self.assertRaisesRegex(ValueError, "signed R2 URL"):
            input_utils.validate_signed_r2_url(
                "https://account.r2.cloudflarestorage.com/bucket/input.png",
                ENDPOINT,
                "front_image_url",
                900,
            )

    def test_download_is_bounded_and_redirect_free(self):
        FakeConnection.response = FakeResponse(b"abcd", headers={"Content-Length": "4"})
        with patch("input_utils.http.client.HTTPSConnection", FakeConnection):
            data = input_utils.download_signed_r2_object(
                signed_url(),
                ENDPOINT,
                "front_image_url",
                4,
                20,
                900,
            )
        self.assertEqual(data, b"abcd")
        self.assertIn("X-Amz-Signature", FakeConnection.requested_target)

        FakeConnection.response = FakeResponse(b"", status=302)
        with patch("input_utils.http.client.HTTPSConnection", FakeConnection):
            with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                input_utils.download_signed_r2_object(
                    signed_url(),
                    ENDPOINT,
                    "front_image_url",
                    4,
                    20,
                    900,
                )

    def test_download_rejects_oversized_content(self):
        FakeConnection.response = FakeResponse(b"abcde", headers={"Content-Length": "5"})
        with patch("input_utils.http.client.HTTPSConnection", FakeConnection):
            with self.assertRaisesRegex(ValueError, "4-byte limit"):
                input_utils.download_signed_r2_object(
                    signed_url(),
                    ENDPOINT,
                    "front_image_url",
                    4,
                    20,
                    900,
                )

    def test_download_failure_does_not_expose_signed_url(self):
        class FailingConnection(FakeConnection):
            def request(self, method, target, headers):
                raise OSError(f"failed request {target}")

        value = signed_url()
        with patch("input_utils.http.client.HTTPSConnection", FailingConnection):
            with self.assertRaises(RuntimeError) as raised:
                input_utils.download_signed_r2_object(
                    value,
                    ENDPOINT,
                    "front_image_url",
                    4,
                    20,
                    900,
                )
        self.assertEqual(str(raised.exception), "front_image_url download failed")
        self.assertNotIn("X-Amz-Signature", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
