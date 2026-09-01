import base64
import binascii
import http.client
import io
import socket
import ssl
import time
import urllib.parse
from datetime import datetime, timezone


SIGNATURE_FIELDS = (
    "X-Amz-Algorithm",
    "X-Amz-Credential",
    "X-Amz-Date",
    "X-Amz-Expires",
    "X-Amz-SignedHeaders",
    "X-Amz-Signature",
)


def select_image_source(inp, view, required):
    base64_field = f"{view}_image_base64"
    url_field = f"{view}_image_url"
    has_base64 = base64_field in inp and inp[base64_field] is not None
    has_url = url_field in inp and inp[url_field] is not None
    if has_base64 and has_url:
        raise ValueError(f"{base64_field} and {url_field} are mutually exclusive")
    if not has_base64 and not has_url:
        if required:
            raise ValueError(f"exactly one of {base64_field} or {url_field} is required")
        return None
    if has_base64:
        value = inp[base64_field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{base64_field} must be a nonempty base64 string")
        return "base64", base64_field, value
    value = inp[url_field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{url_field} must be a nonempty HTTPS URL")
    return "url", url_field, value


def validate_image_bytes(raw, field_name, destination, max_bytes, max_pixels):
    from PIL import Image, UnidentifiedImageError

    if not raw:
        raise ValueError(f"{field_name} resolves to an empty file")
    if len(raw) > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")

    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError(f"{field_name} must contain exactly one image frame")
            width, height = source.size
            if width < 2 or height < 2 or width * height > max_pixels:
                raise ValueError(
                    f"{field_name} dimensions must be at least 2x2 and at most "
                    f"{max_pixels} pixels"
                )
            image = source.convert("RGBA")
            image.load()
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{field_name} is not a valid supported raster image") from exc

    alpha_bbox = image.getchannel("A").getbbox()
    if (
        alpha_bbox is None
        or alpha_bbox[2] - alpha_bbox[0] < 2
        or alpha_bbox[3] - alpha_bbox[1] < 2
    ):
        raise ValueError(f"{field_name} has no usable nontransparent image area")
    image.save(destination, format="PNG", optimize=False, compress_level=6)


def decode_base64_image(value, field_name, destination, max_bytes, max_pixels):
    encoded = value.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError(f"{field_name} has an invalid data URI")
    encoded = "".join(encoded.split())
    max_encoded_length = ((max_bytes + 2) // 3) * 4
    if len(encoded) > max_encoded_length:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte decoded limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} is not valid base64") from exc
    validate_image_bytes(raw, field_name, destination, max_bytes, max_pixels)


def _endpoint_authority(endpoint_url):
    try:
        parsed = urllib.parse.urlsplit(endpoint_url)
        port = parsed.port or 443
    except (TypeError, ValueError):
        raise RuntimeError("BUCKET_ENDPOINT_URL must be a valid HTTPS endpoint") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("BUCKET_ENDPOINT_URL must be a valid HTTPS endpoint")
    return parsed.hostname.lower(), port


def validate_signed_r2_url(
    value,
    endpoint_url,
    field_name,
    max_ttl_seconds,
    now=None,
):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty HTTPS URL")
    value = value.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} is not a valid HTTPS URL")
    allowed_host, allowed_port = _endpoint_authority(endpoint_url)
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port or 443
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} is not a valid HTTPS URL") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.hostname.lower() != allowed_host
        or port != allowed_port
    ):
        raise ValueError(f"{field_name} must use the configured R2 HTTPS host")
    try:
        query = urllib.parse.parse_qs(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=64,
        )
    except ValueError:
        raise ValueError(f"{field_name} has an invalid signed query") from None
    if any(len(query.get(name, ())) != 1 or not query[name][0] for name in SIGNATURE_FIELDS):
        raise ValueError(f"{field_name} must be a signed R2 URL")
    if query["X-Amz-Algorithm"][0] != "AWS4-HMAC-SHA256":
        raise ValueError(f"{field_name} must use AWS Signature Version 4")
    if "host" not in query["X-Amz-SignedHeaders"][0].lower().split(";"):
        raise ValueError(f"{field_name} must sign the host header")
    try:
        ttl = int(query["X-Amz-Expires"][0])
        signed_at = datetime.strptime(
            query["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ"
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} has invalid signing metadata") from None
    if ttl < 1 or ttl > max_ttl_seconds:
        raise ValueError(
            f"{field_name} signed lifetime must be between 1 and {max_ttl_seconds} seconds"
        )
    current = now or datetime.now(timezone.utc)
    age = (current - signed_at).total_seconds()
    if age < -300 or age > ttl:
        raise ValueError(f"{field_name} is expired or not yet valid")
    target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    return parsed.hostname, port, target


def download_signed_r2_object(
    value,
    endpoint_url,
    field_name,
    max_bytes,
    timeout_seconds,
    max_ttl_seconds,
):
    host, port, target = validate_signed_r2_url(
        value,
        endpoint_url,
        field_name,
        max_ttl_seconds,
    )
    if timeout_seconds <= 0:
        raise RuntimeError("download timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    connection = None
    response = None
    try:
        connection = http.client.HTTPSConnection(
            host,
            port=port,
            timeout=timeout_seconds,
            context=ssl.create_default_context(),
        )
        connection.request("GET", target, headers={"Accept-Encoding": "identity"})
        if connection.sock is not None:
            connection.sock.settimeout(max(0.001, deadline - time.monotonic()))
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"{field_name} download returned HTTP {response.status}")
        content_encoding = (response.getheader("Content-Encoding") or "").lower()
        if content_encoding not in {"", "identity"}:
            raise RuntimeError(f"{field_name} download used unsupported content encoding")
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                raise RuntimeError(f"{field_name} download has an invalid size") from None
            if declared_length < 0 or declared_length > max_bytes:
                raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")
        data = bytearray()
        while True:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise RuntimeError(f"{field_name} download timed out")
            if connection.sock is not None:
                connection.sock.settimeout(max(0.001, remaining_time))
            chunk = response.read(min(65536, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")
        return bytes(data)
    except ValueError:
        raise
    except (socket.timeout, TimeoutError):
        raise RuntimeError(f"{field_name} download timed out") from None
    except RuntimeError:
        raise
    except (http.client.HTTPException, OSError, ssl.SSLError):
        raise RuntimeError(f"{field_name} download failed") from None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def download_r2_image(
    value,
    endpoint_url,
    field_name,
    destination,
    max_bytes,
    max_pixels,
    timeout_seconds,
    max_ttl_seconds,
):
    raw = download_signed_r2_object(
        value,
        endpoint_url,
        field_name,
        max_bytes,
        timeout_seconds,
        max_ttl_seconds,
    )
    validate_image_bytes(raw, field_name, destination, max_bytes, max_pixels)
