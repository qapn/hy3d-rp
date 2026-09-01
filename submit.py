import argparse
import hashlib
import json
import re
import stat
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from input_utils import download_signed_r2_object


REQUIRED_ENV = (
    "RUNPOD_API_KEY",
    "RUNPOD_ENDPOINT_ID",
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_INPUT_PIXELS = 25_000_000
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
INPUT_URL_TTL = 900
OUTPUT_URL_TTL = 3600
RUNPOD_TIMEOUT = 3600
POLL_INTERVAL = 5


def load_repo_env(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("the repository .env must be a regular file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise RuntimeError("the repository .env must have mode 600")
    values = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            raise RuntimeError(f"invalid .env entry on line {line_number}")
        name, value = stripped.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise RuntimeError(f"invalid .env name on line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    missing = [name for name in REQUIRED_ENV if not values.get(name)]
    if missing:
        raise RuntimeError("the repository .env is missing: " + ", ".join(missing))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", values["RUNPOD_ENDPOINT_ID"]):
        raise RuntimeError("RUNPOD_ENDPOINT_ID is invalid")
    return values


def read_png(path):
    from PIL import Image, UnidentifiedImageError

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"input is not a regular file: {path}")
    size = path.stat().st_size
    if size < 8 or size > MAX_INPUT_BYTES:
        raise RuntimeError(f"input PNG size is invalid: {path}")
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"input is not a PNG: {path}")
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise RuntimeError(f"input must be a single-frame PNG: {path}")
            width, height = image.size
            if width < 2 or height < 2 or width * height > MAX_INPUT_PIXELS:
                raise RuntimeError(f"input PNG dimensions are invalid: {path}")
            rgba = image.convert("RGBA")
            rgba.load()
    except RuntimeError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise RuntimeError(f"input is not a valid PNG: {path}") from None
    alpha_bbox = rgba.getchannel("A").getbbox()
    if (
        alpha_bbox is None
        or alpha_bbox[2] - alpha_bbox[0] < 2
        or alpha_bbox[3] - alpha_bbox[1] < 2
    ):
        raise RuntimeError(f"input PNG has no usable nontransparent area: {path}")
    return data, hashlib.sha256(data).hexdigest()


def make_r2_client(values):
    try:
        import boto3
        from botocore.config import Config

        return boto3.client(
            "s3",
            endpoint_url=values["BUCKET_ENDPOINT_URL"],
            aws_access_key_id=values["BUCKET_ACCESS_KEY_ID"],
            aws_secret_access_key=values["BUCKET_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
    except Exception:
        raise RuntimeError("could not initialize the R2 client") from None


def upload_input(client, bucket, key, data):
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="image/png",
        )
    except Exception:
        raise RuntimeError("temporary R2 input upload failed") from None


def sign_input(client, bucket, key):
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=INPUT_URL_TTL,
        )
    except Exception:
        raise RuntimeError("temporary R2 input signing failed") from None


def delete_input(client, bucket, key):
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except Exception:
        raise RuntimeError("temporary R2 input cleanup failed") from None


def runpod_request(request, timeout):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024:
                raise RuntimeError("RunPod returned an oversized response")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"RunPod request failed with HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("RunPod request failed") from None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        raise RuntimeError("RunPod returned an invalid response") from None
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("RunPod request failed") from None


def submit_job(values, image_urls):
    endpoint_id = values["RUNPOD_ENDPOINT_ID"]
    base_url = f"https://api.runpod.ai/v2/{urllib.parse.quote(endpoint_id, safe='')}"
    payload = {
        "input": {
            **{f"{view}_image_url": url for view, url in image_urls.items()},
            "seed": 12345,
            "num_inference_steps": 30,
            "octree_resolution": 380,
            "num_chunks": 20000,
            "return_base64": False,
        }
    }
    headers = {
        "Authorization": f"Bearer {values['RUNPOD_API_KEY']}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(
        f"{base_url}/run",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    response = runpod_request(request, 60)
    job_id = response.get("id") if isinstance(response, dict) else None
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise RuntimeError("RunPod did not return a valid job ID")
    deadline = time.monotonic() + RUNPOD_TIMEOUT
    while True:
        if time.monotonic() >= deadline:
            raise RuntimeError("RunPod job timed out")
        status_request = urllib.request.Request(
            f"{base_url}/status/{urllib.parse.quote(job_id, safe='')}",
            headers={"Authorization": headers["Authorization"]},
            method="GET",
        )
        status_response = runpod_request(status_request, 60)
        if not isinstance(status_response, dict):
            raise RuntimeError("RunPod returned an invalid status response")
        status = status_response.get("status")
        if status == "COMPLETED":
            output = status_response.get("output")
            if not isinstance(output, dict):
                raise RuntimeError("RunPod completed without a valid output")
            return output
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError(f"RunPod job ended with status {status}")
        if status not in {"IN_QUEUE", "IN_PROGRESS"}:
            raise RuntimeError("RunPod returned an unknown job status")
        time.sleep(POLL_INTERVAL)


def validate_glb(data, output):
    if len(data) < 20:
        raise RuntimeError("downloaded GLB is empty or truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise RuntimeError("downloaded output is not a valid GLB v2 file")
    digest = hashlib.sha256(data).hexdigest()
    expected_digest = output.get("sha256")
    expected_size = output.get("size_bytes")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_digest
    ):
        raise RuntimeError("worker output has an invalid SHA-256")
    if digest.lower() != expected_digest.lower():
        raise RuntimeError("downloaded GLB SHA-256 does not match the worker output")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise RuntimeError("worker output has an invalid byte count")
    if expected_size != len(data):
        raise RuntimeError("downloaded GLB size does not match the worker output")
    return digest


def download_output(output, values):
    url = output.get("model_url")
    if output.get("expires_in") != OUTPUT_URL_TTL:
        raise RuntimeError("worker output does not have a one-hour lifetime")
    if output.get("format") != "glb":
        raise RuntimeError("worker output format is not GLB")
    return download_signed_r2_object(
        url,
        values["BUCKET_ENDPOINT_URL"],
        "model_url",
        MAX_OUTPUT_BYTES,
        300,
        OUTPUT_URL_TTL,
    )


def run(args):
    env_path = Path(__file__).resolve().with_name(".env")
    values = load_repo_env(env_path)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise RuntimeError(f"output path already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise RuntimeError(f"output directory does not exist: {output_path.parent}")

    paths = {
        "front": args.front.expanduser().resolve(),
        "left": args.left.expanduser().resolve(),
        "back": args.back.expanduser().resolve(),
    }
    if args.right is not None:
        paths["right"] = args.right.expanduser().resolve()
    images = {view: read_png(path) for view, path in paths.items()}
    hashes = {view: result[1] for view, result in images.items()}
    client = make_r2_client(values)
    bucket = values["BUCKET_NAME"]
    prefix = f"inputs/{uuid.uuid4()}"
    uploaded = []
    result = None
    failure = None
    cleanup_failure = None
    try:
        image_urls = {}
        for view, (data, _) in images.items():
            key = f"{prefix}/{view}.png"
            uploaded.append(key)
            upload_input(client, bucket, key, data)
            image_urls[view] = sign_input(client, bucket, key)
        worker_output = submit_job(values, image_urls)
        glb = download_output(worker_output, values)
        model_hash = validate_glb(glb, worker_output)
        try:
            with output_path.open("xb") as destination:
                destination.write(glb)
        except FileExistsError:
            raise RuntimeError(f"output path already exists: {output_path}") from None
        except OSError:
            raise RuntimeError("could not write the local GLB") from None
        result = output_path, hashes, model_hash
    except Exception as exc:
        failure = exc
    finally:
        for key in uploaded:
            try:
                delete_input(client, bucket, key)
            except RuntimeError as exc:
                cleanup_failure = exc
    if failure is not None:
        raise failure
    if cleanup_failure is not None:
        raise cleanup_failure
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--back", type=Path, required=True)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        output_path, hashes, model_hash = run(parse_args(argv))
    except Exception as exc:
        print(f"submission_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"output_path={output_path}")
    for view, digest in hashes.items():
        print(f"{view}_sha256={digest}")
    print(f"model_sha256={model_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
