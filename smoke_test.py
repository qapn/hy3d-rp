import base64
import hashlib
import json
import os
import struct
import time
import urllib.request
from pathlib import Path


DEFAULT_SAMPLE_ROOT = Path(
    os.environ.get(
        "HY3D_SMOKE_SAMPLE_ROOT",
        "/opt/Hunyuan3D-2/assets/example_mv_images/1",
    )
)


def _flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _image_base64(view):
    encoded_name = f"SMOKE_{view.upper()}_IMAGE_BASE64"
    if os.environ.get(encoded_name):
        return os.environ[encoded_name]

    path_name = f"SMOKE_{view.upper()}_IMAGE_PATH"
    path = Path(os.environ.get(path_name, str(DEFAULT_SAMPLE_ROOT / f"{view}.png")))
    if not path.is_file():
        raise FileNotFoundError(
            f"Smoke-test {view} image not found at {path}; set {encoded_name} or {path_name}"
        )
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _validate_glb(data):
    if len(data) < 20:
        raise RuntimeError("Smoke-test GLB is empty or truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise RuntimeError("Smoke-test output is not a valid GLB v2 file")


def _download(url):
    timeout = int(os.environ.get("SMOKE_DOWNLOAD_TIMEOUT", "300"))
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def run_smoke_test(worker_handler):
    include_right = _flag("SMOKE_INCLUDE_RIGHT", False)
    return_base64 = _flag("SMOKE_RETURN_BASE64", True)
    verify_url = _flag("SMOKE_VERIFY_URL", True)
    if not return_base64 and not verify_url:
        raise ValueError("Enable SMOKE_RETURN_BASE64 or SMOKE_VERIFY_URL to inspect the GLB")

    inp = {
        "front_image_base64": _image_base64("front"),
        "left_image_base64": _image_base64("left"),
        "back_image_base64": _image_base64("back"),
        "seed": int(os.environ.get("SMOKE_SEED", "12345")),
        "num_inference_steps": int(os.environ.get("SMOKE_NUM_INFERENCE_STEPS", "30")),
        "octree_resolution": int(os.environ.get("SMOKE_OCTREE_RESOLUTION", "380")),
        "num_chunks": int(os.environ.get("SMOKE_NUM_CHUNKS", "20000")),
        "return_base64": return_base64,
    }
    if include_right:
        inp["right_image_base64"] = _image_base64("right")

    job_id = os.environ.get("SMOKE_JOB_ID", f"hy3d-smoke-{int(time.time())}")
    print(f"[smoke] Starting job {job_id}", flush=True)
    result = worker_handler({"id": job_id, "input": inp})
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError(f"Smoke-test worker failure:\n{json.dumps(result, indent=2)}")
    if not result.get("model_url") or result.get("expires_in") != 3600:
        raise RuntimeError("Smoke test did not receive a one-hour model_url")

    inline_data = None
    if return_base64:
        inline_data = base64.b64decode(result["model_base64"], validate=True)
        _validate_glb(inline_data)

    downloaded_data = None
    if verify_url:
        downloaded_data = _download(result["model_url"])
        _validate_glb(downloaded_data)

    glb = inline_data if inline_data is not None else downloaded_data
    digest = hashlib.sha256(glb).hexdigest()
    if digest != result.get("sha256"):
        raise RuntimeError("Smoke-test GLB digest does not match the handler response")
    if inline_data is not None and downloaded_data is not None and inline_data != downloaded_data:
        raise RuntimeError("R2 object differs from the returned base64 GLB")

    expected = os.environ.get("SMOKE_EXPECTED_SHA256")
    if expected and digest.lower() != expected.strip().lower():
        raise RuntimeError(
            f"Smoke-test digest {digest} does not match SMOKE_EXPECTED_SHA256={expected}"
        )

    summary = {
        key: result[key]
        for key in (
            "format",
            "expires_in",
            "sha256",
            "size_bytes",
            "vertex_count",
            "face_count",
            "seed",
        )
    }
    print(f"[smoke] PASS\n{json.dumps(summary, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    from handler import handler

    raise SystemExit(run_smoke_test(handler))
