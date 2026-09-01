import base64
import binascii
import hashlib
import io
import os
import re
import struct
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

MODEL_PATH = os.environ.get("HY3D_MODEL_PATH", "/models/Hunyuan3D-2mv")
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-mv"
MODEL_REVISION = "3a761b539b29fe4ff64714813aa9560fd66f5de0"
SOURCE_COMMIT = "f8db63096c8282cb27354314d896feba5ba6ff8a"
JOB_TMP_ROOT = os.environ.get("JOB_TMP_ROOT", "/tmp/hy3d-jobs")
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", "25000000"))
PRESIGNED_URL_TTL = 3600

REQUIRED_BUCKET_ENV = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)
REQUIRED_VIEWS = ("front", "left", "back")
OPTIONAL_VIEWS = ("right",)

PIPELINE = None
INIT_ERROR = None
_INFERENCE_LOCK = threading.Lock()


def load_model():
    global PIPELINE

    started = time.monotonic()
    print(
        f"[init] Hunyuan3D source={SOURCE_COMMIT} model_revision={MODEL_REVISION}",
        flush=True,
    )
    print(
        f"[init] Loading full tencent/Hunyuan3D-2mv/{MODEL_SUBFOLDER} from {MODEL_PATH}",
        flush=True,
    )

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")

    checkpoint = Path(MODEL_PATH) / MODEL_SUBFOLDER / "model.fp16.safetensors"
    config = Path(MODEL_PATH) / MODEL_SUBFOLDER / "config.yaml"
    for required_file in (checkpoint, config):
        if not required_file.is_file() or required_file.stat().st_size == 0:
            raise FileNotFoundError(f"Baked model file is missing or empty: {required_file}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    device_name = torch.cuda.get_device_name(0)
    print(f"[init] CUDA device: {device_name}; dtype=torch.float16", flush=True)
    PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_PATH,
        subfolder=MODEL_SUBFOLDER,
        variant="fp16",
        use_safetensors=True,
        device="cuda",
        dtype=torch.float16,
    )
    PIPELINE.model.eval()
    PIPELINE.vae.eval()
    PIPELINE.conditioner.eval()

    allocated_gib = torch.cuda.memory_allocated() / (1024**3)
    print(
        f"[init] Model ready in {time.monotonic() - started:.1f}s "
        f"({allocated_gib:.2f} GiB CUDA allocated).",
        flush=True,
    )


try:
    os.makedirs(JOB_TMP_ROOT, mode=0o700, exist_ok=True)
    load_model()
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def _integer_input(inp, name, default, minimum, maximum):
    value = inp.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be an integer between {minimum} and {maximum}"
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _boolean_input(inp, name, default=False):
    value = inp.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _decode_image(value, field_name, destination):
    from PIL import Image, UnidentifiedImageError

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required and must be a nonempty base64 string")

    encoded = value.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError(f"{field_name} has an invalid data URI")
    encoded = "".join(encoded.split())
    max_encoded_length = ((MAX_IMAGE_BYTES + 2) // 3) * 4
    if len(encoded) > max_encoded_length:
        raise ValueError(f"{field_name} exceeds the {MAX_IMAGE_BYTES}-byte decoded limit")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} is not valid base64") from exc
    if not raw:
        raise ValueError(f"{field_name} decodes to an empty file")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"{field_name} exceeds the {MAX_IMAGE_BYTES}-byte decoded limit")

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError(f"{field_name} must contain exactly one image frame")
            width, height = source.size
            if width < 2 or height < 2 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"{field_name} dimensions must be at least 2x2 and at most "
                    f"{MAX_IMAGE_PIXELS} pixels"
                )
            image = source.convert("RGBA")
            image.load()
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{field_name} is not a valid supported raster image") from exc

    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox is None or alpha_bbox[2] - alpha_bbox[0] < 2 or alpha_bbox[3] - alpha_bbox[1] < 2:
        raise ValueError(f"{field_name} has no usable nontransparent image area")

    image.save(destination, format="PNG", optimize=False, compress_level=6)


def _bucket_config():
    missing = [name for name in REQUIRED_BUCKET_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "R2 is not configured; missing environment variables: " + ", ".join(missing)
        )
    return {name: os.environ[name] for name in REQUIRED_BUCKET_ENV}


def _safe_job_id(job):
    raw = str(job.get("id") or uuid.uuid4())
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    return (safe or str(uuid.uuid4()))[:128]


def _upload_to_r2(job_id, output_path):
    import boto3
    from botocore.config import Config

    bucket = _bucket_config()
    key = f"outputs/{job_id}.glb"
    client = boto3.client(
        "s3",
        endpoint_url=bucket["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=bucket["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=bucket["BUCKET_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    client.upload_file(
        str(output_path),
        bucket["BUCKET_NAME"],
        key,
        ExtraArgs={"ContentType": "model/gltf-binary"},
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket["BUCKET_NAME"], "Key": key},
        ExpiresIn=PRESIGNED_URL_TTL,
    )


def _geometry_only_glb(mesh):
    import numpy as np
    import trimesh

    if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise RuntimeError("Inference did not return a mesh")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("Inference returned a mesh with no vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("Inference returned a mesh with no triangular faces")
    if not np.isfinite(vertices).all():
        raise RuntimeError("Inference returned non-finite mesh vertices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise RuntimeError("Inference returned out-of-range face indices")

    geometry = trimesh.Trimesh(
        vertices=np.ascontiguousarray(vertices),
        faces=np.ascontiguousarray(faces),
        process=False,
        validate=False,
    )
    glb = geometry.export(file_type="glb")
    if not isinstance(glb, bytes):
        glb = bytes(glb)

    if len(glb) < 20:
        raise RuntimeError("GLB export is empty or truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", glb, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(glb):
        raise RuntimeError("GLB export has an invalid header")
    return glb, len(vertices), len(faces)


def handler(job):
    if INIT_ERROR:
        return {"error": f"Model failed to load:\n{INIT_ERROR}"}

    job_id = _safe_job_id(job if isinstance(job, dict) else {})
    try:
        if not isinstance(job, dict):
            raise ValueError("job must be an object")
        inp = job.get("input")
        if not isinstance(inp, dict):
            raise ValueError("input must be an object")

        _bucket_config()
        seed = _integer_input(inp, "seed", 12345, 0, 2**63 - 1)
        num_inference_steps = _integer_input(
            inp, "num_inference_steps", 30, 1, 100
        )
        octree_resolution = _integer_input(inp, "octree_resolution", 380, 64, 512)
        num_chunks = _integer_input(inp, "num_chunks", 20000, 1, 1_000_000)
        return_base64 = _boolean_input(inp, "return_base64", False)

        for view in REQUIRED_VIEWS:
            field = f"{view}_image_base64"
            if not isinstance(inp.get(field), str) or not inp[field].strip():
                raise ValueError(f"{field} is required")

        with tempfile.TemporaryDirectory(
            prefix=f"{job_id[:48]}-", dir=JOB_TMP_ROOT
        ) as job_dir:
            job_path = Path(job_dir)
            images = {}
            for view in REQUIRED_VIEWS + OPTIONAL_VIEWS:
                field = f"{view}_image_base64"
                value = inp.get(field)
                if view in OPTIONAL_VIEWS and value is None:
                    continue
                image_path = job_path / f"{view}.png"
                _decode_image(value, field, image_path)
                images[view] = str(image_path)

            print(
                f"[job:{job_id}] Generating geometry views={list(images)} seed={seed} "
                f"steps={num_inference_steps} octree={octree_resolution} chunks={num_chunks}",
                flush=True,
            )
            started = time.monotonic()
            import torch

            generator = torch.Generator(device="cpu").manual_seed(seed)
            with _INFERENCE_LOCK, torch.inference_mode():
                meshes = PIPELINE(
                    image=images,
                    num_inference_steps=num_inference_steps,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    generator=generator,
                    mc_algo="mc",
                    output_type="trimesh",
                    enable_pbar=False,
                )

            mesh = meshes[0] if isinstance(meshes, (list, tuple)) and meshes else None
            glb, vertex_count, face_count = _geometry_only_glb(mesh)
            output_path = job_path / "model.glb"
            output_path.write_bytes(glb)
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("Inference produced no output GLB")

            digest = hashlib.sha256(glb).hexdigest()
            model_url = _upload_to_r2(job_id, output_path)
            elapsed = time.monotonic() - started
            print(
                f"[job:{job_id}] Complete in {elapsed:.1f}s: {len(glb)} bytes, "
                f"{vertex_count} vertices, {face_count} faces, sha256={digest}",
                flush=True,
            )

            result = {
                "model_url": model_url,
                "format": "glb",
                "expires_in": PRESIGNED_URL_TTL,
                "sha256": digest,
                "size_bytes": len(glb),
                "vertex_count": vertex_count,
                "face_count": face_count,
                "seed": seed,
            }
            if return_base64:
                result["model_base64"] = base64.b64encode(glb).decode("ascii")
            return result

    except ValueError as exc:
        print(f"[job:{job_id}] Invalid input: {exc}", flush=True)
        return {"error": str(exc)}
    except Exception as exc:
        failure = traceback.format_exc()
        print(f"[job:{job_id}] FAILED:\n{failure}", flush=True)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": failure,
        }
