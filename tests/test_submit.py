import hashlib
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import submit


ENV_VALUES = {
    "RUNPOD_API_KEY": "runpod-key",
    "RUNPOD_ENDPOINT_ID": "endpoint-id",
    "BUCKET_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com",
    "BUCKET_ACCESS_KEY_ID": "r2-key",
    "BUCKET_SECRET_ACCESS_KEY": "r2-secret",
    "BUCKET_NAME": "bucket",
}


class EnvironmentTests(unittest.TestCase):
    def test_requires_mode_600_and_all_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "\n".join(f"{name}={value}" for name, value in ENV_VALUES.items()),
                encoding="utf-8",
            )
            path.chmod(0o600)
            self.assertEqual(submit.load_repo_env(path), ENV_VALUES)
            path.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "mode 600"):
                submit.load_repo_env(path)


class SubmissionTests(unittest.TestCase):
    def test_submission_uses_url_fields_and_fixed_inference_settings(self):
        calls = []

        def fake_request(request, timeout):
            calls.append(request)
            if request.get_method() == "POST":
                return {"id": "job-id"}
            return {"status": "COMPLETED", "output": {"format": "glb"}}

        image_urls = {
            "front": "https://signed/front",
            "left": "https://signed/left",
            "back": "https://signed/back",
        }
        with patch("submit.runpod_request", side_effect=fake_request):
            output = submit.submit_job(ENV_VALUES, image_urls)
        payload = json.loads(calls[0].data)
        inp = payload["input"]
        self.assertEqual(inp["front_image_url"], image_urls["front"])
        self.assertEqual(inp["num_inference_steps"], 30)
        self.assertEqual(inp["octree_resolution"], 380)
        self.assertEqual(inp["num_chunks"], 20000)
        self.assertNotIn("front_image_base64", inp)
        self.assertEqual(output, {"format": "glb"})

    def test_glb_header_size_and_hash_are_verified(self):
        glb = struct.pack("<4sII", b"glTF", 2, 20) + b"\0" * 8
        digest = hashlib.sha256(glb).hexdigest()
        output = {"sha256": digest, "size_bytes": len(glb)}
        self.assertEqual(submit.validate_glb(glb, output), digest)
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            submit.validate_glb(glb, {**output, "sha256": "0" * 64})

    def test_temporary_inputs_are_deleted_after_failure(self):
        deleted = []
        args = SimpleNamespace(
            front=Path("front.png"),
            left=Path("left.png"),
            back=Path("back.png"),
            right=None,
            output=Path("result.glb"),
        )
        with (
            patch("submit.load_repo_env", return_value=ENV_VALUES),
            patch("submit.read_png", return_value=(b"png", "a" * 64)),
            patch("submit.make_r2_client", return_value=object()),
            patch("submit.upload_input"),
            patch("submit.sign_input", return_value="https://signed/input"),
            patch("submit.submit_job", side_effect=RuntimeError("job failed")),
            patch("submit.delete_input", side_effect=lambda client, bucket, key: deleted.append(key)),
        ):
            with self.assertRaisesRegex(RuntimeError, "job failed"):
                submit.run(args)
        self.assertEqual(len(deleted), 3)
        self.assertEqual({key.rsplit("/", 1)[-1] for key in deleted}, {
            "front.png",
            "left.png",
            "back.png",
        })


if __name__ == "__main__":
    unittest.main()
