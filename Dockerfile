FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HY3D_SOURCE_COMMIT=f8db63096c8282cb27354314d896feba5ba6ff8a \
    HY3D_MODEL_REVISION=3a761b539b29fe4ff64714813aa9560fd66f5de0 \
    HY3D_MODEL_PATH=/models/Hunyuan3D-2mv \
    HY3D_MODEL_SUBFOLDER=hunyuan3d-dit-v2-mv \
    HY3D_SOURCE_PATH=/opt/Hunyuan3D-2 \
    HF_HOME=/models/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN git init "${HY3D_SOURCE_PATH}" \
    && git -C "${HY3D_SOURCE_PATH}" remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git \
    && git -C "${HY3D_SOURCE_PATH}" fetch --depth 1 origin "${HY3D_SOURCE_COMMIT}" \
    && git -C "${HY3D_SOURCE_PATH}" checkout --detach FETCH_HEAD \
    && test "$(git -C "${HY3D_SOURCE_PATH}" rev-parse HEAD)" = "${HY3D_SOURCE_COMMIT}" \
    && python -m pip install --no-cache-dir --no-deps -e "${HY3D_SOURCE_PATH}"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tencent/Hunyuan3D-2mv', revision='${HY3D_MODEL_REVISION}', local_dir='${HY3D_MODEL_PATH}', allow_patterns=['LICENSE', 'NOTICE', 'README.md', '${HY3D_MODEL_SUBFOLDER}/config.yaml', '${HY3D_MODEL_SUBFOLDER}/model.fp16.safetensors'], max_workers=4)" \
    && test -s "${HY3D_MODEL_PATH}/${HY3D_MODEL_SUBFOLDER}/config.yaml" \
    && test "$(stat -c %s "${HY3D_MODEL_PATH}/${HY3D_MODEL_SUBFOLDER}/model.fp16.safetensors")" -gt 1000000000 \
    && mkdir -p /licenses \
    && cp "${HY3D_SOURCE_PATH}/LICENSE" /licenses/Hunyuan3D-2-LICENSE \
    && cp "${HY3D_SOURCE_PATH}/NOTICE" /licenses/Hunyuan3D-2-NOTICE \
    && cp "${HY3D_MODEL_PATH}/LICENSE" /licenses/Hunyuan3D-2mv-LICENSE \
    && cp "${HY3D_MODEL_PATH}/NOTICE" /licenses/Hunyuan3D-2mv-NOTICE \
    && rm -rf "${HF_HOME}" "${HY3D_MODEL_PATH}/.cache"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY handler.py worker.py smoke_test.py /app/

CMD ["python", "-u", "/app/handler.py"]
