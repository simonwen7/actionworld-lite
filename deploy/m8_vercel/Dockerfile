# ActionWorld Lite M8 — Vercel Container Runtime image (Hobby, CPU only).
# Built only from the allowlisted staging directory produced by stage.py.
# The native Vercel Python runtime has no libOSMesa, which headless MuJoCo needs.
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    MESA_SHADER_CACHE_DISABLE=true \
    PYTORCH_ENABLE_MPS_FALLBACK=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Same PyTorch 2.14.1 release, CPU-only build; the PyPI Linux wheel pulls CUDA libraries.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu \
    && pip install --no-cache-dir -r requirements.txt uvicorn==0.54.0 \
    && python -c "import torch, mujoco, numpy, fastapi, sys; \
assert sys.version_info[:2] == (3, 12), sys.version; \
assert torch.__version__ == '2.14.1+cpu', torch.__version__; \
assert mujoco.__version__ == '3.17.0', mujoco.__version__; \
assert numpy.__version__ == '2.5.4', numpy.__version__; \
assert fastapi.__version__ == '0.117.1', fastapi.__version__; \
print('M8 PINNED RUNTIME', sys.version.split()[0], torch.__version__, mujoco.__version__, numpy.__version__, fastapi.__version__)"

COPY . .

RUN python -c "from m8_web import model_adapter; print('M8 FROZEN ARTIFACTS VERIFIED', model_adapter.assert_frozen_artifacts())"

EXPOSE 80
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
