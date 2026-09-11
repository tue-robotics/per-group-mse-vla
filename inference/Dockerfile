# Minimal CUDA image to run the SmolVLA policy server.
#
# This is a starting point, not a hardened production image. It is the
# shape of container we used while iterating on the policy server, with
# the workarounds that bit us during deployment baked in as build steps.
#
# Build with
#
#   docker build -f inference/Dockerfile -t smolvla-policy-server .
#
# Run with --runtime=nvidia (or with the nvidia-container-runtime daemon
# set as default), otherwise the container will not see the GPU even
# though the host has one.
#
#   docker run --rm --runtime=nvidia --gpus all \
#       -v /path/to/checkpoint:/checkpoint \
#       -p 8000:8000 \
#       smolvla-policy-server
#
# Python 3.12 is required by LeRobot v0.5.1. The CUDA-enabled PyTorch wheel
# supplies the model runtime; the host NVIDIA driver is exposed at run time.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HOME=/root/.cache/huggingface

# Install the CUDA-enabled PyTorch runtime first, then LeRobot and the small
# set of WebSocket server dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "torch==2.7.*" --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install \
        --no-deps \
        "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.5.1.tar.gz"

RUN python -m pip install \
    accelerate av datasets deepdiff diffusers draccus einops gymnasium \
        huggingface-hub imageio jsonlines mergedeep num2words \
        opencv-python-headless packaging pyserial pyyaml \
        safetensors "transformers==5.3.0" websockets msgpack \
    && python -m pip install \
        "torchvision==0.22.*" --index-url https://download.pytorch.org/whl/cu118

# Pre-cache the SmolVLM-2 processor so the server can boot offline. This
# avoids a first-call stall (and the occasional HF rate-limit) when the
# container starts inside an evaluator without internet.
RUN python -c "__import__('transformers').AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')"

WORKDIR /opt/policy
COPY inference/policy_server.py /opt/policy/policy_server.py

# The server expects the checkpoint to be mounted at /checkpoint by the
# operator. The startup command reads $POLICY_CHECKPOINT_PATH so a
# different mount point can be used without rebuilding the image.
ENV POLICY_CHECKPOINT_PATH=/checkpoint \
    POLICY_STATE_INDICES=0,1,2,3,4,5
EXPOSE 8000

CMD ["sh", "-c", "python policy_server.py --checkpoint ${POLICY_CHECKPOINT_PATH} --host 0.0.0.0 --port 8000 --state-indices ${POLICY_STATE_INDICES}"]
