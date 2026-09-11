# Inference skeleton

A self-contained WebSocket policy server for SmolVLA, with the
companion smoke test and the Dockerfile we used while iterating on it.
If `docs/INFERENCE.md` is the contract on paper, this directory is the
contract in code.

Worth saying upfront, this is **not** the workshop evaluator. That one
is the counterpart on the robot side, belongs upstream, and is not
redistributed here (see `docs/CONTEXT.md`). What follows is an
implementation written from scratch, kept small enough to read end to
end.

The directory contains four files. `policy_server.py` loads a SmolVLA
checkpoint, reads the normalization statistics directly from the
checkpoint's safetensors (the choice explained in `docs/CONTEXT.md`),
accepts msgpack-encoded observations on a WebSocket, and returns the
predicted action chunk in the 11-DoF HSR layout. `smoke_test.py` is a
synthetic client that sends a zeros observation, checks the response
shape, finiteness and per-joint ranges, and reports per-call latency.
`local_smoke_test.py` loads the checkpoint and performs one direct GPU
inference before a policy server, ROS, or robot is involved.
`Dockerfile` builds a CUDA 12 + Python 3.12 + LeRobot v0.5.1 image
with the SmolVLM processor pre-cached so the server can boot offline.

## Running it locally

In one terminal start the server.

```bash
python inference/policy_server.py \
    --checkpoint /path/to/pretrained_model \
    --host 0.0.0.0 --port 8000
```

In another, smoke-test it.

```bash
python inference/smoke_test.py --url ws://localhost:8000
```

A healthy run prints five `[ok]` lines and a latency summary at the
end. On failure the script exits with status 1 and the line just above
the exit names which assertion fired.

## Running it in Docker

```bash
docker build -f inference/Dockerfile -t smolvla-policy-server .
docker run --rm --gpus all \
    -v /path/to/huggingface-model-cache:/model-cache:ro \
    -e POLICY_CHECKPOINT_PATH=/model-cache/snapshots/<snapshot-id> \
    -p 8000:8000 \
    smolvla-policy-server
```

Mount the complete Hugging Face model cache, rather than its `snapshots`
subdirectory: checkpoint files link into the cache's `blobs` directory.
The container exposes port 8000. Add `--runtime=nvidia` only on hosts that
require it in addition to `--gpus all`.

## Pre-robot checkpoint test

This command evaluates the deployable inference path without ROS, GPSR,
cameras, or robot actuation:

```bash
docker run --rm --gpus all \
    -v /home/amigo/.cache/huggingface/hub/models--PauMontagut--per-group-mse-smolvla:/model-cache:ro \
    smolvla-policy-server \
    python /opt/policy/local_smoke_test.py \
    --checkpoint /model-cache/snapshots/cb72ca6a3a58e724a3ca8579bea3811f1810be96
```

Success prints `PASS` with a finite `actions_shape=(50, 11)`. This verifies
CUDA access, model loading, the checkpoint's processors and normalization,
state selection, image preprocessing, and one action-chunk prediction. It
does not prove that live ROS camera observations, arm calibration, collision
behavior, or task completion will succeed. Run `smoke_test.py` against the
running WebSocket server next to validate the network boundary.

## I/O contract

Documented in full in [`../docs/INFERENCE.md`](../docs/INFERENCE.md).
The request is a dict with `head_rgb`, `hand_rgb`, `state` and
`instruction`. The response is a dict with `actions` of shape
`(T, 11)`. The 11 columns are the HSR action layout, in the order
`arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, gripper,
head_pan, head_tilt, base_x, base_y, base_theta`.

## Wiring this to a real evaluator

The server is agnostic to who is on the other end of the WebSocket. If
your evaluator already speaks the contract above, point it at the
server's host and port and you are done. If it speaks a different one,
`handle()` in `policy_server.py` is around fifteen lines and is the
only place that needs to change.
