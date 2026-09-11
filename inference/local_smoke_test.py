"""Validate one GPU VLA inference without ROS, GPSR, or a robot.

Run inside the policy image after mounting the complete Hugging Face model
cache (the snapshot files link to its ``blobs`` directory):

    docker run --rm --gpus all \
        -v /home/amigo/.cache/huggingface/hub/models--PauMontagut--per-group-mse-smolvla:/model-cache:ro \
        smolvla-policy-server \
        python /opt/policy/local_smoke_test.py \
        --checkpoint /model-cache/snapshots/cb72ca6a3a58e724a3ca8579bea3811f1810be96

A successful run prints ``PASS`` after loading the checkpoint and producing a
finite ``(T, 11)`` action chunk. This proves that CUDA, LeRobot, checkpoint
processors, state selection, image preprocessing, and model inference work
together. It does not validate live camera topics, robot calibration, motion,
or task completion.
"""

import argparse
from pathlib import Path

import numpy as np

from policy_server import Server


EXPECTED_ACTION_DIM = 11


def main():
    parser = argparse.ArgumentParser(description="Pre-robot SmolVLA GPU inference test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-layout", default="hsr11")
    parser.add_argument("--state-indices", default="0,1,2,3,4,5")
    args = parser.parse_args()

    state_indices = [int(value) for value in args.state_indices.split(",") if value]
    server = Server(
        checkpoint_dir=Path(args.checkpoint),
        device=args.device,
        action_layout=args.action_layout,
        state_indices=state_indices,
    )
    observation = {
        "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "hand_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
        "instruction": "pick up the mug",
    }
    actions = server.infer(observation)
    finite = bool(np.isfinite(actions).all())
    expected_shape = actions.ndim == 2 and actions.shape[1] == EXPECTED_ACTION_DIM
    max_abs_action = float(np.abs(actions).max())

    print("actions_shape={}".format(actions.shape))
    print("finite={}".format(finite))
    print("max_abs_action={:.6f}".format(max_abs_action))

    if not expected_shape:
        raise RuntimeError(
            "Expected an action chunk with {} columns, got {}".format(
                EXPECTED_ACTION_DIM, actions.shape
            )
        )
    if not finite:
        raise RuntimeError("Policy returned NaN or Inf actions")
    print("PASS: checkpoint inference is ready for the WebSocket policy server")


if __name__ == "__main__":
    main()
