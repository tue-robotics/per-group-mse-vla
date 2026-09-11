"""Load and run a checkpoint without ROS, GPSR, or a robot.

Example:
    python inference/local_smoke_test.py \
        --checkpoint /path/to/pretrained_model \
        --device cuda \
        --state-indices 0,1,2,3,4,5
"""

import argparse
from pathlib import Path

import numpy as np

from policy_server import Server


def main():
    parser = argparse.ArgumentParser(description="Local SmolVLA checkpoint smoke test")
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
    print("actions_shape={}".format(actions.shape))
    print("finite={}".format(bool(np.isfinite(actions).all())))
    if not np.isfinite(actions).all():
        raise RuntimeError("Policy returned NaN or Inf actions")


if __name__ == "__main__":
    main()
