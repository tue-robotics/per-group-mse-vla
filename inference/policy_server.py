"""
WebSocket policy server for SmolVLA on the Toyota HSR.

This is the skeleton we used to put a SmolVLA checkpoint behind the kind
of WebSocket contract that mobile-manipulation evaluators tend to expect.
It is NOT the workshop evaluator (that one belongs upstream and is not
redistributed here, see docs/CONTEXT.md). It is a small self-contained
server that you can read end to end in a sitting, deploy in a Docker
container, and bend to your own evaluator if you need to.

The endpoint accepts a JSON-ish payload (msgpack, see below) with the
following keys.

    head_rgb     uint8 (480, 640, 3)   head camera frame
    hand_rgb     uint8 (480, 640, 3)   hand camera frame
    state        float32               robot state; dimensions are selected
                                       from the checkpoint configuration
    instruction  str                   natural-language prompt

It responds with

    actions      float32 (T, 11)       T future actions, 11-DoF layout, robot units
                                       arm(5) + gripper(1) + head(2) + base(3)

The action chunk is what the policy emits in one forward pass, around 50
steps at the configured action horizon. The client typically plays a
prefix and asks for the next chunk before the prefix runs out.

Modern checkpoints use the LeRobot preprocessor and postprocessor pipelines
saved beside the model. This keeps image resizing, empty-camera insertion,
tokenization and normalization aligned with the checkpoint. Older checkpoints
without those pipeline files use the legacy tokenizer/statistics fallback.

Run it with

    python inference/policy_server.py \\
        --checkpoint /path/to/pretrained_model \\
        --host 0.0.0.0 --port 8000

and verify with `inference/smoke_test.py` before connecting any robot.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import msgpack
import numpy as np
import torch
import websockets
from safetensors import safe_open

LOG = logging.getLogger("policy_server")

# Mirrors eval/per_group_mse.py and matches the default dataset layout.
DEFAULT_ACTION_LAYOUT = "hsr11"
ACTION_LAYOUT_TO_DIM = {
    "hsr11": 11,
    "arm6": 6,
}
DEFAULT_STATE_DIM = 8

# Tokenizer's attention_mask comes back as int64 on this code path.
# SmolVLA's eager attention expects bool. See docs/CONTEXT.md ("attention_mask dtype").
ATTENTION_MASK_DTYPE = torch.bool


def load_norm_stats(checkpoint_dir: Path):
    """Load the normalization statistics that the checkpoint shipped with.

    The exact filename varies a bit between LeRobot versions, so we look
    for the usual suspects rather than hard-coding one. Returns a dict
    with float32 numpy arrays.
    """
    candidates = [
        checkpoint_dir / "norm_stats.safetensors",
        checkpoint_dir / "stats.safetensors",
    ]
    safetensors_path = next((p for p in candidates if p.exists()), None)
    if safetensors_path is None:
        raise FileNotFoundError(
            f"No norm_stats.safetensors or stats.safetensors under {checkpoint_dir}. "
            "If the checkpoint stores stats elsewhere, point at it explicitly."
        )

    stats = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            stats[key] = f.get_tensor(key).float().cpu().numpy()
    LOG.info("Loaded %d norm-stats tensors from %s", len(stats), safetensors_path.name)
    return stats


def resolve_action_dim(action_layout: str, action_dim: int = None) -> int:
    if action_dim is not None:
        return int(action_dim)
    if action_layout not in ACTION_LAYOUT_TO_DIM:
        raise ValueError(
            "Unknown action layout '{}'. Expected one of {}".format(
                action_layout, sorted(ACTION_LAYOUT_TO_DIM.keys())
            )
        )
    return ACTION_LAYOUT_TO_DIM[action_layout]


def normalize_state(state: np.ndarray, stats: dict) -> np.ndarray:
    """Apply mean/std normalization to the 8-D state vector.

    Falls back to quantile normalization (q01, q99) if mean/std are not
    in the stats dict, which is the case for some LeRobot versions.
    """
    if "state_mean" in stats and "state_std" in stats:
        return (state - stats["state_mean"]) / (stats["state_std"] + 1e-8)
    if "state_q01" in stats and "state_q99" in stats:
        q01, q99 = stats["state_q01"], stats["state_q99"]
        return 2.0 * (state - q01) / (q99 - q01 + 1e-8) - 1.0
    raise KeyError("Norm stats for the state vector not found in the checkpoint.")


def unnormalize_action(action: np.ndarray, stats: dict, action_dim: int) -> np.ndarray:
    """Undo the same transform on the predicted action chunk."""
    if "action_mean" in stats and "action_std" in stats:
        mean = stats["action_mean"][:action_dim]
        std = stats["action_std"][:action_dim]
        return action * std + mean
    if "action_q01" in stats and "action_q99" in stats:
        q01, q99 = stats["action_q01"][:action_dim], stats["action_q99"][:action_dim]
        return ((action + 1.0) / 2.0) * (q99 - q01) + q01
    raise KeyError("Norm stats for the action vector not found in the checkpoint.")


def _img_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(img).to(device).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def _img_to_frame_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert an HWC RGB observation to the unbatched LeRobot frame format."""
    return torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)


def _feature_dim(features: dict, name: str) -> int:
    feature = features.get(name)
    if feature is None:
        return 0
    shape = feature.shape if hasattr(feature, "shape") else feature.get("shape")
    return int(shape[0])


class Server:
    def __init__(self, checkpoint_dir: Path, device: str = "cuda",
                 action_layout: str = DEFAULT_ACTION_LAYOUT, action_dim: int = None,
                 state_indices=None):
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.device = device
        with open(checkpoint_dir / "config.json") as config_file:
            checkpoint_config = json.load(config_file)

        policy_type = checkpoint_config.get("type")
        if not policy_type:
            raise ValueError("Checkpoint config.json does not declare a policy 'type'")

        policy_cls = get_policy_class(policy_type)
        self.policy = policy_cls.from_pretrained(str(checkpoint_dir))
        self.policy.eval().to(device)
        self.action_layout = action_layout
        input_features = getattr(self.policy.config, "input_features", {})
        output_features = getattr(self.policy.config, "output_features", {})
        self.state_dim = _feature_dim(input_features, "observation.state") or DEFAULT_STATE_DIM
        model_action_dim = _feature_dim(output_features, "action")
        self.action_dim = int(action_dim or model_action_dim or resolve_action_dim(action_layout))
        self.state_indices = None if state_indices is None else tuple(int(i) for i in state_indices)

        self.preprocessor = None
        self.postprocessor = None
        processor_files = (
            checkpoint_dir / "policy_preprocessor.json",
            checkpoint_dir / "policy_postprocessor.json",
        )
        if all(path.exists() for path in processor_files):
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy.config,
                pretrained_path=str(checkpoint_dir),
            )
            LOG.info("Loaded checkpoint-defined LeRobot pre/postprocessors")
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
            self.stats = load_norm_stats(checkpoint_dir)
            LOG.info("Loaded legacy normalization statistics")

        LOG.info("Policy server ready on %s with policy=%s state_dim=%d action_dim=%d",
                 device, policy_type, self.state_dim, self.action_dim)

    def _select_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_indices is not None:
            if len(self.state_indices) != self.state_dim:
                raise ValueError(
                    "state_indices has length {}, checkpoint expects {} values".format(
                        len(self.state_indices), self.state_dim
                    )
                )
            try:
                return state[list(self.state_indices)]
            except IndexError as error:
                raise ValueError("state_indices contains an index outside the robot state") from error
        if state.shape == (self.state_dim,):
            return state
        raise ValueError(
            "Checkpoint expects {} state values but the robot supplied {}. "
            "Configure state_indices for the model layout.".format(self.state_dim, state.size)
        )

    def _infer_with_processors(self, head_rgb, hand_rgb, state, instruction):
        frame = {
            "observation.image.head": _img_to_frame_tensor(head_rgb),
            "observation.image.hand": _img_to_frame_tensor(hand_rgb),
            "observation.state": torch.from_numpy(state),
            "task": instruction,
        }
        batch = self.preprocessor(frame)
        prediction = self.policy.predict_action_chunk(batch)
        actions = self.postprocessor(prediction[0])
        if isinstance(actions, dict):
            actions = actions.get("action")
        if actions is None:
            raise ValueError("Checkpoint postprocessor did not return an action tensor")
        return actions.detach().cpu().numpy()

    @torch.no_grad()
    def infer(self, obs: dict) -> np.ndarray:
        """One forward pass. Returns (T, action_dim) float32 in robot units."""
        head_rgb = np.asarray(obs["head_rgb"], dtype=np.uint8)
        hand_rgb = np.asarray(obs["hand_rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        instruction = str(obs.get("instruction", ""))
        state = self._select_state(state)

        if self.preprocessor is not None:
            return self._infer_with_processors(head_rgb, hand_rgb, state, instruction).astype(np.float32)

        state_norm = normalize_state(state, self.stats).astype(np.float32)
        enc = self.tokenizer(instruction, return_tensors="pt",
                             padding=True, truncation=True)

        batch = {
            "observation.images.camera1": _img_to_tensor(head_rgb, self.device),
            "observation.images.camera2": _img_to_tensor(hand_rgb, self.device),
            "observation.state": torch.from_numpy(state_norm).float()
                                      .to(self.device).unsqueeze(0),
            "input_ids": enc["input_ids"].to(self.device),
            # attention_mask must be bool, not the int64 the tokenizer
            # returns, see docs/CONTEXT.md.
            "attention_mask": enc["attention_mask"].to(self.device).to(ATTENTION_MASK_DTYPE),
        }

        # predict_action_chunk returns (1, T, action_dim). Slicing to
        # self.action_dim is defensive in case the policy was trained with
        # padded actions.
        pred = self.policy.predict_action_chunk(batch)
        chunk_norm = pred[0, :, :self.action_dim].cpu().numpy()
        return unnormalize_action(chunk_norm, self.stats, self.action_dim).astype(np.float32)


async def handle(ws, server: Server):
    LOG.info("Client connected from %s", ws.remote_address)
    async for message in ws:
        t0 = time.perf_counter()
        try:
            obs = msgpack.unpackb(message, raw=False)
            actions = server.infer(obs)
            response = {"actions": actions.tolist()}
            latency_ms = (time.perf_counter() - t0) * 1000
            LOG.info("Inference OK, chunk shape %s, %.1f ms", actions.shape, latency_ms)
            await ws.send(msgpack.packb(response, use_bin_type=True))
        except Exception as e:
            LOG.exception("Inference failed")
            await ws.send(msgpack.packb({"error": str(e)}, use_bin_type=True))


async def serve(checkpoint_dir: Path, host: str, port: int, device: str,
                action_layout: str, action_dim: int = None, state_indices=None):
    server = Server(
        checkpoint_dir,
        device=device,
        action_layout=action_layout,
        action_dim=action_dim,
        state_indices=state_indices,
    )
    # 20 MB cap on incoming payloads. A pair of 480x640x3 uint8 frames is
    # ~1.8 MB raw, msgpack-packed lists are bigger but well below 20 MB.
    async with websockets.serve(lambda ws: handle(ws, server), host, port,
                                max_size=20 * 1024 * 1024):
        LOG.info("Listening on ws://%s:%d", host, port)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="SmolVLA WebSocket policy server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the pretrained_model directory of a SmolVLA checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--action-layout",
        type=str,
        default=DEFAULT_ACTION_LAYOUT,
        choices=sorted(ACTION_LAYOUT_TO_DIM.keys()),
        help="Action layout preset. hsr11 is default, arm6 keeps arm+gripper.",
    )
    parser.add_argument("--action-dim", type=int, default=None,
                        help="Optional explicit action dimension override.")
    parser.add_argument(
        "--state-indices",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated source indices for the checkpoint state vector.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )
    state_indices = [int(value) for value in args.state_indices.split(",") if value]
    asyncio.run(
        serve(
            Path(args.checkpoint),
            args.host,
            args.port,
            args.device,
            args.action_layout,
            args.action_dim,
            state_indices,
        )
    )


if __name__ == "__main__":
    main()
