"""
Task-specific top-up. Continues training from the generalist checkpoint on
a smaller, task-focused dataset.

For the ICRA paper this was the private relocation subset of the AIRoA
ICRA 2026 dataset, distributed only to the competing teams. The top-up
phase is short on purpose. With a small dataset the loss bottoms out
around 3k to 4k steps and starts climbing back up shortly after
(overfitting on the gripper signal). The 3k checkpoint was the one that
won the per-group MSE sweep in our setup. On a different dataset size
you should re-do the curve.

On hardware, the top-up runs use batch 8 with mixed precision so the job
fits in ~4 GB of VRAM, well within the lab's shared-hardware budget. With
AMP off and batch 32 the training is a bit cleaner but takes the whole card.

A typical run looks like

    python training/train_smolvla_task.py \\
        --generalist outputs/smolvla_generalist/checkpoints/020000/pretrained_model \\
        --dataset-root datasets/task6911-v30 \\
        --output-dir outputs/smolvla_topup_task6911 \\
        --steps 5000 --batch-size 8

and with a higher VRAM budget

    python training/train_smolvla_task.py \\
        --generalist .../020000/pretrained_model \\
        --dataset-root datasets/task6911-v30 \\
        --output-dir outputs/smolvla_topup_b32 \\
        --steps 5000 --batch-size 32 --no-amp
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ACTION_LAYOUT_TO_DIM = {
    "hsr11": 11,
    "arm6": 6,
}

# lerobot-train override key that sets the policy action-head width. The exact
# key differs across LeRobot versions, so it is exposed via --action-dim-key.
DEFAULT_ACTION_DIM_KEY = "policy.action_dim"


def build_layout_overrides(
    action_layout: str,
    action_dim: int = None,
    action_dim_key: str = DEFAULT_ACTION_DIM_KEY,
):
    """Return lerobot-train overrides that set the trained action-head width.

    hsr11 (11 DoF) is native and returns no overrides, preserving the
    retrain-from-checkpoint path. Reduced layouts (e.g. arm6) emit an explicit
    action-dim override so the head trains to that width from scratch. A reduced
    width also requires the dataset action to be sliced to match; pass any
    version-specific dataset keys through --policy-override.
    """
    resolved_dim = action_dim if action_dim is not None else ACTION_LAYOUT_TO_DIM[action_layout]
    if action_layout == "hsr11" and resolved_dim == 11:
        return []
    return [f"--{action_dim_key}={resolved_dim}"]


def validate_dataset(root: Path):
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"ERROR: {info_path} not found. Is the dataset in LeRobot v3.0 format?")
    info = json.loads(info_path.read_text())
    print(f"Top-up dataset: {root}")
    print(f"  Episodes: {info.get('total_episodes', 0):,}")
    print(f"  Tasks   : {info.get('total_tasks', 0)}")
    print(f"  FPS     : {info.get('fps', 0)}")
    if info.get("codebase_version") != "v3.0":
        sys.exit(f"ERROR: Expected LeRobot v3.0, got {info.get('codebase_version')}")


def main():
    parser = argparse.ArgumentParser(description="SmolVLA task-specific top-up")
    parser.add_argument("--generalist", type=str, required=True,
                        help="Generalist checkpoint (pretrained_model dir) to start from")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="local/task-dataset")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 is the safest default. Increase only after checking RAM")
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--action-layout", type=str, default="hsr11",
                        choices=sorted(ACTION_LAYOUT_TO_DIM.keys()),
                        help="Action layout preset. hsr11 keeps current behavior.")
    parser.add_argument("--action-dim", type=int, default=None,
                        help="Optional explicit action dimension override.")
    parser.add_argument(
        "--action-dim-key",
        type=str,
        default=DEFAULT_ACTION_DIM_KEY,
        help="lerobot-train key that sets the policy action-head width "
        "(adapt to your LeRobot version).",
    )
    parser.add_argument("--policy-override", action="append", default=[],
                        help="Extra raw lerobot-train overrides, repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    validate_dataset(dataset_root)

    if not Path(args.generalist).exists():
        sys.exit(f"ERROR: generalist checkpoint not found at {args.generalist}")

    cmd = [
        "lerobot-train",
        f"--policy.path={args.generalist}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--output_dir={args.output_dir}",
        f"--job_name={Path(args.output_dir).name}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        '--rename_map={"observation.image.head": "observation.images.camera1", '
        '"observation.image.hand": "observation.images.camera2"}',
        "--policy.empty_cameras=1",
        "--dataset.video_backend=pyav",
        f"--seed={args.seed}",
        f"--num_workers={args.num_workers}",
        "--save_checkpoint=true",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
    ]
    if not args.no_amp:
        cmd.append("--policy.use_amp=true")

    cmd.extend(
        build_layout_overrides(args.action_layout, args.action_dim, args.action_dim_key)
    )
    cmd.extend(args.policy_override)

    if args.dry_run:
        print("\nWould run:\n " + " \\\n  ".join(cmd))
        return

    print(f"\nTop-up from {args.generalist}")
    print(f"  Steps: {args.steps}, batch: {args.batch_size}, amp={not args.no_amp}")
    print(f"  Output: {args.output_dir}\n")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
