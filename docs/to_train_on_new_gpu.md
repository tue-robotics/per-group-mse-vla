# Training on a new GPU

Working notes from porting the SmolVLA training scripts in this repo off the
lab's RTX 3090 onto other hardware (a Jetson AGX Orin 64GB and a laptop RTX
3050). General enough to reuse for whatever GPU you try next.

## The GPUs, and the reasoning

| GPU | Arch / compute cap. | VRAM | Memory bandwidth | PyTorch install | Role |
|---|---|---|---|---|---|
| RTX 3090 (paper's GPU) | Ampere, 8.6 | 24 GB dedicated | ~936 GB/s | standard PyPI wheel | reference, full-scale runs |
| Laptop RTX 3050 | Ampere, 8.6 | 4-6 GB dedicated (check yours) | ~192-224 GB/s | standard PyPI wheel | **start here** |
| Jetson AGX Orin 64GB | Ampere, **8.7** | 64 GB unified (shared w/ CPU) | ~200 GB/s | Jetson-specific wheel, not on standard PyPI | on-device validation, later real deployment target |

**Why start on the RTX 3050 laptop, not the Jetson:**

- Same compute capability (8.6) as the 3090 the whole repo was built and
  documented against. No architecture surprises.
- It's a normal x86_64 + discrete NVIDIA CUDA stack. `pip install torch`
  from the standard index just works, matched to your driver's CUDA version.
  This is exactly the setup [`docs/SETUP.md`](SETUP.md) already assumes.
- The Jetson (aarch64, Tegra, JetPack 7.2 / CUDA 13.2 at the time of writing)
  has **no confirmed prebuilt PyTorch wheel yet** on the community index
  (`pypi.jetson-ai-lab.io` only publishes up to `jp6`, and Orin's compute
  capability 8.7 differs from the `sbsa` builds meant for server-grade ARM
  + Hopper/Blackwell GPUs). That's a real, open-ended risk to debug through,
  independent of whether the training code itself is correct.
- Low VRAM (4-6 GB) is a real constraint, but the fine-tuning recipe here
  freezes the vision-language backbone and only trains the action head (see
  the README's pipeline figure), so the optimizer only holds state for a
  small parameter subset. Batch size 1-2 with AMP should fit and is enough
  to prove the training loop, data loading, and checkpointing all work.

**Recommended order:**

1. **RTX 3050 laptop** — get the environment and the training script
   mechanics working end to end with a tiny smoke run. Fast iteration loop,
   standard tooling, cheapest place to catch bugs.
2. **Jetson AGX Orin 64GB** — once the code path is proven, port it here to
   validate it also runs on the actual edge target, dealing with the
   aarch64/JetPack wheel problem in isolation from any code bugs.
3. **A bigger GPU (3090 or cloud A100/H100/etc.)** — once both of the above
   work, move real, full-length training runs (the 20k-step generalist
   phase, multi-thousand-step top-up sweeps) to a GPU with real throughput.
   Neither the laptop nor the Jetson is meant to run the full schedule; both
   are for validating the pipeline.

## General setup steps (portable to any CUDA GPU)

### 1. Identify the platform

```bash
uname -m                 # x86_64 (normal desktop/laptop/cloud) vs aarch64 (Jetson)
nvidia-smi                # driver version, CUDA version, GPU name
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

- `x86_64` + a normal NVIDIA driver → standard route (section 3a).
- `aarch64` on a Jetson → special wheel route (section 3b), see
  `docs/SETUP.md` is not enough by itself here.

### 2. Isolated environment, never a global install

Always a fresh venv per machine. Don't `pip install` into the system Python,
especially on any shared box.

```bash
python3.12 -m venv ~/venvs/vla
source ~/venvs/vla/bin/activate
pip install --upgrade pip
```

LeRobot v0.5.1 requires Python 3.12. Check `python3.12 --version` first; if
it's missing, install it via `pyenv` (user-local, no sudo) rather than a
system package manager on a shared machine.

### 3a. Standard x86_64 + discrete NVIDIA GPU (laptop, workstation, cloud)

Match the torch build to your driver's supported CUDA version (`nvidia-smi`
top-right corner shows the max CUDA the driver supports).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or cu124/cu126, whatever matches
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `cuda.is_available()` is `False`, the wheel doesn't match your driver.
Try a different `cuXXX` tag before going further.

### 3b. Jetson (aarch64, JetPack/L4T)

```bash
cat /etc/nv_tegra_release          # L4T / JetPack version
jtop 2>/dev/null || jetson_release  # if jetson-stats is installed
```

Standard PyPI torch wheels are CPU-only on aarch64. Check
[pypi.jetson-ai-lab.io](https://pypi.jetson-ai-lab.io/) for a tag matching
your JetPack (`jpN/cuXXX`), install and verify the same way as above. If no
tag matches your JetPack version, options in order of effort: try the
closest `jpN` tag anyway and verify empirically, use an NVIDIA NGC
`l4t-pytorch`/`l4t-jetpack` container or `dusty-nv/jetson-containers`, or
build PyTorch from source with `TORCH_CUDA_ARCH_LIST` set to your GPU's
compute capability (`8.7` for Orin).

### 4. Install LeRobot with the SmolVLA extras

```bash
pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot@v0.5.1"
python -c "import torch; print(torch.cuda.is_available())"   # re-check, this step can silently swap torch
```

If this step replaced your GPU-matched torch build with a generic one,
reinstall the correct one last with `--no-deps` to force it back.

### 5. Verify the install

```bash
lerobot-info
lerobot-train --help | head -40
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy"
```

### 6. Get a small slice of data

Full AIRoA MoMa 5k is ~4 TB (gated dataset, HF account + accepted terms
required). For validating a pipeline, only pull a handful of episodes:

```bash
pip install -U huggingface_hub
hf auth login   # paste your own token
```

Accept the terms once at
https://huggingface.co/datasets/airoa-org/airoa-moma-5k, then:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("airoa-org/airoa-moma-5k", episodes=list(range(50)))
print(ds.root)   # local path, pass this as --dataset-root
```

### 7. Dry-run before actually launching

Both launchers in `training/` accept `--dry-run`, which prints the resolved
`lerobot-train` command without running it. Always do this first on new
hardware, it catches path/flag mistakes for free.

```bash
python training/train_smolvla_generalist.py \
    --dataset-root <ds.root> \
    --output-dir outputs/smoketest \
    --steps 50 --batch-size 2 --dry-run
```

### 8. Pick batch size / AMP by VRAM tier

| VRAM | `--batch-size` | AMP |
|---|---|---|
| 4-6 GB (RTX 3050 laptop) | 1-2 | on (default, don't pass `--no-amp`) |
| 8-12 GB (RTX 3060/3070/4070-class) | 4-8 | on |
| 24 GB (RTX 3090/4090) | 8-32 | on for 8, `--no-amp` fine for lower step-count 32 runs |
| 64 GB unified (Jetson Orin) | 8-16 | on — memory isn't the limit here, compute is; expect much slower wall-clock per step than the table implies |

If you hit a CUDA OOM, drop batch size before touching anything else.

### 9. Smoke test with a low `--save-freq`

Default `--save-freq` values (5000 for generalist, 1000 for top-up) mean a
short smoke run may never write a checkpoint. Override it so it's less than
or equal to `--steps`:

```bash
python training/train_smolvla_generalist.py \
    --dataset-root <ds.root> \
    --output-dir outputs/smoketest_generalist \
    --steps 50 --batch-size 2 --save-freq 25

python training/train_smolvla_task.py \
    --generalist outputs/smoketest_generalist/checkpoints/000025/pretrained_model \
    --dataset-root <ds.root> \
    --output-dir outputs/smoketest_topup \
    --steps 20 --batch-size 2 --save-freq 10
```

### 10. Evaluate before trusting anything

```bash
python eval/per_group_mse.py \
    --checkpoint outputs/smoketest_topup/checkpoints/000010/pretrained_model \
    --dataset-root <ds.root> \
    --n-samples 40
```

## Gotchas that show up regardless of GPU

- **`num2words` hidden dependency.** The SmolVLM processor needs it; some
  installs don't pull it transitively. `pip install num2words` if imports fail.
- **`attention_mask` dtype.** Cast to `.bool()` if you hit a cryptic shape
  error from the eager attention path.
- **Episode 3997** is corrupt in some AIRoA MoMa snapshots; excluded by
  `scripts/find_corrupt_episodes.py` if you re-run the check on a new
  snapshot.
- **`ffmpeg` missing at the OS level** doesn't necessarily block you — `av`
  wheels usually bundle their own codecs. Only chase installing system
  `ffmpeg` if `pip install` of the `av` package itself fails to find a wheel.
- **Docker.** The provided `inference/Dockerfile` is x86_64 only (`nvidia/cuda`
  base image); it will not run on a Jetson. For any of the setups above, a
  plain venv is simpler and sufficient — only build a GPU-specific Docker
  image if you need strict isolation between multiple users on a shared box.
