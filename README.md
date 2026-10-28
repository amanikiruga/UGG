<div align="center">

# UGG

One video diffusion model that predicts **RGB, depth and camera rays** for every frame of a clip.

</div>

UGG fine-tunes Wan2.2-TI2V-5B into a mixture-of-transformers with three experts:
RGB, camera rays (Plücker direction and moment) and depth. They share self-attention over the
concatenated sequence `[rgb | ray | depth]`. Training uses diffusion forcing: any subset of
frames and modalities can be clean (given), and the model generates the rest. One checkpoint
therefore handles:

video → 4D, image (+ cameras) → 4D, first/last frame (+ cameras) → 4D, text → 4D,
cameras → 4D and depth → 4D (see [the six tasks](#the-six-tasks)).

The code layout follows [Large Video Planner](https://github.com/buoyancy99/large-video-planner):
Hydra configs in `configurations/`, Lightning modules in `algorithms/`, data in `datasets/`,
and the training loop in `experiments/`, all launched by `main.py`.

```
UGG/
├── ugg/                  inference API: UGG.from_pretrained(), the six tasks, camera + depth coding, video IO
├── assets/               sample clip (Bonn balloon2) + cameras/depth for it
├── infer.py              command-line inference (Hydra)
├── main.py               training entry point (Hydra + Lightning), as in LVP
├── algorithms/wan/       the model (wan_t2v_ray_depth_mot_concat_5b.py) and Wan modules
├── datasets/             the 14 training datasets, the mixture, and a dummy dataset
├── experiments/          Lightning trainer / FSDP setup
├── configurations/
│   ├── algorithm/ugg.yaml        release model: weights path, sampling defaults, training recipe
│   ├── algorithm/ugg_toy.yaml    2-block debug model
│   ├── experiment/ugg_train.yaml release training recipe (optimizer, batch, checkpointing)
│   ├── dataset/mixture_parallax14.yaml   release training mixture
│   └── inference.yaml            inference defaults (mode, seed, cameras, depth decode)
├── scripts/train_release.sbatch  multi-node launcher
└── tests/
```

# Checkpoints

| | path |
|---|---|
| Release weights (28 GB, bf16) | `/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/final_release_checkpoints/spdepth_2250_NVS25.7263_BEST.ckpt` |
| Wan2.2-TI2V-5B (DiT config, VAE, UMT5-XXL) | Hugging Face `Wan-AI/Wan2.2-TI2V-5B`, cached under `$HF_HOME` |

The release path is set in [configurations/algorithm/ugg.yaml](configurations/algorithm/ugg.yaml)
(`model.tuned_ckpt_path`); override it with `algorithm.model.tuned_ckpt_path=...`. To fetch the
base model on a new machine:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --include "config.json" "Wan2.2_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/*"
```

# Environment setup

```bash
conda create -n ugg python=3.12 && conda activate ugg
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation     # optional; FA3 is picked up if importable
```

```bash
export HF_HOME=/path/to/hf_cache   # where the Wan2.2 files are cached
export HF_HUB_OFFLINE=1            # optional: for compute nodes without internet
```

Inference needs one 80 GB GPU (about 40 GB peak; add 11 GB for T5 when using prompts).

# Inference

### The six tasks

Every task is the same model with a different set of clean (given) inputs; the rest is generated.

| task | given | generated | Python | trained setting |
|---|---|---|---|---|
| video → 4D | all frames | depth, cameras | `model.video_to_4d(video)` | yes |
| image + cameras → 4D | first frame, cameras | frames, depth | `model.image_to_4d(img, cameras=...)` | yes |
| first/last + cameras → 4D | first + last frame, cameras | frames, depth | `model.first_last_to_4d(a, b, cameras=...)` | yes |
| text → 4D | caption | frames, depth, cameras | `model.text_to_4d(prompt)` | approximately* |
| cameras → 4D | cameras (+ caption) | frames, depth | `model.camera_to_4d(cameras)` | zero-shot |
| depth → 4D | depth (+ caption) | frames, cameras | `model.depth_to_4d(depth)` | zero-shot |

Also available: `image_to_4d(img)` (image only) and `model(video, mode="sparse")` (first + last
frame, no cameras: the paper's sparse protocol). Training drew from five settings: first frame;
all frames; first + last frame + cameras; first frame + cameras; first frame + depth. Their history
tokens were themselves partially noised, so *text-only matches a fully-noised first frame. Cameras-
or depth-only without any frame was never trained and is zero-shot.

`model.generate(video=... | images=[...], observe="all"|"first"|"first_last", cameras=..., depth=..., prompt=...)`
takes any combination directly.

### Sample inputs

`assets/` holds a 50-frame clip from the Bonn RGB-D Dynamic dataset (`rgbd_bonn_balloon2`), its
first and last frames, and `bonn_balloon2_geometry.npz` with cameras (`c2w`, `intrinsics` at
288×384) and depth for the same frames, estimated by UGG from the video, for the camera and depth tasks.

### Python

```python
import numpy as np
from ugg import UGG

model = UGG.from_pretrained()                      # release weights + protocol defaults
geo = np.load("assets/bonn_balloon2_geometry.npz")
cams = (geo["c2w"], geo["intrinsics"])             # c2w (T,4,4) OpenCV, K (3,3) or (T,3,3) at 288x384

pred = model.video_to_4d("assets/bonn_balloon2.mp4")
pred = model.image_to_4d("assets/bonn_balloon2_first.png", cameras=cams)
pred = model.first_last_to_4d("assets/bonn_balloon2_first.png", "assets/bonn_balloon2_last.png", cameras=cams)
pred = model.text_to_4d("a person walks through an office holding balloons", num_frames=50)
pred = model.camera_to_4d(cams, prompt="an office")
pred = model.depth_to_4d(geo["depth"])

pred.rgb            # (T, H, W, 3) uint8
pred.depth          # (T, H, W) relative depth (up to scale)
pred.depth_norm     # (T, H, W) raw output: normalized disparity in [-1, 1]
pred.c2w            # (T, 4, 4) camera-to-world (translation up to scale)
pred.intrinsics     # (T, 3, 3)
pred.save("out/")   # rgb.mp4, depth.mp4, *.npy, meta.json
```

Any call takes `seed`, `sample_steps`, `hist_guidance`, `text_guidance`, `prompt`,
`shared_intrinsics`, `percentile_clip` and `max_frames`. Unset arguments take the config
defaults. Cameras from one prediction can drive another (`cameras=pred`). To change defaults
once, pass Hydra overrides:

```python
model = UGG.from_pretrained(overrides=["algorithm.sample_steps=20", "seed=42", "percentile_clip=10"])
model = UGG.from_pretrained(ckpt="/path/to/other.ckpt")
```

### Command line

```bash
python infer.py task=video      input=assets/bonn_balloon2.mp4 output_dir=out/video
python infer.py task=video      input=assets/bonn_balloon2.mp4 mode=sparse seed=1
python infer.py task=image      input=assets/bonn_balloon2_first.png cameras=assets/bonn_balloon2_geometry.npz
python infer.py task=first_last input=assets/bonn_balloon2_first.png last=assets/bonn_balloon2_last.png \
                                cameras=assets/bonn_balloon2_geometry.npz
python infer.py task=text       prompt="a person walks through an office holding balloons" algorithm.lang_guidance=2.0
python infer.py task=camera     cameras=assets/bonn_balloon2_geometry.npz prompt="an office"
python infer.py task=depth      depth=assets/bonn_balloon2_geometry.npz
python infer.py task=video      input=my_clip.mp4 algorithm.sample_steps=20 algorithm.hist_guidance=2.0
```

### Defaults and what they mean

| key | default | |
|---|---|---|
| `task` | `video` | CLI only: which inputs are given (table above). |
| `mode` | `dense` | `task=video` only. `dense`: all RGB frames are history. `sparse`: frames 0 and T-1 only. |
| `num_frames` | 50 | clip length when no video or cameras fix it (text, image alone). |
| `algorithm.sample_steps` | 40 | UniPC steps (shift 3.0). 10 steps costs little on outdoor depth. |
| `algorithm.hist_guidance` | 1.0 | guidance on the clean frames: `f·(1+h) − h·f_no_history`. 0 skips the extra forward pass. |
| `algorithm.lang_guidance` | 0.0 | text classifier-free guidance; needs `prompt`. |
| `prompt` | null | null means an all-zero text embedding, which is what the paper numbers use. |
| `seed` | 0 | the depth/pose protocol seed (the NVS protocol uses 42). |
| `max_frames`, `frame_mode` | 50, `contig_first` | uses the first M frames, where M is the largest value ≤ 50 with M % 4 == 2. `uniform` spreads them over the video instead. |
| `dataset.height`, `dataset.width` | 288, 384 | the input is center-cropped to 3:4 and then resized. |
| `shared_intrinsics` | true | fits one K per clip when recovering cameras from rays (much better rotation). |
| `percentile_clip` | 2.0 | disparity floor for the depth decode; 10 measured better outdoors. |
| `dit_bf16` | true | runs the DiT in bf16. |


# Training

### Smoke test (one GPU, a few minutes)

```bash
python -m main +name=smoke experiment=ugg_train algorithm=ugg_toy dataset=dummy \
  dataset.n_frames=10 dataset.height=160 dataset.width=192 dataset.raymap_height=20 dataset.raymap_width=24 \
  experiment.training.batch_size=1 experiment.training.max_steps=4 wandb.mode=disabled
```

`ugg_toy` is the release model truncated to 2 blocks, and `dummy` produces random tensors in the
training-sample format. Together they run the full path: VAE encode, diffusion-forcing noise,
loss, FSDP, the optimizer step and checkpointing.

### Training data

The mixture reads every dataset from one root, `data_dir` (set in
[configurations/config.yaml](configurations/config.yaml) or on the command line):

```
<data_dir>/
├── tartanair/  tartanair_v2/  spring/  bedlam/  bedlam2/  vkitti2/  mid_air/  mvs_synth/
├── ase/  scenenet/  scenenet_gt_poses/  matrixcity/  omniworld_game/
├── re10k/  dl3dv_colmap/  dl3dv_colmap_prompt_embeds/
└── captions/<dataset>/      precomputed UMT5 caption embeddings (optional per dataset)
```

Each loader's docstring in `datasets/` describes the layout it expects. Symlinks are fine.

### The release recipe

```bash
python -m main +name=ugg_finetune experiment=ugg_train algorithm=ugg dataset=mixture_parallax14 \
  data_dir=/path/to/data experiment.num_nodes=6
# or on SLURM (6 nodes x 4 GPUs, requeue-safe):
REPO=$PWD HF_HOME=... sbatch scripts/train_release.sbatch data_dir=/path/to/data
```

To check that the full model trains on your hardware before a long run (2 GPUs, CPU-offloaded
optimizer, random data, about 15 minutes, mostly weight loading):

```bash
VWM_SKIP_STARTUP_CHECKPOINT=1 python -m main +name=full_smoke experiment=ugg_train algorithm=ugg dataset=dummy \
  dataset.n_frames=10 dataset.height=160 dataset.width=192 dataset.raymap_height=20 dataset.raymap_width=24 \
  experiment.training.batch_size=1 experiment.training.max_steps=2 +experiment.fsdp_cpu_offload=true wandb.mode=disabled
```

| | release fine-tuning |
|---|---|
| init | weights-only warm start (`algorithm.model.tuned_ckpt_path`) |
| trainable | RGB + ray experts (8.78 B). The depth expert (3.78 B) is frozen with `gf_freeze_depth`. |
| objective | flow matching, diffusion forcing `ray_depth_prediction_firstlast`, clean-token loss masked |
| data | `mixture_parallax14`: 14 datasets, 288×384, 50 frames |
| batch | 8 per GPU × 24 H200 = 192 |
| optimizer | AdamW (fp32 state), lr 6e-6 constant after 150 warmup steps, betas (0.9, 0.95), wd 0.05, clip 1.0 |
| memory | FSDP hybrid shard within node, `use_orig_params`, gradient checkpointing 0.93 |
| speed | FA3, compiled VAE encoder, merged FLF chunks: ≈ 8.5 s/step |
| release | step 2250 |


### Training arguments

- `experiment=ugg_train`: [configurations/experiment/ugg_train.yaml](configurations/experiment/ugg_train.yaml):
  LR, batch size, max steps, checkpoint cadence, FSDP flags.
- `algorithm=ugg`: [configurations/algorithm/ugg.yaml](configurations/algorithm/ugg.yaml):
  initial weights, expert freeze, LR schedule, diffusion-forcing settings per dataset.
- `dataset=mixture_parallax14`: [configurations/dataset/mixture_parallax14.yaml](configurations/dataset/mixture_parallax14.yaml):
  the subsets, their sampling weights and `mixture_seed`.
- Resume: `load=<run>/checkpoints/last.ckpt` restores weights, optimizer and step
  (`train_release.sbatch` does this automatically on requeue).
- Logging: set `wandb.entity` in [configurations/config.yaml](configurations/config.yaml), or `wandb.mode=disabled`.

# Tests

```bash
python tests/test_cpu.py                                # CPU: imports, configs, video IO, depth decode, cameras
python tests/gpu_smoke.py                               # GPU: all six tasks on assets/, one model load
```

# Acknowledgements

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template
[repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the
above sentence in `README.md` and the `LICENSE` file to credit the author. The model code builds on
[Wan2.2](https://github.com/Wan-Video/Wan2.2) and follows the layout of
[Large Video Planner](https://github.com/buoyancy99/large-video-planner). The sample clip in
`assets/` is from the Bonn RGB-D Dynamic dataset.
