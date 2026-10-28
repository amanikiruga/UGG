"""UGG inference: one video in, per-frame RGB, depth and cameras out.

    from ugg import UGG
    model = UGG.from_pretrained()                      # release weights, protocol defaults
    pred = model("clip.mp4")                           # dense: all frames observed
    pred = model("clip.mp4", mode="sparse", seed=1)    # only first + last frame observed
    pred.save("out/")

Defaults live in configurations/inference.yaml and configurations/algorithm/ugg.yaml and are
changed with Hydra overrides: UGG.from_pretrained(overrides=["algorithm.sample_steps=20"]).
"""
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import torch.nn.functional as F

from .cameras import Cameras, encode_cameras, recover_cameras
from .depth import depth_from_norm, encode_depth
from .io import center_crop_resize, legal_num_frames, prepare_video

REPO = Path(__file__).resolve().parents[1]
_UNSET = object()  # percentile_clip=None is meaningful (no clip), so it needs its own sentinel
RELEASE_CKPT = "/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/final_release_checkpoints/spdepth_2250_NVS25.7263_BEST.ckpt"


def compose_config(overrides: Sequence[str] = ()) -> DictConfig:
    """The resolved inference config (configurations/inference.yaml + overrides)."""
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):
        cfg = compose(config_name="inference", overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Prediction:
    rgb: np.ndarray            # (T, H, W, 3) uint8, generated/reconstructed frames
    depth_norm: np.ndarray     # (T, H, W) raw depth channel: normalized disparity in [-1, 1]
    depth: np.ndarray          # (T, H, W) relative depth (up to scale), from depth_norm
    c2w: np.ndarray            # (T, 4, 4) camera-to-world, first camera at the origin
    intrinsics: np.ndarray     # (T, 3, 3) pixel intrinsics at H x W
    raymaps: np.ndarray        # (T, 6, H, W) predicted Pluecker rays [direction | moment]
    frame_indices: np.ndarray  # (T,) indices into the input video
    meta: dict = field(default_factory=dict)

    def save(self, out_dir, fps: int = 10) -> Path:
        import imageio.v3 as iio
        import matplotlib

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k in ("depth_norm", "depth", "c2w", "intrinsics", "frame_indices"):
            np.save(out / f"{k}.npy", getattr(self, k))
        with open(out / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)
        iio.imwrite(out / "rgb.mp4", self.rgb, fps=fps)
        disp = (self.depth_norm + 1) / 2
        lo, hi = np.percentile(disp, [2, 98])
        cmap = matplotlib.colormaps["Spectral_r"]
        vis = (cmap(np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)
        iio.imwrite(out / "depth.mp4", vis, fps=fps)
        return out


class UGG:
    def __init__(self, model, cfg: DictConfig):
        self.model = model
        self.cfg = cfg

    @property
    def device(self):
        return self.model.device

    @classmethod
    def from_pretrained(cls, ckpt: Optional[str] = None, device: Optional[str] = None,
                        overrides: Sequence[str] = (), cfg: Optional[DictConfig] = None) -> "UGG":
        """Build the model and load weights (release checkpoint unless `ckpt` is given)."""
        from algorithms.wan import WanTextToVideoRayDepthMoTConcat5B

        if cfg is None:
            overrides = list(overrides)
            if ckpt is not None:
                overrides.append(f"algorithm.model.tuned_ckpt_path='{ckpt}'")
            cfg = compose_config(overrides)
        model = WanTextToVideoRayDepthMoTConcat5B(cfg.algorithm)
        model.configure_model()
        model = model.eval().to(device or cfg.device)
        if cfg.dit_bf16:
            model.model = model.model.to(torch.bfloat16)
        model.vae_scale = [model.vae_mean, model.vae_inv_std]
        return cls(model, cfg)

    def load_text_encoder(self) -> None:
        """Load UMT5-XXL (bf16) on first use; only needed for prompts or text guidance."""
        if self.model.text_encoder is not None:
            return
        from huggingface_hub import hf_hub_download

        from algorithms.wan.modules.t5 import umt5_xxl

        enc = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16,
                       device=torch.device("cpu")).eval().requires_grad_(False)
        repo_id, filename = os.path.split(self.cfg.algorithm.text_encoder.ckpt_path)
        enc.load_state_dict(torch.load(hf_hub_download(repo_id=repo_id, filename=filename),
                                       map_location="cpu", weights_only=True))
        self.model.text_encoder = enc.to(self.device)

    def _set_num_frames(self, m: int) -> int:
        """Reconfigure the model for an m-frame clip (mirrors the algorithm's __init__)."""
        model = self.model
        per_modality_t = 2 + (m - 2 + model.vae_stride[0] - 1) // model.vae_stride[0]
        model.n_frames = m
        model.lat_t = 4 * per_modality_t
        model.max_tokens = (3 * per_modality_t) * (model.lat_h // model.patch_size[1]) \
            * (model.lat_w // model.patch_size[2])
        return per_modality_t

    # ---- the six tasks -------------------------------------------------------------------
    def video_to_4d(self, video, **kw) -> Prediction:
        """All RGB frames observed -> depth + cameras (training setting 1)."""
        return self.generate(video=video, observe="all", **kw)

    def image_to_4d(self, image, cameras=None, **kw) -> Prediction:
        """One image (+ optional camera trajectory) -> video + depth + cameras (settings 0 / 3)."""
        return self.generate(images=[image], cameras=cameras, **kw)

    def first_last_to_4d(self, first, last, cameras=None, **kw) -> Prediction:
        """First + last image (+ optional cameras) -> in-between video, depth, cameras (setting 2)."""
        return self.generate(images=[first, last], cameras=cameras, **kw)

    def text_to_4d(self, prompt: str, num_frames: int = 50, **kw) -> Prediction:
        """Text only -> video + depth + cameras. Not a trained pattern on its own (closest:
        setting 0 at full history noise); quality is best with a descriptive caption."""
        return self.generate(prompt=prompt, num_frames=num_frames, **kw)

    def camera_to_4d(self, cameras, **kw) -> Prediction:
        """Camera trajectory only (+ optional prompt) -> video + depth. Zero-shot: training
        always paired rays with at least the first RGB frame."""
        return self.generate(cameras=cameras, **kw)

    def depth_to_4d(self, depth, **kw) -> Prediction:
        """Depth maps only (+ optional prompt) -> video + cameras. Zero-shot: training always
        paired depth with the first RGB frame."""
        return self.generate(depth=depth, **kw)

    def __call__(self, video, mode: Optional[str] = None, **kw) -> Prediction:
        """Reconstruction shortcut: mode="dense" (all frames observed) or "sparse" (first + last
        only, the paper's sparse protocol)."""
        mode = mode or self.cfg.mode
        if mode not in ("dense", "sparse"):
            raise ValueError(f"mode must be 'dense' or 'sparse', got {mode!r}")
        return self.generate(video=video, observe="all" if mode == "dense" else "first_last", **kw)

    reconstruct = __call__

    # ---- general entry point -------------------------------------------------------------
    @torch.no_grad()
    def generate(self, video=None, observe: str = "all", images=None, cameras=None, depth=None,
                 prompt: Optional[str] = None, num_frames: Optional[int] = None,
                 seed: Optional[int] = None, sample_steps: Optional[int] = None,
                 hist_guidance: Optional[float] = None, text_guidance: Optional[float] = None,
                 shared_intrinsics: Optional[bool] = None, percentile_clip=_UNSET,
                 max_frames: Optional[int] = None, frame_mode: Optional[str] = None) -> Prediction:
        """Condition on any mix of RGB frames, cameras, depth and text; generate the rest.

        video:   path / (N, H, W, 3) uint8 / (N, 3, H, W) float in [0, 1]; of its frames, those
                 named by `observe` are given to the model: "all", "first" or "first_last".
        images:  instead of a video, one image (frame 0) or two (frames 0 and M-1), each a path,
                 (H, W, 3) uint8 array or (3, H, W) float tensor; M comes from the cameras,
                 depth or `num_frames`.
        cameras: (c2w (T, 4, 4) OpenCV camera-to-world, K (3, 3) or (T, 3, 3)) in pixels of the
                 288x384 model input, or a Prediction / Cameras whose cameras to reuse.
        depth:   (T, H, W) metric or relative depth; <= 0 marks invalid pixels.
        prompt:  optional caption (loads T5 on first use); `text_guidance` > 0 needs one.
        Unset sampling arguments take the config defaults.
        """
        cfg, algo, model = self.cfg, self.cfg.algorithm, self.model
        H, W = model.height, model.width
        prompt = prompt if prompt is not None else cfg.prompt
        seed = cfg.seed if seed is None else seed
        model.sample_steps = algo.sample_steps if sample_steps is None else sample_steps
        model.hist_guidance = algo.hist_guidance if hist_guidance is None else hist_guidance
        model.lang_guidance = algo.lang_guidance if text_guidance is None else text_guidance
        shared_intrinsics = cfg.shared_intrinsics if shared_intrinsics is None else shared_intrinsics
        percentile_clip = cfg.percentile_clip if percentile_clip is _UNSET else percentile_clip
        max_frames = max_frames or cfg.max_frames

        # -- frame count: from the video, the cameras, the depth, or num_frames -------------
        idx = None
        if video is not None and observe == "all":
            frames, idx = prepare_video(video, H, W, max_frames, frame_mode or cfg.frame_mode)
        elif video is not None:
            if observe not in ("first", "first_last"):
                raise ValueError(f"observe must be all | first | first_last, got {observe!r}")
            frames, idx = prepare_video(video, H, W, max_frames, frame_mode or cfg.frame_mode)
        elif images is not None:
            if len(images) not in (1, 2):
                raise ValueError("images: give one (first frame) or two (first and last)")
            observe = "first" if len(images) == 1 else "first_last"
            imgs = torch.from_numpy(_as_clip(*images)).permute(0, 3, 1, 2).float() / 255.0
            imgs = center_crop_resize(imgs, H, W) * 2 - 1
            m = legal_num_frames(_length(cameras, depth, num_frames, cfg.max_frames), max_frames)
            frames = torch.zeros(m, 3, H, W)
            frames[0], frames[-1] = imgs[0], imgs[-1]
        else:
            m = legal_num_frames(_length(cameras, depth, num_frames, cfg.max_frames), max_frames)
            frames = torch.zeros(m, 3, H, W)
        m = len(frames)
        T_lat = self._set_num_frames(m)

        # -- which latent tokens are clean (history): [rgb | ray_d | ray_m | depth] ----------
        hist = []
        if video is not None or images is not None:
            hist += {"all": list(range(T_lat)), "first": [0], "first_last": [0, T_lat - 1]}[observe]
        rh, rw = cfg.dataset.raymap_height, cfg.dataset.raymap_width
        raymaps = torch.zeros(m, 6, rh, rw)
        scale = None
        if cameras is not None:
            c2w, K = _unpack_cameras(cameras)
            if len(c2w) < m:
                raise ValueError(f"need {m} cameras, got {len(c2w)}")
            raymaps, scale = encode_cameras(c2w[:m], K if np.ndim(K) == 2 else K[:m], (H, W), (rh, rw))
            hist += list(range(T_lat, 3 * T_lat))
        depths = torch.zeros(m, 1, H, W)
        if depth is not None:
            d = torch.as_tensor(np.asarray(depth, dtype=np.float32))[:m, None]
            if d.shape[-2:] != (H, W):  # same center crop as the RGB, nearest resize
                d = F.interpolate(center_crop_depth(d, H, W), size=(H, W), mode="nearest")
            depths = torch.from_numpy(encode_depth(d[:, 0].numpy(), scale))[:, None]
            hist += list(range(3 * T_lat, 4 * T_lat))
        if not hist and not prompt:
            raise ValueError("nothing to condition on: give a video, cameras, depth or a prompt")
        if not hist:
            model.hist_guidance = 0.0  # no clean tokens -> history guidance is undefined
        hist_indexes = torch.tensor(sorted(hist), dtype=torch.long)

        # -- text ---------------------------------------------------------------------------
        if model.lang_guidance and not prompt:
            raise ValueError("text_guidance > 0 needs a prompt")
        if prompt:
            self.load_text_encoder()
            prompt_embeds = model.encode_text([prompt])
        else:
            # What the release model was evaluated with: an all-zero text embedding.
            prompt_embeds = [torch.zeros(cfg.dataset.max_text_tokens, 4096, device=self.device)]

        batch = {
            "videos": frames[None].to(self.device),
            "raymaps": raymaps[None].to(self.device),
            "depths": depths[None].to(self.device),
            "prompts": [prompt or ""],
            "prompt_embeds": prompt_embeds,
        }
        seed_everything(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.sample_seq(batch, hist_indexes=hist_indexes, pbar=_pbar(model.sample_steps))
        out = out.squeeze(0).float()                      # (4T, C, H, W): [rgb | ray_d | ray_m | depth]
        rgb, ray_d, ray_m, depth_out = out.split(m)
        pred_rays = torch.cat([ray_d, ray_m], dim=1)
        cams = recover_cameras(pred_rays, shared_intrinsics=shared_intrinsics)
        depth_norm = depth_out.mean(dim=1).cpu().numpy().astype(np.float32)
        given_rgb = video is not None or images is not None
        conditioned = [k for k, on in (("rgb", given_rgb),
                                       ("cameras", cameras is not None), ("depth", depth is not None),
                                       ("text", bool(prompt))) if on]
        return Prediction(
            rgb=((rgb.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy(),
            depth_norm=depth_norm,
            depth=depth_from_norm(depth_norm, percentile_clip),
            c2w=cams.c2w,
            intrinsics=cams.intrinsics,
            raymaps=pred_rays.cpu().numpy(),
            frame_indices=np.arange(m) if idx is None else idx,
            meta=dict(conditioned=conditioned, observe=observe if given_rgb else "none",
                      prompt=prompt, seed=seed, sample_steps=model.sample_steps,
                      hist_guidance=model.hist_guidance, text_guidance=model.lang_guidance,
                      shared_intrinsics=shared_intrinsics, percentile_clip=percentile_clip,
                      ckpt=algo.model.tuned_ckpt_path, height=H, width=W),
        )


def _pbar(n):
    from tqdm import tqdm
    return tqdm(total=n, desc="sampling", leave=False)


def _as_clip(*images):
    """One or two images (path, HxWx3 uint8 array or 3xHxW float tensor) -> a short clip."""
    import imageio.v3 as iio

    out = []
    for im in images:
        if isinstance(im, (str, os.PathLike)):
            im = np.asarray(iio.imread(im))[..., :3]
        if isinstance(im, torch.Tensor):
            im = (im.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        out.append(np.asarray(im))
    return np.stack(out)


def _unpack_cameras(cameras):
    if isinstance(cameras, (Prediction, Cameras)):
        return cameras.c2w, cameras.intrinsics
    c2w, K = cameras
    return np.asarray(c2w), np.asarray(K)


def _length(cameras, depth, num_frames, default):
    if cameras is not None:
        return len(_unpack_cameras(cameras)[0])
    if depth is not None:
        return len(depth)
    return num_frames or default


def center_crop_depth(d, H, W):
    """(T, 1, h, w) depth -> its largest centered crop with the model's aspect ratio."""
    from datasets._crop_utils import apply_spatial_crop, center_crop_box_for_aspect
    return apply_spatial_crop(d, center_crop_box_for_aspect(*d.shape[-2:], H, W))
