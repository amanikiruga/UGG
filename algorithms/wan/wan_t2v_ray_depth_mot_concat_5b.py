import logging
import gc
import io
import os
from datetime import timedelta

import torch
import bitsandbytes as bnb
import torch.nn.functional as F
import wandb
import numpy as np
import torch.distributed as dist
from einops import rearrange, repeat
from tqdm import tqdm
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from transformers import get_scheduler
from huggingface_hub import hf_hub_download
from PIL import Image
from utils.training_correctness import (
    collect_all_rank_rng_states,
    masked_flow_mse,
    restore_local_rng_state,
    seed_rank_training_streams,
    select_rank_rng_state,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .modules.model_ray_depth_mot_concat_5b import WanModel, WanAttentionBlock
from .modules.model import WanModel as BaseWanModel
from .modules.t5 import umt5_xxl, T5CrossAttention, T5SelfAttention
from .modules.tokenizers import HuggingfaceTokenizer
from .modules.vae_v2 import video_vae_flf_factory_v2
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.distributed_utils import is_rank_zero
from datasets.re10k import raymap_to_camera


# ---- validation metrics and visualizations ---------------------------------------------

def _depth_to_colormap(depth: torch.Tensor, ref: torch.Tensor = None) -> torch.Tensor:
    """Turbo colormap of depth. depth: [B, T, C, H, W] -> [B, T, 3, H, W] in [0, 1].
    Min-max normalized per sample over the clip; pass `ref` (e.g. GT) to share its range."""
    cmap = plt.get_cmap("turbo")
    B, T = depth.shape[:2]
    d = depth.mean(dim=2, keepdim=True)  # 1-ch GT and 3-ch VAE-decoded predictions alike
    src = ref.mean(dim=2, keepdim=True) if ref is not None else d
    src_flat = src.reshape(B, -1).float()
    d_min = src_flat.amin(dim=1).view(B, 1, 1, 1, 1)
    d_max = src_flat.amax(dim=1).view(B, 1, 1, 1, 1)
    rng = (d_max - d_min).clamp(min=1e-6)
    depth_01 = ((d.float() - d_min) / rng).clamp(0, 1).squeeze(2).cpu().numpy()  # [B, T, H, W]
    colored = cmap(depth_01.reshape(-1, *depth_01.shape[2:]))[:, :, :, :3]  # [B*T, H, W, 3]
    return torch.from_numpy(colored).permute(0, 3, 1, 2).float().view(B, T, 3, *depth.shape[-2:])


def _psnr_batch(pred: torch.Tensor, gt: torch.Tensor, skip_first_last: bool = True) -> float:
    """pred, gt: [B, T, 3, H, W] in [-1,1]. Mean PSNR over the interior (generated) frames."""
    pred_u = (pred * 0.5 + 0.5).clamp(0, 1).float()
    gt_u   = (gt   * 0.5 + 0.5).clamp(0, 1).float()
    if skip_first_last and pred_u.shape[1] > 2:
        pred_u = pred_u[:, 1:-1]
        gt_u   = gt_u[:, 1:-1]
    mse  = ((pred_u - gt_u) ** 2).mean(dim=(2, 3, 4)).clamp(min=1e-10)  # [B, T]
    psnr = -10.0 * torch.log10(mse)
    return float(psnr.mean().item())


def _depth_delta_batch(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> dict:
    """delta1/2/3 after per-sample least-squares scale+shift alignment.

    pred_depth: [B, T, 3, H, W] in [-1,1] (3-ch VAE decoded); gt_depth: [B, T, 1, H, W] in [-1,1].
    """
    B = pred_depth.shape[0]
    pred_u = (pred_depth.mean(dim=2) * 0.5 + 0.5).clamp(0, 1).float()  # [B, T, H, W]
    gt_u   = (gt_depth.squeeze(2)   * 0.5 + 0.5).clamp(0, 1).float()   # [B, T, H, W]
    d1, d2, d3 = [], [], []
    for b in range(B):
        pv = pred_u[b].reshape(-1).cpu()
        gv = gt_u[b].reshape(-1).cpu()
        valid = (gv > 1e-4) & pv.isfinite() & gv.isfinite()
        if valid.sum() < 10:
            continue
        pv, gv = pv[valid], gv[valid]
        if pv.shape[0] > 50000:  # subsample for speed
            idx = torch.randperm(pv.shape[0])[:50000]
            pv, gv = pv[idx], gv[idx]
        A   = torch.stack([pv, torch.ones_like(pv)], dim=1)
        sol = torch.linalg.lstsq(A, gv.unsqueeze(1)).solution
        scale, shift = float(sol[0].item()), float(sol[1].item())
        if scale <= 0:
            continue
        aligned = (scale * pred_u[b] + shift).clamp(min=1e-4)
        gt_b    = gt_u[b].clamp(min=1e-4)
        ratio   = torch.max(aligned / gt_b, gt_b / aligned)
        d1.append(float((ratio < 1.25).float().mean().item()))
        d2.append(float((ratio < 1.25 ** 2).float().mean().item()))
        d3.append(float((ratio < 1.25 ** 3).float().mean().item()))
    if not d1:
        return {"delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    return {"delta1": float(np.mean(d1)), "delta2": float(np.mean(d2)), "delta3": float(np.mean(d3))}


def _compute_relative_pose_errors(w2c_pred: np.ndarray, w2c_gt: np.ndarray) -> tuple:
    """Pairwise relative-pose errors in degrees: (max(rot, trans), rot, trans)."""
    N = len(w2c_pred)
    combined, rots, trans = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            rel_pred = w2c_pred[j] @ np.linalg.inv(w2c_pred[i])
            rel_gt   = w2c_gt[j]   @ np.linalg.inv(w2c_gt[i])
            R_err  = rel_pred[:3, :3] @ rel_gt[:3, :3].T
            cos_a  = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            rot_e  = float(np.degrees(np.arccos(cos_a)))
            tp, tg = rel_pred[:3, 3], rel_gt[:3, 3]
            np_n, ng_n = np.linalg.norm(tp), np.linalg.norm(tg)
            if np_n < 1e-8 or ng_n < 1e-8:
                trans_e = 0.0
            else:
                cos_t   = np.clip(np.dot(tp / np_n, tg / ng_n), -1.0, 1.0)
                trans_e = float(np.degrees(np.arccos(cos_t)))
            combined.append(max(rot_e, trans_e))
            rots.append(rot_e)
            trans.append(trans_e)
    return np.array(combined), np.array(rots), np.array(trans)


def _compute_auc(errors: np.ndarray, max_threshold: float, n_points: int = 1000) -> float:
    thresholds = np.linspace(0.0, max_threshold, n_points)
    return float(np.array([(errors <= t).mean() for t in thresholds]).mean()) * 100.0


def _pose_metrics_batch(pred_ray_d: torch.Tensor, pred_ray_m: torch.Tensor,
                        gt_raymaps: torch.Tensor) -> dict:
    """AUC@3/30, rotation and translation error, averaged over all frame pairs and the batch.

    pred_ray_d, pred_ray_m: [B, T, 3, H, W]; gt_raymaps: [B, T, 6, H, W].
    """
    all_combined, all_rot, all_trans = [], [], []
    for b in range(pred_ray_d.shape[0]):
        pred_rm = torch.cat([pred_ray_d[b], pred_ray_m[b]], dim=1)  # [T, 6, H, W]
        gt_rm   = gt_raymaps[b]                                       # [T, 6, H, W]
        try:
            _, pred_ext = raymap_to_camera(pred_rm.float())
            _, gt_ext   = raymap_to_camera(gt_rm.float())
        except Exception:
            continue
        pred_w2c = np.linalg.inv(pred_ext.cpu().numpy().astype(np.float64))
        gt_w2c   = np.linalg.inv(gt_ext.cpu().numpy().astype(np.float64))
        combined, rots, trans = _compute_relative_pose_errors(pred_w2c, gt_w2c)
        all_combined.append(combined)
        all_rot.append(rots)
        all_trans.append(trans)
    if not all_combined:
        return {"auc3": 0.0, "auc30": 0.0, "rot_err": 0.0, "trans_err": 0.0}
    cat_c = np.concatenate(all_combined)
    cat_r = np.concatenate(all_rot)
    cat_t = np.concatenate(all_trans)
    return {
        "auc3":      _compute_auc(cat_c, max_threshold=3.0),
        "auc30":     _compute_auc(cat_c, max_threshold=30.0),
        "rot_err":   float(cat_r.mean()),
        "trans_err": float(cat_t.mean()),
    }


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align source camera centers to target via optimal rotation + scale (no reflection)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    s_c, t_c = source - mu_s, target - mu_t
    U, _, Vt = np.linalg.svd(t_c.T @ s_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    ss = (s_c ** 2).sum()
    scale = (t_c * (s_c @ R.T)).sum() / ss if ss > 0 else 1.0
    return scale * (source - mu_s) @ R.T + mu_t


def _make_trajectory_figure(pred_raymaps: torch.Tensor, gt_raymaps: torch.Tensor) -> Image.Image:
    """3-D plot of the predicted trajectory (Procrustes-aligned) against GT. Inputs: [T, 6, H, W]."""
    try:
        _, pred_ext = raymap_to_camera(pred_raymaps.float())  # [T, 4, 4]
        _, gt_ext   = raymap_to_camera(gt_raymaps.float())    # [T, 4, 4]
    except Exception as e:
        logging.warning(f"raymap_to_camera failed, skipping trajectory viz: {e}")
        return None

    pred_c = pred_ext[:, :3, 3].cpu().numpy()   # [T, 3]
    gt_c   = gt_ext[:, :3, 3].cpu().numpy()     # [T, 3]
    pred_aligned = _procrustes_align(pred_c, gt_c)

    T_val = len(pred_c)
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / max(T_val - 1, 1)) for i in range(T_val)]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(T_val - 1):
        ax.plot(pred_aligned[i:i+2, 0], pred_aligned[i:i+2, 1], pred_aligned[i:i+2, 2],
                color=colors[i], linewidth=2)
    ax.scatter(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
               c=colors, s=30, zorder=5, label="pred (aligned)")
    ax.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2],
            color="black", linewidth=1.5, linestyle="--")
    ax.scatter(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2],
               color="black", s=20, zorder=4, label="GT")
    ax.legend()
    ax.set_title("Camera Trajectories (pred Procrustes-aligned to GT)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_img


class WanTextToVideoRayDepthMoTConcat5B(BasePytorchAlgo):
    """
    UGG: Wan2.2-TI2V-5B with a mixture of transformers over RGB, Pluecker raymaps and depth.

    Token layout: [rgb | ray_d | ray_m | depth], T latent frames each. The model sees three
    expert streams: RGB, rays (ray_d and ray_m channel-concatenated into one block) and depth.
    Diffusion forcing gives every latent frame its own noise level, so any subset of the
    modalities can be given as clean history and the rest generated.
    """

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        """Clip with FSDP's own clip_grad_norm_ (total norm across shards); Lightning's
        FSDPPrecision refuses trainer-level norm clipping."""
        if gradient_clip_val is None:
            return
        model = getattr(self.trainer.strategy, "model", None)
        if model is not None and hasattr(model, "clip_grad_norm_"):
            norm = model.clip_grad_norm_(gradient_clip_val)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        self._grad_norm_last = float(norm)  # pre-clip global norm, logged in training_step

    def __init__(self, cfg):
        self.num_train_timesteps = cfg.num_train_timesteps
        self.height = cfg.height
        self.width = cfg.width
        self.n_frames = cfg.n_frames
        self.gradient_checkpointing_rate = cfg.gradient_checkpointing_rate
        self.sample_solver = cfg.sample_solver
        self.sample_steps = cfg.sample_steps
        self.sample_shift = cfg.sample_shift
        self.lang_guidance = cfg.lang_guidance
        self.neg_prompt = cfg.neg_prompt
        self.hist_guidance = cfg.hist_guidance
        self.diffusion_forcing = cfg.diffusion_forcing
        self.vae_stride = cfg.vae.stride
        self.patch_size = cfg.model.patch_size

        if not (self.diffusion_forcing.enabled
                and self.diffusion_forcing.mode == "ray_depth_prediction_firstlast"):
            raise ValueError("UGG requires diffusion_forcing.enabled with mode=ray_depth_prediction_firstlast")

        self.lat_h = self.height // self.vae_stride[1]
        self.lat_w = self.width // self.vae_stride[2]
        # First-last-frame VAE: first and last frames get their own latent, the interior is
        # compressed by vae_stride[0]. Four modality blocks of per_modality_t frames each.
        if self.n_frames <= 1:
            per_modality_t = self.n_frames
        else:
            interior = self.n_frames - 2
            per_modality_t = 2 + (interior + self.vae_stride[0] - 1) // self.vae_stride[0]
        self.lat_t = 4 * per_modality_t
        self.lat_c = cfg.vae.z_dim
        self.max_area = self.height * self.width
        # The model fuses ray_d + ray_m into one block, so its sequence spans 3 blocks.
        self.max_tokens = (
            (3 * self.lat_t // 4)
            * (self.lat_h // self.patch_size[1])
            * (self.lat_w // self.patch_size[2])
        )

        self.load_prompt_embed = cfg.load_prompt_embed
        self.load_video_latent = cfg.load_video_latent
        if self.load_video_latent:
            raise NotImplementedError("Loading video latent is not implemented yet")
        super().__init__(cfg)

    @staticmethod
    def classes_to_shard():
        return {WanAttentionBlock, T5CrossAttention, T5SelfAttention}

    @property
    def is_inference(self) -> bool:
        return self._trainer is None or not self.trainer.training

    def _set_pg_timeout(self):
        """Raise the default process-group timeout (cfg.pg_timeout_min, default 180 min).
        A full-state checkpoint gather can exceed NCCL's 10-minute default."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        minutes = int(self.cfg.get("pg_timeout_min", 180))
        dist.distributed_c10d._set_pg_timeout(timedelta(minutes=minutes), dist.group.WORLD)

    def configure_model(self):
        self._set_pg_timeout()  # before checkpoint restore and dataloader setup
        logging.info("Building model...")
        if not self.cfg.load_prompt_embed:
            text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=torch.bfloat16 if self.is_inference else self.dtype,
                    device=torch.device("cpu"),
                )
                .eval()
                .requires_grad_(False)
            )
            if self.cfg.text_encoder.ckpt_path is not None:
                text_encoder_repo_id = os.path.dirname(self.cfg.text_encoder.ckpt_path)
                text_encoder_filename = os.path.basename(self.cfg.text_encoder.ckpt_path)
                text_encoder_model_path = hf_hub_download(
                    repo_id=text_encoder_repo_id, filename=text_encoder_filename
                )
                text_encoder.load_state_dict(
                    torch.load(text_encoder_model_path, map_location="cpu", weights_only=True)
                )
            if self.cfg.text_encoder.compile:
                text_encoder = torch.compile(text_encoder)
        else:
            text_encoder = None
        self.text_encoder = text_encoder

        self.tokenizer = HuggingfaceTokenizer(
            name=self.cfg.text_encoder.name,
            seq_len=self.cfg.text_encoder.text_len,
            clean="whitespace",
        )
        vae_repo_id = os.path.dirname(self.cfg.vae.ckpt_path)
        vae_filename = os.path.basename(self.cfg.vae.ckpt_path)
        vae_model_path = hf_hub_download(repo_id=vae_repo_id, filename=vae_filename)
        self.vae = (
            video_vae_flf_factory_v2(pretrained_path=vae_model_path, z_dim=self.cfg.vae.z_dim)
            .eval()
            .requires_grad_(False)
        ).to(self.dtype)
        self.register_buffer("vae_mean", torch.tensor(self.cfg.vae.mean, dtype=self.dtype))
        self.register_buffer("vae_inv_std", 1.0 / torch.tensor(self.cfg.vae.std, dtype=self.dtype))
        self.vae_scale = [self.vae_mean, self.vae_inv_std]
        # The frozen VAE and text encoder are not saved in checkpoints (see on_save_checkpoint)
        # and are rebuilt from their pretrained files, so checkpoints load non-strictly.
        self.strict_loading = False
        if self.cfg.vae.compile:
            self.vae = torch.compile(self.vae)
        if self.cfg.vae.merge_flf_chunks:
            # Encode the interior 4-frame chunks in one pass; same result, more memory.
            self.vae.merge_flf_chunks = True
        if self.cfg.vae.encoder_compile:
            # The VAE runs through encode(), not forward(), so compile the encoder module itself.
            self.vae.encoder = torch.compile(self.vae.encoder, dynamic=False)

        if self.cfg.model.tuned_ckpt_path is not None and is_rank_zero:
            print(
                "[WARM START] algorithm.model.tuned_ckpt_path loads weights only: fresh optimizer, "
                "LR warmup and global_step. To resume a run use load=<ckpt>.",
                flush=True,
            )

        if self.cfg.model.tuned_ckpt_path is None:
            self.model = WanModel.from_t2v_pretrained(
                self.cfg.model.ckpt_path, local_files_only=True
            )
        else:
            model_repo_id = self.cfg.model.ckpt_path
            config_path = hf_hub_download(repo_id=model_repo_id, filename="config.json")
            loaded_state_dict = self._load_tuned_state_dict()

            is_mot_checkpoint = any(
                k.startswith("blocks.0.self_attn.q.0.") for k in loaded_state_dict
            )

            if is_mot_checkpoint:
                # Build on the meta device and materialize each parameter once from the
                # checkpoint (assign=True), instead of allocating a full fp32 model first.
                with torch.device("meta"):
                    self.model = WanModel.from_config(
                        WanModel._dict_from_json_file(config_path)
                    )
                if self.is_inference:
                    self.model.to(torch.bfloat16)
                self.model.load_state_dict(loaded_state_dict, assign=True, strict=False)
                if "patch_embedding_ray.weight" not in loaded_state_dict:
                    self.model.init_ray_embedding_from_scratch()
                if "patch_embedding_depth.weight" not in loaded_state_dict:
                    self.model.init_depth_embedding_from_rgb()
                # freqs is a plain attribute, left on meta; rebuild it on a real device.
                self.model.freqs = self.model._build_freqs()
            else:
                base_model = BaseWanModel.from_config(WanModel._dict_from_json_file(config_path))
                base_model.load_state_dict(loaded_state_dict, assign=True, strict=False)
                self.model = WanModel.from_config(base_model.config)
                self.model.load_from_t2v_pretrained(base_model)

        n_debug = self.cfg.model.get("debug_num_layers", None)
        if n_debug:
            # Smoke tests: a truncated stack exercises every code path at a fraction of the cost.
            self.model.blocks = self.model.blocks[:n_debug]
            self.model.num_layers = n_debug

        expected_ray_ch = 2 * self.vae_stride[0] * 3
        assert self.model.patch_embedding_ray.in_channels == expected_ray_ch, (
            f"patch_embedding_ray expects {self.model.patch_embedding_ray.in_channels} "
            f"in-channels but the fused ray block has {expected_ray_ch} "
            f"(= 2 * vae_stride[0]({self.vae_stride[0]}) * 3). Set model.ray_in_dim accordingly."
        )

        if not self.is_inference:
            self.model.to(self.dtype).train()
        if self.gradient_checkpointing_rate > 0:
            self.model.gradient_checkpointing_enable(p=self.gradient_checkpointing_rate)
        if self.cfg.model.compile:
            self.model = torch.compile(self.model)

        # Freeze the depth expert and train the RGB/ray experts. Must happen here: FSDP wraps
        # the module after this hook and snapshots requires_grad at wrap time. Needs
        # experiment.fsdp_use_orig_params=true (mixed requires_grad within a block).
        if bool(self.cfg.get("gf_freeze_depth", False)):
            from .fg_expert import is_depth_expert, classify_gf
            names = [n for n, _ in self.model.named_parameters()]
            frozen_names, train_names, contra = classify_gf(names)
            assert not contra, f"GF classifier contradiction on {contra[:5]}"
            assert frozen_names, "GF freeze requested but no depth-expert params matched -- wrong model?"
            n_train = n_froze = 0
            for n, p in self.model.named_parameters():
                if is_depth_expert(n):
                    p.requires_grad_(False)
                    n_froze += p.numel()
                else:
                    n_train += p.numel()
            if is_rank_zero:
                tot = n_train + n_froze
                print(
                    f"[GF FREEZE] frozen depth expert {n_froze/1e9:.3f}B "
                    f"({100*n_froze/tot:.2f}%) across {len(frozen_names)} tensors; "
                    f"trainable rgb/ray/pose {n_train/1e9:.3f}B ({100*n_train/tot:.2f}%) "
                    f"across {len(train_names)} tensors.",
                    flush=True,
                )

        self.training_scheduler, self.training_timesteps = self.build_scheduler(True)

    def configure_optimizers(self):
        # "adamw" (fp32) supports full-state checkpoints. "adamw8bit" saves optimizer memory,
        # but FSDP cannot gather its mixed-dtype state, so it needs save_weights_only=true.
        opt_impl = str(self.cfg.get("optimizer_impl", "adamw")).lower()
        if opt_impl == "adamw8bit":
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.betas,
            )
        elif opt_impl == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.betas,
            )
        else:
            raise ValueError(
                f"optimizer_impl must be 'adamw' or 'adamw8bit', got {opt_impl!r}"
            )
        print(f"[OPTIM] {opt_impl} lr={self.cfg.lr} betas={list(self.cfg.betas)} "
              f"wd={self.cfg.weight_decay}", flush=True)
        scheduler_cfg = dict(self.cfg.lr_scheduler)
        scheduler_name = scheduler_cfg.pop("name")
        if scheduler_name == "warmup_hold_cosine":
            from utils.lr_schedulers import get_warmup_hold_cosine_scheduler
            scheduler = get_warmup_hold_cosine_scheduler(optimizer, **scheduler_cfg)
        else:
            scheduler = get_scheduler(
                name=scheduler_name, optimizer=optimizer, **scheduler_cfg
            )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _load_tuned_state_dict(self, prefix="model."):
        path = self.cfg.model.tuned_ckpt_path
        if not os.path.exists(path):
            # <hf_user>/<hf_repo>/<file> on the Hugging Face Hub
            repo_id, filename = os.path.split(path)
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            full = load_file(path, device="cpu")
        else:
            full = torch.load(path, mmap=True, map_location="cpu", weights_only=False)["state_dict"]
        state_dict = {
            k[len(prefix):]: v
            for k, v in full.items()
            if k.startswith(prefix)
        }
        del full
        gc.collect()
        return state_dict

    def build_scheduler(self, is_training=True):
        if self.sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                scheduler.set_timesteps(self.sample_steps, device=self.device, shift=self.sample_shift)
            timesteps = scheduler.timesteps
        elif self.sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                sampling_sigmas = get_sampling_sigmas(self.sample_steps, self.sample_shift)
                timesteps, _ = retrieve_timesteps(scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return scheduler, timesteps

    def encode_text(self, texts):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

    def encode_video(self, videos):
        """videos: [B, C, T, H, W]"""
        return self.vae.encode(videos, self.vae_scale)

    def decode_video(self, zs):
        return self.vae.decode(zs, self.vae_scale).clamp_(-1, 1)

    def clone_batch(self, batch):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _pack_ray_latent(self, rays):
        """Rays bypass the VAE: resample to the latent grid and stack vae_stride[0] frames per
        latent slot along channels (first/last frame repeated), zero-padded to lat_c channels.

        rays: [B, T, 3, H, W] -> [B, lat_c, T_lat, lat_h, lat_w]
        """
        B, T_in, C, H, W = rays.shape
        if H != self.lat_h or W != self.lat_w:
            rays = F.interpolate(
                rays.reshape(B * T_in, C, H, W),
                size=(self.lat_h, self.lat_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T_in, C, self.lat_h, self.lat_w)
        T_lat = self.lat_t // 4
        stride = self.vae_stride[0]
        r = rays

        slots = [torch.cat([r[:, 0]] * stride, dim=1)]  # first frame, repeated
        for i in range(1, T_lat - 1):  # interior: stride consecutive frames per slot
            start = 1 + (i - 1) * stride
            group = [r[:, min(start + j, T_in - 1)] for j in range(stride)]
            slots.append(torch.cat(group, dim=1))
        slots.append(torch.cat([r[:, -1]] * stride, dim=1))  # last frame, repeated

        packed = torch.stack(slots, dim=2)
        pad = packed.new_zeros(B, self.lat_c - stride * C, T_lat, self.lat_h, self.lat_w)
        return torch.cat([packed, pad], dim=1)

    def _unpack_ray_latent(self, ray_lat, upsample_spatial=True):
        """Inverse of _pack_ray_latent: [B, lat_c, T_lat, lat_h, lat_w] -> [B, 3, n_frames, H, W]."""
        B = ray_lat.shape[0]
        stride = self.vae_stride[0]
        C = 3
        real = ray_lat[:, :stride * C]

        per = real.view(B, stride, C, -1, self.lat_h, self.lat_w)
        r = per.mean(dim=1)  # [B, 3, T_lat, lat_h, lat_w]
        if upsample_spatial:
            r = rearrange(r, "b c t h w -> (b t) c h w")
            r = F.interpolate(r, size=(self.height, self.width), mode="bilinear", align_corners=False)
            r = rearrange(r, "(b t) c h w -> b c t h w ", b = B)

        r = F.interpolate(
            r,
            size=(self.n_frames, r.shape[-2], r.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )
        return r

    @torch.no_grad()
    def prepare_embeds(self, batch):
        videos  = batch["videos"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        raymaps = batch["raymaps"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        depths  = batch["depths"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        prompts = batch["prompts"]

        # Per-sample modality presence flags (all present if the batch does not say).
        B = videos.shape[0]
        has_raymap = batch.get("has_raymap")
        has_depth  = batch.get("has_depth")
        if has_raymap is None:
            has_raymap = torch.ones(B, dtype=torch.bool, device=self.device)
        else:
            has_raymap = has_raymap.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if has_depth is None:
            has_depth = torch.ones(B, dtype=torch.bool, device=self.device)
        else:
            has_depth = has_depth.to(device=self.device, dtype=torch.bool, non_blocking=True)
        batch["has_raymap"] = has_raymap
        batch["has_depth"]  = has_depth

        batch["videos"]  = videos
        batch["raymaps"] = raymaps
        batch["depths"]  = depths

        batch_size, t, _, h, w = videos.shape
        if t != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, got {t}")
        if h != self.height or w != self.width:
            raise ValueError(f"Expected {self.height}x{self.width}, got {h}x{w}")

        if not self.cfg.load_prompt_embed:
            prompt_embeds = self.encode_text(prompts)
        elif isinstance(batch["prompt_embeds"], list):
            prompt_embeds = batch["prompt_embeds"]  # already processed
        else:
            prompt_embeds = batch["prompt_embeds"].to(self.dtype)
            prompt_embed_len = batch["prompt_embed_len"]
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, prompt_embed_len)]

        video_lat = self.encode_video(rearrange(videos, "b t c h w -> b c t h w"))

        rays_d = raymaps[:, :, :3, :, :]  # [B, T, 3, H, W]
        rays_m = raymaps[:, :, 3:, :, :]  # [B, T, 3, H, W]
        ray_d_lat = self._pack_ray_latent(rays_d)
        ray_m_lat = self._pack_ray_latent(rays_m)

        depths_3ch = depths.repeat(1, 1, 3, 1, 1)
        depth_lat = self.encode_video(rearrange(depths_3ch, "b t c h w -> b c t h w"))

        T_lat = video_lat.shape[2]
        # modality_mask over the model's 3 blocks: 0=RGB, 1=fused ray, 2=depth. video_lat keeps
        # the 4-block layout because ray_d and ray_m are noised per block.
        modality_mask = torch.cat([
            torch.zeros (batch_size, T_lat,     device=self.device, dtype=torch.long),
            torch.ones  (batch_size, T_lat,     device=self.device, dtype=torch.long),
            torch.full  ((batch_size, T_lat), 2, device=self.device, dtype=torch.long),
        ], dim=1)  # [B, 3*T_lat]

        video_lat = torch.cat([video_lat, ray_d_lat, ray_m_lat, depth_lat], dim=2)  # [B, C, 4*T_lat, H, W]
        assert video_lat.shape[2] == self.lat_t, \
            f"Expected lat_t={self.lat_t}, got {video_lat.shape[2]}"

        batch["prompt_embeds"]  = prompt_embeds
        batch["video_lat"]      = video_lat
        batch["modality_mask"]  = modality_mask
        batch["image_embeds"]   = None
        batch["clip_embeds"]    = None
        return batch

    def add_training_noise(self, video_lat, batch=None):
        """Continuous-time flow matching with diffusion forcing over [rgb | ray_d | ray_m | depth].

        Each sample draws one conditioning setting, restricted to those its subset allows
        (diffusion_forcing.subset_settings):
          0  first RGB history; everything else generated
          1  all RGB history; rays and depth generated
          2  first + last RGB and all rays history; middle RGB and depth generated
          3  first RGB and all rays history; remaining RGB and depth generated
          4  first RGB and all depth history; remaining RGB and rays generated
        History frames share one noise level, set to clean with probability clean_hist_prob.
        Returns noisy_lat, noise and t in [0, num_train_timesteps] of shape [b, f].
        """
        b, _, f = video_lat.shape[:3]
        device = video_lat.device
        noise = torch.randn_like(video_lat)  # [b, c, f, h, w]

        if batch is None or "subset_id" not in batch:
            raise KeyError("ray_depth_prediction_firstlast requires batch['subset_id']")
        subset_settings = self.diffusion_forcing.subset_settings
        DEFAULT_P = np.array([0.10, 0.37, 0.38, 0.10, 0.05])
        T = f // 4  # per-modality latent frames
        t = np.zeros((b, f), dtype=np.float32)
        for i in range(b):
            subset_name = batch["subset_id"][i]
            if subset_name not in subset_settings:
                raise KeyError(
                    f"diffusion_forcing.subset_settings has no entry for "
                    f"subset_id={subset_name!r}; add it to the algorithm yaml"
                )
            allowed = list(subset_settings[subset_name])
            p = DEFAULT_P[allowed]
            p = p / p.sum()
            setting = int(np.random.choice(allowed, p=p))
            future_t = np.random.uniform(0, 1)

            if setting == 0:
                history_t = np.random.uniform(0, 1)
                t[i, :] = future_t
                t[i, 0] = history_t
                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                    t[i, 0] = 0

            elif setting == 1:
                history_t = np.random.uniform(0, 1)
                t[i, :] = future_t
                t[i, :T] = history_t
                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                    t[i, :T] = 0

            elif setting == 2:
                history_t = np.random.uniform(0, 1)
                t[i, :] = future_t
                t[i, 0] = history_t       # first RGB
                t[i, T - 1] = history_t   # last RGB
                t[i, T:3*T] = history_t   # all rays (ray_d + ray_m)
                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                    t[i, 0] = 0
                    t[i, T - 1] = 0
                    t[i, T:3*T] = 0

            elif setting == 3:
                history_t = np.random.uniform(0, 1)
                t[i, :] = future_t
                t[i, 0] = history_t        # first RGB
                t[i, T:3*T] = history_t    # all rays (ray_d + ray_m)
                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                    t[i, 0] = 0
                    t[i, T:3*T] = 0
            else:
                history_t = np.random.uniform(0, 1)
                t[i, :] = future_t
                t[i, 0] = history_t        # first RGB
                t[i, 3*T:] = history_t     # all depth
                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                    t[i, 0] = 0
                    t[i, 3*T:] = 0

            # depth is generated in every setting except 4
            if setting != 4:
                t[i, 3*T:] = future_t

        t = torch.from_numpy(t).to(device)
        t = t.float()

        # A modality the sample does not have is pure noise (t = 1) whatever the setting;
        # its loss is masked in training_step.
        has_raymap = batch.get("has_raymap")
        has_depth  = batch.get("has_depth")
        if has_raymap is not None:
            missing_ray = (~has_raymap.bool()).to(device).view(b, 1).float()
            t[:, T:3*T] = t[:, T:3*T] * (1.0 - missing_ray) + missing_ray
        if has_depth is not None:
            missing_depth = (~has_depth.bool()).to(device).view(b, 1).float()
            t[:, 3*T:]   = t[:, 3*T:]   * (1.0 - missing_depth) + missing_depth

        # t in [0, 1], 0 = clean. Shift, noise, then scale to [0, num_train_timesteps].
        t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
        t_expanded = t.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [b, 1, f, 1, 1]
        noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
        t = t * self.num_train_timesteps  # [b, f]
        return noisy_lat, noise, t

    def remove_noise(self, flow_pred, t, video_pred_lat):
        b, _, f = video_pred_lat.shape[:3]
        video_pred_lat = rearrange(video_pred_lat, "b c f h w -> (b f) c h w")
        flow_pred = rearrange(flow_pred, "b c f h w -> (b f) c h w")
        if t.ndim == 1:
            t = repeat(t, "b -> (b f)", f=f)
        elif t.ndim == 2:
            t = t.flatten()
        video_pred_lat = self.inference_scheduler.step(
            flow_pred, t, video_pred_lat, return_dict=False
        )[0]
        video_pred_lat = rearrange(video_pred_lat, "(b f) c h w -> b c f h w", b=b)
        return video_pred_lat

    def training_step(self, batch, batch_idx=None):
        batch = self.prepare_embeds(batch)
        prompt_embeds   = batch["prompt_embeds"]
        modality_mask   = batch["modality_mask"]
        video_lat       = batch["video_lat"]
        clip_embeds     = batch["clip_embeds"]
        image_embeds    = batch["image_embeds"]

        noisy_lat, noise, t = self.add_training_noise(video_lat, batch=batch)
        flow = noise - video_lat

        assert noisy_lat.shape[2] == self.lat_t, \
            f"noisy_lat temporal dim should be {self.lat_t}, got {noisy_lat.shape[2]}"

        T = self.lat_t // 4
        video_noise_lat = noisy_lat[:, :, :T]        # RGB [B, C, T, H, W]
        # Fused ray block: ray_d and ray_m channel-concatenated, real channels only.
        ray_ch = self.vae_stride[0] * 3              # stride frames x 3 Pluecker channels
        ray_noise_lat = torch.cat([
            noisy_lat[:, :ray_ch, T:2*T],
            noisy_lat[:, :ray_ch, 2*T:3*T],
        ], dim=1)
        depth_noise_lat = noisy_lat[:, :, 3*T:]      # depth [B, C, T, H, W]

        # Timesteps for the model's 3 blocks (ray_d and ray_m share a noise level).
        t_model = torch.cat([t[:, :T], t[:, T:2*T], t[:, 3*T:]], dim=1)

        flow_pred = self.model(
            video_noise_lat,
            ray_noise_lat,
            depth_noise_lat,
            modality_mask,
            t=t_model,
            context=prompt_embeds,
            clip_fea=clip_embeds,
            seq_len=self.max_tokens,
            y=image_embeds,
        )

        # Ray losses use only the real packed channels; the zero-pad channels are not supervised.
        flow_rgb_pred   = flow_pred[:, :, :T]
        flow_ray_d_pred = flow_pred[:, :ray_ch, T:2*T]
        flow_ray_m_pred = flow_pred[:, :ray_ch, 2*T:3*T]
        flow_depth_pred = flow_pred[:, :, 3*T:]
        flow_rgb_target   = flow[:, :, :T]
        flow_ray_d_target = flow[:, :ray_ch, T:2*T]
        flow_ray_m_target = flow[:, :ray_ch, 2*T:3*T]
        flow_depth_target = flow[:, :, 3*T:]

        # Clean (t == 0) tokens are context, not targets: their flow target is unobservable noise.
        mask_clean = bool(self.cfg.diffusion_forcing.get("mask_clean_token_loss", True))
        has_raymap = batch["has_raymap"]
        has_depth  = batch["has_depth"]
        rgb_loss = masked_flow_mse(
            flow_rgb_pred, flow_rgb_target, timestep=t[:, :T], mask_clean=mask_clean,
        )
        ray_d_loss = masked_flow_mse(
            flow_ray_d_pred, flow_ray_d_target, timestep=t[:, T:2*T],
            available=has_raymap, mask_clean=mask_clean,
        )
        ray_m_loss = masked_flow_mse(
            flow_ray_m_pred, flow_ray_m_target, timestep=t[:, 2*T:3*T],
            available=has_raymap, mask_clean=mask_clean,
        )
        depth_loss = masked_flow_mse(
            flow_depth_pred, flow_depth_target, timestep=t[:, 3*T:],
            available=has_depth, mask_clean=mask_clean,
        )
        dw = float(self.cfg.diffusion_forcing.get("depth_loss_weight", 1.0))
        total_loss = (rgb_loss + ray_d_loss + ray_m_loss + dw * depth_loss) / (3.0 + dw)

        if self.global_step % self.cfg.logging.loss_freq == 0:
            self.log("train/loss_total", total_loss, sync_dist=True)
            self.log("train/loss_rgb",   rgb_loss,   sync_dist=True)
            self.log("train/loss_ray_d", ray_d_loss, sync_dist=True)
            self.log("train/loss_ray_m", ray_m_loss, sync_dist=True)
            self.log("train/loss_depth", depth_loss, sync_dist=True)
            self.log("global_step", self.global_step, sync_dist=True)
            gn = getattr(self, "_grad_norm_last", None)  # previous step's pre-clip norm
            if gn is not None:
                self.log("train/grad_norm", gn, sync_dist=False)

        return total_loss

    # ---- checkpointing and resume --------------------------------------------------------

    def on_save_checkpoint(self, checkpoint) -> None:
        # One RNG state per rank, so a resume continues each rank's own noise/task stream.
        checkpoint["vwm_rank_rng_states"] = collect_all_rank_rng_states()
        # Drop the frozen text encoder and VAE (rebuilt from pretrained files) and store the
        # trained weights in bf16, halving the checkpoint.
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k in list(sd.keys()):
                if k.startswith("text_encoder.") or k.startswith("vae."):
                    del sd[k]
                elif torch.is_tensor(sd[k]) and sd[k].is_floating_point() and sd[k].dtype != torch.bfloat16:
                    sd[k] = sd[k].to(torch.bfloat16)

    def on_load_checkpoint(self, checkpoint) -> None:
        self._maybe_rescale_lr_on_resume(checkpoint)
        rank_states = checkpoint.get("vwm_rank_rng_states")
        if rank_states is None:
            self._vwm_checkpoint_missing_rng = True
            print(
                "[RNG] checkpoint has no vwm_rank_rng_states; continuation "
                "will be rank-distinct but not main-process exact",
                flush=True,
            )
            return
        # Saved per-rank streams cannot be reused on a different number of GPUs. With
        # rng_allow_world_size_change, fall back to fresh rank-distinct streams instead of failing.
        _ws = getattr(getattr(self, "_trainer", None), "world_size", None)
        if _ws is None and dist.is_available() and dist.is_initialized():
            _ws = dist.get_world_size()
        if _ws is not None and len(rank_states) != int(_ws):
            if bool(self.cfg.get("rng_allow_world_size_change", False)):
                self._vwm_checkpoint_missing_rng = True
                print(
                    f"[RNG] world-size change on resume: checkpoint has "
                    f"{len(rank_states)} rank RNG states, current world_size={int(_ws)}. "
                    f"Opt-in rng_allow_world_size_change=true -> deriving rank-distinct "
                    f"streams instead. Weights/optimizer/LR/step are unaffected.",
                    flush=True,
                )
                return
        self._pending_vwm_rng_state = select_rank_rng_state(
            rank_states,
            global_rank=self.global_rank,
            world_size=_ws,
        )

    def _maybe_rescale_lr_on_resume(self, checkpoint) -> None:
        """With rescale_lr_on_resume, make the configured LR win over the one stored in a
        full-state checkpoint (optimizer initial_lr/lr and scheduler base_lrs)."""
        if not bool(self.cfg.get("rescale_lr_on_resume", False)):
            return
        new_lr = float(self.cfg.lr)
        old = None
        for opt_state in checkpoint.get("optimizer_states", []):
            for pg in opt_state.get("param_groups", []):
                if old is None:
                    old = pg.get("initial_lr", pg.get("lr"))
                if "initial_lr" in pg:
                    pg["initial_lr"] = new_lr
                if "lr" in pg:
                    pg["lr"] = new_lr
        for sched in checkpoint.get("lr_schedulers", []):
            if "base_lrs" in sched:
                sched["base_lrs"] = [new_lr for _ in sched["base_lrs"]]
        print(
            f"[LR] rescale_lr_on_resume: peak LR {old} -> {new_lr} "
            f"(base_lrs/initial_lr rewritten; step position preserved)",
            flush=True,
        )

    def _checkpoint_dir(self) -> str:
        from lightning.pytorch.callbacks import ModelCheckpoint
        for cb in getattr(self.trainer, "checkpoint_callbacks", []):
            if isinstance(cb, ModelCheckpoint) and getattr(cb, "dirpath", None):
                return cb.dirpath
        if getattr(self.logger, "save_dir", None):
            return os.path.join(self.logger.save_dir, "checkpoints")
        return os.path.join(self.trainer.default_root_dir, "checkpoints")

    def on_train_start(self) -> None:
        # Release memory fragmented by checkpoint loading and FSDP sharding.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        global_rank = int(self.global_rank)
        base_seed = int(os.environ.get("PL_GLOBAL_SEED", "0") or 0)
        pending_rng_state = getattr(self, "_pending_vwm_rng_state", None)
        if pending_rng_state is not None:
            restore_local_rng_state(pending_rng_state)
            del self._pending_vwm_rng_state
            print(f"[RNG] restored checkpoint stream for rank={global_rank}", flush=True)
        elif self.global_step == 0:
            rank_seed = seed_rank_training_streams(base_seed, global_rank)
            print(f"[RNG] fresh training seed={rank_seed} rank={global_rank}", flush=True)
        elif getattr(self, "_vwm_checkpoint_missing_rng", False):
            # No saved streams: still give every rank a distinct one.
            fallback_seed = (
                base_seed + int(self.global_step) * 1_000_003
            ) % (2**32)
            rank_seed = seed_rank_training_streams(fallback_seed, global_rank)
            print(
                f"[RNG] non-exact legacy-resume seed={rank_seed} rank={global_rank}",
                flush=True,
            )

        # Save a checkpoint before the first step (collective under FSDP: every rank enters).
        if self.global_step == 0 and self.cfg.get("save_startup_checkpoint", True):
            checkpoint_dir = self._checkpoint_dir()
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.trainer.save_checkpoint(os.path.join(checkpoint_dir, "startup.ckpt"))

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        self.log("epoch", float(self.current_epoch), sync_dist=True)
        checkpoint_dir = self._checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Collective under FSDP; do not wrap this call in a rank-zero guard.
        self.trainer.save_checkpoint(
            os.path.join(checkpoint_dir, f"epoch={self.current_epoch:04d}.ckpt")
        )

    # ---- sampling --------------------------------------------------------------------------

    @torch.no_grad()
    def sample_seq(self, batch, pbar=None, hist_indexes=None):
        """Denoise [rgb | ray_d | ray_m | depth] with the latent frames in `hist_indexes` held
        clean (default: first and last frame of RGB and both ray blocks).

        Guidance: flow * (1 + lang + hist) - lang * flow(neg prompt) - hist * flow(history noised).
        Returns [B, 4 * n_frames, 3, H, W].
        """
        self.inference_scheduler, self.inference_timesteps = self.build_scheduler(False)
        lang_guidance = self.lang_guidance if self.lang_guidance else 0
        hist_guidance = self.hist_guidance if self.hist_guidance else 0

        batch = self.prepare_embeds(batch)
        clip_embeds   = batch["clip_embeds"]
        image_embeds  = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat     = batch["video_lat"]
        modality_mask = batch["modality_mask"]

        batch_size = video_lat.shape[0]
        T = self.lat_t // 4  # frames per modality in latent space

        video_pred_lat = torch.randn_like(video_lat)

        if lang_guidance:
            neg_prompt_embeds = self.encode_text([self.neg_prompt] * len(batch["prompts"]))

        if pbar is None:
            pbar = tqdm(range(len(self.inference_timesteps)), desc="Sampling")

        if hist_indexes is None:
            hist_indexes = torch.tensor([0, T - 1, T, 2*T - 1, 2*T, 3*T - 1])

        ray_ch = self.vae_stride[0] * 3
        def _call_model(pred_lat, ctx):
            ray_lat = torch.cat([
                pred_lat[:, :ray_ch, T:2*T],
                pred_lat[:, :ray_ch, 2*T:3*T],
            ], dim=1)  # fused ray block, real channels only: [B, 2*ray_ch, T, H, W]
            t_m = torch.cat([t_expanded[:, :T], t_expanded[:, T:2*T], t_expanded[:, 3*T:]], dim=1)
            return self.model(
                pred_lat[:, :, :T],
                ray_lat,
                pred_lat[:, :, 3*T:],    # depth
                modality_mask,
                t=t_m,
                context=ctx,
                seq_len=self.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
            )

        for t in self.inference_timesteps:
            video_pred_lat[:, :, hist_indexes] = video_lat[:, :, hist_indexes]
            t_expanded = torch.full((batch_size, self.lat_t), t, device=self.device)
            t_expanded[:, hist_indexes] = self.inference_timesteps[-1]

            flow_pred = _call_model(video_pred_lat, prompt_embeds)

            if lang_guidance:
                no_lang_flow_pred = _call_model(video_pred_lat, neg_prompt_embeds)
            else:
                no_lang_flow_pred = torch.zeros_like(flow_pred)

            if hist_guidance:
                no_hist_lat = video_pred_lat.clone()
                no_hist_lat[:, :, hist_indexes] = torch.randn_like(no_hist_lat[:, :, hist_indexes])
                t_expanded[:, hist_indexes] = self.inference_timesteps[0]
                no_hist_flow_pred = _call_model(no_hist_lat, prompt_embeds)
            else:
                no_hist_flow_pred = torch.zeros_like(flow_pred)

            flow_pred = flow_pred * (1 + lang_guidance + hist_guidance)
            flow_pred = flow_pred - lang_guidance * no_lang_flow_pred - hist_guidance * no_hist_flow_pred

            video_pred_lat = self.remove_noise(flow_pred, t, video_pred_lat)
            pbar.update(1)

        video_pred_lat[:, :, hist_indexes] = video_lat[:, :, hist_indexes]

        rgb_lat   = video_pred_lat[:, :, :T]
        ray_d_lat = video_pred_lat[:, :, T:2*T]
        ray_m_lat = video_pred_lat[:, :, 2*T:3*T]
        depth_lat = video_pred_lat[:, :, 3*T:]

        video_rgb   = self.decode_video(rgb_lat)
        video_ray_d = self._unpack_ray_latent(ray_d_lat)
        video_ray_m = self._unpack_ray_latent(ray_m_lat)
        video_depth = self.decode_video(depth_lat)

        video_pred = torch.cat([video_rgb, video_ray_d, video_ray_m, video_depth], dim=2)
        video_pred = rearrange(video_pred, "b c t h w -> b t c h w")
        return video_pred

    # ---- validation ------------------------------------------------------------------------

    def validation_step(self, batch, batch_idx=None):
        T = self.lat_t // 4

        # Alternate between two tasks across validation rounds (all ranks pick the same one):
        # RGB -> rays + depth, and first/last RGB + rays -> RGB + depth.
        val_every = self.trainer.val_check_interval if self.trainer is not None else 100
        do_rgb_to_raymap = (self.global_step // max(1, val_every)) % 2 == 0

        # raymap -> RGB needs real rays as history; fall back if any sample has none.
        has_raymap_all = bool(batch["has_raymap"].all().item())
        has_depth_all  = bool(batch["has_depth"].all().item())
        if not do_rgb_to_raymap and not has_raymap_all:
            do_rgb_to_raymap = True

        if do_rgb_to_raymap:
            hist_rgb_to_raymap = torch.arange(T)
            video_pred = self.sample_seq(batch, hist_indexes=hist_rgb_to_raymap)
            self.visualize(video_pred, batch, prefix="rgb_to_raymap_depth/",
                           compute_rgb_metrics=False,
                           compute_pose_metrics=has_raymap_all,
                           compute_depth_metrics=has_depth_all)
        else:
            hist_raymap_to_rgb = torch.cat([
                torch.tensor([0, T - 1]),       # first + last RGB
                torch.arange(T, 3 * T),         # all ray_d + ray_m
            ])
            video_pred = self.sample_seq(batch, hist_indexes=hist_raymap_to_rgb)
            self.visualize(video_pred, batch, prefix="raymap_to_rgb_depth/",
                           compute_rgb_metrics=True,
                           compute_pose_metrics=False,
                           compute_depth_metrics=has_depth_all)

    def visualize(self, video_pred, batch, prefix="", compute_rgb_metrics=False,
                  compute_pose_metrics=False, compute_depth_metrics=False):
        """Log prediction/GT panels, metrics and trajectory plots.

        video_pred: [B, 4T, 3, H, W] -- [rgb | ray_d | ray_m | depth] frames
        """
        video_gt  = batch["videos"]              # [B, T, 3, H, W]
        raymaps   = batch.get("raymaps", None)   # [B, T, 6, H, W]
        depth_gt  = batch.get("depths", None)    # [B, T, 1, H, W]

        T = video_pred.shape[1] // 4

        pred_rgb   = video_pred[:, :T]        # [B, T, 3, H, W]
        pred_ray_d = video_pred[:, T:2*T]
        pred_ray_m = video_pred[:, 2*T:3*T]
        pred_depth = video_pred[:, 3*T:]

        if raymaps is not None:
            B_r, T_r = raymaps.shape[:2]
            H_pred, W_pred = pred_ray_d.shape[-2:]
            rays_d_gt = rearrange(raymaps[:, :, :3], "b t c h w -> (b t) c h w")
            rays_m_gt = rearrange(raymaps[:, :, 3:], "b t c h w -> (b t) c h w")
            if rays_d_gt.shape[-2] != H_pred or rays_d_gt.shape[-1] != W_pred:
                rays_d_gt = F.interpolate(rays_d_gt, size=(H_pred, W_pred), mode="bilinear", align_corners=False)
                rays_m_gt = F.interpolate(rays_m_gt, size=(H_pred, W_pred), mode="bilinear", align_corners=False)
            rays_d_gt = rearrange(rays_d_gt, "(b t) c h w -> b t c h w", b=B_r, t=T_r)
            rays_m_gt = rearrange(rays_m_gt, "(b t) c h w -> b t c h w", b=B_r, t=T_r)
            vis_rgb   = torch.cat([pred_rgb,   video_gt],  dim=-1)
            vis_ray_d = torch.cat([pred_ray_d, rays_d_gt], dim=-1)
            vis_ray_m = torch.cat([pred_ray_m, rays_m_gt], dim=-1)
        else:
            vis_rgb   = pred_rgb
            vis_ray_d = pred_ray_d
            vis_ray_m = pred_ray_m

        # With valid GT depth, color the prediction on the GT's range; otherwise on its own.
        has_depth_t = batch.get("has_depth")
        gt_is_valid = (depth_gt is not None and has_depth_t is not None
                       and bool(has_depth_t.all().item()))
        if gt_is_valid:
            pred_depth_color = _depth_to_colormap(pred_depth, ref=depth_gt) * 2 - 1
            vis_depth = torch.cat([pred_depth_color, _depth_to_colormap(depth_gt) * 2 - 1], dim=-1)
        elif depth_gt is not None:
            pred_depth_color = _depth_to_colormap(pred_depth) * 2 - 1
            vis_depth = torch.cat([pred_depth_color, _depth_to_colormap(depth_gt) * 2 - 1], dim=-1)
        else:
            pred_depth_color = _depth_to_colormap(pred_depth) * 2 - 1
            vis_depth = pred_depth_color

        vis_depth = vis_depth.to(vis_rgb.device)
        vis = torch.cat([vis_rgb, vis_ray_d, vis_ray_m, vis_depth], dim=-1) * 0.5 + 0.5
        vis = vis.cpu()

        pred_rgb_cpu   = pred_rgb.cpu()
        pred_depth_cpu = pred_depth.cpu()
        video_gt_cpu   = video_gt.cpu() if isinstance(video_gt, torch.Tensor) else None
        depth_gt_cpu   = depth_gt.cpu() if depth_gt is not None else None

        if dist.is_available() and dist.is_initialized():
            vis = rearrange(self.all_gather(vis), "p b ... -> (p b) ...")
            pred_ray_d = rearrange(self.all_gather(pred_ray_d.cpu()), "p b ... -> (p b) ...")
            pred_ray_m = rearrange(self.all_gather(pred_ray_m.cpu()), "p b ... -> (p b) ...")
            pred_rgb_cpu   = rearrange(self.all_gather(pred_rgb_cpu),   "p b ... -> (p b) ...")
            pred_depth_cpu = rearrange(self.all_gather(pred_depth_cpu), "p b ... -> (p b) ...")
            if video_gt_cpu is not None:
                video_gt_cpu = rearrange(self.all_gather(video_gt_cpu), "p b ... -> (p b) ...")
            if depth_gt_cpu is not None:
                depth_gt_cpu = rearrange(self.all_gather(depth_gt_cpu), "p b ... -> (p b) ...")
            if raymaps is not None:
                raymaps = rearrange(self.all_gather(raymaps.cpu()), "p b ... -> (p b) ...")

        if is_rank_zero:
            # Metrics are logged globally and, when the batch comes from one subset, per subset.
            subset_ids = batch.get("subset_id")
            if isinstance(subset_ids, str):
                subset_ids = [subset_ids]
            elif subset_ids is None:
                subset_ids = []
            unique_sid = subset_ids[0] if (subset_ids and all(s == subset_ids[0] for s in subset_ids)) else None

            def _log_metric(name: str, value):
                logs = {f"val_metrics/global/{prefix}{name}": value}
                if unique_sid is not None:
                    logs[f"val_metrics/{unique_sid}/{prefix}{name}"] = value
                self.logger.experiment.log(logs, step=self.global_step)

            self.log_video(
                f"validation_vis/global/{prefix}rgb_rayd_raym_depth",
                vis[:16],
                fps=self.cfg.logging.fps,
                step=self.global_step,
            )
            if unique_sid is not None:
                self.log_video(
                    f"validation_vis/{unique_sid}/{prefix}rgb_rayd_raym_depth",
                    vis[:16],
                    fps=self.cfg.logging.fps,
                    step=self.global_step,
                )

            if compute_rgb_metrics and video_gt_cpu is not None:
                psnr = _psnr_batch(pred_rgb_cpu, video_gt_cpu, skip_first_last=True)
                _log_metric("psnr", psnr)

            # Gated on the flag, not on depth_gt: missing depth arrives as a placeholder tensor.
            if compute_depth_metrics and depth_gt_cpu is not None:
                dm = _depth_delta_batch(pred_depth_cpu, depth_gt_cpu)
                for k, v in dm.items():
                    _log_metric(k, v)

            if compute_pose_metrics and raymaps is not None:
                pm = _pose_metrics_batch(pred_ray_d, pred_ray_m, raymaps)
                for k, v in pm.items():
                    _log_metric(k, v)

            if compute_pose_metrics and raymaps is not None:
                B_traj = min(pred_ray_d.shape[0], 16)
                traj_imgs = []
                for i in range(B_traj):
                    pred_raymaps_i = torch.cat([pred_ray_d[i], pred_ray_m[i]], dim=1)
                    traj_img = _make_trajectory_figure(pred_raymaps_i, raymaps[i])
                    if traj_img is not None:
                        traj_imgs.append(traj_img)
                if traj_imgs:
                    n = len(traj_imgs)
                    ncols = min(n, 4)
                    nrows = int(np.ceil(n / ncols))
                    w, h = traj_imgs[0].size
                    grid_img = Image.new("RGB", (ncols * w, nrows * h), color=(255, 255, 255))
                    for idx, img in enumerate(traj_imgs):
                        row, col = divmod(idx, ncols)
                        grid_img.paste(img, (col * w, row * h))
                    traj_logs = {f"validation_viz/global/{prefix}trajectory": wandb.Image(grid_img)}
                    if unique_sid is not None:
                        traj_logs[f"validation_viz/{unique_sid}/{prefix}trajectory"] = wandb.Image(grid_img)
                    self.logger.experiment.log(traj_logs, step=self.global_step)
