"""Small, testable invariants shared by the multimodal training variants."""

import random

import numpy as np
import torch
import torch.distributed as dist


def _split_modality_time_embeddings(e, per_modality_frames):
    """Return timestep embeddings aligned with RGB, ray, and depth heads.

    The transformer uses ``[rgb | ray | depth]`` temporal blocks. Each output
    head processes only one block, so a concatenated ``[B, 3*T, C]`` embedding
    must be split before modulation. A shared ``[B, T, C]`` embedding is valid
    for the scalar-timestep inference path and is reused unchanged.
    """
    if e.ndim != 3:
        raise ValueError(
            "Expected time embeddings with shape [B, T, C] or [B, 3*T, C] "
            f"for T={per_modality_frames}, but got {tuple(e.shape)}"
        )
    if e.shape[1] == per_modality_frames:
        return e, e, e
    if e.shape[1] != 3 * per_modality_frames:
        raise ValueError(
            "Expected time embeddings with shape [B, T, C] or [B, 3*T, C] "
            f"for T={per_modality_frames}, but got {tuple(e.shape)}"
        )
    return e.split(per_modality_frames, dim=1)


def distributed_masked_mean(values, mask):
    """Mean over globally valid values with DDP/FSDP-correct gradients.

    DDP averages gradients across ranks. A rank-local mean therefore weights
    every rank equally even when valid counts differ. Scaling the local
    numerator by ``world_size / global_count`` makes the post-DDP gradient and
    a ``sync_dist=True`` logged scalar equal the true global masked mean.
    """
    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have identical shape, got "
            f"{tuple(values.shape)} and {tuple(mask.shape)}"
        )
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    local_sum = (values * mask_f).sum()
    local_count = mask_f.sum().detach()

    if dist.is_available() and dist.is_initialized():
        global_count = local_count.clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        scaled = local_sum * (dist.get_world_size() / global_count.clamp_min(1.0))
        return torch.where(global_count > 0, scaled, local_sum * 0.0)

    mean = local_sum / local_count.clamp_min(1.0)
    return torch.where(local_count > 0, mean, local_sum * 0.0)


def masked_flow_mse(
    pred,
    target,
    *,
    timestep=None,
    available=None,
    mask_clean=False,
):
    """Compute the branch's per-sample flow MSE with global normalization.

    When H011 is enabled, exact-clean frames (timestep zero) are removed before
    each sample's frame mean. The final mean remains per valid sample, matching
    `clean-training-code-aug-4`; only its cross-rank denominator is corrected.
    """
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError(
            "pred and target must have identical [B,C,T,H,W] shapes, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )

    batch = pred.shape[0]
    if available is None:
        available_mask = torch.ones(batch, dtype=torch.bool, device=pred.device)
    else:
        available_mask = available.to(device=pred.device, dtype=torch.bool).reshape(-1)
        if available_mask.shape != (batch,):
            raise ValueError(
                f"available must contain one value per sample, got {tuple(available.shape)}"
            )

    if not mask_clean:
        per_sample = ((pred - target) ** 2).mean(dim=(1, 2, 3, 4))
        return distributed_masked_mean(per_sample, available_mask)

    expected_timestep_shape = (batch, pred.shape[2])
    if timestep is None or tuple(timestep.shape) != expected_timestep_shape:
        actual = None if timestep is None else tuple(timestep.shape)
        raise ValueError(
            f"mask_clean requires timestep [B,T]={expected_timestep_shape}, got {actual}"
        )
    per_frame = ((pred - target) ** 2).mean(dim=(1, 3, 4))
    eligible = timestep.to(device=pred.device).gt(0)
    eligible = eligible & available_mask[:, None]
    eligible_f = eligible.to(per_frame.dtype)
    eligible_per_sample = eligible_f.sum(dim=1)
    per_sample = (
        (per_frame * eligible_f).sum(dim=1)
        / eligible_per_sample.clamp_min(1.0)
    )
    return distributed_masked_mean(per_sample, eligible_per_sample > 0)


def seed_rank_training_streams(base_seed, global_rank):
    """Seed stochastic training streams independently after model creation."""
    rank_seed = int(base_seed) + int(global_rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
    return rank_seed


def collect_local_rng_state():
    """Collect main-process RNG streams used by task/noise sampling."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_local_rng_state(state):
    """Restore a state produced by :func:`collect_local_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])


def collect_all_rank_rng_states(local_state=None):
    """Return one serializable main-process RNG state per distributed rank."""
    local_state = collect_local_rng_state() if local_state is None else local_state
    if not (dist.is_available() and dist.is_initialized()):
        return [local_state]
    rank_states = [None] * dist.get_world_size()
    dist.all_gather_object(rank_states, local_state)
    return rank_states


def select_rank_rng_state(rank_states, global_rank, world_size=None):
    """Select one rank's state and fail loudly on a world-size mismatch."""
    global_rank = int(global_rank)
    if world_size is None and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    if world_size is not None:
        world_size = int(world_size)
        if len(rank_states) != world_size:
            raise RuntimeError(
                f"Checkpoint has {len(rank_states)} RNG states but current "
                f"world_size={world_size}"
            )
    if global_rank < 0 or global_rank >= len(rank_states):
        raise RuntimeError(
            f"Checkpoint has {len(rank_states)} RNG states but global_rank={global_rank}"
        )
    return rank_states[global_rank]
