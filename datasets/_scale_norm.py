"""Scheme-C "parallax" normalization — opt-in cross-dataset scale scheme.

Documented in full in repo-root SCALE_NORMALIZATION_ANALYSIS.md. Enabled per-run by the
hydra flag `dataset.scale_mode=parallax` (propagated to every subset by MixtureDataset);
`dataset.parallax_kappa` sets kappa (default 4.0). When the flag is absent every loader
keeps its exact prior behavior (per-clip max|moment| or the scale_preserving constant).

Recipe (identical across all geometry datasets):
  s = kappa * median(valid metric depth over the clip)          # per-clip scalar
  moments <- clamp(moments / s, -1, 1)                          # -> scene-relative parallax
  depth   <- 2 / (1 + clamp(depth / s, 0)) - 1                  # -> fixed operating point

Why one shared scalar for BOTH channels: dividing camera translation (carried by the
Plucker moments m = t x d) and depth by the same s is a global similarity (uniform scene
scale), under which projection / point-cloud formation is exactly invariant — so predicted
depth and pose stay aligned (verified to machine precision; see doc 4b). Normalizing the two
channels by *different* scalars would desync them and is therefore forbidden.

Moments are deliberately NOT clamped to [-1, 1] here: the model does not use a fixed-range
VAE latent for the raymap channels, so out-of-range moments are fine, and skipping the clamp
keeps the (depth, pose) mapping EXACTLY invertible / alignment-preserving for 100% of clips
(clamping was the only op that could desync moments from depth). kappa=4 merely keeps the
bulk near unit magnitude (pooled normalized-moment P99 ~= 5.3 across 1,459 clips / 12
datasets) so the channel is well-conditioned; the tail is allowed to exceed 1.
"""
from typing import Optional

import torch

PARALLAX_MODE = "parallax"
GLOBAL_METRIC_MODE = "global_metric"
PARALLAX_KAPPA_DEFAULT = 4.0
GLOBAL_METRIC_SCALE_DEFAULT = 10.0   # metres; ONE global constant for scale_mode=global_metric
PARALLAX_ZBAR_FLOOR_M = 1e-2   # metres; guards degenerate near-zero-depth clips
# scheme-C depth channel map. "log" (default) = 1 - 2*clip(alpha*ln(d/median + 1), 0, 1); "disparity"
# = 2/(1+d/s)-1 (the original). The log map is FAR more robust through the VAE latent: the frozen
# Wan2.2 VAE round-trips log-mapped depth at ~3% abs_rel vs ~45% for disparity (which crushes far/
# deep scenes into a razor-thin slice near -1 whose 1/u inversion explodes under VAE channel noise).
# Both orient near->+1, far->-1, invalid(sentinel)-> -1. alpha=0.30 puts the P99 depth at the far end.
PARALLAX_DEPTH_MAP_DEFAULT = "log"
PARALLAX_LOG_ALPHA_DEFAULT = 0.30
# scheme-C ray channels 3:6. "companded_origin" (default) = Aether-style: replace the Plucker
# moment with the camera ORIGIN t (first-frame gauge), normalized by the shared s and mu-law
# companded  co = sign(t/s)*ln(1+mu*|t/s|)/ln(1+mu). Fixes (a) the moment's cross-product
# attenuation of forward/dolly motion (|m| = |t|sin(theta), ~0 at image center) and (b) tiny
# values in low-parallax clips (mu=255 gives ~46x gain near 0; spring med 0.0035 -> 0.116).
# Exactly invertible; decode = spatial mean -> mu-law expand -> *s (see re10k.raymap_to_camera).
# "moment" keeps the original Plucker moment / s. Direction channels 0:3 unchanged either way.
PARALLAX_RAY_ENCODING_DEFAULT = "companded_origin"
PARALLAX_MU_DEFAULT = 255.0
# Ceiling on the gauge-normalised Plucker moment. In PARALLAX mode every dataset routes
# through apply_parallax / _divide_and_disparity below, and BOTH were unclamped -- whereas
# every dataset's own MAX-NORM path already does clamp(moments/scale, -1, 1). That asymmetry
# is why gauge-normalised runs see unbounded ray inputs and max-norm runs never do, and
# it is how a single diverged COLMAP pose once reached the model as 9.24e10 (one DL3DV
# frame, grad_norm 9.9e6).
#
# The moment IS signal: |m|/s ~ |t|/(2*zbar), i.e. translation per unit scene depth, so the
# large values are high-parallax clips, not noise. Measured element-level magnitudes under
# the gauge: median 0.035-0.074, p99 0.87-1.40, largest legitimate value seen 8.40
# (tartanair_v2). A bound of 10 therefore alters ~0% of real data while capping the
# pathology; a bound of 1 would clip ~2.1% of tartanair elements (60% of the mixture) and
# 0.6% of tartanair_v2 -- the most pose-informative content. Set to None to restore the
# old unbounded behaviour.
PARALLAX_MOMENT_CLAMP_DEFAULT = 10.0


def parallax_moment_clamp(cfg):
    """Ceiling for |gauge-normalised moment|; None disables clamping."""
    try:
        v = cfg.get("parallax_moment_clamp", PARALLAX_MOMENT_CLAMP_DEFAULT)
    except Exception:
        v = PARALLAX_MOMENT_CLAMP_DEFAULT
    if v is None:
        return None
    v = float(v)
    return v if v > 0 else None


def clamp_moments_(t, cfg):
    """In-place clamp of the moment channels to +/- parallax_moment_clamp(cfg)."""
    c = parallax_moment_clamp(cfg)
    if c is not None:
        t[:, 3:6] = t[:, 3:6].clamp(-c, c)
    return t


_EPS = 1e-8
_INV_EPS = 1e-9    # saturation guard shared by the depth-channel inverses


def is_parallax_mode(cfg) -> bool:
    """True iff dataset.scale_mode == 'parallax'. cfg is a DictConfig or dict."""
    try:
        return str(cfg.get("scale_mode", None)) == PARALLAX_MODE
    except Exception:
        return False


def parallax_kappa(cfg) -> float:
    try:
        return float(cfg.get("parallax_kappa", PARALLAX_KAPPA_DEFAULT) or PARALLAX_KAPPA_DEFAULT)
    except Exception:
        return PARALLAX_KAPPA_DEFAULT


def parallax_depth_map(cfg) -> str:
    """'log' (default, VAE-robust) or 'disparity' (original) for the parallax depth channel."""
    try:
        return str(cfg.get("parallax_depth_map", PARALLAX_DEPTH_MAP_DEFAULT) or PARALLAX_DEPTH_MAP_DEFAULT)
    except Exception:
        return PARALLAX_DEPTH_MAP_DEFAULT


def parallax_log_alpha(cfg) -> float:
    try:
        return float(cfg.get("parallax_log_alpha", PARALLAX_LOG_ALPHA_DEFAULT) or PARALLAX_LOG_ALPHA_DEFAULT)
    except Exception:
        return PARALLAX_LOG_ALPHA_DEFAULT


def parallax_ray_encoding(cfg) -> str:
    """'companded_origin' (default) or 'moment' for the parallax ray channels 3:6."""
    try:
        return str(cfg.get("parallax_ray_encoding", PARALLAX_RAY_ENCODING_DEFAULT)
                   or PARALLAX_RAY_ENCODING_DEFAULT)
    except Exception:
        return PARALLAX_RAY_ENCODING_DEFAULT


def parallax_mu(cfg) -> float:
    try:
        return float(cfg.get("parallax_mu", PARALLAX_MU_DEFAULT) or PARALLAX_MU_DEFAULT)
    except Exception:
        return PARALLAX_MU_DEFAULT


def mulaw_compand(x: torch.Tensor, mu: float) -> torch.Tensor:
    """sign(x)*ln(1+mu|x|)/ln(1+mu): [-1,1]->[-1,1], ~(mu/ln(1+mu))x near 0, log tail beyond."""
    import math
    return torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)


def mulaw_expand(y: torch.Tensor, mu: float) -> torch.Tensor:
    """Exact inverse of mulaw_compand: sign(y)*((1+mu)^|y| - 1)/mu."""
    import math
    return torch.sign(y) * torch.expm1(y.abs() * math.log1p(mu)) / mu


def origins_from_plucker(raymaps: torch.Tensor, max_pixels: int = 4096) -> torch.Tensor:
    """Recover the per-frame camera origin o from a Plucker raymap [d | m=o x d], (T,6,H,W)
    -> (T,3). Closed form: d x m = d x (o x d) = (I - d d^T) o for unit d, so summing over
    pixels gives the 3x3 system  [sum(I - d d^T)] o = sum(d x m). Exact for loader-built
    raymaps (all rays share one origin); A is well-conditioned for any nonzero FOV."""
    T, C, H, W = raymaps.shape
    d = raymaps[:, 0:3].reshape(T, 3, -1).permute(0, 2, 1)   # (T, N, 3)
    m = raymaps[:, 3:6].reshape(T, 3, -1).permute(0, 2, 1)
    n = d.shape[1]
    if n > max_pixels:
        idx = torch.linspace(0, n - 1, max_pixels, device=d.device).long()
        d, m = d[:, idx], m[:, idx]
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    A = (d.shape[1] * torch.eye(3, dtype=d.dtype, device=d.device).expand(T, 3, 3)
         - torch.einsum("tni,tnj->tij", d, d))
    b = torch.cross(d, m, dim=-1).sum(dim=1)                 # (T, 3)
    return torch.linalg.solve(A, b)


def companded_origin_ray_norm(raymaps: torch.Tensor, cfg, min_scale: float = 1.0):
    """NVS variant of the companded-origin encoding for loaders WITHOUT metric depth
    (re10k / dl3dv_colmap): the shared depth gauge s = kappa*median(depth) doesn't exist, so
    gauge by the per-clip translation extent instead — s = max(|origin|, min_scale) (the same
    per-clip-max convention their moment normalization used, floor keeps near-static clips
    conditioned). Channels 3:6 <- mu-law companded origin, spatially constant, identical
    encoding semantics to apply_parallax's companded_origin. Returns (raymaps_norm, scale)."""
    origins = origins_from_plucker(raymaps)                  # (T, 3)
    scale = max(float(origins.abs().max().item()), min_scale)
    if scale < _EPS:
        scale = 1.0
    co = mulaw_compand(origins / scale, parallax_mu(cfg))
    out = raymaps.clone()
    out[:, 3:6] = co.view(-1, 3, 1, 1).expand_as(out[:, 3:6])
    return out, scale


def parallax_scale(depth_valid_metric: torch.Tensor, kappa: float) -> float:
    """s = kappa * median(valid metric depth), floored. `depth_valid_metric` is a 1-D (or
    any-shape) tensor of ONLY the valid, positive metric depths in the clip."""
    if depth_valid_metric.numel() > 0:
        zbar = float(depth_valid_metric.median().item())
    else:
        zbar = 0.0
    zbar = max(zbar, PARALLAX_ZBAR_FLOOR_M)
    scale = kappa * zbar
    return scale if scale > _EPS else 1.0


def parallax_scale_from_sentinel(depth_metric: torch.Tensor, cfg,
                                 invalid_sentinel: float) -> float:
    """Convenience for the many loaders that stuff invalid pixels with a large sentinel:
    take the per-clip parallax scale over pixels with depth < sentinel and > 0."""
    valid = (depth_metric < invalid_sentinel) & (depth_metric > 0)
    return parallax_scale(depth_metric[valid], parallax_kappa(cfg))


def apply_parallax(raymaps_raw: torch.Tensor, depth_metric: torch.Tensor, cfg,
                   invalid_sentinel: float):
    """Full scheme-C normalization, returning (raymaps_norm, depth_channel, scale) exactly like
    each loader's `_apply_mutual_scale`. Moments are divided by the single per-clip scalar
    s = kappa*median(valid depth) and NOT clamped (no VAE range constraint) so depth<->pose stay
    exactly aligned. The DEPTH channel uses parallax_depth_map(cfg): 'log' (default, VAE-robust)
    or 'disparity' (original). `invalid_sentinel` is the value the caller stuffs into invalid
    pixels (use float('inf') if invalid pixels are +inf)."""
    kappa = parallax_kappa(cfg)
    scale = parallax_scale_from_sentinel(depth_metric, cfg, invalid_sentinel)
    raymaps_norm = raymaps_raw.clone()
    if parallax_ray_encoding(cfg) == "companded_origin":
        # Aether-style: camera origin, shared scale s, mu-law companded, broadcast spatially.
        origins = origins_from_plucker(raymaps_raw)              # (T, 3)
        co = mulaw_compand(origins / scale, parallax_mu(cfg))
        raymaps_norm[:, 3:6] = co.view(-1, 3, 1, 1).expand_as(raymaps_norm[:, 3:6])
    else:
        raymaps_norm[:, 3:6] = raymaps_raw[:, 3:6] / scale      # moments -> scene-relative parallax
        clamp_moments_(raymaps_norm, cfg)   # bound the pathology; see PARALLAX_MOMENT_CLAMP_DEFAULT
    if parallax_depth_map(cfg) == "log":
        depth_out = _log_depth(depth_metric, scale, kappa, parallax_log_alpha(cfg))
    else:
        depth_out = _disparity_depth(depth_metric, scale)
    return raymaps_norm, depth_out, scale


def _disparity_depth(depth_metric, scale):
    """Original scheme-C depth map: 2/(1+d/s)-1 in (-1,1], invalid(large d)-> -1."""
    depth_scaled = torch.clamp(depth_metric / scale, min=0.0)
    return 1.0 / (1.0 + depth_scaled) * 2.0 - 1.0


def _disparity_depth_inv(depth_channel, scale):
    """Exact inverse of `_disparity_depth`. x -> d = (2/(x+1) - 1)*s.
    v = (x+1)/2 collapses to 0 for the invalid sentinel and for depth far beyond s, where
    1/v is numerically meaningless; those pixels are reported unrecoverable."""
    v = (depth_channel + 1.0) / 2.0
    unrecoverable = v <= _INV_EPS
    return (1.0 / v.clamp_min(_INV_EPS) - 1.0) * scale, unrecoverable


def _log_depth(depth_metric, scale, kappa, alpha):
    """VAE-robust depth map: v = clip(alpha*ln(d/median + 1), 0, 1); depth = 1 - 2v.
    median = scale/kappa. Orientation matches disparity (near->+1, far->-1); invalid pixels
    (stuffed with a large/inf sentinel) -> v=1 -> -1 automatically. See PARALLAX_DEPTH_MAP note."""
    median = scale / max(kappa, _EPS)
    d_n = torch.clamp(depth_metric, min=0.0) / max(median, _EPS)
    v = torch.clamp(alpha * torch.log(d_n + 1.0), 0.0, 1.0)
    return 1.0 - 2.0 * v


def _log_depth_inv(depth_channel, scale, kappa, alpha):
    """Exact inverse of `_log_depth`. x -> d = (exp(v/alpha) - 1)*median, v = (1-x)/2,
    median = scale/kappa. The forward clamp saturates at v = 1 (d >= (exp(1/alpha)-1)*median,
    i.e. ~27x the median at alpha=0.30), which is also where invalid pixels land; those
    carry no depth information and are reported unrecoverable."""
    median = scale / max(kappa, _EPS)
    v = (1.0 - depth_channel) / 2.0
    unrecoverable = v >= 1.0 - _INV_EPS
    return torch.expm1(v.clamp(0.0, 1.0 - _INV_EPS) / alpha) * median, unrecoverable


def invert_depth_channel(depth_channel, scale: float, cfg):
    """Normalized depth channel in [-1, 1] -> metric depth, dispatching on `cfg` exactly as
    `apply_scale_norm` does on the way in. Returns (metric_depth, unrecoverable_mask).

    Training only needs the forward direction, but anything that reads a predicted depth
    channel -- evaluation, visualisation, point-cloud export -- needs this one, and a caller
    forced to reimplement it is a caller whose formula will drift from the encoder's."""
    if is_parallax_mode(cfg) and parallax_depth_map(cfg) == "log":
        return _log_depth_inv(depth_channel, scale,
                              parallax_kappa(cfg), parallax_log_alpha(cfg))
    return _disparity_depth_inv(depth_channel, scale)


def _divide_and_disparity(raymaps_raw, depth_metric, scale, cfg=None):
    """Shared tail for global_metric (strategy D): divide moments+depth by ONE scalar `scale`,
    depth -> signed disparity 2/(1+d/s)-1 (kept for metric bijectivity). Moments are clamped
    to +/- parallax_moment_clamp(cfg) when a cfg is supplied (default 10.0)."""
    raymaps_norm = raymaps_raw.clone()
    raymaps_norm[:, 3:6] = raymaps_raw[:, 3:6] / scale
    if cfg is not None:
        clamp_moments_(raymaps_norm, cfg)
    return raymaps_norm, _disparity_depth(depth_metric, scale), scale


# ---------------- strategy D: single global metric scale ----------------

def is_global_metric_mode(cfg) -> bool:
    """True iff dataset.scale_mode == 'global_metric'."""
    try:
        return str(cfg.get("scale_mode", None)) == GLOBAL_METRIC_MODE
    except Exception:
        return False


def global_metric_scale(cfg) -> float:
    try:
        return float(cfg.get("global_metric_scale", GLOBAL_METRIC_SCALE_DEFAULT)
                     or GLOBAL_METRIC_SCALE_DEFAULT)
    except Exception:
        return GLOBAL_METRIC_SCALE_DEFAULT


def apply_global_metric(raymaps_raw: torch.Tensor, depth_metric: torch.Tensor, cfg):
    """Strategy D: divide moments AND depth by ONE global constant s (default 10 m, same for
    every dataset/clip). Fully metric-BIJECTIVE — recover metres by moment*s and inverse
    disparity then *s. Depth -> signed modified disparity 2/(1+d/s)-1 (bounded). Invalid pixels
    (stuffed with a large sentinel by the caller) map to disparity ~ -1 automatically."""
    return _divide_and_disparity(raymaps_raw, depth_metric, global_metric_scale(cfg))


def is_scale_norm_mode(cfg) -> bool:
    """True for any opt-in scale scheme handled by apply_scale_norm (parallax or global_metric)."""
    return is_parallax_mode(cfg) or is_global_metric_mode(cfg)


def apply_scale_norm(raymaps_raw: torch.Tensor, depth_metric: torch.Tensor, cfg,
                     invalid_sentinel: float):
    """Dispatch: scale_mode=parallax -> per-clip kappa*median(depth); global_metric -> fixed s.
    Both divide moments+depth by that one scalar and disparity-transform depth (see the two
    apply_* helpers). Returns (raymaps_norm, depth_signed_disparity, scale)."""
    if is_global_metric_mode(cfg):
        return apply_global_metric(raymaps_raw, depth_metric, cfg)
    return apply_parallax(raymaps_raw, depth_metric, cfg, invalid_sentinel)
