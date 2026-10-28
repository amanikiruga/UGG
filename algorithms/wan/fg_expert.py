"""Canonical definition of the FG depth expert -- ONE source of truth.

Both consumers import from here so they can never disagree:
  * scripts/splice_fg_checkpoint.py  -- which tensors come from G
  * the training algorithm            -- which tensors stay trainable when the rest is frozen

A disagreement between those two would be silent and severe: a tensor spliced from G but
left frozen would never adapt, and a tensor kept from F but left trainable would drift the
"frozen" branches. Hence one module.

Layout of the MoT-concat 5B (algorithms/wan/modules/model_ray_depth_mot_concat_5b.py):
per-modality weights live in ModuleList(3) slots [0]=rgb, [1]=ray, [2]=depth, over a single
UNMASKED joint attention across the concatenated [rgb | ray | depth] token sequence.
"""
import re

DEPTH_SLOT = 2

# Whole tensors belonging exclusively to the depth expert.
DEPTH_EXPERT_PATTERNS = (
    r"^(?:model\.)?blocks\.\d+\.self_attn\.[qkvo]\.2\.(?:weight|bias)$",
    r"^(?:model\.)?blocks\.\d+\.self_attn\.norm_[qk]\.2\.weight$",
    r"^(?:model\.)?blocks\.\d+\.ffn\.2\.\d+\.(?:weight|bias)$",
    r"^(?:model\.)?patch_embedding_depth\.(?:weight|bias)$",
    r"^(?:model\.)?head_depth\.(?:modulation|head\.(?:weight|bias))$",
)

# One tensor holding all three modality embeddings, (3, 1, 1, dim); depth is slice [2].
# It is SHARED, so under a freeze it must stay trainable as a whole (slots 0/1 receive no
# gradient anyway: rgb's slot is initialised to zeros and each slot is only read by its own
# modality's tokens).
PARTIAL_MODALITY_PATTERNS = (
    r"^(?:model\.)?blocks\.\d+\.self_attn\.ray_embedding$",
)

# Tensors that must NEVER be treated as depth. Asserted, not merely implied by the patterns.
MUST_STAY_PATTERNS = (
    r"^(?:model\.)?blocks\.\d+\.modulation$",                 # (1,6,dim) SHARED across modalities
    r"^(?:model\.)?blocks\.\d+\.cross_attn\.",                # collapsed to one shared slot (text)
    r"^(?:model\.)?head_(?:rgb|ray|mod)\.",                   # head_ray AND head_mod are both RAY
    r"^(?:model\.)?patch_embedding(?:_ray)?\.(?:weight|bias)$",
    r"^(?:model\.)?blocks\.\d+\.(?:self_attn\.[qkvo]|self_attn\.norm_[qk]|ffn)\.[01]\.",
)
# NOTE: norm1 / norm2 are nn.ModuleList(3) of WanLayerNorm(elementwise_affine=False) and so
# hold NO parameters at all -- there is nothing to classify or transplant for them.

_DEPTH   = tuple(re.compile(p) for p in DEPTH_EXPERT_PATTERNS)
_PARTIAL = tuple(re.compile(p) for p in PARTIAL_MODALITY_PATTERNS)
_STAY    = tuple(re.compile(p) for p in MUST_STAY_PATTERNS)


def is_depth_expert(name: str) -> bool:
    """True for a tensor owned entirely by the depth expert."""
    return any(p.match(name) for p in _DEPTH)


def is_partial_modality(name: str) -> bool:
    """True for a shared tensor whose depth slice ([2]) belongs to the depth expert."""
    return any(p.match(name) for p in _PARTIAL)


def must_stay(name: str) -> bool:
    """True for a tensor that must remain F's under every circumstance."""
    return any(p.match(name) for p in _STAY)


def is_trainable_under_fg(name: str) -> bool:
    """FG freezes rgb/ray/pose and trains the depth expert (plus the shared modality
    embedding, whose non-depth slots receive no gradient)."""
    return is_depth_expert(name) or is_partial_modality(name)


def classify(names):
    """(depth, partial, frozen, contradictions) for a list of parameter names."""
    depth   = [n for n in names if is_depth_expert(n)]
    partial = [n for n in names if is_partial_modality(n)]
    frozen  = [n for n in names if not is_trainable_under_fg(n)]
    contra  = [n for n in depth + partial if must_stay(n)]
    return depth, partial, frozen, contra


def expected_depth_key_count(num_blocks: int) -> int:
    """30 blocks x (8 qkvo + 2 qk-norm + 4 ffn) + patch_embedding_depth(2) + head_depth(3)."""
    return num_blocks * 14 + 2 + 3


# ---------------------------------------------------------------------------
# GF (the inverse of FG): freeze the DEPTH expert, train rgb/ray/pose.
#
# Motivation (2026-09-12): eff-2500 has the best depth we have measured
# (bonn 0.059745 / kitti 0.083316 / sintel 0.367010) but weak NVS (23.068),
# while F has the best NVS (24.5393) and worse depth. Splicing eff-2500's depth
# expert into F FAILED outright (NVS 21.6978, bonn 0.084365, kitti 0.123652 --
# worse than BOTH parents), because the transplanted expert had never co-adapted
# to F's joint attention. So instead of moving weights, start FROM eff-2500 and
# move the rgb/ray side by gradient descent, holding depth fixed.
#
# SYMMETRIC CAVEAT to the one in the FG hook: freezing the depth expert does NOT
# make depth invariant. Self-attention is unmasked over [rgb | ray | depth], so
# the depth tokens read rgb/ray k/v and depth OUTPUTS drift as rgb/ray adapts.
# This bounds the drift; it does not remove it. Depth must be re-measured on
# every checkpoint, and the run stopped when it regresses past tolerance.
#
# The shared modality embedding (self_attn.ray_embedding, slots [0]=rgb [1]=ray
# [2]=depth) stays TRAINABLE here: its rgb/ray slices are exactly what we want to
# move, and slot [2] receives gradient only from depth tokens. If that turns out
# to matter, zero its [2] grad in an optimizer hook rather than freezing the tensor.
# ---------------------------------------------------------------------------

def is_trainable_under_gf(name: str) -> bool:
    """GF freezes the depth expert and trains everything else (rgb/ray/pose,
    cross-attention, the shared modulation, and the shared modality embedding)."""
    return not is_depth_expert(name)


def classify_gf(names):
    """(frozen_depth, trainable, contradictions) for a list of parameter names."""
    frozen    = [n for n in names if is_depth_expert(n)]
    trainable = [n for n in names if not is_depth_expert(n)]
    contra    = [n for n in frozen if must_stay(n)]
    return frozen, trainable, contra
