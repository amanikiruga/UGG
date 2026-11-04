"""Scale-preserving constants for geometry datasets.

With `scale_preserving=True`, `_apply_mutual_scale` divides both the Pluecker camera moments
and the metric depth by one constant (instead of a per-clip max/median), so metric scale is
preserved (multiply back to recover metres) and camera/depth parallax stays metric.

Each dataset declares its own constant as `mutual_scale` in its yaml (roughly its depth P90),
so every dataset's disparity lands in a similar range. GLOBAL_SCALE is the fallback when a
yaml omits it. NVS-only subsets (re10k, dl3dv_colmap) have no depth and normalize only the
camera moments; see their module constants.
"""

# Fallback: about the pooled depth P90 of the geometry datasets and well above their moment P95,
# so typical camera motion is not clipped and the bulk of depths spread over a usable range.
GLOBAL_SCALE = 30.0
