"""Scale-preserving constants for geometry datasets.

PRIMARY (2026-06-29): each dataset declares its OWN `mutual_scale` in its config yaml
(= that dataset's measured depth-P90), and `_apply_mutual_scale` uses it when
scale_preserving=True (`float(self.cfg.get("mutual_scale") or GLOBAL_SCALE)`). This divides
BOTH depth and camera-moments by a PER-DATASET constant so every dataset's depth-disparity
distribution lands in a similar range (datasets "look the same") while metric scale is
preserved WITHIN each dataset. GLOBAL_SCALE below is only the FALLBACK when a yaml omits
mutual_scale. Per-dataset values (depth-P90): tartanair 21, tartanair_v2 23, spring 42,
bedlam 63 (bedlam2 reuses it), point_odyssey 10, vkitti2 89, ase 4.5
(datasets/global_scale_all.json); omniworld_game 0.19 (game-engine-arbitrary units,
datasets/global_scale_omniworld.json). NVS-only subsets (re10k, dl3dv_colmap) have no
depth and normalize only camera moments — see their own module constants.

----- (original single-global-constant design, now the fallback) -----

When a geometry dataset runs with `cfg.scale_preserving=True`, `_apply_mutual_scale`
divides BOTH the Plucker camera-moments AND the metric depth by this ONE constant
(instead of a per-clip max/median). That preserves the global metric scale across the
whole mixture (recover metres by multiplying back) and keeps camera/depth parallax metric.

GLOBAL_SCALE is chosen (scripts/compute_global_scale.py over ~50 clips x each dataset) so:
  - it is >= the ~P95 of camera-moment magnitude over the mixture, so raymap moments are
    NOT clipped to +/-1 for the bulk of clips and are large enough to be predictable;
  - it sits around the combined depth P90, so modified-disparity `1/(1+d/scale)*2-1`
    spreads the bulk of depths across a usable range (near -> ~+1, far -> ~-0.5) rather
    than collapsing to a narrow band.
See datasets/global_scale_all.json for the measured per-dataset depth/moment percentiles.
"""

# Chosen from 50 clips x each of 7 datasets (scripts/compute_global_scale.py ->
# datasets/global_scale_all.json), 2026-06-29:
#   dataset       depthP50 depthP90 depthP95   momP95  momMax
#   tartanair       5.5      21       42         5.3     20
#   tartanair_v2    4.9      23       50         8.4     33
#   spring          8.7      42       69         0.45     4   (near-static cam)
#   bedlam          7.1      63       68         0.20     6   (near-static cam)
#   point_odyssey   3.3      10       15         0.78     4
#   vkitti2        17        89      129         8.4     43   (driving: far depth, big motion)
#   ase             2.2       4.5      5.4       2.5      6   (indoor walking: small scenes)
# 30 = pooled depth-P90 (~36) and 3.6x the max moment-P95 (8.4): typical camera motion of
# every dataset stays UNCLIPPED (momP95/30 = 0.06-0.28, predictable / not vanishing), far
# depths stay non-saturated (disparity ~ -0.6), and the bulk of depths spread across a usable
# range. Single constant => global metric scale preserved (recover metres by x30). The cost
# (unavoidable for ONE constant over a 60x depth range): ASE 2-5m scenes compress near +1, and
# vkitti2's rare >30m driving moments clip — both accepted to keep cross-dataset scale metric.
GLOBAL_SCALE = 30.0
