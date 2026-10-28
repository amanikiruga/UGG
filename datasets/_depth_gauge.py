"""Scheme-G metric depth gauge for pose-only (NVS) datasets — "G3".

WHY THIS EXISTS
---------------
Scheme G (`dataset.scale_mode=parallax`, `parallax_ray_encoding=moment`) divides the
Plucker moments AND the depth channel by one shared per-clip scalar

    s = kappa * median(valid metric depth of the clip)

so the moment channel encodes *scene-relative parallax*. Geometry datasets (TartanAir,
vkitti2, ...) have metric depth, so they can compute `s`. The NVS datasets do not:
RE10K/DL3DV carry COLMAP poses with an arbitrary per-clip scale and no depth at all.

Before this module those loaders silently fell back to per-clip `max|moment|`
normalization. In a mixture that is a real bug, not a cosmetic one: the SAME ray channel
then carries two incompatible magnitude codes at once —

    geometry half : |m| = metric parallax, unclamped, varies freely per clip
    NVS half      : |m| = fraction of this clip's own max, always exactly 1.0 at the peak

They are spatially indistinguishable (both are Plucker moments), so the shared ray branch
is pulled toward two different readings of the same number. Measured on the training
loaders: mean |m| 0.047 (TartanAir, scheme G) vs 0.14 (RE10K, max-norm fallback).

WHAT THIS MODULE PROVIDES
-------------------------
A per-scene table, built offline, giving each NVS scene its missing gauge:

    s_colmap_units = kappa * median_metric_depth_m * cu_per_m

`cu_per_m` converts metres into that scene's arbitrary COLMAP units. Both numbers come
from running a metric geometry model (DA3NESTED-GIANT-LARGE) over frames of the scene
and solving for the depth scale that is multi-view consistent with the scene's OWN
annotation cameras — NOT from the model's predicted camera scale, which was measured
wrong by up to 3x on TartanAir ground truth. Validated against ScanNet v2 metric GT to
~3% (median reprojection-consistency residual 0.38% over 72,519 RE10K scenes).

The table is a plain .npz; see `scripts/build_depth_gauge.py` (not shipped) for
how it is produced. The loader only reads it.

USAGE (dataset yaml)
--------------------
    scale_mode: parallax
    parallax_kappa: 0.5
    parallax_ray_encoding: moment
    parallax_depth_map: log
    parallax_gauge_file: /path/to/gauge_v1.npz

KAPPA IS FREE TO RETUNE
-----------------------
Under the log depth map the median is `scale/kappa` while `scale = kappa*median`, so
kappa CANCELS EXACTLY in the depth channel (verified 0.0 difference across kappa in
{1,2,4,8}). It is purely a moment-channel gain knob, which is why G3 can use kappa=0.5
to restore moment amplitude without touching the depth representation that scheme G is
good at.
"""
from typing import Optional, Sequence

import numpy as np

# One table shared by every dataset instance in a worker process. The table is ~30 MB
# and is read once per worker; keep it UNCOMPRESSED on disk (decompressing 70k unicode
# scene ids costs ~17 s per worker vs 0.02 s raw).
_TABLE_CACHE: dict = {}


def _load_table(path: str) -> dict:
    """{scene_id: row index} plus the flat arrays; slices are taken on demand.

    Storing an index rather than 72k pre-sliced tuples costs ~0 MB instead of ~52 MB and
    builds 10x faster -- and that memory is paid once PER WORKER PROCESS (num_workers x
    ranks), so the pre-sliced form was ~1 GB across a 20-worker launch for no benefit.
    """
    if path in _TABLE_CACHE:
        return _TABLE_CACHE[path]
    z = np.load(path)
    tbl = {
        "idx": {s: i for i, s in enumerate(z["scene_ids"].astype(str).tolist())},
        "off": z["offsets"], "ts": z["ts_flat"], "med": z["med_flat"], "cu": z["cu_per_m"],
        "path": path,
    }
    _TABLE_CACHE[path] = tbl
    return tbl


def gauge_scale(cfg, scene_id: str, clip_timestamps: Optional[Sequence[int]] = None,
                kappa: Optional[float] = None) -> float:
    """s in COLMAP units for this clip.

    `clip_timestamps` are the annotation timestamps of the frames actually sampled; the
    clip's median depth is taken over the nearest gauged frames, so a 50-frame crop of a
    long scene is gauged by the part of the scene it actually shows.

    A scene with no gauge is a FATAL error, deliberately. The alternatives are both worse:
    a population-median prior trains the clip on a scale we have no evidence for, and a
    max-norm fallback puts a second, incompatible magnitude code on the ray channel --
    the exact bug this gauge exists to remove. Ungauged scenes must be pruned from the
    dataset's records instead, so they are excluded loudly and once, not silently per batch.
    """
    path = cfg.get("parallax_gauge_file", None)
    if not path:
        raise ValueError("gauge_scale called without parallax_gauge_file")
    if kappa is None:
        from ._scale_norm import parallax_kappa
        kappa = parallax_kappa(cfg)
    tbl = _load_table(str(path))
    i = tbl["idx"].get(str(scene_id))
    if i is None:
        raise KeyError(
            f"scene {scene_id!r} has no entry in {tbl['path']}. Prune ungauged scenes "
            "from the dataset records; there is no scale fallback by design.")
    o = tbl["off"]
    ts_s, med_s = tbl["ts"][o[i]:o[i + 1]], tbl["med"][o[i]:o[i + 1]]
    if clip_timestamps is None or len(ts_s) == 0:
        med_clip = float(np.median(med_s))
    else:
        q = np.asarray(list(clip_timestamps), dtype=np.int64)
        j = np.clip(np.searchsorted(ts_s, q), 0, len(ts_s) - 1)
        jm = np.clip(j - 1, 0, len(ts_s) - 1)
        nearest = np.where(np.abs(ts_s[jm] - q) <= np.abs(ts_s[j] - q), jm, j)
        med_clip = float(np.median(med_s[nearest]))
    s = kappa * med_clip * float(tbl["cu"][i])
    if not (s > 1e-8):
        raise ValueError(f"scene {scene_id!r} gauged to a degenerate scale s={s!r}")
    return s


def is_gauge_mode(cfg) -> bool:
    """True iff this cfg selects scheme G *with* a depth gauge (i.e. G3)."""
    from ._scale_norm import is_parallax_mode, parallax_ray_encoding
    return (is_parallax_mode(cfg)
            and parallax_ray_encoding(cfg) == "moment"
            and bool(cfg.get("parallax_gauge_file", None)))
