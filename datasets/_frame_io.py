"""Threaded per-frame reads, shared by the geometry loaders.

Every loader read its 50 frames in a serial Python loop. Measured on BEDLAM PNGs, that
runs at 53 MB/s while the same filesystem sustains ~726 MB/s under 16 threads -- a 13.8x
speedup, because PIL/imageio/numpy all release the GIL inside the decoder. Profiling put
these loops at 63-90% of __getitem__ for the slowest subsets.

`ex.map` preserves input order, so stacked frames stay in temporal order.
"""
import os
from concurrent.futures import ThreadPoolExecutor

FRAME_THREADS = int(os.environ.get("VWM_FRAME_THREADS", "16"))


def map_frames(fn, items):
    """[fn(x) for x in items], on a thread pool. Order preserved. Falls back to a plain
    loop for tiny inputs or when VWM_FRAME_THREADS<=1 (set that to debug a reader)."""
    items = list(items)
    if len(items) < 2 or FRAME_THREADS <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(FRAME_THREADS, len(items))) as ex:
        return list(ex.map(fn, items))
