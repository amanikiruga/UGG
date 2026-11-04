"""Threaded per-frame reads, shared by the geometry loaders.

PIL/imageio/numpy release the GIL inside their decoders, so reading a clip's frames on a
thread pool is much faster than a serial loop.

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
