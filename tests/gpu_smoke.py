"""GPU smoke test of every inference task with one model load (short clips, few steps).

    python tests/gpu_smoke.py [--out DIR]

Uses the sample clip in assets/. Covers the six tasks (video, image+cameras, first/last+cameras,
text, cameras only, depth only), the sparse protocol, seed determinism, history guidance on/off,
T5 loading + text guidance, and Prediction.save().
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ugg import UGG  # noqa: E402

A = ROOT / "assets"
VIDEO = A / "bonn_balloon2" / "frames.mp4"
FIRST, LAST = A / "re10k_bedroom" / "first.png", A / "re10k_bedroom" / "last.png"
M = 10  # frames per test clip


def check(pred, m=M, H=288, W=384):
    assert pred.rgb.shape == (m, H, W, 3) and pred.rgb.dtype == np.uint8
    assert pred.depth_norm.shape == (m, H, W) and np.isfinite(pred.depth_norm).all()
    assert pred.depth.shape == (m, H, W) and (pred.depth > 0).all()
    assert pred.c2w.shape == (m, 4, 4) and np.isfinite(pred.c2w).all()
    assert pred.intrinsics.shape == (m, 3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: the release checkpoint)")
    args = ap.parse_args()
    out = Path(args.out or tempfile.mkdtemp())
    geo = np.load(A / "re10k_bedroom" / "cameras.npz")
    cams = (geo["c2w"][:M], geo["intrinsics"][:M])
    depth = np.load(A / "dl3dv_treatment_room" / "depth.npz")["depth"][:M].astype(np.float32)
    kw = dict(sample_steps=4, seed=0)

    t = time.time()
    model = UGG.from_pretrained(ckpt=sys.argv[sys.argv.index('--ckpt') + 1] if '--ckpt' in sys.argv else None)
    print(f"load {time.time() - t:.0f}s")

    # 1. video -> 4D (dense) and the sparse protocol
    a = model.video_to_4d(VIDEO, max_frames=M, **kw)
    check(a)
    assert np.array_equal(a.depth_norm, model(VIDEO, max_frames=M, **kw).depth_norm), "same seed must reproduce"
    assert not np.array_equal(a.depth_norm, model(VIDEO, max_frames=M, sample_steps=4, seed=1).depth_norm)
    check(model(VIDEO, mode="sparse", max_frames=M, **kw))
    g0 = model.video_to_4d(VIDEO, max_frames=M, hist_guidance=0.0, **kw)
    assert not np.array_equal(g0.depth_norm, a.depth_norm)
    print("video->4D ok (dense, sparse, seeds, hist_guidance=0)")

    # 2. image + cameras -> 4D, and image alone
    p = model.image_to_4d(FIRST, cameras=cams, **kw)
    check(p)
    assert p.meta["conditioned"] == ["rgb", "cameras"]
    check(model.image_to_4d(FIRST, num_frames=M, **kw))
    print("image(+cameras)->4D ok")

    # 3. first + last + cameras -> 4D
    p = model.first_last_to_4d(FIRST, LAST, cameras=cams, **kw)
    check(p)
    print("first/last+cameras->4D ok")

    # 4. text -> 4D (loads T5), with text guidance
    t = time.time()
    p = model.text_to_4d("a person walks through an office holding balloons", num_frames=M,
                         text_guidance=2.0, **kw)
    check(p)
    assert model.model.text_encoder is not None and p.meta["hist_guidance"] == 0.0
    print(f"text->4D ok ({time.time() - t:.0f}s incl. T5 load)")

    # 5. cameras only -> 4D; the given trajectory comes back out of the rays
    p = model.camera_to_4d(cams, **kw)
    check(p)
    rel = lambda c: np.linalg.inv(c[0]) @ c  # noqa: E731
    t_in, t_out = rel(cams[0])[:, :3, 3], rel(p.c2w)[:, :3, 3]
    cos = (t_in[1:] * t_out[1:]).sum(-1) / (np.linalg.norm(t_in[1:], axis=-1) * np.linalg.norm(t_out[1:], axis=-1) + 1e-9)
    assert np.median(cos) > 0.9, f"conditioned trajectory not reproduced (median cos {np.median(cos):.2f})"
    print(f"camera->4D ok (trajectory direction cos {np.median(cos):.3f})")

    # 6. depth only -> 4D; the given depth comes back out (up to scale)
    p = model.depth_to_4d(depth, **kw)
    check(p)
    v = depth > 0
    corr = np.corrcoef(np.log(p.depth[v]), np.log(depth[v]))[0, 1]
    room = A / "dl3dv_treatment_room"
    check(model.depth_to_4d(depth, images=[room / "first.png", room / "last.png"], **kw))
    assert corr > 0.9, f"conditioned depth not reproduced (log-depth corr {corr:.2f})"
    print(f"depth->4D ok (log-depth corr {corr:.3f})")

    try:
        model.video_to_4d(VIDEO, max_frames=M, text_guidance=1.0, sample_steps=4)
        raise AssertionError("text_guidance without a prompt must raise")
    except ValueError:
        pass
    a.save(out / "video")
    assert all((out / "video" / f).exists() for f in ("rgb.mp4", "depth.mp4", "depth.npy", "c2w.npy", "meta.json"))
    print(f"save ok -> {out}")
    print("GPU SMOKE PASS")


if __name__ == "__main__":
    main()
