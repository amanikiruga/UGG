"""Fast CPU checks: config composition, video IO, depth decoding, camera recovery.

    python tests/test_cpu.py        (or: python -m pytest tests/test_cpu.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ugg import RELEASE_CKPT, compose_config, depth_from_norm, prepare_video  # noqa: E402
from ugg.cameras import _MOMENT_ABS_MAX, recover_cameras  # noqa: E402
from ugg.io import legal_num_frames, select_frames  # noqa: E402


def test_every_module_imports():
    import importlib

    pkgs = ["algorithms", "datasets", "experiments", "utils", "ugg"]
    mods = [".".join(f.relative_to(ROOT).with_suffix("").parts) for p in pkgs
            for f in sorted((ROOT / p).rglob("*.py"))]
    mods = [m[: -len(".__init__")] if m.endswith(".__init__") else m for m in mods]
    for m in mods:
        importlib.import_module(m)


def test_inference_config_defaults():
    cfg = compose_config()
    a = cfg.algorithm
    assert a.model.tuned_ckpt_path == RELEASE_CKPT
    assert (a.sample_steps, a.hist_guidance, a.lang_guidance) == (40, 1.0, 0.0)
    assert a.diffusion_forcing.mode == "ray_depth_prediction_firstlast"
    assert a.load_prompt_embed and cfg.seed == 0 and cfg.mode == "dense"
    assert (a.height, a.width) == (288, 384)
    assert not a.vae.encoder_compile and not a.vae.merge_flf_chunks  # inference encodes eagerly
    cfg = compose_config(["algorithm.sample_steps=7", "seed=3"])
    assert cfg.algorithm.sample_steps == 7 and cfg.seed == 3


def test_training_recipe():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configurations")):
        cfg = compose("config", overrides=["+name=t", "experiment=ugg_train", "data_dir=/data"])
    a, e = cfg.algorithm, cfg.experiment
    subsets = {k: v for k, v in cfg.dataset.items() if k.startswith("subset/")}
    assert len(subsets) == 14
    for name, sub in subsets.items():  # every path hangs off the one data_dir
        assert sub.data_root.startswith("/data/"), (name, sub.data_root)
    assert e.training.lr == 6e-6 and e.training.batch_size == 8 and e.training.optim.gradient_clip_val == 1.0
    assert e.fsdp_use_orig_params and a.gf_freeze_depth  # frozen depth expert needs use_orig_params
    assert a.lr_scheduler.name == "warmup_hold_cosine" and a.lr_scheduler.min_lr_rate == 1.0
    assert a.gradient_checkpointing_rate == 0.93 and a.vae.encoder_compile and a.vae.merge_flf_chunks
    df = a.diffusion_forcing
    assert df.mode == "ray_depth_prediction_firstlast" and df.train_rgb and df.mask_clean_token_loss
    assert list(df.subset_settings.re10k) == [0, 1, 2, 3]
    assert (cfg.dataset.height, cfg.dataset.width, cfg.dataset.n_frames) == (288, 384, 50)


def test_legal_num_frames():
    for n, m in [(2, 2), (5, 2), (6, 6), (49, 46), (50, 50), (110, 50)]:
        assert legal_num_frames(n) == m and m % 4 == 2


def test_prepare_video_crops_and_normalizes():
    video = (np.random.rand(60, 480, 854, 3) * 255).astype(np.uint8)
    frames, idx = prepare_video(video)
    assert frames.shape == (50, 3, 288, 384) and idx.tolist() == list(range(50))
    assert frames.min() >= -1 and frames.max() <= 1
    _, idx = prepare_video(video, frame_mode="uniform")
    assert idx[0] == 0 and idx[-1] == 59 and len(idx) == 50


def test_depth_decode():
    norm = np.random.RandomState(0).uniform(-1, 1, (4, 8, 8)).astype(np.float32)
    d = depth_from_norm(norm)
    disp = np.clip((norm + 1) / 2, max(np.percentile((norm + 1) / 2, 2.0), 1e-3), 1 - 1e-3)
    np.testing.assert_allclose(d, 1 / disp - 1, rtol=1e-6)
    # monotone: larger disparity -> smaller depth
    d0 = depth_from_norm(np.linspace(-0.9, 0.9, 10)[None, None], percentile_clip=None)
    assert np.all(np.diff(d0.ravel()) < 0)


def _raymap(K, c2w, H, W):
    v, u = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    pix = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    d = pix @ c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    m = np.cross(np.broadcast_to(c2w[:3, 3], d.shape), d)
    return np.concatenate([d, m / _MOMENT_ABS_MAX], -1).transpose(2, 0, 1)


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_camera_recovery_roundtrip():
    for shared in (False, True):
        _check_roundtrip(shared)


def _check_roundtrip(shared):
    H, W = 36, 48
    K = np.array([[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1]])
    poses = []
    for i in range(6):
        T = np.eye(4)
        T[:3, :3] = _rot("y", 4 * i) @ _rot("x", 2 * i)
        T[:3, 3] = [0.3 * i, 0.05 * i, 0.1 * i]
        poses.append(T)
    rays = torch.from_numpy(np.stack([_raymap(K, T, H, W) for T in poses])).float()
    cams = recover_cameras(rays, shared_intrinsics=shared)
    # Camera centres are defined up to one global scale (the shared-K path returns them in
    # model-native units; pose metrics are Sim(3)-aligned).
    t_gt = np.stack([T[:3, 3] for T in poses])
    t = cams.c2w[:, :3, 3]
    s = (t * t_gt).sum() / (t * t).sum()
    np.testing.assert_allclose(s * t, t_gt, atol=2e-2)
    for T, Tp, Kp in zip(poses, cams.c2w, cams.intrinsics):
        np.testing.assert_allclose(Tp[:3, :3], T[:3, :3], atol=2e-3)
        np.testing.assert_allclose(Kp[:2, :3], K[:2, :3], atol=0.5)  # focal + principal point, px


def test_camera_and_depth_encoding_roundtrip():
    from ugg.cameras import encode_cameras
    from ugg.depth import encode_depth

    K = np.array([[300.0, 0, 192], [0, 300.0, 144], [0, 0, 1]])
    poses = []
    for i in range(6):
        T = np.eye(4)
        T[:3, :3] = _rot("y", 3 * i)
        T[:3, 3] = [0.2 * i, 0.0, 0.1 * i]
        poses.append(T)
    poses = np.stack(poses)
    rays, scale = encode_cameras(poses, K, (288, 384), (36, 48))
    assert rays.shape == (6, 6, 36, 48) and rays[:, 3:].abs().max() <= 1 and scale >= 1
    cams = recover_cameras(rays, shared_intrinsics=True)  # model-native units: one global scale
    t_gt, t = poses[:, :3, 3], cams.c2w[:, :3, 3]
    s = (t * t_gt).sum() / (t * t).sum()
    np.testing.assert_allclose(s * t, t_gt, atol=2e-2)
    np.testing.assert_allclose(cams.c2w[:, :3, :3], poses[:, :3, :3], atol=3e-3)
    d = np.random.RandomState(0).uniform(0.5, 8, (2, 4, 4)).astype(np.float32)
    d[0, 0, 0] = 0  # invalid
    e = encode_depth(d)
    assert e[0, 0, 0] == -1 and (e[d > 0] > -1).all() and (e <= 1).all()
    back = depth_from_norm(np.where(d > 0, e, 0.0), percentile_clip=None)
    np.testing.assert_allclose(back[d > 0] / np.median(back[d > 0]), d[d > 0] / np.median(d[d > 0]), rtol=2e-3)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
