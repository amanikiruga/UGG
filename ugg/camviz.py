"""Camera-trajectory video: the whole path in 3D with the current camera highlighted.

Every pose gets a faint frustum, the path is a line with white nodes, and the current frame's
camera is drawn larger in amber with its axes. Each frustum carries an up-marker triangle, so
roll is visible. The view orbits slowly around the path over a floor grid.

Pure numpy + PIL, deterministic frame for frame.

    from ugg.camviz import render_cameras
    frames = render_cameras(pred.c2w, 640, 480)            # list of PIL images, one per pose
"""
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

BG = (247, 247, 249)
PATH = (59, 99, 199)
ACTIVE = (230, 160, 0)
GRID = (167, 174, 185)
SS = 2  # supersample factor, then LANCZOS down for anti-aliasing


def _frustum(length: float) -> np.ndarray:
    """Line segments of a frustum glyph in camera space (OpenCV: +z forward, -y up), (S, 2, 3)."""
    w = length * 0.52
    h = w * 0.75
    c = [(-w, -h, length), (w, -h, length), (w, h, length), (-w, h, length)]
    segs = []
    for i, cor in enumerate(c):
        segs.append(((0, 0, 0), cor))
        segs.append((cor, c[(i + 1) % 4]))
    segs.append(((-w * 0.4, -h, length), (0, -h * 1.5, length)))  # up-marker
    segs.append(((0, -h * 1.5, length), (w * 0.4, -h, length)))
    return np.array(segs, np.float64)


def _look_at(eye, target, up) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    n = np.linalg.norm(s)
    if n < 1e-9:
        s = np.cross(f, np.array([1.0, 0, 0]))
        n = np.linalg.norm(s)
    s /= n
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def _xf(segs: np.ndarray, M: np.ndarray) -> np.ndarray:
    p = segs.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    return p.reshape(segs.shape)


def render_cameras(c2w: np.ndarray, width: int = 640, height: int = 480, num_frames: Optional[int] = None,
                   fov_deg: float = 48.0, orbit_deg: float = 26.0, view_dir: Sequence[float] = (1.2, -0.85, -1.5),
                   fit: float = 0.9) -> List[Image.Image]:
    """Render `num_frames` frames (default: one per pose); the amber camera walks c2w[0..T-1].

    c2w: (T, 4, 4) OpenCV camera-to-world. view_dir: direction from the path centre to the
    viewer (default: behind, right and above the first camera). fit: fraction of the frame the
    path and frusta fill, fixed for the whole clip; 0 uses a plain distance-based framing.
    """
    c2w = np.asarray(c2w, np.float64)
    T = len(c2w)
    N = num_frames or T
    W, H = width, height
    P = c2w[:, :3, 3]
    lo, hi = P.min(0), P.max(0)
    center = (lo + hi) / 2
    radius = max(np.linalg.norm(hi - lo) / 2, 1e-3)

    ghost = _frustum(radius * 0.075)
    active = _frustum(radius * 0.23)
    axes = np.array([((0, 0, 0), (radius * 0.13, 0, 0)),
                     ((0, 0, 0), (0, radius * 0.13, 0)),
                     ((0, 0, 0), (0, 0, radius * 0.16))], np.float64)
    # floor grid just below the lowest camera (OpenCV world is y-down)
    g, gn = radius * 2.1, 10
    gy = hi[1] + radius * 0.18
    grid = []
    for i in range(gn + 1):
        a = -g / 2 + g * i / gn
        grid.append(((center[0] + a, gy, center[2] - g / 2), (center[0] + a, gy, center[2] + g / 2)))
        grid.append(((center[0] - g / 2, gy, center[2] + a), (center[0] + g / 2, gy, center[2] + a)))
    grid = np.array(grid, np.float64)

    up = np.array([0.0, -1.0, 0.0])
    half = min(np.radians(fov_deg / 2), np.arctan(np.tan(np.radians(fov_deg / 2)) * (W / H)))
    dist = radius / np.sin(half) * 0.72
    base = np.asarray(view_dir, float)
    base = base / np.linalg.norm(base)

    def view(k):
        az = np.radians(orbit_deg * (k / max(N - 1, 1) - 0.5))
        ca, sa = np.cos(az), np.sin(az)
        d = np.array([base[0] * ca - base[2] * sa, base[1], base[0] * sa + base[2] * ca])
        return _look_at(center + d * dist, center, up)

    fpx = (H * SS / 2) / np.tan(np.radians(fov_deg / 2))
    cx, cy = W * SS / 2, H * SS / 2
    if fit:
        # one fixed 2D framing for the clip so the path and every active frustum fit across the orbit
        pts = np.concatenate([P] + [_xf(active, c2w[j]).reshape(-1, 3) for j in range(T)])
        xs, ys = [], []
        for k in range(N):
            V = view(k)
            q = pts @ V[:3, :3].T + V[:3, 3]
            z = np.maximum(-q[:, 2], 1e-6)
            xs.append(q[:, 0] / z)
            ys.append(-q[:, 1] / z)
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        a, b, c_, d_ = xs.min(), xs.max(), ys.min(), ys.max()
        fpx = fit * min(W * SS / max(b - a, 1e-9), H * SS / max(d_ - c_, 1e-9))
        cx, cy = W * SS / 2 - fpx * (a + b) / 2, H * SS / 2 - fpx * (c_ + d_) / 2

    out = []
    for k in range(N):
        V = view(k)
        im = Image.new("RGB", (W * SS, H * SS), BG)
        dr = ImageDraw.Draw(im, "RGBA")

        def proj(p):
            q = p @ V[:3, :3].T + V[:3, 3]
            z = -q[..., 2]
            ok = z > 1e-6
            u = np.where(ok, q[..., 0] / np.maximum(z, 1e-6) * fpx + cx, -1e9)
            v = np.where(ok, -q[..., 1] / np.maximum(z, 1e-6) * fpx + cy, -1e9)
            return u, v, ok

        def lines(segs, col, w):
            u, v, ok = proj(segs)
            for i in range(len(segs)):
                if ok[i, 0] and ok[i, 1]:
                    dr.line([u[i, 0], v[i, 0], u[i, 1], v[i, 1]], fill=col, width=w)

        lines(grid, GRID + (56,), SS)
        for i in range(T):
            lines(_xf(ghost, c2w[i]), PATH + (61,), SS)
        pu, pv, pok = proj(P)
        nodes = [(pu[i], pv[i]) for i in range(T) if pok[i]]
        if len(nodes) > 1:
            dr.line([c for p in nodes for c in p], fill=PATH + (166,), width=2 * SS)
        r = max(2.0 * SS, radius * 0.016 * fpx / dist)
        for x, y in nodes:
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255), outline=PATH + (200,), width=SS)
        j = int(round(k * (T - 1) / max(N - 1, 1)))
        lines(_xf(active, c2w[j]), ACTIVE + (255,), 2 * SS)
        lines(_xf(axes, c2w[j]), ACTIVE + (190,), 2 * SS)
        au, av, aok = proj(P[j:j + 1])
        if aok[0]:
            s = max(3.0 * SS, radius * 0.026 * fpx / dist)
            dr.ellipse([au[0] - s, av[0] - s, au[0] + s, av[0] + s], fill=ACTIVE + (255,))
        out.append(im.resize((W, H), Image.LANCZOS))
    return out
