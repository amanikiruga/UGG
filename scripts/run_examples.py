"""Run every task on the bundled samples with the settings that work best for each.

    python scripts/run_examples.py [--out outputs/examples] [--only text camera]

Each example writes rgb.mp4, depth.mp4, *.npy and meta.json under <out>/<name>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ugg import UGG  # noqa: E402

A = ROOT / "assets"


def cams(scene):
    g = np.load(A / scene / "cameras.npz")
    return g["c2w"], g["intrinsics"]


def first(scene):
    return A / scene / "first.png"


def last(scene):
    return A / scene / "last.png"


def caption(scene):
    """The dataset's caption for the scene, used as the prompt (as during training)."""
    return (A / scene / "caption.txt").read_text().strip()


PROMPT = json.load(open(A / "text_prompts.json"))["prompts"][0]

# name -> (task, lambda model: prediction). Settings are per task; see README "The six tasks".
EXAMPLES = {
    "video_bonn": lambda m: m.video_to_4d(A / "bonn_balloon2" / "frames.mp4"),
    "image_re10k_staircase": lambda m: m.image_to_4d(
        first("re10k_staircase"), cameras=cams("re10k_staircase"),
        hist_guidance=1.5, sample_shift=0.9, seed=2),
    "image_dl3dv_gallery": lambda m: m.image_to_4d(
        first("dl3dv_gallery"), cameras=cams("dl3dv_gallery"), prompt=caption("dl3dv_gallery"),
        hist_guidance=1.5, seed=0),
    "first_last_re10k_bedroom": lambda m: m.first_last_to_4d(
        first("re10k_bedroom"), last("re10k_bedroom"), cameras=cams("re10k_bedroom"),
        hist_guidance=1.5, sample_shift=0.7, sample_steps=60, seed=97),
    "first_last_dl3dv_gallery": lambda m: m.first_last_to_4d(
        first("dl3dv_gallery"), last("dl3dv_gallery"), cameras=cams("dl3dv_gallery"),
        prompt=caption("dl3dv_gallery"), hist_guidance=1.5, seed=0),
    "text": lambda m: m.text_to_4d(PROMPT, num_frames=50, text_guidance=5.0, seed=11),
    "camera_dl3dv_gallery": lambda m: m.camera_to_4d(cams("dl3dv_gallery"), prompt=caption("dl3dv_gallery"),
                                                   hist_guidance=2.0, seed=29),
    "depth_dl3dv_treatment_room": lambda m: m.depth_to_4d(
        np.load(A / "dl3dv_treatment_room" / "depth.npz")["depth"].astype(np.float32),
        images=[first("dl3dv_treatment_room"), last("dl3dv_treatment_room")],
        prompt=caption("dl3dv_treatment_room"), hist_guidance=2.0, seed=59),
    "depth_only_dl3dv_treatment_room": lambda m: m.depth_to_4d(
        np.load(A / "dl3dv_treatment_room" / "depth.npz")["depth"].astype(np.float32),
        prompt=caption("dl3dv_treatment_room"), hist_guidance=2.0, seed=59),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "examples"))
    ap.add_argument("--only", nargs="*", default=None, help="substrings of example names to run")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    names = [n for n in EXAMPLES if not args.only or any(s in n for s in args.only)]
    model = UGG.from_pretrained(ckpt=args.ckpt)
    for name in names:
        t = time.time()
        pred = EXAMPLES[name](model)
        pred.save(Path(args.out) / name)
        print(f"{name}: {len(pred.rgb)} frames, given {pred.meta['conditioned']} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
