"""Command-line inference for the six UGG tasks (sample inputs in assets/).

    python infer.py task=video      input=assets/bonn_balloon2/frames.mp4
    python infer.py task=video      input=assets/bonn_balloon2/frames.mp4 mode=sparse
    python infer.py task=image      input=assets/re10k_staircase/first.png cameras=assets/re10k_staircase/cameras.npz
    python infer.py task=first_last input=assets/re10k_bedroom/first.png last=assets/re10k_bedroom/last.png \
                                    cameras=assets/re10k_bedroom/cameras.npz
    python infer.py task=text       prompt="The camera glides forward down a long marble corridor..." algorithm.lang_guidance=5
    python infer.py task=camera     cameras=assets/dl3dv_gallery/cameras.npz algorithm.hist_guidance=2
    python infer.py task=depth      depth=assets/dl3dv_treatment_room/depth.npz algorithm.hist_guidance=2 \
                                    input=assets/dl3dv_treatment_room/first.png last=assets/dl3dv_treatment_room/last.png

Writes rgb.mp4, depth.mp4, depth{,_norm}.npy, c2w.npy, intrinsics.npy and meta.json to
`output_dir`. All defaults: configurations/inference.yaml and configurations/algorithm/ugg.yaml.
"""
import hydra
import numpy as np
from omegaconf import DictConfig

from ugg import UGG


def _cameras(path):
    g = np.load(path)
    return g["c2w"], g["intrinsics"]


def _depth(path):
    g = np.load(path)
    return (g["depth"] if hasattr(g, "files") else g).astype(np.float32)


@hydra.main(version_base=None, config_path="configurations", config_name="inference")
def main(cfg: DictConfig) -> None:
    need = {"video": ["input"], "image": ["input"], "first_last": ["input", "last"], "text": ["prompt"],
            "camera": ["cameras"], "depth": ["depth"]}
    if cfg.task not in need:
        raise SystemExit(f"task must be one of {list(need)}")
    missing = [k for k in need[cfg.task] if cfg[k] is None]
    if missing:
        raise SystemExit(f"task={cfg.task} needs " + ", ".join(f"{k}=..." for k in missing))
    cams = _cameras(cfg.cameras) if cfg.cameras else None
    model = UGG.from_pretrained(cfg=cfg)
    if cfg.task == "video":
        pred = model(cfg.input)
    elif cfg.task == "image":
        pred = model.image_to_4d(cfg.input, cameras=cams, num_frames=cfg.num_frames)
    elif cfg.task == "first_last":
        pred = model.first_last_to_4d(cfg.input, cfg.last, cameras=cams, num_frames=cfg.num_frames)
    elif cfg.task == "text":
        pred = model.text_to_4d(cfg.prompt, num_frames=cfg.num_frames)
    elif cfg.task == "camera":
        pred = model.camera_to_4d(cams)
    else:  # optional anchors: input= (first frame) and last= (last frame)
        anchors = [x for x in (cfg.input, cfg.last) if x]
        pred = model.depth_to_4d(_depth(cfg.depth), images=anchors or None)
    out = pred.save(cfg.output_dir, fps=cfg.dataset.fps)
    print(f"saved {len(pred.rgb)} frames ({cfg.task}; given: {', '.join(pred.meta['conditioned'])}) to {out}")


if __name__ == "__main__":
    main()
