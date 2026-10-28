"""UGG: one video diffusion model for RGB, depth and camera rays."""
import sys
from pathlib import Path

# The repo root holds the training packages (algorithms/, datasets/, utils/).
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .depth import depth_from_norm  # noqa: E402
from .io import prepare_video  # noqa: E402
from .pipeline import RELEASE_CKPT, UGG, Prediction, compose_config  # noqa: E402

__all__ = ["UGG", "Prediction", "RELEASE_CKPT", "compose_config", "depth_from_norm", "prepare_video"]
