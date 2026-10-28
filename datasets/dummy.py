"""Random-tensor dataset with the exact sample schema of MixtureDataset.

Used for smoke tests of the training loop (`dataset=dummy`) without any data on disk.
"""
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class DummyVideoDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        self.cfg = cfg
        self.split = split

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        T, H, W = self.cfg.n_frames, self.cfg.height, self.cfg.width
        Hr, Wr = self.cfg.raymap_height, self.cfg.raymap_width
        L = self.cfg.max_text_tokens
        return {
            "videos": torch.rand(T, 3, H, W, generator=g) * 2 - 1,
            "raymaps": torch.randn(T, 6, Hr, Wr, generator=g),
            "depths": torch.rand(T, 1, H, W, generator=g) * 2 - 1,
            "has_raymap": torch.tensor(True),
            "has_depth": torch.tensor(True),
            "subset_id": self.cfg.subset_id,
            "prompts": "",
            "prompt_embeds": torch.zeros(L, 4096),
            "prompt_embed_len": L,
            "video_metadata": {"num_frames": T, "height": H, "width": W, "has_caption": False},
        }
