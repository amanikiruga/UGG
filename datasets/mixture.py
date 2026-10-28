from typing import List
import time
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from omegaconf import DictConfig, open_dict
import torch
import numpy as np
# from datasets.openx_base import OpenXVideoDataset
# from datasets.droid import DroidVideoDataset
# from datasets.something_something import SomethingSomethingDataset
# from datasets.epic_kitchen import EpicKitchenDataset
# from datasets.pandas import PandasVideoDataset
# from datasets.deprecated.video_1x_wm import WorldModel1XDataset
# from datasets.ego4d import Ego4DVideoDataset
# from datasets.aria import AriaDataset
from datasets.dl3dv_colmap import DL3DVColmapDataset
from datasets.mid_air import MidAirDataset
from datasets.mvs_synth import MVSSynthDataset
from datasets.matrixcity import MatrixCityDataset
from datasets.omniworld_game import OmniWorldGameDataset
from datasets.re10k import RE10KDataset
from datasets.scenenet import SceneNetRGBDDataset
from datasets.vkitti2 import VirtualKitti2Dataset
from datasets.tartanair import TartanAirDataset
from datasets.tartanair_v2 import TartanAirV2Dataset
from datasets.spring import SpringDataset
from datasets.bedlam import BedlamDataset
from datasets.ase import ASEDataset

subset_classes = dict(
    # something_something=SomethingSomethingDataset,
    # epic_kitchen=EpicKitchenDataset,
    # pandas=PandasVideoDataset,
    # video_1x_wm=WorldModel1XDataset,
    # ego4d=Ego4DVideoDataset,
    # droid=DroidVideoDataset,
    # austin_buds=OpenXVideoDataset,
    # austin_sailor=OpenXVideoDataset,
    # austin_sirius=OpenXVideoDataset,
    # bc_z=OpenXVideoDataset,
    # berkeley_autolab=OpenXVideoDataset,
    # berkeley_cable=OpenXVideoDataset,
    # berkeley_fanuc=OpenXVideoDataset,
    # bridge=OpenXVideoDataset,
    # cmu_stretch=OpenXVideoDataset,
    # dlr_edan=OpenXVideoDataset,
    # dobbe=OpenXVideoDataset,
    # fmb=OpenXVideoDataset,
    # fractal=OpenXVideoDataset,
    # iamlab_cmu=OpenXVideoDataset,
    # jaco_play=OpenXVideoDataset,
    # language_table=OpenXVideoDataset,
    # nyu_franka=OpenXVideoDataset,
    # roboturk=OpenXVideoDataset,
    # stanford_hydra=OpenXVideoDataset,
    # taco_play=OpenXVideoDataset,
    # toto=OpenXVideoDataset,
    # ucsd_kitchen=OpenXVideoDataset,
    # utaustin_mutex=OpenXVideoDataset,
    # viola=OpenXVideoDataset,
    
    # --- 3D video datasets ---
    # aria=AriaDataset,
    dl3dv_colmap=DL3DVColmapDataset,  # original COLMAP poses, NVS-only (no depth)
    mid_air=MidAirDataset,
    mvs_synth=MVSSynthDataset,
    matrixcity=MatrixCityDataset,
    omniworld_game=OmniWorldGameDataset,
    re10k=RE10KDataset,
    scenenet=SceneNetRGBDDataset,   # camera-aware SceneNet (real GT poses), used in mixture_parallax14
    vkitti2=VirtualKitti2Dataset,
    tartanair=TartanAirDataset,
    tartanair_v2=TartanAirV2Dataset,
    spring=SpringDataset,
    bedlam=BedlamDataset,
    bedlam2=BedlamDataset,        # BEDLAM v2 tree (separate data_root); same loader (handles v1+v2 layout)
    ase=ASEDataset,
)


class MixtureDataset(IterableDataset):
    """
    A fault tolerant mixture of video datasets
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        self.split = split
        self.random_seed = (
            self._make_time_seed()
            if split in ["training", "train"]
            else int(np.random.get_state()[1][0])
        )
        # Optional deterministic seed for cross-run comparability (dataset.mixture_seed).
        # The time seed above differs per rank because each rank constructs its own
        # dataset; a naive shared constant would make every rank yield the SAME stream
        # (32 duplicated batches per step). Mix the rank in with a stride larger than
        # any num_workers so (rank, worker) pairs stay disjoint in _build_iteration_rng.
        _cfg_seed = self.cfg.get("mixture_seed", None)
        if _cfg_seed is not None and split in ["training", "train"]:
            import os as _os
            _rank = int(_os.environ.get("RANK", _os.environ.get("SLURM_PROCID", "0")))
            self.random_seed = (int(_cfg_seed) + 100003 * _rank) % (2**32)
        self.subset_cfg = {
            k.split("/")[1]: v for k, v in self.cfg.items() if k.startswith("subset/")
        }
        if split == "all":
            raise ValueError("split cannot be `all` for MixtureDataset`")
        weight = dict(self.cfg[split].weight)
        # Check if all keys in weight exist in subset_cfg
        for key in weight:
            if key not in self.subset_cfg:
                raise ValueError(
                    f"Dataset '{key}' specified in weights but not found in configuration"
                )
        self.subset_cfg = {k: v for k, v in self.subset_cfg.items() if k in weight}
        weight_type = self.cfg[split].weight_type  # one of relative or absolute
        self.subsets: List[Dataset] = []
        for subset_name, subset_cfg in self.subset_cfg.items():
            subset_cfg["height"] = self.cfg.height
            subset_cfg["width"] = self.cfg.width
            subset_cfg["n_frames"] = self.cfg.n_frames
            # Do NOT override cfg.fps — each subset's yaml carries its own
            # calibrated fps so per-frame motion magnitude is comparable across
            # datasets (see datasets/geometry_verification/all/tune_fps.py).
            # The mixture's top-level `fps:` is just a playback/reference rate.
            if "fps" not in subset_cfg:
                subset_cfg["fps"] = self.cfg.fps
            subset_cfg["load_video_latent"] = self.cfg.load_video_latent
            subset_cfg["load_prompt_embed"] = self.cfg.load_prompt_embed
            subset_cfg["max_text_tokens"] = self.cfg.max_text_tokens
            subset_cfg["image_to_video"] = self.cfg.image_to_video
            # scale_preserving / scale_mode / parallax_kappa are not pre-declared in the
            # subset yamls, so open_dict to bypass struct-mode key check. scale_mode=parallax
            # is scheme C (SCALE_NORMALIZATION_ANALYSIS.md); propagated to every subset so a
            # single top-level dataset.scale_mode=parallax switches the whole mixture. NVS
            # subsets without metric depth (re10k, dl3dv_colmap) ignore it and keep per-clip
            # moment normalization.
            with open_dict(subset_cfg):
                subset_cfg["scale_preserving"] = self.cfg.get("scale_preserving", False)
                subset_cfg["scale_mode"] = self.cfg.get("scale_mode", None)
                subset_cfg["parallax_kappa"] = self.cfg.get("parallax_kappa", None)
                subset_cfg["parallax_depth_map"] = self.cfg.get("parallax_depth_map", None)
                subset_cfg["parallax_log_alpha"] = self.cfg.get("parallax_log_alpha", None)
                subset_cfg["parallax_ray_encoding"] = self.cfg.get("parallax_ray_encoding", None)
                subset_cfg["parallax_mu"] = self.cfg.get("parallax_mu", None)
                # G3: the metric depth-gauge table that lets the pose-only NVS subsets
                # (re10k, dl3dv_colmap) compute the SAME s = kappa*median(depth) gauge as
                # the geometry loaders. Without it they silently fall back to per-clip
                # max-norm and the mixture carries two incompatible moment codes at once.
                subset_cfg["parallax_gauge_file"] = self.cfg.get("parallax_gauge_file", None)
                # Per-subset alpha override for the indoor/outdoor alpha arms (GalphaIO).
                # The block above sets ONE alpha for every subset; this lets a named subset
                # carry its own. Alpha is consumed only by the depth-channel encode
                # (_scale_norm._log_depth), so it is a no-op for the depth-less NVS subsets.
                # Inert unless `parallax_log_alpha_by_subset` is set, so every other run is
                # bit-identical to before this change. Same block the 1.3B GalphaIO arm
                # trained with.
                _by_sub = self.cfg.get("parallax_log_alpha_by_subset", None)
                if _by_sub is not None and subset_name in _by_sub:
                    subset_cfg["parallax_log_alpha"] = float(_by_sub[subset_name])
            # Only override raymap_{h,w} on subsets that already declare them
            # (e.g. scenenet_depth has no raymap generation and lacks these keys).
            if "raymap_height" in self.cfg and "raymap_height" in subset_cfg:
                subset_cfg["raymap_height"] = self.cfg.raymap_height
            if "raymap_width" in self.cfg and "raymap_width" in subset_cfg:
                subset_cfg["raymap_width"] = self.cfg.raymap_width
            self.subsets.append(subset_classes[subset_name](subset_cfg, split))
            if weight_type == "relative":
                weight[subset_name] = weight[subset_name] * len(self.subsets[-1])

        # --- Hard shape assertion --------------------------------------------
        # Mixture batches stack tensors across subsets, so every subset must
        # emit `videos` of the same (H, W). Catch aspect/orientation mismatches
        # up-front here rather than getting an opaque collate_fn error later.
        expected_hw = (self.cfg.height, self.cfg.width)
        for name, subset in zip(self.subset_cfg.keys(), self.subsets):
            got_hw = (getattr(subset, "height", None), getattr(subset, "width", None))
            if got_hw != expected_hw:
                raise ValueError(
                    f"MixtureDataset: subset '{name}' has (H, W) = {got_hw} but "
                    f"mixture expects {expected_hw}. All subsets must share the "
                    "same output shape (e.g. aria should be landscape 240×320 too)."
                )

        # Normalize weights to sum to 1
        total_weight = sum(weight.values())
        self.normalized_weights = {k: v / total_weight for k, v in weight.items()}

        # Store dataset sizes for printing
        dataset_sizes = {
            subset_name: len(subset)
            for subset_name, subset in zip(self.subset_cfg.keys(), self.subsets)
        }

        # Print normalized weights and dataset sizes in a nice format
        print("\nDataset information for split '{}':".format(self.split))
        print("-" * 60)
        print(f"{'Dataset':<25} {'Size':<10} {'Weight':<10} {'Normalized':<10}")
        print("-" * 60)
        for subset_name, norm_weight in sorted(
            self.normalized_weights.items(), key=lambda x: -x[1]
        ):
            size = dataset_sizes[subset_name]
            orig_weight = self.cfg[split].weight[subset_name]
            print(
                f"{subset_name:<25} {size:<10,d} {orig_weight:<10.4f} {norm_weight:<10.4f}"
            )
        print("-" * 60)

        # Calculate cumulative probabilities for sampling
        self.cumsum_weights = {}
        cumsum = 0
        for k, v in self.normalized_weights.items():
            cumsum += v
            self.cumsum_weights[k] = cumsum

        # some scripts want to access the records
        self.records = []
        for subset in self.subsets:
            self.records.extend(subset.records)

    @staticmethod
    def _make_time_seed() -> int:
        return time.time_ns() % (2**32)

    def _build_iteration_rng(self, worker_id: int = None):
        if worker_id is None:
            worker = get_worker_info()
            worker_id = 0 if worker is None else worker.id
        seed = (int(self.random_seed) + int(worker_id)) % (2**32)
        return np.random.RandomState(seed)

    # Required keys in every yielded sample. Auxiliary keys (extrinsics_gt,
    # intrinsics_gt, raymap_scale, depths_raw, depths_valid, sky_mask, etc.)
    # are stripped here because individual subsets emit different sets of them
    # and default DataLoader collate cannot stack heterogeneous dict shapes.
    _REQUIRED_KEYS = frozenset({
        "videos", "raymaps", "depths",
        "has_raymap", "has_depth",
        "subset_id",
        "prompts", "prompt_embeds", "prompt_embed_len",
        "video_metadata", "bbox_render", "has_bbox",
    })

    def _standardize_sample(self, sample: dict, subset_name: str) -> dict:
        """Enforce a fixed schema across all subsets.

        - Missing `raymaps` → randn placeholder, has_raymap=False
        - Missing `depths`  → randn placeholder, has_depth=False
        randn (not zeros) so that even if downstream t=1 enforcement fails, the
        model sees pure-Gaussian input rather than a structured zero pattern
        that would correlate with "this is a missing-modality sample".
        """
        T = self.cfg.n_frames
        H = self.cfg.height
        W = self.cfg.width
        Hr = self.cfg.raymap_height
        Wr = self.cfg.raymap_width

        has_raymap = "raymaps" in sample
        has_depth  = "depths"  in sample
        if not has_raymap:
            sample["raymaps"] = torch.randn(T, 6, Hr, Wr)
        if not has_depth:
            sample["depths"]  = torch.randn(T, 1, H,  W)
        sample["has_raymap"] = torch.tensor(has_raymap, dtype=torch.bool)
        sample["has_depth"]  = torch.tensor(has_depth,  dtype=torch.bool)
        # subset_id as a string survives default DataLoader collate (becomes a
        # list[str] of length B). Used by validation to bin metrics per subset.
        sample["subset_id"] = subset_name
        return {k: v for k, v in sample.items() if k in self._REQUIRED_KEYS}

    def _skip_iterations(self, rng, n):
        """Advance `rng` by exactly `n` loop iterations WITHOUT loading any data.

        Why this is faithful: the body of __iter__ consumes the RNG in exactly two
        places per iteration -- rng.random() to pick the subset, then
        rng.randint(len(dataset)) to pick the index -- and neither depends on the
        CONTENT of the sample. So replaying just those two calls reproduces the RNG
        state that the real loop would have reached.

        The subset selection MUST be replicated rather than skipped: numpy's randint
        uses masked rejection sampling, so the number of underlying draws depends on
        the BOUND, i.e. on len(dataset) for the selected subset. Picking the wrong
        subset here would desynchronise the stream.

        Not reproduced: the `except` path, which consumes both draws and yields
        nothing. Measured over a full run -- 25 failures in 8.9M samples (2.8e-6) --
        so across 1120 (rank, worker) streams almost every stream has zero, and an
        affected one is off by a single sample in ~1620.
        """
        for _ in range(int(n)):
            rand = rng.random()
            selected_subset = None
            for subset_name, cumsum in self.cumsum_weights.items():
                if rand <= cumsum:
                    selected_subset = subset_name
                    break
            if selected_subset is None:          # mirrors the real loop's fallthrough
                continue
            subset_idx = list(self.subset_cfg.keys()).index(selected_subset)
            rng.randint(len(self.subsets[subset_idx]))
        return rng

    def __iter__(self):
        rng = self._build_iteration_rng()
        # Default 0 -> byte-for-byte the previous behaviour. Only a deliberate
        # dataset.stream_skip_iters= override can change the stream.
        _skip = self.cfg.get("stream_skip_iters", 0) or 0
        if _skip:
            w = get_worker_info()
            print(f"[STREAM-SKIP] worker {0 if w is None else w.id}: "
                  f"burning {int(_skip)} iterations ({2*int(_skip)} rng draws) before yielding",
                  flush=True)
            rng = self._skip_iterations(rng, _skip)
        while True:
            # Sample a random subset based on weights using numpy random
            rand = rng.random()
            for subset_name, cumsum in self.cumsum_weights.items():
                if rand <= cumsum:
                    selected_subset = subset_name
                    break

            # Get the corresponding dataset index
            subset_idx = list(self.subset_cfg.keys()).index(selected_subset)

            try:
                # Sample randomly from the selected dataset using numpy random
                dataset = self.subsets[subset_idx]
                idx = rng.randint(len(dataset))
                sample = dataset[idx]
                yield self._standardize_sample(sample, selected_subset)
            except Exception as e:
                if self.debug:
                    raise e
                else:
                    print(f"Error sampling from {selected_subset}: {str(e)}")
                    continue
