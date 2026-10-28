from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
import random
import threading
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
# # import decord  # isort:skip

# decord.bridge.set_bridge("torch")

import imageio.v3 as iio
import decord



class VideoDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        self.split = split
        self.data_root = Path(cfg.data_root)
        self.metadata_path = Path(cfg.metadata_path)
        self.auto_download = cfg.auto_download
        self.force_download = cfg.force_download
        self.test_percentage = cfg.test_percentage
        self.id_token = cfg.id_token or ""
        self.height = cfg.height
        self.width = cfg.width
        self.n_frames = cfg.n_frames
        self.fps = cfg.fps
        self.trim_mode = cfg.trim_mode
        self.pad_mode = cfg.pad_mode
        self.filtering = cfg.filtering
        self.load_video_latent = cfg.load_video_latent
        self.load_prompt_embed = cfg.load_prompt_embed
        # Opt-in caption+embedding cache (annotations/<ds>/ with captions.csv +
        # prompt_embeds/). Off unless cfg.caption_cache_dir is set -> zero change
        # for existing runs. A loader sets SCENE_ID_KEY to the record field whose
        # value matches captions.csv 'scene_id'.
        self.caption_cache = None
        _ccd = getattr(cfg, "caption_cache_dir", None)
        if _ccd:
            from ._caption_cache import CaptionCache
            self.caption_cache = CaptionCache(
                _ccd, require_embed=self.load_prompt_embed,
                scene_key_cols=getattr(self, "CAPTION_SCENE_KEY_COLS", None))
            print(f"{type(self).__name__}: caption cache {_ccd} "
                  f"({self.caption_cache.n_scenes} scenes, {self.caption_cache.n_rows} variants)")
        self.augmentation = cfg.augmentation
        self.image_to_video = cfg.image_to_video
        self.max_text_tokens = cfg.max_text_tokens

        # trigger auto-download if not already downloaded
        trigger_download = False
        if not self.data_root.is_dir():
            print(f"Dataset root folder {self.data_root} does not exist.")
            if not self.auto_download:
                raise ValueError(
                    f"Attempting to automatically download the dataset since dataset root folder {self.data_root} does not exist. "
                    "If this is the intended behavior, append `dataset.auto_download=True` in your command to pass this check."
                )
            trigger_download = True
        if self.force_download:
            trigger_download = True
        if trigger_download:
            # if threading.current_thread() is not threading.main_thread():
            if torch.distributed.is_initialized():
                raise ValueError(
                    "Download must be called from the main thread with single-process training. Did you call this inside a multi-worker dataloader?"
                )
            print(f"Attempting to download dataset to {self.data_root}...")
            self.download()

        self.records = self._load_records_cached()  # a list of dictionaries
        self.augment_transforms = self._build_video_transforms(augment=True)
        self.no_augment_transforms = self._build_video_transforms(augment=False)
        self.img_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )

        if self.trim_mode not in ["speedup", "random_cut"]:
            raise ValueError(
                f"Invalid trim_mode: {self.trim_mode}. Must be one of ['speedup', 'random_cut']."
            )
        if self.pad_mode not in ["slowdown", "pad_last", "discard"]:
            raise ValueError(
                f"Invalid pad_mode: {self.pad_mode}. Must be one of ['slowdown', 'pad_last', 'discard']."
            )

    def _build_video_transforms(self, augment: bool = True):
        trans = []
        if augment and self.augmentation.random_flip is not None:
            raise ValueError("random_flip is not supported for video augmentation")
            trans.append(transforms.RandomHorizontalFlip(self.augmentation.random_flip))

        aspect_ratio = self.width / self.height
        aspect_ratio = [aspect_ratio, aspect_ratio]
        if augment and self.augmentation.ratio is not None:
            raise ValueError("ratio is not supported for video augmentation")
            aspect_ratio[0] *= self.augmentation.ratio[0]
            aspect_ratio[1] *= self.augmentation.ratio[1]

        scale = [1.0, 1.0]
        if augment and self.augmentation.scale is not None:
            raise ValueError("scale is not supported for video augmentation")
            scale[0] *= self.augmentation.scale[0]
            scale[1] *= self.augmentation.scale[1]

        trans.append(
            transforms.RandomResizedCrop(
                size=(self.height, self.width),
                scale=scale,
                ratio=aspect_ratio,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
        return transforms.Compose(trans)

    def _load_records_cached(self) -> List[Dict[str, Any]]:
        """Records-manifest cache. Walk-based loaders (tartanair/ase/bedlam/matrixcity/
        mid_air/mvs_synth/scenenet/…) rebuild their record list by stat-storming the
        filesystem in `_load_records()` at EVERY process start — painful at 32-rank
        distributed launch (every rank walks the data tree). If `cfg.records_manifest` is
        set, the post-split/post-preprocess record list is cached to a per-split JSON:
        first run walks + writes it, later runs load it instantly. Records are plain
        dicts (paths/ints/lists) so JSON round-trips exactly.

        Build once with `scripts/build_records_manifest.py`. Manifest is stale if the
        data changed — delete it (or omit the cfg key) to force a re-walk."""
        mrel = self.cfg.get("records_manifest", None)
        if not mrel:
            return self._load_records()
        if os.path.isabs(str(mrel)):
            mp = Path(mrel)
            mp = mp.with_name(f"{mp.stem}.{self.split}{mp.suffix or '.json'}")
        else:
            # A relative value resolves to a single WRITABLE manifest dir (data_roots are
            # often read-only), keyed by data_root so bedlam vs
            # bedlam2 vs tartanair get distinct files. Override dir via VWM_MANIFEST_DIR.
            mroot = Path(os.environ.get(
                "VWM_MANIFEST_DIR",
                os.path.expanduser("~/.cache/ugg/records_manifests")))
            key = str(self.data_root).strip("/").replace("/", "_")
            mp = mroot / f"{key}.{self.split}.json"
        if mp.exists():
            recs = json.loads(mp.read_text())
            self._warn_if_manifest_stale(mp, recs)
            print(f"{type(self).__name__}: loaded {len(recs)} records from manifest {mp}")
            return recs
        recs = self._load_records()
        try:
            mp.parent.mkdir(parents=True, exist_ok=True)
            tmp = mp.with_suffix(mp.suffix + ".tmp")
            tmp.write_text(json.dumps(recs, default=str))
            os.replace(tmp, mp)                       # atomic: concurrent ranks can't half-read
            self._write_manifest_signature(mp, recs)
            print(f"{type(self).__name__}: wrote records manifest {mp} ({len(recs)})")
        except Exception as e:
            print(f"WARN: could not write records manifest {mp}: {e}")
        return recs


    # ---- manifest staleness guard -------------------------------------------------
    # A records manifest is a CACHE of a filesystem walk. When the data underneath it
    # changes, the cache keeps working and silently misrepresents the corpus -- there is
    # no error, just wrong data. Three separate instances of this were found in the
    # 2026-08-25 dataset audit, each silent:
    #   * ASE finished downloading after its manifest was built -> training drew from
    #     37,373 of 91,368 scenes (59% of the corpus unreachable).
    #   * dl3dv's manifest had frozen absolute prompt-embed paths, so repointing the
    #     config at a regenerated embed store had no effect until the manifest was deleted.
    #   * scenenet's manifest listed 16,865 records whose pose files had been purged;
    #     every sample raised, was swallowed by MixtureDataset, and the subset silently
    #     contributed nothing.
    # The guard below is deliberately cheap -- one listdir of data_root, no walk -- and
    # only WARNS. It never refuses to train and never rebuilds behind your back; the fix
    # is always to delete the manifest so the next run re-walks.
    @staticmethod
    def _is_rank_zero() -> bool:
        for k in ("SLURM_PROCID", "RANK", "LOCAL_RANK"):
            v = os.environ.get(k)
            if v is not None:
                try:
                    return int(v) == 0
                except ValueError:
                    return True
        return True

    def _manifest_signature(self):
        """O(1) fingerprint of the data root: its mtime, which the filesystem bumps
        whenever an entry is added or removed. Deliberately NOT a child count -- counting
        ASE's 92,290 entries costs 1.3 s cold per rank, and 512 ranks doing that at launch
        is precisely the filesystem metadata storm this manifest cache exists to avoid."""
        try:
            st = os.stat(str(self.data_root))
            return {"data_root": str(self.data_root), "mtime": int(st.st_mtime)}
        except Exception:
            return None

    @staticmethod
    def _signature_path(mp: Path) -> Path:
        return mp.with_suffix(mp.suffix + ".sig.json")

    def _write_manifest_signature(self, mp: Path, recs) -> None:
        sig = self._manifest_signature()
        if sig is None:
            return
        sig["n_records"] = len(recs)
        try:
            self._signature_path(mp).write_text(json.dumps(sig))
        except Exception:
            pass                                      # advisory only

    def _warn_if_manifest_stale(self, mp: Path, recs) -> None:
        if not self._is_rank_zero():
            return                                    # one voice per launch, not 512
        sp = self._signature_path(mp)
        cur = self._manifest_signature()
        if cur is None:
            return
        if not sp.exists():
            print(f"NOTE [{type(self).__name__}]: manifest {mp.name} predates the staleness "
                  f"guard (no .sig.json), so it cannot be checked. If the data under "
                  f"{cur['data_root']} changed since it was built, delete the manifest to "
                  f"re-walk ({len(recs)} records cached).")
            return
        try:
            old = json.loads(sp.read_text())
        except Exception:
            return
        if old.get("mtime") != cur["mtime"]:
            import datetime as _dt
            def _fmt(t):
                try:
                    return _dt.datetime.fromtimestamp(int(t)).isoformat(timespec="seconds")
                except Exception:
                    return str(t)
            print(f"*** STALE MANIFEST [{type(self).__name__}] {mp.name}: {cur['data_root']} "
                  f"was last modified {_fmt(cur['mtime'])} but the manifest was built against "
                  f"{_fmt(old.get('mtime'))}. Entries were added or removed, so the cached "
                  f"{len(recs)} records may no longer cover the corpus. DELETE {mp} to "
                  f"re-walk. ***")

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # a hook to modify the original record on the fly
        return record

    # Record field holding the value that matches captions.csv 'scene_id'.
    # Loaders using the caption cache set this (e.g. "seq_id", "video_path").
    SCENE_ID_KEY: str = "scene_id"
    # For datasets whose scene_id repeats across a sub-index, the captions.csv
    # columns (== record fields) that identify a distinct clip, e.g. OmniWorld:
    # ["scene_id", "split_idx"]. None -> key on SCENE_ID_KEY alone.
    CAPTION_SCENE_KEY_COLS = None

    def _select_caption(self, record: Dict[str, Any]):
        """Opt-in: return (caption, abs_embed_path) picked at __getitem__ time
        (random caption-variant augmentation) or None to use record fields."""
        if self.caption_cache is None:
            return None
        if self.CAPTION_SCENE_KEY_COLS is not None:
            if any(c not in record for c in self.CAPTION_SCENE_KEY_COLS):
                return None
            key = tuple(str(record[c]) for c in self.CAPTION_SCENE_KEY_COLS)
        else:
            sid = record.get(self.SCENE_ID_KEY)
            if sid is None:
                return None
            key = sid
        return self.caption_cache.pick(key)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        # Load video data - either raw or preprocessed latents
        videos = self._load_video(record)
        # images = videos[:1].clone() if self.image_to_video else None
        image_latents, video_latents = None, None
        video_metadata = {
            "num_frames": videos.shape[0],
            "height": videos.shape[2],
            "width": videos.shape[3],
        }

        if self.load_video_latent:
            image_latents, video_latents = self._load_video_latent(record)
            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                n_frames = latent_num_frames * 4
            else:
                n_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            assert video_metadata["num_frames"] == n_frames, "num_frames changed"
            assert video_metadata["height"] == height, "height changed"
            assert video_metadata["width"] == width, "width changed"

        # Load prompt data - either raw or preprocessed embeddings
        embed_override = None
        sel = self._select_caption(record)
        if sel is not None:
            caption, embed_override = sel
        else:
            caption = ""
            if "caption" in record:
                caption = record["caption"]
            elif "gemini_caption" in record:
                caption = record["gemini_caption"]
            elif "original_caption" in record:
                caption = record["original_caption"]
        video_metadata["has_caption"] = caption != ""
        prompts = self.id_token + caption
        prompt_embeds = None
        prompt_embed_len = None
        if self.load_prompt_embed:
            prompt_embeds, prompt_embed_len = self._load_prompt_embed(record, embed_override)

        has_bbox, bbox_render = self._render_bbox(record)

        output = {
            "videos": videos,
            "video_metadata": video_metadata,
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
        }

        if prompts is not None:
            output["prompts"] = prompts
        # if images is not None:
        #     output["images"] = images
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        if image_latents is not None:
            output["image_latents"] = image_latents
        if video_latents is not None:
            output["video_latents"] = video_latents

        return output

    def _n_frames_in_src(self, src_fps):
        """
        Given the fps of the source video, return the number of frames in it we shall
        use in order to generate a target video of self.n_frames frames at self.fps.

        Note the definition of fps of the source video is described in README.md as,
        for a real-world task that requires 1 second to finish, how many frames does it
        take this source video to capture? This is usually just the fps of the source
        video, but if the source video is already a slow motion video, this may be
        different.


        """
        return round(self.n_frames / self.fps * src_fps)

    def _temporal_sample(self, n_frames: int, fps: int) -> torch.Tensor:
        """
        Given number of frames and fps, return a sequence of frame indices to downsample / upsample the video temporally.
        This shall consider self.n_frames and fps.
        """

        # target_len is the number of frames in the source video that we shall use to generate a target video of self.n_frames frames at self.fps
        target_len = self._n_frames_in_src(fps)

        if n_frames < target_len:
            if self.pad_mode == "pad_last":
                indices = np.linspace(0, target_len - 1, self.n_frames)
                indices = np.clip(indices, 0, n_frames - 1)
            elif self.pad_mode == "slowdown":
                indices = np.linspace(0, n_frames - 1, self.n_frames)
            elif self.pad_mode == "discard":
                raise ValueError(
                    "pad_mode is set to 'discard', but this short video is not filtered out."
                )
            else:
                raise ValueError(f"Invalid pad_mode: {self.pad_mode}")
        elif n_frames > target_len:
            if getattr(self.cfg, 'no_augmentations', False):
                start = (n_frames - target_len) // 2
                indices = start + np.linspace(0, target_len - 1, self.n_frames)
            elif self.trim_mode == "random_cut":
                start = np.random.randint(0, n_frames - target_len)
                indices = start + np.linspace(0, target_len - 1, self.n_frames)
            elif self.trim_mode == "speedup":
                indices = np.linspace(0, n_frames - 1, self.n_frames)
            elif self.trim_mode == "discard":
                raise ValueError(
                    "trim_mode is set to 'discard', but this long video is not filtered out."
                )
            else:
                raise ValueError(f"Invalid trim_mode: {self.trim_mode}")
        else:
            indices = np.linspace(0, n_frames - 1, self.n_frames)

        indices = np.round(indices).astype(int)
        return indices

    def _load_video(self, record: Dict[str, Any]) -> torch.Tensor:
        """
        Given a record, return a tensor of shape (n_frames, 3, H, W)
        """

        video_path = self.data_root / record["video_path"]
        # Seek-read only the sampled frames instead of decoding the whole video
        # (iio.imread(index=None) loaded entire clips into RAM -> per-worker OOM).
        # num_threads=1 avoids decord's threaded-decoder EAGAIN bug.
        vr = decord.VideoReader(video_path.as_posix(), num_threads=1)
        n_frames = len(vr)
        start = record.get("trim_start", 0)
        end = record.get("trim_end", n_frames)
        indices = self._temporal_sample(end - start, record["fps"])
        indices = list(start + indices)
        frames = torch.from_numpy(vr.get_batch(indices).asnumpy())  # (T, H, W, 3)
        # do some padding
        if len(frames) != self.n_frames:
            raise ValueError(
                f"Expected {len(frames)=} to be equal to {self.n_frames=}."
            )

        # crop if specified in the record
        if "crop_top" in record and "crop_bottom" in record:
            frames = frames[:, record["crop_top"] : record["crop_bottom"]]
        if "crop_left" in record and "crop_right" in record:
            frames = frames[:, :, record["crop_left"] : record["crop_right"]]

        frames = frames.float().permute(0, 3, 1, 2).contiguous() / 255.0

        if "has_bbox" in record and record["has_bbox"]:
            frames = self.no_augment_transforms(frames)
        else:
            frames = self.augment_transforms(frames)
        frames = self.img_normalize(frames)

        return frames

    def _render_bbox(self, record: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a record, return a tensor of shape (H, W)
        """

        # first frame and last frame forms 2 channels
        bbox_render = torch.zeros(2, record["height"], record["width"])
        has_bbox = torch.zeros(2, dtype=torch.bool)
        # if "first_frame_has_bbox" in record and record["first_frame_has_bbox"]:
        #     has_bbox[0] = True
        #     bbox_top = int(record["first_frame_bbox_top"])
        #     bbox_bottom = int(record["first_frame_bbox_bottom"])
        #     bbox_left = int(record["first_frame_bbox_left"])
        #     bbox_right = int(record["first_frame_bbox_right"])
        #     bbox_render[0, bbox_top:bbox_bottom, bbox_left:bbox_right] = 1
        # if "last_frame_has_bbox" in record and record["last_frame_has_bbox"]:
        #     has_bbox[-1] = True
        #     bbox_top = int(record["last_frame_bbox_top"])
        #     bbox_bottom = int(record["last_frame_bbox_bottom"])
        #     bbox_left = int(record["last_frame_bbox_left"])
        #     bbox_right = int(record["last_frame_bbox_right"])
        #     bbox_render[-1, bbox_top:bbox_bottom, bbox_left:bbox_right] = 1
        bbox_render = self.no_augment_transforms(bbox_render)
        return has_bbox, bbox_render

    def _load_records(self) -> Tuple[List[str], List[str]]:
        """
        Given the metadata file, loads the records as a list.
        Each record is a dictionary containing a datapoint's video path / caption etc.
        Require these entries: "video_path", "caption", "height", "width", "n_frames", "fps"
        Optional entry: "split" - if present, will be used instead of test_percentage
        """

        records = pd.read_csv(self.data_root / self.metadata_path, na_filter=False)
        records = records.to_dict("records")
        len_pre_filter = len(records)
        if not self.filtering.disable:
            records = [record for record in records if self._filter_record(record)]
        len_post_filter = len(records)
        
        print(
            f"{self.data_root / self.metadata_path}: filtered {len_pre_filter - len_post_filter} records from {len_pre_filter} to {len_post_filter}, rataining rate: {len_post_filter / len_pre_filter}"
        )

        if self.cfg.check_video_path and not self.debug:
            print("Checking records such that all video_path are valid...")
            print(
                "This could take a while. To skip, append `dataset.check_video_path=False` to your command."
            )
            for r in tqdm(records, desc="Checking video paths"):
                self._check_record(r)
            print("Done checking records")

        # Handle split selection
        if self.split != "all":
            if "split" in records[0]:
                # Use split field from records. Accept {validation, val, test}
                # as mutually-aliased holdout splits — individual dataset csvs
                # disagree on which label they use (scenenet uses "val",
                # others use "validation", exp_base requests "test"), so we
                # treat them all as equivalent for the non-training splits.
                aliases = {
                    "training": {"training"},
                    "test": {"test", "validation", "val"},
                    "validation": {"validation", "val", "test"},
                }.get(self.split, {self.split})
                records = [r for r in records if r["split"] in aliases]
                if not records:
                    raise ValueError(f"No records found for split '{self.split}'")
            else:
                # Use test_percentage
                if self.split == "training":
                    records = records[: -int(len(records) * self.test_percentage)] if self.test_percentage > 0 else records
                else:  # validation/test
                    records = records[-int(len(records) * self.test_percentage) :]
        random.Random(0).shuffle(records)

        records = [self.preprocess_record(record) for record in records]

        return records

    def _filter_record(self, x: Dict[str, Any]) -> bool:
        """
        x is a record dictionary containing a datapoint's video path / caption etc.
        Returns True if the record should be kept, False otherwise.
        """
        h, w, fps = x["height"], x["width"], x["fps"]

        # if record specified a crop, use that
        if "crop_left" in x and "crop_right" in x:
            w = x["crop_right"] - x["crop_left"]
        if "crop_top" in x and "crop_bottom" in x:
            h = x["crop_bottom"] - x["crop_top"]
        if "trim_start" in x and "trim_end" in x:
            n_frames = x["trim_end"] - x["trim_start"]
        elif "n_frames" in x:
            n_frames = x["n_frames"]
        else:
            raise ValueError(
                "Record missing required key 'n_frames', if trim not specified"
            )

        h_range = self.filtering.height
        if h_range is not None and h < h_range[0] or h > h_range[1]:
            print(f"Height {h} is out of range {h_range}")
            return False
        w_range = self.filtering.width
        if w_range is not None and w < w_range[0] or w > w_range[1]:
            print(f"Width {w} is out of range {w_range}")
            return False
        f_range = self.filtering.n_frames
        if f_range is not None and n_frames < f_range[0] or n_frames > f_range[1]:
            print(f"Number of frames {n_frames} is out of range {f_range}")
            return False
        fps_range = self.filtering.fps
        if fps_range is not None and fps < fps_range[0] or fps > fps_range[1]:
            print(f"FPS {fps} is out of range {fps_range}")
            return False
        if n_frames < self._n_frames_in_src(fps) and self.pad_mode == "discard":
            print(f"Number of frames {n_frames} is less than {self._n_frames_in_src(fps)} and pad_mode is discard")
            return False

        # then filter using stable_background, stable_brightness,
        # note that some datasets may not have these keys
        if "stable_background" in x and not x["stable_background"]:
            print(f"Stable background is False")
            return False
        if "stable_brightness" in x and not x["stable_brightness"]:
            print(f"Stable brightness is False")
            return False

        return True

    def _check_record(self, x: Dict[str, Any]) -> bool:
        """
        x is a record dictionary containing a datapoint's video path / caption etc.
        raise an error if the record is not valid. e.g.
        """
        video_path = self.data_root / x["video_path"]
        if not video_path.is_file():
            msg = f"Expected `{video_path=}` to be a valid file but found it to be invalid."
            if self.debug:
                print(msg)
            else:
                raise ValueError(msg)

    def _load_video_latent(
        self, record: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "video_latent_path" not in record:
            raise ValueError("Record missing required key 'video_latent_path'")
        video_latent_path = self.data_root / record["video_latent_path"]

        image_latent = None
        if self.image_to_video:
            if "image_latent_path" not in record:
                raise ValueError("Record missing required key 'image_latent_path'")
            image_latent_path = self.data_root / record["image_latent_path"]
            image_latent = torch.load(
                image_latent_path, map_location="cpu", weights_only=True
            )
        video_latent = torch.load(
            video_latent_path, map_location="cpu", weights_only=True
        )

        return image_latent, video_latent

    # Rate-limited miss reporting. A high miss rate is a data problem worth seeing,
    # but printing per sample inside a DataLoader worker floods the log, so report
    # the first miss and then at exponentially growing intervals.
    _caption_miss_count = 0
    _caption_hit_count = 0

    def _note_caption_miss(self, hit: bool) -> None:
        cls = type(self)
        if hit:
            cls._caption_hit_count += 1
            return
        cls._caption_miss_count += 1
        n = cls._caption_miss_count
        if n & (n - 1) == 0:  # 1, 2, 4, 8, ... powers of two
            total = n + cls._caption_hit_count
            print(
                f"[{cls.__name__}] caption-cache miss {n}/{total} "
                f"({n / max(total, 1):.1%}); using the null text embedding for these. "
                f"This is a defined fallback, NOT a dropped sample."
            )

    def _load_prompt_embed_or_null(self, record: Dict[str, Any], path_override=None):
        """Caption-cache-aware embed load with a DEFINED fallback.

        Returns (embed, len, has_caption).

        Datasets whose captions were generated for only part of the corpus (ASE 59%,
        OmniWorld 54%, MatrixCity 69% missing) have no `prompt_embed_path` on the
        record and no cache entry. Calling `_load_prompt_embed` for those raises
        ValueError, and `MixtureDataset.__iter__` swallows the exception and redraws
        -- so the scene becomes silently UNREACHABLE and its decoded video is thrown
        away. That trades a caption bug for a much worse coverage bug.

        Instead we fall back to the zero embedding, which is exactly what these
        subsets trained against before captions were wired up, so coverage is
        unchanged and captions are used wherever they exist.

        This lives here rather than inside `_load_prompt_embed` on purpose: datasets
        that genuinely require `prompt_embed_path` (re10k, dl3dv_colmap) must keep
        failing loudly rather than quietly training on zeros.
        """
        if path_override is None and "prompt_embed_path" not in record:
            self._note_caption_miss(hit=False)
            return torch.zeros(self.max_text_tokens, 4096), self.max_text_tokens, False
        embed, length = self._load_prompt_embed(record, path_override=path_override)
        self._note_caption_miss(hit=True)
        return embed, length, True

    def _load_prompt_embed(self, record: Dict[str, Any], path_override=None) -> torch.Tensor:
        # if self.debug:
        #     return torch.zeros(self.max_text_tokens, 4096), self.max_text_tokens

        if path_override is not None:
            prompt_embed_path = Path(path_override)   # absolute path from the caption cache
        else:
            if "prompt_embed_path" not in record:
                raise ValueError("Record missing required key 'prompt_embed_path'")
            prompt_embed_path = self.data_root / record["prompt_embed_path"]
        prompt_embed = torch.load(
            prompt_embed_path, map_location="cpu", weights_only=True
        )

        prompt_embed_len = prompt_embed.size(0)
        if prompt_embed_len < self.max_text_tokens:
            # Pad with zeros to max_text_tokens
            padding = torch.zeros(
                self.max_text_tokens - prompt_embed.size(0),
                prompt_embed.size(1),
                dtype=prompt_embed.dtype,
                device=prompt_embed.device,
            )
            prompt_embed = torch.cat([prompt_embed, padding], dim=0)

        return prompt_embed, prompt_embed_len

    def download(self):
        """
        Automatically download the dataset to self.data_root. Optional.
        """
        raise NotImplementedError(
            "Automatic download not implemented for this dataset."
        )
