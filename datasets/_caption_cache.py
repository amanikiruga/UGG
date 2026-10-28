"""Shared, opt-in caption + prompt-embedding cache lookup for the geometry
datasets.

The captioning pipeline (video_4d_dataset) writes, per dataset,
    <cache_dir>/captions.csv                      (split,scene_id[,variant,...],frame_indices,caption)
    <cache_dir>/prompt_embeds/<key>.pt            (raw bf16 [seq,4096] UMT5-XXL)
where <key> = the row's identifying columns (everything except
frame_indices/caption) joined by "__" with "/" and whitespace sanitized —
IDENTICAL to scripts/cache_prompt_embeds.py. This module reconstructs that
mapping so a loader can, per record, attach a real caption and an ABSOLUTE
prompt_embed_path (video_base._load_prompt_embed resolves data_root/<path>;
an absolute path passes through unchanged).

A dataset opts in by setting cfg.caption_cache_dir; a loader then calls
`cache.attach(record, scene_id)` in preprocess_record. Multiple caption
"variants" (distinct Gemini seeds) per scene are chosen at random each call
for text augmentation. Default off → zero behaviour change for existing runs.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_NON_ID = ("frame_indices", "caption")


def _sanitize(v: str) -> str:
    return str(v).replace("/", "_").replace(" ", "_").replace("\t", "_")


def _key_for(row: Dict[str, str], id_cols) -> str:
    return "__".join(_sanitize(row[c]) for c in id_cols)


class CaptionCache:
    """Index of scene_id -> list of (caption, absolute_embed_path) variants."""

    def __init__(self, cache_dir: str | Path, require_embed: bool = True,
                 scene_key_cols=None):
        """scene_key_cols: columns that identify a *distinct scene/clip* (lookup
        key). Default None -> key on 'scene_id' alone (unique per scene). Set it
        for datasets whose scene_id repeats across a sub-index (e.g. OmniWorld:
        ['scene_id','split_idx']) so each clip gets ITS caption, not a sibling's.
        The remaining id column ('variant') is the augmentation axis grouped
        under one key. Embed filenames always use the FULL id columns."""
        cache_dir = Path(cache_dir)
        csv_path = cache_dir / "captions.csv"
        embed_dir = cache_dir / "prompt_embeds"
        if not csv_path.is_file():
            raise FileNotFoundError(f"caption cache missing captions.csv: {csv_path}")
        self.scene_key_cols = scene_key_cols
        self.by_scene: Dict[Any, list] = {}
        n_rows = n_missing_embed = 0
        with open(csv_path, newline="") as f:
            rd = csv.DictReader(f)
            id_cols = [c for c in rd.fieldnames if c not in _NON_ID]
            for row in rd:
                cap = row.get("caption", "")
                if not cap or cap.startswith("__ERROR__"):
                    continue
                emb = embed_dir / f"{_key_for(row, id_cols)}.pt"
                if require_embed and not emb.exists():
                    n_missing_embed += 1
                    continue
                key = (row["scene_id"] if scene_key_cols is None
                       else tuple(str(row[c]) for c in scene_key_cols))
                self.by_scene.setdefault(key, []).append((cap.strip(), str(emb)))
                n_rows += 1
        self.n_rows = n_rows
        self.n_scenes = len(self.by_scene)
        self.n_missing_embed = n_missing_embed

    def has(self, scene_id: str) -> bool:
        return scene_id in self.by_scene

    def pick(self, scene_id: str, rng: Optional[random.Random] = None
             ) -> Optional[Tuple[str, str]]:
        variants = self.by_scene.get(scene_id)
        if not variants:
            return None
        return (rng or random).choice(variants)

    def attach(self, record: Dict[str, Any], scene_id: str,
               rng: Optional[random.Random] = None) -> bool:
        """Set record['caption'] and record['prompt_embed_path'] (absolute).
        Returns False if scene_id has no cached caption (record left unchanged)."""
        hit = self.pick(scene_id, rng)
        if hit is None:
            return False
        record["caption"], record["prompt_embed_path"] = hit
        return True
