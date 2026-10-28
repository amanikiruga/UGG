"""Pure helpers for RE10K record and temporal-selection invariants."""

from pathlib import Path

import numpy as np


def exclude_invalid_video_records(records, invalid_video_ids):
    """Filter confirmed-invalid IDs after either manifest load or directory scan."""
    invalid_video_ids = set(invalid_video_ids or [])
    if not invalid_video_ids:
        return records, 0
    filtered = [
        record
        for record in records
        if Path(record["video_path"]).stem not in invalid_video_ids
    ]
    return filtered, len(records) - len(filtered)


def select_re10k_clip(valid_frame_indices, valid_annos, sample_indices):
    """Select aligned decoded-frame IDs and camera annotations before I/O."""
    if len(valid_frame_indices) != len(valid_annos):
        raise ValueError(
            "valid_frame_indices and valid_annos must have the same length, got "
            f"{len(valid_frame_indices)} and {len(valid_annos)}"
        )
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if sample_indices.ndim != 1:
        raise ValueError(
            f"sample_indices must be one-dimensional, got {sample_indices.shape}"
        )
    if sample_indices.size and (
        sample_indices.min() < 0 or sample_indices.max() >= len(valid_frame_indices)
    ):
        raise IndexError("sample_indices contain an out-of-range RE10K frame")
    selected_frames = np.asarray(valid_frame_indices, dtype=np.int64)[sample_indices]
    selected_annos = [valid_annos[int(index)] for index in sample_indices]
    return selected_frames.tolist(), selected_annos
