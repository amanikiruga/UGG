"""Project-specific learning-rate schedules with explicit, testable boundaries."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def warmup_hold_cosine_multiplier(
    step: int,
    *,
    warmup_steps: int,
    hold_steps: int,
    total_steps: int,
    min_lr_rate: float,
) -> float:
    """Linear warmup, constant peak, then half-cosine decay to ``min_lr_rate``."""
    if not 0 <= warmup_steps <= hold_steps < total_steps:
        raise ValueError("require 0 <= warmup_steps <= hold_steps < total_steps")
    if not 0.0 <= min_lr_rate <= 1.0:
        raise ValueError("min_lr_rate must be in [0, 1]")
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if step <= hold_steps:
        return 1.0
    progress = min(1.0, float(step - hold_steps) / float(total_steps - hold_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine


def get_warmup_hold_cosine_scheduler(
    optimizer,
    *,
    num_warmup_steps: int,
    num_hold_steps: int,
    num_training_steps: int,
    min_lr_rate: float,
    last_epoch: int = -1,
):
    return LambdaLR(
        optimizer,
        lambda step: warmup_hold_cosine_multiplier(
            step,
            warmup_steps=num_warmup_steps,
            hold_steps=num_hold_steps,
            total_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        ),
        last_epoch=last_epoch,
    )

