"""Speaker-balanced WeightedRandomSampler.

Train set is dominated by spk_E (~45%). Plain shuffling lets the model
memorize that speaker's voice. We reweight so every speaker contributes
equally per epoch, roughly leveling per-speaker gradient signal.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def build_speaker_balanced_sampler(csv_path: str, num_samples: int | None = None) -> WeightedRandomSampler:
    df = pd.read_csv(csv_path)
    spk_counts = df["spk_id"].value_counts().to_dict()
    weights = df["spk_id"].map(lambda s: 1.0 / spk_counts[s]).to_numpy()
    n = num_samples if num_samples is not None else len(df)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=n,
        replacement=True,
    )
