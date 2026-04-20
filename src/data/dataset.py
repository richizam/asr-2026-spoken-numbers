"""CSV-driven dataset for train/dev/test splits.

Audio is lazily loaded, resampled to ``target_sr`` (16 kHz), peak-normalized,
and (for train/dev) converted to a CTC target using the char vocabulary and
``num2words`` Russian normalizer. The dataset can optionally read from a
pre-cached ``.npy`` shard directory produced by ``scripts/prepare_data.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..text import CyrillicVocab, int_to_words


@dataclass
class SampleMeta:
    filename: str
    transcription: Optional[int]
    spk_id: Optional[str]
    gender: Optional[str]
    samplerate: int
    ext: str


class SpokenNumbersDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        audio_root: str | Path,
        target_sr: int = 16_000,
        vocab: Optional[CyrillicVocab] = None,
        augment: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        cache_dir: Optional[str | Path] = None,
        max_seconds: float = 6.0,
        has_labels: bool = True,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.audio_root = Path(audio_root)
        self.target_sr = target_sr
        self.vocab = vocab or CyrillicVocab()
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = int(max_seconds * target_sr)
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, relpath: str) -> np.ndarray:
        if self.cache_dir is not None:
            npy_path = self.cache_dir / (Path(relpath).stem + ".npy")
            if npy_path.exists():
                return np.load(npy_path)
        path = self.audio_root / relpath
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr, res_type="kaiser_fast")
        peak = float(np.max(np.abs(data))) if data.size else 1.0
        if peak > 0:
            data = data / max(peak, 1e-6)
        return data.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        relpath = row["filename"]
        wav = self._load_audio(relpath)
        if self.augment is not None:
            wav = self.augment(wav, self.target_sr)
        if wav.shape[0] > self.max_samples:
            wav = wav[: self.max_samples]
        item: dict = {
            "wav": torch.from_numpy(np.asarray(wav, dtype=np.float32)),
            "filename": relpath,
        }
        if self.has_labels and "transcription" in row:
            n = int(row["transcription"])
            words = int_to_words(n)
            target_ids = self.vocab.encode(words)
            item["target"] = torch.tensor(target_ids, dtype=torch.long)
            item["target_int"] = n
            item["target_text"] = words
            if "spk_id" in row and isinstance(row["spk_id"], str):
                item["spk_id"] = row["spk_id"]
        return item


def collate_fn(batch: list[dict]) -> dict:
    wav_lens = [b["wav"].size(0) for b in batch]
    max_len = max(wav_lens)
    wavs = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, b in enumerate(batch):
        wavs[i, : wav_lens[i]] = b["wav"]
    out = {
        "wav": wavs,
        "wav_lens": torch.tensor(wav_lens, dtype=torch.long),
        "filenames": [b["filename"] for b in batch],
    }
    if "target" in batch[0]:
        tgt_lens = [b["target"].size(0) for b in batch]
        max_tgt = max(tgt_lens)
        targets = torch.zeros(len(batch), max_tgt, dtype=torch.long)
        for i, b in enumerate(batch):
            targets[i, : tgt_lens[i]] = b["target"]
        out["targets"] = targets
        out["target_lens"] = torch.tensor(tgt_lens, dtype=torch.long)
        out["target_ints"] = [b["target_int"] for b in batch]
        out["target_texts"] = [b["target_text"] for b in batch]
        out["spk_ids"] = [b.get("spk_id") for b in batch]
    return out
