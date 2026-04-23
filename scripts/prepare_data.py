"""Resample train/dev/test audio to 16 kHz float32 .npy shards.

Usage:
    python scripts/prepare_data.py \\
        --data-root asr-2026-spoken-numbers-recognition-challenge \\
        --cache-root cache_16k

Reads every file referenced by train.csv / dev.csv / test.csv, resamples to
16 kHz mono, peak-normalizes, saves as ``<cache_root>/<split>/<stem>.npy``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def resample(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    if peak > 0:
        data = data / max(peak, 1e-6)
    return data.astype(np.float32)


def _write_one(rel: str, audio_root: Path, cache_dir: Path, target_sr: int) -> None:
    out = cache_dir / Path(rel).with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    arr = resample(audio_root / rel, target_sr)
    np.save(out, arr)


def process_split(
    csv_path: Path,
    audio_root: Path,
    cache_dir: Path,
    target_sr: int,
    num_workers: int,
) -> None:
    df = pd.read_csv(csv_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = df["filename"].tolist()
    if num_workers <= 1:
        for rel in tqdm(files, desc=f"-> {cache_dir.name}"):
            _write_one(rel, audio_root, cache_dir, target_sr)
        return
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        tasks = [ex.submit(_write_one, rel, audio_root, cache_dir, target_sr) for rel in files]
        for fut in tqdm(tasks, desc=f"-> {cache_dir.name}"):
            fut.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--cache-root", required=True, type=Path)
    ap.add_argument("--target-sr", type=int, default=16_000)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    for split in ("train", "dev", "test"):
        csv = args.data_root / f"{split}.csv"
        if not csv.exists():
            print(f"skip: {csv} missing")
            continue
        process_split(csv, args.data_root, args.cache_root, args.target_sr, args.num_workers)


if __name__ == "__main__":
    main()
