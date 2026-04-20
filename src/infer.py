"""Run inference on a test CSV and produce submission.csv.

Usage:
    python -m src.infer --config configs/conformer_ctc.yaml --ckpt runs/.../best.ckpt \\
        --test-csv asr-2026-spoken-numbers-recognition-challenge/test.csv \\
        --data-root asr-2026-spoken-numbers-recognition-challenge \\
        --out submission.csv

Supports logits averaging across multiple checkpoints (ensemble):
    --ckpt runA/best.ckpt runB/best.ckpt runC/best.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import SpokenNumbersDataset, collate_fn
from .decode.ctc_beam import CTCBeamDecoder
from .models.conformer_ctc import ConformerCTC, ConformerCTCConfig
from .text import CyrillicVocab
from .text.denormalizer import TrieSnapper


def load_model(ckpt_path: Path, cfg: dict, device: torch.device) -> ConformerCTC:
    model = ConformerCTC(ConformerCTCConfig(**cfg["model"])).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if "model" in state else state
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def run(
    config_path: str,
    ckpts: list[str],
    test_csv: str,
    data_root: str,
    out_path: str,
    method: str = "beam_lm",
    batch_size: int = 8,
    num_workers: int = 2,
    cache_root: str | None = None,
) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = CyrillicVocab()

    models = [load_model(Path(p), cfg, device) for p in ckpts]
    print(f"[infer] loaded {len(models)} model(s)")

    decoder = CTCBeamDecoder(
        vocab=vocab,
        beam_width=cfg["decode"]["beam_width"],
        alpha=cfg["decode"]["alpha"],
        beta=cfg["decode"]["beta"],
        temperature=cfg["decode"]["temperature"],
        lm_model_path=cfg["decode"].get("lm_path") or None,
        snap=TrieSnapper(),
    )

    ds = SpokenNumbersDataset(
        test_csv, data_root,
        target_sr=cfg["data"]["target_sr"],
        vocab=vocab,
        augment=None,
        cache_dir=cache_root,
        max_seconds=cfg["data"]["max_seconds"],
        has_labels=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)

    rows: list[tuple[str, int]] = []
    for batch in tqdm(loader, desc="infer"):
        wav = batch["wav"].to(device)
        wav_lens = batch["wav_lens"].to(device)
        # Ensemble via log-softmax averaging.
        acc_log_probs = None
        out_lens = None
        for m in models:
            lp, ol = m(wav, wav_lens)  # (T, B, V), (B,)
            lp = lp.transpose(0, 1)    # (B, T, V)
            acc_log_probs = lp if acc_log_probs is None else acc_log_probs + lp
            out_lens = ol
        acc_log_probs = acc_log_probs / len(models)
        for b in range(acc_log_probs.size(0)):
            lp_b = acc_log_probs[b, : int(out_lens[b].item())].cpu()
            pred = decoder.decode_to_int(lp_b, method=method)
            rows.append((batch["filenames"][b], int(pred)))

    df = pd.DataFrame(rows, columns=["filename", "transcription"])
    df.to_csv(out_path, index=False)
    print(f"[infer] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--method", default="beam_lm")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cache-root", default=None)
    args = ap.parse_args()
    run(
        args.config, args.ckpt, args.test_csv, args.data_root, args.out,
        method=args.method, batch_size=args.batch_size, num_workers=args.num_workers,
        cache_root=args.cache_root,
    )
