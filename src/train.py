"""Training loop for Conformer-CTC on Russian spoken numbers.

Highlights:
    - fp16 autocast (``amp: true`` in config).
    - Grad accumulation + grad clipping.
    - Linear warmup then cosine decay.
    - SpecAugment on log-mel features (inline, during forward).
    - Speaker-balanced WeightedRandomSampler for train.
    - Per-``spk_id`` dev CER logging + harmonic-mean stopping criterion.
    - Resumable checkpoints (model, optimizer, scheduler, scaler, epoch,
      global_step, RNG state) to survive Kaggle/Colab session caps.

Run:
    python -m src.train --config configs/conformer_ctc.yaml
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.augment import build_train_augment, spec_augment
from .data.dataset import SpokenNumbersDataset, collate_fn
from .data.sampler import build_speaker_balanced_sampler
from .decode.ctc_beam import CTCBeamDecoder
from .models.conformer_ctc import ConformerCTC, ConformerCTCConfig
from .text import CyrillicVocab, int_to_words
from .text.denormalizer import TrieSnapper


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEMA:
    """Exponential moving average of model parameters.

    Standard generalization trick: keep a shadow state dict, update on every
    optimizer step with decay=0.999. Swap weights at validation and save the
    shadow alongside the live weights so inference can prefer the smoother
    EMA copy. Integer buffers (if any) are copied rather than averaged.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            sh = self.shadow[k]
            if v.dtype.is_floating_point:
                sh.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                sh.copy_(v.detach())

    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}


def linear_warmup_cosine(step: int, warmup: int, total: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))


def build_model(cfg: dict) -> ConformerCTC:
    m_cfg = ConformerCTCConfig(**cfg["model"])
    return ConformerCTC(m_cfg)


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    # Levenshtein on chars
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m] / n


def int_cer(ref_int: int, hyp_int: int) -> float:
    return cer(str(ref_int), str(hyp_int))


@torch.no_grad()
def validate(
    model: ConformerCTC,
    loader: DataLoader,
    decoder: CTCBeamDecoder,
    device: torch.device,
    method: str = "greedy",
) -> dict[str, float]:
    model.eval()
    per_spk_errs: dict[str, list[float]] = {}
    all_errs: list[float] = []
    empty_preds = 0
    pred_1000 = 0
    blank_id = decoder.blank_id
    # Token-argmax histogram across all dev frames, to catch blank-collapse.
    argmax_hist = torch.zeros(decoder.vocab.__len__(), dtype=torch.long)
    samples_shown: list[tuple[int, int, str]] = []
    for batch in tqdm(loader, desc="val", leave=False):
        wav = batch["wav"].to(device)
        wav_lens = batch["wav_lens"].to(device)
        log_probs, out_lens = model(wav, wav_lens)  # (T, B, V)
        log_probs = log_probs.transpose(0, 1)       # (B, T, V)
        argmax = log_probs.argmax(dim=-1)           # (B, T)
        for b in range(log_probs.size(0)):
            T_valid = int(out_lens[b].item())
            lp = log_probs[b, :T_valid]
            argmax_hist += torch.bincount(argmax[b, :T_valid].cpu(), minlength=argmax_hist.numel())
            raw_text = decoder.decode_text(lp.cpu(), method=method)
            hyp_int = decoder.snap.decode_to_int(raw_text)
            ref_int = batch["target_ints"][b]
            err = int_cer(ref_int, hyp_int)
            all_errs.append(err)
            if not raw_text.strip():
                empty_preds += 1
            if hyp_int == 1000:
                pred_1000 += 1
            if len(samples_shown) < 3:
                samples_shown.append((ref_int, hyp_int, raw_text[:80]))
            spk = batch["spk_ids"][b] or "unknown"
            per_spk_errs.setdefault(spk, []).append(err)
    mean_cer = float(np.mean(all_errs)) if all_errs else 1.0
    per_spk = {s: float(np.mean(v)) for s, v in per_spk_errs.items()}
    # Harmonic mean over speakers (competition-style robustness signal).
    if per_spk:
        inv = [1.0 / max(v, 1e-6) for v in per_spk.values()]
        hmean = len(per_spk) / sum(inv)
    else:
        hmean = 1.0
    model.train()
    n = max(1, len(all_errs))
    total_frames = int(argmax_hist.sum().item()) or 1
    blank_frac = float(argmax_hist[blank_id].item()) / total_frames
    for ref_int, hyp_int, raw in samples_shown:
        print(f"  sample: ref={ref_int} hyp={hyp_int} text={raw!r}")
    print(f"  argmax blank_frac={blank_frac:.3f}  total_dev_frames={total_frames}")
    return {
        "cer": mean_cer,
        "per_spk": per_spk,
        "hmean_cer": hmean,
        "empty_pred_rate": empty_preds / n,
        "pred_1000_rate": pred_1000 / n,
        "blank_frac": blank_frac,
    }


def save_ckpt(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = CyrillicVocab()
    augment = build_train_augment(cfg["data"].get("noise_dir"), cfg["data"].get("rir_dir"))

    train_ds = SpokenNumbersDataset(
        cfg["data"]["train_csv"],
        cfg["data"]["data_root"],
        target_sr=cfg["data"]["target_sr"],
        vocab=vocab,
        augment=augment,
        cache_dir=cfg["data"].get("cache_root"),
        max_seconds=cfg["data"]["max_seconds"],
    )
    dev_ds = SpokenNumbersDataset(
        cfg["data"]["dev_csv"],
        cfg["data"]["data_root"],
        target_sr=cfg["data"]["target_sr"],
        vocab=vocab,
        augment=None,
        cache_dir=cfg["data"].get("cache_root"),
        max_seconds=cfg["data"]["max_seconds"],
    )

    sampler = build_speaker_balanced_sampler(cfg["data"]["train_csv"], num_samples=len(train_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[init] model params: {n_params:,}")
    assert n_params < 5_000_000, "model exceeds 5M-parameter budget"

    decoder = CTCBeamDecoder(
        vocab=vocab,
        beam_width=cfg["decode"]["beam_width"],
        alpha=cfg["decode"]["alpha"],
        beta=cfg["decode"]["beta"],
        temperature=cfg["decode"]["temperature"],
        lm_model_path=cfg["decode"].get("lm_path") or None,
        snap=TrieSnapper(),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    total_steps = cfg["train"]["max_epochs"] * math.ceil(len(train_loader) / cfg["train"]["grad_accum"])
    warmup = cfg["train"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: linear_warmup_cosine(s, warmup, total_steps)
    )
    scaler = torch.amp.GradScaler(enabled=cfg["train"]["amp"])

    ema_cfg = cfg["train"].get("ema", {}) or {}
    ema = (
        ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999)))
        if ema_cfg.get("enabled", True)
        else None
    )

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    best_cer = float("inf")
    val_method = cfg["decode"].get("method", "beam_lm")

    last_ckpt = out_dir / "last.ckpt"
    if last_ckpt.exists():
        state = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        global_step = state["step"]
        best_metric = state.get("best_metric", state.get("best_hmean_cer", state.get("best_cer", float("inf"))))
        best_cer = state.get("best_cer", float("inf"))
        torch.set_rng_state(state["rng_cpu"])
        if torch.cuda.is_available() and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        if ema is not None:
            if state.get("ema") is not None:
                ema.load_state_dict(state["ema"])
                print("[resume] restored EMA weights from checkpoint")
            else:
                # First run that introduces EMA — seed shadow from live weights.
                ema = ModelEMA(model, decay=ema.decay)
                print("[resume] no EMA in checkpoint; seeding shadow from live weights")
        print(
            f"[resume] epoch {start_epoch}, step {global_step}, "
            f"best_metric {best_metric:.4f}, best_cer {best_cer:.4f}"
        )

    ctc = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

    for epoch in range(start_epoch, cfg["train"]["max_epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_unalign = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for it, batch in enumerate(pbar):
            wav = batch["wav"].to(device, non_blocking=True)
            wav_lens = batch["wav_lens"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lens = batch["target_lens"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=cfg["train"]["amp"]):
                # Forward, but intercept log-mel to apply SpecAugment before subsampling.
                # Reimplement the body of ConformerCTC.forward() to allow feature aug.
                mel = model.frontend(wav)
                t_mel = mel.size(-1)
                mel_lens = model.compute_mel_lens(wav_lens, t_mel)
                mel = model.normalize_mel(mel, mel_lens)
                mel = spec_augment(
                    mel,
                    num_freq_masks=cfg["train"]["specaug"]["num_freq_masks"],
                    freq_mask_param=cfg["train"]["specaug"]["freq_mask_param"],
                    num_time_masks=cfg["train"]["specaug"]["num_time_masks"],
                    time_mask_param=cfg["train"]["specaug"]["time_mask_param"],
                )
                x = model.subsampling(mel)
                t_out = x.size(1)
                x = x + model.pe[:t_out].to(x.dtype)
                x = model.input_drop(x)
                out_lens = ((mel_lens + 1) // 2 + 1) // 2
                out_lens = torch.clamp(out_lens, max=t_out)
                mask = torch.arange(t_out, device=x.device)[None, :] >= out_lens[:, None]
                for block in model.blocks:
                    x = block(x, mask=mask)
                logits = model.head(x)

            # Compute CTC outside autocast in fp32. log_softmax in fp16 can
            # yield -inf and zero_infinity=True then silently zeros the loss.
            log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # (T, B, V)
            # Guard: any sample where out_lens < target_lens is unalignable and
            # would be masked by zero_infinity, learning nothing. Count them.
            unalignable = int((out_lens < target_lens).sum().item())
            loss = ctc(log_probs, targets, out_lens, target_lens) / cfg["train"]["grad_accum"]

            scaler.scale(loss).backward()

            if (it + 1) % cfg["train"]["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if ema is not None:
                    ema.update(model)

            running_loss = 0.9 * running_loss + 0.1 * loss.item() * cfg["train"]["grad_accum"]
            running_unalign = 0.9 * running_unalign + 0.1 * unalignable
            if it % cfg["train"]["log_every"] == 0:
                pbar.set_postfix(
                    loss=f"{running_loss:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    unalign=f"{running_unalign:.1f}",
                )

        # Validate on EMA weights when available — they generalize better,
        # so the hmean-CER curve based on them is a cleaner early-stop signal.
        if (epoch + 1) % cfg["train"]["val_every"] == 0:
            backup = ema.apply_to(model) if ema is not None else None
            try:
                metrics = validate(model, dev_loader, decoder, device, method=val_method)
            finally:
                if backup is not None and ema is not None:
                    ema.restore(model, backup)
            print(
                f"[epoch {epoch}] dev CER = {metrics['cer']:.4f}  "
                f"hmean CER = {metrics['hmean_cer']:.4f}  "
                f"empty = {metrics['empty_pred_rate']:.1%}  "
                f"pred1000 = {metrics['pred_1000_rate']:.1%}  "
                f"per_spk = {metrics['per_spk']}"
            )
            cur = metrics["hmean_cer"]
            if cur < best_metric:
                best_metric = cur
                best_cer = metrics["cer"]
                save_ckpt(
                    out_dir / "best.ckpt",
                    model=model.state_dict(),
                    model_ema=ema.state_dict() if ema is not None else None,
                    cer=metrics["cer"],
                    hmean_cer=metrics["hmean_cer"],
                    epoch=epoch,
                    cfg=cfg,
                )
                print(
                    f"[epoch {epoch}] new best dev hmean CER: {cur:.4f} "
                    f"(mean CER {metrics['cer']:.4f}) -> saved best.ckpt"
                )

        save_ckpt(
            last_ckpt,
            model=model.state_dict(),
            opt=opt.state_dict(),
            sched=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            ema=ema.state_dict() if ema is not None else None,
            epoch=epoch,
            step=global_step,
            best_metric=best_metric,
            best_cer=best_cer,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            cfg=cfg,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)
