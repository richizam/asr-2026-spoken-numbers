import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.conformer_ctc import ConformerCTC, ConformerCTCConfig, build_default_model


def test_param_count_under_5m() -> None:
    model = build_default_model()
    n = model.num_parameters()
    assert n < 5_000_000, f"model has {n:,} params, over 5M budget"
    assert n > 1_000_000, f"model has only {n:,} params, too small"
    print(f"ConformerCTC param count: {n:,}")


def test_forward_shape() -> None:
    model = build_default_model()
    wav = torch.randn(2, 16000 * 3)
    lens = torch.tensor([16000 * 3, 16000 * 2])
    log_probs, out_lens = model(wav, lens)
    # log_probs: (T, B, V)
    assert log_probs.dim() == 3
    assert log_probs.size(1) == 2
    assert log_probs.size(2) == 35
    assert (out_lens > 0).all()
    assert (out_lens <= log_probs.size(0)).all()


def test_backward() -> None:
    model = build_default_model()
    wav = torch.randn(2, 16000 * 2)
    log_probs, out_lens = model(wav)
    # dummy CTC loss: targets are zeros of length 3
    targets = torch.tensor([[1, 2, 3], [1, 2, 3]])
    target_lens = torch.tensor([3, 3])
    loss = torch.nn.functional.ctc_loss(
        log_probs, targets, out_lens, target_lens, blank=0, zero_infinity=True
    )
    loss.backward()
    assert torch.isfinite(loss).item()
