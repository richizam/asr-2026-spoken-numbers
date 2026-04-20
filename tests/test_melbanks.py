import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.melbanks import LogMelFilterBanks


def test_shape_single_utterance() -> None:
    mel = LogMelFilterBanks(samplerate=16000, n_fft=400, hop_length=160, n_mels=80)
    x = torch.randn(1, 16000 * 3)  # 3-second utterance
    y = mel(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 80
    # center=True -> ceil(T / hop) + 1 frames
    expected_frames = 16000 * 3 // 160 + 1
    assert abs(y.shape[2] - expected_frames) <= 1


def test_shape_batched() -> None:
    mel = LogMelFilterBanks()
    x = torch.randn(4, 16000 * 2)
    y = mel(x)
    assert y.shape[:2] == (4, 80)


def test_finite() -> None:
    mel = LogMelFilterBanks()
    x = torch.randn(2, 16000)
    y = mel(x)
    assert torch.isfinite(y).all()
