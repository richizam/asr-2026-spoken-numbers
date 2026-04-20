"""Log-mel filterbank front-end.

Adapted from assignments/assignment1/melbanks.py (the skeleton).  All the
``<YOUR CODE GOES HERE>`` holes are filled in so ``forward(x)`` returns a
``(batch, n_mels, n_frames)`` tensor of natural-log mel energies ready for
the acoustic model.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
        self,
        n_fft: int = 400,
        samplerate: int = 16000,
        hop_length: int = 160,
        n_mels: int = 80,
        pad_mode: str = "reflect",
        power: float = 2.0,
        normalize_stft: bool = False,
        onesided: bool = True,
        center: bool = True,
        return_complex: bool = True,
        f_min_hz: float = 0.0,
        f_max_hz: Optional[float] = None,
        norm_mel: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.pad_mode = pad_mode
        self.power = power
        self.normalize_stft = normalize_stft
        self.onesided = onesided
        self.center = center
        self.return_complex = return_complex
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz if f_max_hz is not None else samplerate / 2
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        self.register_buffer("window", torch.hann_window(self.window_length), persistent=False)
        self.register_buffer("mel_fbanks", self._init_melscale_fbanks(), persistent=False)

    def _init_melscale_fbanks(self) -> torch.Tensor:
        return F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale,
        )

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )
        mag = stft.abs().pow(self.power)  # (B, n_freqs, T)
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time) waveform at ``self.samplerate``.
        Returns:
            (batch, n_mels, n_frames) natural-log mel energies.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        spec = self.spectrogram(x)                              # (B, n_freqs, T)
        mel = torch.matmul(spec.transpose(-1, -2), self.mel_fbanks)  # (B, T, n_mels)
        mel = mel.transpose(-1, -2)                             # (B, n_mels, T)
        return torch.log(mel + 1e-6)
