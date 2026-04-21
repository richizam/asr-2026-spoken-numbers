"""Conformer-CTC small (~4M params) for Russian spoken-number ASR.

Architecture:
  LogMel front-end  ->  Conv2d subsampling (4x)  ->  linear proj
    ->  N x Conformer block (half-FFN + MHSA + Conv + half-FFN + LN)
    ->  linear head over the character vocab.

Standard sinusoidal positional encoding (not relative) keeps the model
small. For the scale of this dataset (~12k utterances, ~300 frames each
after subsampling) absolute PE is sufficient and saves ~0.5M params vs.
relative-positional MHA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .melbanks import LogMelFilterBanks


@dataclass
class ConformerCTCConfig:
    vocab_size: int = 35
    n_mels: int = 80
    samplerate: int = 16000
    d_model: int = 128
    ffn_expansion: int = 2
    num_heads: int = 4
    num_layers: int = 12
    conv_kernel: int = 15
    dropout: float = 0.1
    max_len: int = 3000
    subsampling_channels: int = 32


def _sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ConvSubsampling(nn.Module):
    """Two stride-2 Conv2d layers => 4x time reduction, 4x freq reduction."""

    def __init__(self, in_mels: int, out_channels: int, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        sub_mels = (in_mels + 1) // 2
        sub_mels = (sub_mels + 1) // 2
        self.linear = nn.Linear(out_channels * sub_mels, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_mels, T)
        x = x.unsqueeze(1)              # (B, 1, n_mels, T)
        x = self.conv(x)                # (B, C, n_mels/4, T/4)
        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * m)  # (B, T', C*mels')
        return self.linear(x)           # (B, T', d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(self.norm(x))))))


class MHSABlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        y = self.norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        return self.drop(y)


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model
        )
        # LayerNorm over the channel dim — no running stats, no train/eval gap
        # under AMP + SpecAugment, unlike the original BatchNorm1d.
        self.mid_norm = nn.LayerNorm(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x).transpose(1, 2)          # (B, D, T)
        y = self.pw1(y)
        y = F.glu(y, dim=1)                       # (B, D, T)
        y = self.dw(y)
        y = self.mid_norm(y.transpose(1, 2)).transpose(1, 2)
        y = F.silu(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y.transpose(1, 2)                  # (B, T, D)


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ConformerCTCConfig) -> None:
        super().__init__()
        self.ff1 = FeedForward(cfg.d_model, cfg.ffn_expansion, cfg.dropout)
        self.mhsa = MHSABlock(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.conv = ConvModule(cfg.d_model, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(cfg.d_model, cfg.ffn_expansion, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask=mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerCTC(nn.Module):
    def __init__(self, cfg: ConformerCTCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = LogMelFilterBanks(samplerate=cfg.samplerate, n_mels=cfg.n_mels)
        # Learnable channel-wise affine stacked on top of CMVN. CMVN alone
        # zeroes the per-utt mean/var but cannot scale individual mel bins;
        # the LayerNorm gives the front-end that extra degree of freedom.
        self.input_norm = nn.LayerNorm(cfg.n_mels)
        self.subsampling = ConvSubsampling(cfg.n_mels, cfg.subsampling_channels, cfg.d_model)
        self.input_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("pe", _sinusoidal_pe(cfg.max_len, cfg.d_model), persistent=False)
        self.blocks = nn.ModuleList([ConformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.subsample_ratio = 4

    def compute_mel_lens(self, lengths: Tensor, t_mel: int) -> Tensor:
        # audio samples -> mel frames (center=True, pad): floor(N/hop)+1
        mel_lens = torch.div(lengths, self.frontend.hop_length, rounding_mode="floor") + 1
        return torch.clamp(mel_lens, max=t_mel)

    def normalize_mel(self, mel: Tensor, mel_lens: Tensor | None = None) -> Tensor:
        """Per-utterance CMVN (across time) then a learnable LayerNorm over mel bins.

        CMVN removes speaker-level spectral tilt / DC offset — the single
        largest numerical difference between an in-domain and an OOD voice.
        Respecting mel_lens matters when the batch is padded: the padded
        frames sit at log(1e-6) and would otherwise drag the mean down.

        After normalization, padded frames are zeroed out. Without this,
        the LN affine's beta bias leaks into the padded tail and the
        stride-2 Conv2d subsampling bleeds that bias into the last
        ~kernel/stride valid frames at each utterance end.
        """
        mask_time: Tensor | None = None
        if mel_lens is None:
            mean = mel.mean(dim=-1, keepdim=True)
            var = mel.var(dim=-1, keepdim=True, unbiased=False)
        else:
            t = mel.size(-1)
            mask_time = (torch.arange(t, device=mel.device)[None, :] < mel_lens[:, None]).unsqueeze(1).to(mel.dtype)
            valid = mask_time.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean = (mel * mask_time).sum(dim=-1, keepdim=True) / valid
            var = ((mel - mean) ** 2 * mask_time).sum(dim=-1, keepdim=True) / valid
        std = torch.sqrt(var.clamp(min=1e-5))
        mel = (mel - mean) / std
        mel = self.input_norm(mel.transpose(1, 2)).transpose(1, 2)
        if mask_time is not None:
            mel = mel * mask_time
        return mel

    def forward(self, waveform: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            waveform: (B, T_audio) float32 16 kHz.
            lengths: (B,) long, number of valid audio samples per utt.
        Returns:
            log_probs: (T_out, B, V) log-softmax, time-first for CTCLoss.
            out_lens:  (B,) long, valid frame counts after subsampling.
        """
        mel = self.frontend(waveform)               # (B, n_mels, T_mel)
        t_mel = mel.size(-1)
        mel_lens = self.compute_mel_lens(lengths, t_mel) if lengths is not None else None
        mel = self.normalize_mel(mel, mel_lens)
        x = self.subsampling(mel)                   # (B, T', D)
        t_out = x.size(1)
        x = x + self.pe[:t_out].to(x.dtype)
        x = self.input_drop(x)

        if mel_lens is not None:
            # subsampling: each Conv2d stride-2 -> ceil(n/2)
            out_lens = ((mel_lens + 1) // 2 + 1) // 2
            out_lens = torch.clamp(out_lens, max=t_out)
            mask = torch.arange(t_out, device=x.device)[None, :] >= out_lens[:, None]
        else:
            out_lens = torch.full((x.size(0),), t_out, dtype=torch.long, device=x.device)
            mask = None

        for block in self.blocks:
            x = block(x, mask=mask)

        logits = self.head(x)                       # (B, T', V)
        # log_softmax in fp32 — under AMP the fp16 version can produce -inf
        # and poison the CTC alignment grid via zero_infinity.
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return log_probs.transpose(0, 1), out_lens  # (T, B, V), (B,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_default_model(vocab_size: int = 35) -> ConformerCTC:
    return ConformerCTC(ConformerCTCConfig(vocab_size=vocab_size))
