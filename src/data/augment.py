"""Audio augmentations for Russian spoken-numbers ASR.

All augmentations operate on float32 mono numpy arrays at a fixed
samplerate. They are composed via ``AugmentChain(p=...)`` which rolls an
independent probability per transform. SpecAugment runs separately on
log-mel features inside the training loop.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


Transform = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class AugmentChain:
    transforms: Sequence[tuple[float, Transform]]

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        for p, t in self.transforms:
            if random.random() < p:
                wav = t(wav, sr)
        return wav.astype(np.float32, copy=False)


def speed_perturb(wav: np.ndarray, sr: int, factors: Sequence[float] = (0.9, 1.0, 1.1)) -> np.ndarray:
    factor = random.choice(factors)
    if factor == 1.0:
        return wav
    import librosa
    return librosa.effects.time_stretch(wav, rate=factor)


def pitch_shift(wav: np.ndarray, sr: int, semitones_range: tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    n_steps = random.uniform(*semitones_range)
    import librosa
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps)


def vtlp(wav: np.ndarray, sr: int, alpha_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Vocal-tract-length perturbation via a piecewise warp of the STFT axis."""
    alpha = random.uniform(*alpha_range)
    if abs(alpha - 1.0) < 1e-3:
        return wav
    import librosa
    n_fft = 512
    hop = 128
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(D), np.angle(D)
    freqs = np.arange(mag.shape[0])
    f_lo = 0.85 * (mag.shape[0] - 1)
    warped = np.where(freqs <= f_lo, freqs * alpha, (mag.shape[0] - 1) - (mag.shape[0] - 1 - freqs) * alpha)
    warped = np.clip(warped, 0, mag.shape[0] - 1)
    new_mag = np.zeros_like(mag)
    for t in range(mag.shape[1]):
        new_mag[:, t] = np.interp(np.arange(mag.shape[0]), warped, mag[:, t])
    D_new = new_mag * np.exp(1j * phase)
    return librosa.istft(D_new, hop_length=hop, length=len(wav)).astype(np.float32)


def mp3_recode(wav: np.ndarray, sr: int, bitrates_kbps: Sequence[int] = (32, 48, 64, 96, 128)) -> np.ndarray:
    """Encode the waveform to MP3 in memory, decode back. Mimics dev/test codec artifacts."""
    try:
        from pydub import AudioSegment
    except ImportError:
        return wav
    br = random.choice(bitrates_kbps)
    pcm16 = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
    seg = AudioSegment(pcm16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    buf = io.BytesIO()
    try:
        seg.export(buf, format="mp3", bitrate=f"{br}k")
        buf.seek(0)
        decoded = AudioSegment.from_file(buf, format="mp3")
    except Exception:
        return wav
    samples = np.array(decoded.get_array_of_samples(), dtype=np.float32) / 32768.0
    if decoded.frame_rate != sr:
        import librosa
        samples = librosa.resample(samples, orig_sr=decoded.frame_rate, target_sr=sr, res_type="kaiser_fast")
    return samples[: len(wav)] if len(samples) >= len(wav) else np.pad(samples, (0, len(wav) - len(samples)))


class NoiseMixer:
    """Additive background noise from a directory of MUSAN-style .wav files."""

    def __init__(self, noise_dir: str | Path, snr_db_range: tuple[float, float] = (5.0, 20.0)) -> None:
        self.snr_range = snr_db_range
        self.files: list[Path] = []
        if noise_dir:
            root = Path(noise_dir)
            if root.exists():
                self.files = [p for p in root.rglob("*.wav")]

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if not self.files:
            return wav
        import soundfile as sf
        path = random.choice(self.files)
        noise, nsr = sf.read(str(path), dtype="float32", always_2d=False)
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if nsr != sr:
            import librosa
            noise = librosa.resample(noise, orig_sr=nsr, target_sr=sr, res_type="kaiser_fast")
        if len(noise) < len(wav):
            reps = int(np.ceil(len(wav) / max(len(noise), 1)))
            noise = np.tile(noise, reps)
        start = random.randint(0, len(noise) - len(wav))
        noise = noise[start : start + len(wav)]
        snr = random.uniform(*self.snr_range)
        sig_p = np.mean(wav ** 2) + 1e-10
        noise_p = np.mean(noise ** 2) + 1e-10
        scale = np.sqrt(sig_p / (noise_p * 10 ** (snr / 10)))
        return wav + scale * noise


class ReverbMixer:
    """Convolve with a random RIR from a directory."""

    def __init__(self, rir_dir: str | Path) -> None:
        self.files: list[Path] = []
        if rir_dir:
            root = Path(rir_dir)
            if root.exists():
                self.files = [p for p in root.rglob("*.wav")]

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if not self.files:
            return wav
        import soundfile as sf
        import scipy.signal as signal
        rir, rsr = sf.read(str(random.choice(self.files)), dtype="float32", always_2d=False)
        if rir.ndim > 1:
            rir = rir.mean(axis=1)
        if rsr != sr:
            import librosa
            rir = librosa.resample(rir, orig_sr=rsr, target_sr=sr, res_type="kaiser_fast")
        rir = rir[: sr // 2]  # cap RIR at 0.5s to keep this fast
        rir = rir / (np.abs(rir).max() + 1e-9)
        out = signal.fftconvolve(wav, rir, mode="full")[: len(wav)]
        peak = np.abs(out).max()
        if peak > 0:
            out = out / peak
        return out.astype(np.float32)


def gain_perturb(wav: np.ndarray, sr: int, db_range: tuple[float, float] = (-10.0, 10.0)) -> np.ndarray:
    db = random.uniform(*db_range)
    return wav * (10.0 ** (db / 20.0))


def spec_augment(feat: "torch.Tensor", num_freq_masks: int = 2, freq_mask_param: int = 15,
                 num_time_masks: int = 2, time_mask_param: int = 35) -> "torch.Tensor":
    """Apply SpecAugment to a log-mel feature tensor of shape (B, n_mels, T)."""
    import torch
    out = feat.clone()
    B, M, T = out.shape
    for b in range(B):
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            if f == 0:
                continue
            f0 = random.randint(0, max(0, M - f))
            out[b, f0 : f0 + f, :] = out[b].mean()
        for _ in range(num_time_masks):
            t = random.randint(0, time_mask_param)
            if t == 0:
                continue
            t0 = random.randint(0, max(0, T - t))
            out[b, :, t0 : t0 + t] = out[b].mean()
    return out


def build_train_augment(noise_dir: str | Path | None = None, rir_dir: str | Path | None = None) -> AugmentChain:
    noise = NoiseMixer(noise_dir) if noise_dir else None
    reverb = ReverbMixer(rir_dir) if rir_dir else None
    transforms: list[tuple[float, Transform]] = [
        (0.7, speed_perturb),
        (0.5, pitch_shift),
        (0.3, vtlp),
        (0.4, mp3_recode),
        (0.6, gain_perturb),
    ]
    if reverb is not None and reverb.files:
        transforms.append((0.3, reverb))
    if noise is not None and noise.files:
        transforms.append((0.5, noise))
    return AugmentChain(transforms)
