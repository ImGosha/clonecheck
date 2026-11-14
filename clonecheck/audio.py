"""Audio preparation: the same steps for every clip, real or fake.

Why this matters: a detector will happily learn any accidental difference
between the two piles of clips (fake ones being louder, or starting without
a pause). So before a clip reaches a model it is trimmed, cut to the same
maximum length and set to the same loudness.

`phone_line` imitates a telephone call, which is how a scam call would
actually arrive: only the 300 to 3400 Hz band survives, at 8,000 samples a
second, squeezed with the mu-law code used by landline networks (G.711).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from .config import MAX_SECONDS, RATE


def load(path) -> np.ndarray:
    """Read a file as 16 kHz mono float32."""
    wave, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != RATE:
        g = np.gcd(rate, RATE)
        wave = resample_poly(wave, RATE // g, rate // g).astype(np.float32)
    return wave


def trim_silence(wave: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    """Cut quiet stretches from the start and the end."""
    if len(wave) < RATE // 10:
        return wave
    frame = RATE // 50  # 20 ms
    n = len(wave) // frame
    energy = (wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    level = 10 * np.log10(energy + 1e-12)
    loud = np.where(level > level.max() - top_db)[0]
    if len(loud) == 0:
        return wave
    return wave[loud[0] * frame: (loud[-1] + 1) * frame]


def set_loudness(wave: np.ndarray, rms: float = 0.05) -> np.ndarray:
    """Scale to a fixed average loudness, without clipping."""
    current = np.sqrt((wave ** 2).mean() + 1e-12)
    wave = wave * (rms / current)
    peak = np.abs(wave).max()
    return (wave / peak * 0.99 if peak > 0.99 else wave).astype(np.float32)


def prepare(wave: np.ndarray) -> np.ndarray:
    """Trim, cut to MAX_SECONDS and set the loudness."""
    wave = trim_silence(np.asarray(wave, dtype=np.float32))
    wave = wave[: MAX_SECONDS * RATE]
    return set_loudness(wave)


def _mu_law(wave: np.ndarray, levels: int = 256) -> np.ndarray:
    """Squeeze to 8 bits with the mu-law curve and expand again."""
    mu = levels - 1
    squeezed = np.sign(wave) * np.log1p(mu * np.abs(wave)) / np.log1p(mu)
    stepped = np.round((squeezed + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(stepped) * np.expm1(np.abs(stepped) * np.log1p(mu)) / mu


def phone_line(wave: np.ndarray) -> np.ndarray:
    """Imitate a landline phone call. Input and output are 16 kHz."""
    band = butter(4, [300, 3400], btype="bandpass", fs=RATE, output="sos")
    narrow = sosfiltfilt(band, wave)
    at_8k = resample_poly(narrow, 1, 2)
    peak = np.abs(at_8k).max() + 1e-9
    coded = _mu_law(np.clip(at_8k / peak, -1, 1)) * peak
    return set_loudness(resample_poly(coded, 2, 1).astype(np.float32))


def add_noise(wave: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add steady background hiss at a chosen signal-to-noise ratio."""
    white = rng.standard_normal(len(wave))
    # make the hiss "pink" (more low rumble, like a room), not pure white
    spectrum = np.fft.rfft(white)
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    pink = np.fft.irfft(spectrum, n=len(wave))
    pink *= np.sqrt((wave ** 2).mean() / ((pink ** 2).mean() + 1e-12)) / (10 ** (snr_db / 20))
    return set_loudness((wave + pink).astype(np.float32))


def condition(wave: np.ndarray, name: str) -> np.ndarray:
    """Apply one listening condition to a prepared clip."""
    if name == "clean":
        return wave
    if name == "phone":
        return phone_line(wave)
    raise ValueError(name)
