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


def noise_gate(wave: np.ndarray, depth_db: float = 40.0) -> np.ndarray:
    """Imitate the noise suppression in browsers and phones: silence every quiet moment.

    Anything more than `depth_db` below the loudest part of the clip is set to
    exact digital silence, which is what my own browser recordings looked like.
    """
    frame = RATE // 100  # 10 ms
    n = len(wave) // frame
    if n < 3:
        return wave
    level = 10 * np.log10((wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    keep = (level > level.max() - depth_db).astype(np.float32)
    keep = np.convolve(keep, np.ones(3) / 3, mode="same")          # soften the edges
    gain = np.repeat(keep, frame)
    out = wave.copy()
    out[: n * frame] *= gain
    return set_loudness(out)


def reverb(wave: np.ndarray, seconds: float, wet: float, rng: np.random.Generator) -> np.ndarray:
    """Add the echo of a room: the clip plus a decaying, blurred copy of itself."""
    from scipy.signal import fftconvolve
    n = int(seconds * RATE)
    tail = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    tail /= np.sqrt((tail ** 2).sum())
    echo = fftconvolve(wave, tail)[: len(wave)]
    return set_loudness((wave + wet * echo).astype(np.float32))


def random_recording(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random mix of the things that happen to real recordings (used for training only).

    Applied to real and fake clips alike, so that a detector cannot use "is
    there background noise?" or "is it suspiciously clean?" to tell them apart.
    """
    if rng.random() < 0.4:
        wave = reverb(wave, rng.uniform(0.15, 0.7), rng.uniform(0.2, 0.6), rng)
    choice = rng.random()
    if choice < 0.55:
        wave = add_noise(wave, rng.uniform(5, 30), rng)
    elif choice < 0.8:
        wave = noise_gate(wave, rng.uniform(30, 45))
    if rng.random() < 0.5:
        wave = phone_line(wave)
    return wave


def condition(wave: np.ndarray, name: str, clip_id: str = "") -> np.ndarray:
    """Apply one listening condition to a prepared clip.

    clean   as recorded                phone   simulated phone line
    noisy   background hiss at 20 dB   gated   noise suppression (quiet moments silenced)
    aug1, aug2   two different random mixes of all of these (training only)
    """
    import zlib
    rng = np.random.default_rng(zlib.crc32(f"{clip_id}|{name}".encode()))
    if name == "clean":
        return wave
    if name == "phone":
        return phone_line(wave)
    if name == "noisy":
        return add_noise(wave, 20.0, rng)
    if name == "gated":
        return noise_gate(wave)
    if name.startswith("aug"):
        return random_recording(wave, rng)
    raise ValueError(name)
