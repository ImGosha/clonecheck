"""Turning a clip into numbers a model can learn from.

  mfcc_features   80 numbers describing the average sound spectrum. The
                  classic, simple description of audio, used for the baseline.
  xlsr_features   the output of XLS-R, a speech model pretrained by Meta on
                  436,000 hours of speech in 128 languages. We do not train it.
                  Each clip is passed through it and its internal activations
                  are averaged over time, once for each of its 25 layers
                  (1,024 numbers per layer).
"""
from __future__ import annotations

import numpy as np

from .config import RATE, ROOT

XLSR = "facebook/wav2vec2-xls-r-300m"
_model = None


def mfcc_features(wave: np.ndarray) -> np.ndarray:
    import librosa
    m = librosa.feature.mfcc(y=wave, sr=RATE, n_mfcc=20)
    d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


def xlsr_features(wave: np.ndarray) -> np.ndarray:
    """Average activation of every XLS-R layer: an array of shape (25, 1024)."""
    import torch
    global _model
    if _model is None:
        from transformers import AutoFeatureExtractor, AutoModel
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        _model = (AutoFeatureExtractor.from_pretrained(XLSR, cache_dir=ROOT / "models_cache"),
                  AutoModel.from_pretrained(XLSR, cache_dir=ROOT / "models_cache").to(device).eval(),
                  device)
    extractor, model, device = _model
    x = extractor(wave, sampling_rate=RATE, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        layers = model(x, output_hidden_states=True).hidden_states
    return torch.stack([h[0].mean(0) for h in layers]).float().cpu().numpy().astype(np.float16)
