"""Turn every clip into numbers a model can learn from.

Two kinds of features are saved for each clip, under each listening
condition (clean, and squeezed through a simulated phone line):

  mfcc   80 numbers describing the average sound spectrum. This is the
         classic, simple description of audio, used for the baseline.
  xlsr   the output of XLS-R, a speech model pretrained by Meta on 436,000
         hours of speech in 128 languages. We do not train it. We pass each
         clip through it and average its internal activations over time,
         once for each of its 25 layers (1,024 numbers per layer).

Results go to data/features/<condition>.npz. Clips already done are
skipped, so the script can be stopped and restarted.

Run:  python scripts/features.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.audio import condition, load, prepare
from clonecheck.config import CONDITIONS, FEATURES, MANIFEST, RATE, ROOT

XLSR = "facebook/wav2vec2-xls-r-300m"
_model = None


def mfcc_features(wave: np.ndarray) -> np.ndarray:
    m = librosa.feature.mfcc(y=wave, sr=RATE, n_mfcc=20)
    d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


def xlsr_features(wave: np.ndarray) -> np.ndarray:
    """Average activation of every XLS-R layer: an array of shape (25, 1024)."""
    global _model
    if _model is None:
        from transformers import AutoFeatureExtractor, AutoModel
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        _model = (AutoFeatureExtractor.from_pretrained(XLSR, cache_dir=ROOT / "models_cache"),
                  AutoModel.from_pretrained(XLSR, cache_dir=ROOT / "models_cache").to(device).eval(),
                  device)
    extractor, model, device = _model
    x = extractor(wave, sampling_rate=RATE, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        layers = model(x, output_hidden_states=True).hidden_states
    return torch.stack([h[0].mean(0) for h in layers]).float().cpu().numpy().astype(np.float16)


def load_store(path: Path) -> dict:
    if not path.exists():
        return {"ids": [], "mfcc": [], "xlsr": []}
    saved = np.load(path, allow_pickle=False)
    return {"ids": list(saved["ids"]), "mfcc": list(saved["mfcc"]), "xlsr": list(saved["xlsr"])}


def save_store(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(store["ids"]), mfcc=np.stack(store["mfcc"]),
             xlsr=np.stack(store["xlsr"]))
    tmp.replace(path)


def main() -> None:
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    clips = clips[[(ROOT / p).exists() for p in clips["path"]]]
    stores = {c: load_store(FEATURES / f"{c}.npz") for c in CONDITIONS}
    assert len({tuple(s["ids"]) for s in stores.values()}) == 1, "feature files are out of step"
    done = set(stores[CONDITIONS[0]]["ids"])
    todo = clips[~clips["clip_id"].isin(done)]
    print(f"{len(todo)} clips to process ({len(done)} already done)", flush=True)
    started = time.time()
    for n, r in enumerate(todo.itertuples(), 1):
        try:
            wave = prepare(load(ROOT / r.path))
            if len(wave) < RATE // 2:
                raise ValueError("clip shorter than half a second")
            # work out both conditions first, then store them together
            ready = {}
            for name in CONDITIONS:
                heard = condition(wave, name)
                ready[name] = (mfcc_features(heard), xlsr_features(heard))
            for name, (mfcc, xlsr) in ready.items():
                stores[name]["mfcc"].append(mfcc)
                stores[name]["xlsr"].append(xlsr)
                stores[name]["ids"].append(r.clip_id)
        except Exception as error:
            print(f"  skipped {r.clip_id}: {type(error).__name__}: {str(error)[:100]}", flush=True)
        if n % 250 == 0 or n == len(todo):
            for name in CONDITIONS:
                save_store(FEATURES / f"{name}.npz", stores[name])
            print(f"  {n}/{len(todo)} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
