"""Create the Chatterbox clips listed in data/clips.csv.

Chatterbox (Resemble AI) needs older library versions than the rest of the
project, so it has its own environment:

    uv venv --python 3.11 .venv-chatterbox
    uv pip install --python .venv-chatterbox/bin/python chatterbox-tts "setuptools<81" pandas soundfile scipy
    .venv-chatterbox/bin/python scripts/generate_chatterbox.py

With --clone OUT_DIR it instead reads data/own_sentences.csv in a copy of my
own voice, using data/audio/own/_reference_<lang>.wav as the voice to copy
(used for the "clone of my own voice" test; see scripts/own_voice.py).
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RATE = 16_000
device = "mps" if torch.backends.mps.is_available() else "cpu"
_models = {}


def model_for(lang: str):
    if lang not in _models:
        if lang == "en":
            from chatterbox.tts import ChatterboxTTS
            _models[lang] = ChatterboxTTS.from_pretrained(device=device)
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            _models[lang] = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return _models[lang]


def speak(text: str, lang: str, reference: str | None = None) -> np.ndarray:
    model = model_for(lang)
    extra = {"audio_prompt_path": reference} if reference else {}
    if lang != "en":
        extra["language_id"] = lang
    wave = model.generate(text, **extra).squeeze().cpu().numpy().astype(np.float32)
    g = np.gcd(model.sr, RATE)
    return resample_poly(wave, RATE // g, model.sr // g).astype(np.float32)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--clone":
        out_dir = Path(sys.argv[2])
        own = ROOT / "data" / "audio" / "own"
        jobs = [(r.lang, r.text, out_dir / f"{r.id}.wav", str(own / f"_reference_{r.lang}.wav"))
                for r in pd.read_csv(ROOT / "data" / "own_sentences.csv").itertuples()]
    else:
        clips = pd.read_csv(ROOT / "data" / "clips.csv", keep_default_na=False)
        jobs = [(r.lang, r.text, ROOT / r.path, None)
                for r in clips[clips["engine"] == "chatterbox"].itertuples()]
    only = os.environ.get("ONLY", "")          # e.g. ONLY=minds to make just those clips
    jobs = [j for j in jobs if not j[2].exists() and only in j[2].name]
    print(f"chatterbox: {len(jobs)} clips to make", flush=True)
    started = time.time()
    for n, (lang, text, out, reference) in enumerate(jobs, 1):
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            sf.write(out, speak(text, lang, reference), RATE, subtype="PCM_16")
        except Exception as error:
            print(f"  failed {out.name}: {type(error).__name__}: {str(error)[:120]}", flush=True)
        if n % 10 == 0:
            print(f"  {n}/{len(jobs)} in {time.time() - started:.0f}s", flush=True)
    print(f"chatterbox: done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
