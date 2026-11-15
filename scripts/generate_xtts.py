"""Create the XTTS clips listed in data/clips.csv (the cloning tool held out for testing).

XTTS-v2 (Coqui) copies a voice from a few seconds of audio. Its model is
released under the Coqui Public Model License (non-commercial use), which
this script accepts by setting COQUI_TOS_AGREED.

Run:  python scripts/generate_xtts.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd
import soundfile as sf

os.environ["COQUI_TOS_AGREED"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import MANIFEST, RATE, ROOT
from clonecheck.voices import to_16k


def main() -> None:
    import torch
    from TTS.api import TTS
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    todo = clips[clips["engine"] == "xtts_clone"]
    todo = todo[[not (ROOT / p).exists() for p in todo["path"]]]
    print(f"xtts: {len(todo)} clips to make", flush=True)
    if todo.empty:
        return
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
        "mps" if torch.backends.mps.is_available() else "cpu")
    started = time.time()
    for n, r in enumerate(todo.itertuples(), 1):
        out = ROOT / r.path
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                model.tts_to_file(text=r.text, speaker_wav=str(ROOT / r.voice), language=r.lang,
                                  file_path=f"{tmp}/a.wav")
                wave, rate = sf.read(f"{tmp}/a.wav", dtype="float32")
            sf.write(out, to_16k(wave, rate), RATE, subtype="PCM_16")
        except Exception as error:
            print(f"  failed {r.clip_id}: {type(error).__name__}: {str(error)[:120]}", flush=True)
        if n % 25 == 0:
            print(f"  xtts: {n}/{len(todo)} in {time.time() - started:.0f}s", flush=True)
    print(f"xtts: done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
