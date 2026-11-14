"""Create the fake clips listed in data/clips.csv.

Each AI voice generator reads its sentences aloud and the result is saved as
a 16 kHz wav file. Clips that already exist are skipped, so the script can
be stopped and restarted.

Run:  python scripts/generate.py            (all generators)
      python scripts/generate.py piper mms  (only some)
      SHARD=0/2 python scripts/generate.py say   (half the clips; run SHARD=1/2 alongside)
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import MANIFEST, RATE, ROOT
from clonecheck.voices import speak


def main() -> None:
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    fake = clips[clips["label"] == "fake"]
    engines = sys.argv[1:] or sorted(fake["engine"].unique())
    for engine in engines:
        todo = fake[(fake["engine"] == engine)]
        todo = todo[[not (ROOT / p).exists() for p in todo["path"]]]
        shard, shards = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
        todo = todo.iloc[shard::shards]
        print(f"{engine}: {len(todo)} clips to make", flush=True)
        started, failed = time.time(), 0
        for n, r in enumerate(todo.itertuples(), 1):
            out = ROOT / r.path
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                wave = speak(engine, r.voice, r.lang, r.text)
                if len(wave) < RATE // 2 or not np.isfinite(wave).all():
                    raise ValueError("empty or broken audio")
                sf.write(out, wave, RATE, subtype="PCM_16")
            except Exception as error:  # keep going; missing clips are reported at the end
                failed += 1
                print(f"  failed {r.clip_id}: {type(error).__name__}: {str(error)[:120]}", flush=True)
            if n % 100 == 0:
                print(f"  {engine}: {n}/{len(todo)} in {time.time() - started:.0f}s", flush=True)
        print(f"{engine}: done, {failed} failed, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
