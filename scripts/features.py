"""Turn every clip into numbers a model can learn from.

For each clip, under each listening condition (clean, and squeezed through a
simulated phone line), this saves the two kinds of features described in
clonecheck/features.py: 80 MFCC numbers and the 25 x 1,024 XLS-R numbers.

Results go to data/features/<condition>.<run>.npz. Clips already done are
skipped, so the script can be stopped and restarted. To use several
processes at once, give each a shard:

    python scripts/features.py          (clean and phone, every clip, one process)
    python scripts/features.py 0 3      (shard 0 of 3; also run 1 3 and 2 3)

Other listening conditions (see clonecheck/audio.py) for some splits only:

    CONDITIONS=aug1,aug2 SPLITS=train python scripts/features.py
    CONDITIONS=noisy,gated SPLITS=dev,test python scripts/features.py 0 2
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.audio import condition, load, prepare
from clonecheck.config import FEATURES, MANIFEST, RATE, ROOT
from clonecheck.features import mfcc_features, xlsr_features


CONDITIONS = os.environ.get("CONDITIONS", "clean,phone").split(",")
SPLITS = os.environ.get("SPLITS", "train,dev,test").split(",")


def already_done() -> set:
    done = None
    for name in CONDITIONS:
        ids = set()
        for file in list(FEATURES.glob(f"{name}.*.npz")) + list(FEATURES.glob(f"{name}.npz")):
            ids |= set(np.load(file, allow_pickle=False)["ids"])
        done = ids if done is None else done & ids
    return done or set()


def save(path: Path, store: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(store["ids"]), mfcc=np.stack(store["mfcc"]),
             xlsr=np.stack(store["xlsr"]))
    tmp.replace(path)


def main() -> None:
    shard, shards = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    stamp = time.strftime("%H%M%S")
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    clips = clips[clips["split"].isin(SPLITS)]
    clips = clips[[(ROOT / p).exists() for p in clips["path"]]]
    clips = clips[~clips["clip_id"].isin(already_done())]
    clips = clips[[zlib.crc32(c.encode()) % shards == shard for c in clips["clip_id"]]]
    print(f"shard {shard}/{shards}: {len(clips)} clips to process", flush=True)
    FEATURES.mkdir(parents=True, exist_ok=True)
    stores = {c: {"ids": [], "mfcc": [], "xlsr": []} for c in CONDITIONS}
    started = time.time()
    for n, r in enumerate(clips.itertuples(), 1):
        try:
            wave = prepare(load(ROOT / r.path))
            if len(wave) < RATE // 2:
                raise ValueError("clip shorter than half a second")
            # work out both conditions first, then store them together
            ready = {}
            for name in CONDITIONS:
                heard = condition(wave, name, r.clip_id)
                ready[name] = (mfcc_features(heard), xlsr_features(heard))
            for name, (mfcc, xlsr) in ready.items():
                stores[name]["mfcc"].append(mfcc)
                stores[name]["xlsr"].append(xlsr)
                stores[name]["ids"].append(r.clip_id)
        except Exception as error:
            print(f"  skipped {r.clip_id}: {type(error).__name__}: {str(error)[:100]}", flush=True)
        if (n % 250 == 0 or n == len(clips)) and stores[CONDITIONS[0]]["ids"]:
            for name in CONDITIONS:
                save(FEATURES / f"{name}.{stamp}-{shard}.npz", stores[name])
            print(f"  shard {shard}: {n}/{len(clips)} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
