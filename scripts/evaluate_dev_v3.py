"""Version 3 on dev data: can the "clean means AI" shortcut be trained away?

After the test, my own recordings showed that the detector partly relies on
how clean a recording is. Version 3 trains on every clip four times: as
recorded, through the phone line, and twice with a random mix of background
noise, room echo, noise suppression and phone line (clonecheck/audio.py,
random_recording). Real and fake clips get the same treatment, so "is it
clean?" can no longer separate them.

This script compares version 2 and version 3 on dev data under four
listening conditions (clean, phone, noisy, gated = noise suppression).
My first 30 recordings and their AI copies are shown too: they revealed the
problem, so from here on they count as dev data, not test data.

Run:  python scripts/evaluate_dev_v3.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import RESULTS, TRAIN_SOURCES
from clonecheck.detect import load_clips, load_features, report, set_threshold, train

HEARD = ("clean", "phone", "noisy", "gated")
RECIPES = {
    "version 2": (("clean", "phone"), ("clean", "phone")),          # (train on, alarm level from)
    "version 3": (("clean", "phone", "aug1", "aug2"), HEARD),
}


def main() -> None:
    layer = json.loads((RESULTS / "chosen.json").read_text())["layer"]
    features = load_features(layers=[layer], conditions=list(HEARD) + ["aug1", "aug2"])
    clips = load_clips()
    clips = clips.assign(who=[r.source if r.label == "real" else r.engine for r in clips.itertuples()])
    train_clips = clips[clips["split"] == "train"]
    first_session = clips[clips["text_source"] == "own"]
    dev = pd.concat([clips[clips["split"] == "dev"], first_session])
    real_dev = dev[(dev["label"] == "real") & dev["source"].isin(TRAIN_SOURCES)]

    tables = []
    for name, (train_on, level_from) in RECIPES.items():
        for kind in ["mfcc", "xlsr"]:
            detector = train(features, train_clips, kind, layer if kind == "xlsr" else None, train_on)
            set_threshold(detector, features, real_dev, level_from)
            table = report(detector, features, dev, ["label", "who", "lang"], HEARD)
            tables.append(table.assign(features=kind, recipe=name, threshold=detector.threshold))
            if kind == "xlsr":
                print(f"== {name}  (alarm level {detector.threshold:.3f}) - % flagged as AI")
                wide = table.pivot_table(index=["label", "who"], columns=["lang", "condition"],
                                         values="share", sort=False)
                print((100 * wide).round(0).to_string(), "\n")
    pd.concat(tables).to_csv(RESULTS / "dev_v3_recipes.csv", index=False)


if __name__ == "__main__":
    main()
