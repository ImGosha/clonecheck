"""Choose settings on dev data. Nothing here touches the test split.

1. Train a detector on the train split for the MFCC features and for every
   layer of XLS-R, and score each on dev against Silero, a generator that
   was not in training.
2. Pick the XLS-R layer that separates real from Silero best.
3. Compare two training recipes for that layer:
       clean only            train on clips as recorded
       clean + phone         train on each clip twice, once through the phone line

Run:  python scripts/evaluate_dev.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import CONDITIONS, RESULTS, TRAIN_ENGINES
from clonecheck.detect import auc, load_clips, load_features, report, set_threshold, train


def main() -> None:
    features = load_features()
    clips = load_clips()
    clips = clips[clips["clip_id"].isin(features["clean"]["index"])]
    train_clips = clips[clips["split"] == "train"]
    dev = clips[clips["split"] == "dev"]
    real_dev = dev[dev["label"] == "real"]
    unseen = dev[(dev["label"] == "real") | (dev["engine"] == "silero")]
    seen = dev[(dev["label"] == "real") | (dev["engine"].isin(TRAIN_ENGINES))]
    print(f"train {len(train_clips)} clips, dev {len(dev)} clips "
          f"({len(real_dev)} real, {(dev.engine == 'silero').sum()} Silero)\n")

    # ---- 1. every feature type, trained on clean clips
    rows = []
    for kind, layer in [("mfcc", None)] + [("xlsr", i) for i in range(25)]:
        detector = train(features, train_clips, kind, layer)
        rows.append({"features": kind if layer is None else f"xlsr layer {layer}", "layer": layer,
                     **{f"seen_{c}": auc(detector, features, seen, c) for c in CONDITIONS},
                     **{f"unseen_{c}": auc(detector, features, unseen, c) for c in CONDITIONS}})
    layers = pd.DataFrame(rows)
    layers["unseen_mean"] = layers[[f"unseen_{c}" for c in CONDITIONS]].mean(axis=1)
    print("Ranking quality (AUC: 1.0 perfect, 0.5 coin flip) on dev")
    print(layers.drop(columns="layer").round(3).to_string(index=False))
    best = int(layers[layers["layer"].notna()].sort_values("unseen_mean").iloc[-1]["layer"])
    print(f"\nChosen XLS-R layer: {best}")

    # ---- 2. training recipes for the chosen layer and for the baseline
    results = {}
    for name, kind, layer, conditions in [
            ("mfcc, clean only", "mfcc", None, ("clean",)),
            ("mfcc, clean + phone", "mfcc", None, ("clean", "phone")),
            ("xlsr, clean only", "xlsr", best, ("clean",)),
            ("xlsr, clean + phone", "xlsr", best, ("clean", "phone"))]:
        detector = train(features, train_clips, kind, layer, conditions)
        set_threshold(detector, features, real_dev)
        table = report(detector, features, dev, ["label", "engine", "lang"])
        results[name] = table
        print(f"\n== {name}  (alarm level {detector.threshold:.3f})")
        wide = table.pivot_table(index=["label", "engine"], columns=["lang", "condition"],
                                 values="share", sort=False)
        print((100 * wide).round(0).to_string())

    RESULTS.mkdir(exist_ok=True)
    layers.to_csv(RESULTS / "dev_layers.csv", index=False)
    pd.concat([t.assign(recipe=n) for n, t in results.items()]).to_csv(RESULTS / "dev_recipes.csv", index=False)
    (RESULTS / "chosen.json").write_text(json.dumps({"layer": best}, indent=2))


if __name__ == "__main__":
    main()
