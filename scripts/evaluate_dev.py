"""Choose settings on dev data. Nothing here touches the test split.

Three training recipes are compared:

    one source       real voices from FLEURS only (the first version)
    many sources     real voices from every training source
    many + phone     the same, with each training clip also heard through
                     the simulated phone line

Each is scored on dev clips: fakes from the training generators and from
Silero (never trained on), real voices from the training sources (new
speakers) and from two sources never trained on (Dialects, SOVA).

The XLS-R layer is chosen here too, by how well it separates real dev
voices from Silero.

Run:  python scripts/evaluate_dev.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import CONDITIONS, RESULTS, TRAIN_SOURCES
from clonecheck.detect import auc, load_clips, load_features, report, set_threshold, train

RECIPES = {  # name: (which training clips, which listening conditions)
    "one source": ("fleurs", ("clean",)),
    "many sources": ("all", ("clean",)),
    "many + phone": ("all", ("clean", "phone")),
}


def recipe_clips(train_clips: pd.DataFrame, which: str) -> pd.DataFrame:
    return train_clips[train_clips["text_source"] == "fleurs"] if which == "fleurs" else train_clips


def show(table: pd.DataFrame) -> None:
    wide = table.pivot_table(index=["label", "who"], columns=["lang", "condition"],
                             values="share", sort=False)
    print((100 * wide).round(0).to_string())


def main() -> None:
    features = load_features()
    clips = load_clips()
    clips = clips[clips["clip_id"].isin(features["clean"]["index"])]
    clips = clips.assign(who=[r.source if r.label == "real" else r.engine for r in clips.itertuples()])
    train_clips = clips[clips["split"] == "train"]
    dev = clips[clips["split"] == "dev"]
    real_dev = dev[dev["label"] == "real"]
    real_dev_seen = real_dev[real_dev["source"].isin(TRAIN_SOURCES)]
    silero = dev[dev["engine"] == "silero"]
    print(f"train {len(train_clips)} clips, dev {len(dev)} clips\n")

    # ---- 1. which XLS-R layer? (trained on every source, clean audio)
    rows = []
    hard = pd.concat([real_dev, silero])
    for layer in range(25):
        detector = train(features, train_clips, "xlsr", layer)
        rows.append({"layer": layer, **{c: auc(detector, features, hard, c) for c in CONDITIONS}})
    layers = pd.DataFrame(rows)
    layers["mean"] = layers[CONDITIONS].mean(axis=1)
    print("XLS-R layers: real dev voices against Silero (AUC: 1.0 perfect, 0.5 coin flip)")
    print(layers.round(3).to_string(index=False))
    best = int(layers.sort_values("mean").iloc[-1]["layer"])
    print(f"\nChosen layer: {best}\n")

    # ---- 2. training recipes, for the baseline features and for XLS-R
    tables = []
    for kind, layer in [("mfcc", None), ("xlsr", best)]:
        for name, (which, conditions) in RECIPES.items():
            detector = train(features, recipe_clips(train_clips, which), kind, layer, conditions)
            # the alarm level comes from real dev voices of the sources this recipe trained on
            level_from = real_dev_seen if which == "all" else real_dev[real_dev["source"] == "fleurs"]
            set_threshold(detector, features, level_from)
            table = report(detector, features, dev, ["label", "who", "lang"])
            print(f"== {kind}, {name}  (alarm level {detector.threshold:.3f}) - % flagged as fake")
            show(table)
            print()
            tables.append(table.assign(features=kind, recipe=name, threshold=detector.threshold))

    RESULTS.mkdir(exist_ok=True)
    layers.to_csv(RESULTS / "dev_layers.csv", index=False)
    pd.concat(tables).to_csv(RESULTS / "dev_recipes.csv", index=False)
    (RESULTS / "chosen.json").write_text(json.dumps({"layer": best}, indent=2))


if __name__ == "__main__":
    main()
