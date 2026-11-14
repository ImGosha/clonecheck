"""Score version 3 after its settings were frozen (results/chosen_v3.json).

Two things are scored, and they are not equally trustworthy:

  fresh        my second recording session (a phone voice memo) and a new AI
               copy made from it. Nothing was chosen with these in view.
  second look  the held-out test set again, now under four listening
               conditions. Version 2 was already scored on it once and the
               own-voice failure seen there led to version 3, so this is no
               longer a first look. My first recordings are left out of it:
               they are dev data now.

Also saves version 3 as the detector the app uses (models/detector.npz).

Run:  python scripts/evaluate_v3.py
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
from clonecheck.config import ENGINE_ROLE, REAL_SOURCE_ROLE, RESULTS, ROOT, TRAIN_SOURCES
from clonecheck.detect import load_clips, load_features, report, set_threshold, train

HEARD = ("clean", "phone", "noisy", "gated")


def save_detector(detector, path: Path, layer: int, name: str) -> None:
    scaler, logistic = detector.model.steps[0][1], detector.model.steps[1][1]
    np.savez(path, mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             weights=logistic.coef_[0].astype(np.float32), bias=np.float32(logistic.intercept_[0]),
             settings=json.dumps({"layer": layer, "threshold": detector.threshold, "recipe": name}))


def main() -> None:
    chosen = json.loads((RESULTS / "chosen_v3.json").read_text())
    layer = chosen["layer"]
    features = load_features(layers=[layer], conditions=list(HEARD) + ["aug1", "aug2"])
    clips = load_clips()
    clips = clips[clips["clip_id"].isin(features["noisy"]["index"]) | (clips["split"] == "train")]
    clips = clips.assign(
        who=[r.source if r.label == "real" else r.engine for r in clips.itertuples()],
        seen=["seen in training" if (REAL_SOURCE_ROLE.get(r.source) if r.label == "real"
                                     else ENGINE_ROLE[r.engine]) == "train" else "never seen"
              for r in clips.itertuples()])
    train_clips = clips[clips["split"] == "train"]
    real_dev = clips[(clips["split"] == "dev") & (clips["label"] == "real")
                     & clips["source"].isin(TRAIN_SOURCES)]
    fresh = clips[clips["text_source"] == "own2"]
    second_look = clips[(clips["split"] == "test") & ~clips["text_source"].isin(["own", "own2"])]

    detectors = {}
    for name, train_on, level_from in [("version 2", ("clean", "phone"), ("clean", "phone")),
                                       ("version 3", tuple(chosen["train_on"]),
                                        tuple(chosen["alarm_level_from"]))]:
        detectors[name] = train(features, train_clips, "xlsr", layer, train_on)
        set_threshold(detectors[name], features, real_dev, level_from)

    tables = []
    for name, detector in detectors.items():
        tables.append(report(detector, features, fresh, ["label", "who", "lang"], HEARD)
                      .assign(version=name, data="fresh"))
        tables.append(report(detector, features, fresh, ["label", "who"], HEARD)
                      .assign(version=name, data="fresh", lang="all"))
        tables.append(report(detector, features, second_look, ["label", "seen", "who", "lang"], HEARD)
                      .assign(version=name, data="second look"))
        tables.append(report(detector, features, second_look, ["label", "seen"], HEARD)
                      .assign(version=name, data="second look", who="ALL", lang="all"))
    results = pd.concat(tables)
    results.to_csv(RESULTS / "v3_results.csv", index=False)

    scores = fresh[["clip_id", "label", "who", "lang"]].copy()
    for name, detector in detectors.items():
        for c in HEARD:
            scores[f"{name} | {c}"] = detector.scores(features, fresh, c)
    scores.to_csv(RESULTS / "v3_fresh_scores.csv", index=False)
    (RESULTS / "v3_thresholds.json").write_text(
        json.dumps({n: d.threshold for n, d in detectors.items()}, indent=2))

    def show(table, index):
        wide = table.pivot_table(index=index, columns=["lang", "condition"], values="flagged",
                                 aggfunc="first", sort=False)
        total = table.pivot_table(index=index, columns=["lang", "condition"], values="clips",
                                  aggfunc="first", sort=False)
        print((wide.astype(int).astype(str) + "/" + total.astype(int).astype(str)).to_string(), "\n")

    print("FRESH: my second recording (phone memo) and a new AI copy of it. Clips flagged as AI.")
    part = results[(results["data"] == "fresh") & (results["lang"] != "all")]
    show(part, ["version", "label", "who"])
    print("SECOND LOOK at the test set: % flagged as AI")
    part = results[(results["data"] == "second look") & (results["who"] == "ALL")]
    wide = part.pivot_table(index=["version", "label", "seen"], columns="condition", values="share", sort=False)
    print((100 * wide).round(0).to_string(), "\n")
    part = results[(results["data"] == "second look") & (results["who"] != "ALL") & (results["version"] == "version 3")]
    wide = part.pivot_table(index=["label", "seen", "who"], columns=["lang", "condition"], values="share", sort=False)
    print("version 3, by generator and dataset")
    print((100 * wide).round(0).to_string())

    (ROOT / "models").mkdir(exist_ok=True)
    save_detector(detectors["version 2"], ROOT / "models" / "detector_v2.npz", layer, "version 2")
    save_detector(detectors["version 3"], ROOT / "models" / "detector.npz", layer, "version 3")


if __name__ == "__main__":
    main()
