"""Score everything on the held-out test split. Run once, after settings are frozen.

Settings (the XLS-R layer and the training recipe) are read from
results/chosen.json, which was written from dev data only.

Methods compared:
    always real              the simplest baseline: never raise the alarm
    mfcc                     logistic regression on classic audio features
    xlsr, one source         the first version (real voices from FLEURS only)
    xlsr, <chosen recipe>    CloneCheck

For every method the alarm level is set on real dev voices so that at most
5% of them are flagged, and is not changed afterwards.

Run:  python scripts/evaluate_test.py
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
from clonecheck.config import (CONDITIONS, ENGINE_ROLE, REAL_SOURCE_ROLE, RESULTS, ROOT,
                               TRAIN_SOURCES)
from clonecheck.detect import load_clips, load_features, report, set_threshold, train

RECIPES = {"one source": ("fleurs", ("clean",)), "many sources": ("all", ("clean",)),
           "many + phone": ("all", ("clean", "phone"))}


def main() -> None:
    chosen = json.loads((RESULTS / "chosen.json").read_text())
    layer, recipe = chosen["layer"], chosen["recipe"]
    features = load_features(layers=[layer])
    clips = load_clips()
    clips = clips[clips["clip_id"].isin(features["clean"]["index"])]
    clips = clips.assign(
        who=[r.source if r.label == "real" else r.engine for r in clips.itertuples()],
        seen=["seen in training" if (REAL_SOURCE_ROLE.get(r.source) if r.label == "real"
                                     else ENGINE_ROLE[r.engine]) == "train" else "never seen"
              for r in clips.itertuples()])
    train_clips = clips[clips["split"] == "train"]
    real_dev = clips[(clips["split"] == "dev") & (clips["label"] == "real")]
    test = clips[clips["split"] == "test"]
    print(f"train {len(train_clips)}, test {len(test)} clips "
          f"({(test.label == 'real').sum()} real, {(test.label == 'fake').sum()} fake)\n")

    def build(kind, which, conditions):
        used = train_clips[train_clips["text_source"] == "fleurs"] if which == "fleurs" else train_clips
        detector = train(features, used, kind, layer if kind == "xlsr" else None, conditions)
        level_from = (real_dev[real_dev["source"].isin(TRAIN_SOURCES)] if which == "all"
                      else real_dev[real_dev["source"] == "fleurs"])
        set_threshold(detector, features, level_from)
        return detector

    methods = {
        "always real": None,
        f"mfcc, {recipe}": build("mfcc", *RECIPES[recipe]),
        "xlsr, one source": build("xlsr", *RECIPES["one source"]),
        f"xlsr, {recipe}": build("xlsr", *RECIPES[recipe]),
    }

    tables, scores = [], test[["clip_id", "label", "who", "seen", "lang", "gender", "text_source"]].copy()
    for name, detector in methods.items():
        tables.append(report(detector, features, test, ["label", "seen", "who", "lang"]).assign(method=name))
        tables.append(report(detector, features, test, ["label", "seen"]).assign(method=name, who="ALL", lang="all"))
        if detector is not None:
            for c in CONDITIONS:
                scores[f"{name} | {c}"] = detector.scores(features, test, c)
        print(f"== {name}" + ("" if detector is None else f"  (alarm level {detector.threshold:.3f})")
              + " - % flagged as fake")
        wide = tables[-2].pivot_table(index=["label", "seen", "who"], columns=["lang", "condition"],
                                      values="share", sort=False)
        print((100 * wide).round(0).to_string(), "\n")

    results = pd.concat(tables)
    results.to_csv(RESULTS / "test_results.csv", index=False)
    scores.to_csv(RESULTS / "test_scores.csv", index=False)
    thresholds = {n: d.threshold for n, d in methods.items() if d is not None}
    (RESULTS / "test_thresholds.json").write_text(json.dumps(thresholds, indent=2))

    summary = results[results["who"] == "ALL"].pivot_table(
        index="method", columns=["label", "seen", "condition"], values="share", sort=False)
    print("SUMMARY - % flagged as fake (for fakes: caught; for real voices: false alarms)")
    print((100 * summary).round(0).to_string())

    # save the final detector so the app can use it
    final = methods[f"xlsr, {recipe}"]
    scaler, logistic = final.model.steps[0][1], final.model.steps[1][1]
    (ROOT / "models").mkdir(exist_ok=True)
    np.savez(ROOT / "models" / "detector.npz", mean=scaler.mean_.astype(np.float32),
             scale=scaler.scale_.astype(np.float32), weights=logistic.coef_[0].astype(np.float32),
             bias=np.float32(logistic.intercept_[0]),
             settings=json.dumps({"layer": layer, "threshold": final.threshold, "recipe": recipe}))


if __name__ == "__main__":
    main()
