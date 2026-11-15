"""The cloning-tool experiment: does training on one cloning tool help against another?

Version 4 is version 3 plus 800 extra training clips in which Chatterbox
copies the voices of real training speakers. Nothing else changes: same
XLS-R layer, same training conditions, same rule for the alarm level.
The plan was written down before any of this was scored (results/chosen_v4.json).

What is scored:

  new tool, my voice        XTTS copies of my voice (never generated before)
  new tool, new speakers    XTTS copies of speakers from never-seen datasets
  known tool, my voice      Chatterbox copies of my voice. Chatterbox is now
                            a tool the detector trained on, but never my voice
  my real voice             both recording sessions: false alarms
  everything else           the earlier never-seen test clips, to check that
                            nothing got worse (a repeat look)

Run:  python scripts/evaluate_v4.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import ENGINE_ROLE, REAL_SOURCE_ROLE, RESULTS, TRAIN_SOURCES
from clonecheck.detect import load_clips, load_features, report, set_threshold, train

HEARD = ("clean", "phone", "noisy", "gated")
AUGMENTED = ("clean", "phone", "aug1", "aug2")
VERSIONS = {  # name: (train on the extra cloned clips?, training conditions, alarm level from)
    "version 2": (False, ("clean", "phone"), ("clean", "phone")),
    "version 3": (False, AUGMENTED, HEARD),
    "version 4": (True, AUGMENTED, HEARD),
    "version 2 + cloned training clips": (True, ("clean", "phone"), ("clean", "phone")),
}


def main() -> None:
    layer = json.loads((RESULTS / "chosen_v4.json").read_text())["layer"]
    features = load_features(layers=[layer], conditions=list(HEARD) + ["aug1", "aug2"])
    clips = load_clips()
    clips = clips.assign(who=[r.source if r.label == "real" else r.engine for r in clips.itertuples()])
    base = clips[clips["split"] == "train"]
    cloned = clips[clips["split"] == "train_clones"]
    cloned = cloned[cloned["clip_id"].isin(features["aug2"]["index"])]
    real_dev = clips[(clips["split"] == "dev") & (clips["label"] == "real")
                     & clips["source"].isin(TRAIN_SOURCES)]
    scored = clips[clips["clip_id"].isin(features["gated"]["index"])]
    new_generator = scored["engine"].map(lambda e: ENGINE_ROLE.get(e) == "test")
    new_dataset = scored["source"].map(lambda s: REAL_SOURCE_ROLE.get(s) == "test")
    sets = {
        "new tool (XTTS), my voice": scored[scored["text_source"].isin(["own_xtts", "own2_xtts"])],
        "new tool (XTTS), new speakers": scored[(scored["engine"] == "xtts_clone")
                                                & ~scored["text_source"].isin(["own_xtts", "own2_xtts"])],
        "known tool (Chatterbox), my voice": scored[scored["engine"].isin(["chatterbox_clone", "chatterbox_clone2"])],
        "known tool (Chatterbox), built-in voice": scored[scored["engine"] == "chatterbox"],
        "other new generators (Kokoro, Bark, Vosk)": scored[scored["engine"].isin(["kokoro", "bark", "vosk"])],
        "my real voice, laptop": scored[scored["source"] == "own"],
        "my real voice, phone": scored[scored["source"] == "own2"],
        "real people, new datasets": scored[(scored["label"] == "real") & new_dataset
                                            & ~scored["source"].isin(["own", "own2"])],
        "real people, known datasets": scored[(scored["label"] == "real") & (scored["split"] == "test")
                                              & scored["source"].isin(TRAIN_SOURCES)],
    }
    print(f"extra cloned training clips: {len(cloned)}")
    print({name: len(part) for name, part in sets.items()}, "\n")

    tables = []
    for name, (use_cloned, train_on, level_from) in VERSIONS.items():
        detector = train(features, pd.concat([base, cloned]) if use_cloned else base, "xlsr", layer, train_on)
        set_threshold(detector, features, real_dev, level_from)
        for set_name, part in sets.items():
            for lang_group, rows in [("all", part)] + [(g, r) for g, r in part.groupby("lang")]:
                table = report(detector, features, rows.assign(group=set_name), ["group"], HEARD)
                tables.append(table.assign(version=name, lang=lang_group, label=rows["label"].iloc[0],
                                           threshold=detector.threshold))
    results = pd.concat(tables)
    results.to_csv(RESULTS / "v4_results.csv", index=False)

    everything = results[results["lang"] == "all"]
    everything = everything.assign(cell=[f"{int(k)}/{int(n)}" for k, n in zip(everything["flagged"], everything["clips"])])
    wide = everything.pivot_table(index=["group", "version"], columns="condition", values="cell",
                                  aggfunc="first", sort=False)[list(HEARD)]
    print("Clips flagged as AI (for copies and generators: caught; for real voices: false alarms)")
    print(wide.to_string())


if __name__ == "__main__":
    main()
