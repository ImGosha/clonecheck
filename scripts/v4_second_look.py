"""A second look at the cloning-tool experiment, added AFTER its results were seen.

Version 4 catches many more copies from the unseen tool (XTTS) than version 3,
but it also flags more real people. So is it a better detector, or did the bar
for raising the alarm simply get lower? Two checks that do not depend on the
alarm level:

  AUC            how well the scores rank fakes above real voices
                 (1.0 perfect, 0.5 coin flip)
  matched catch  copies caught when each version's alarm level is moved so that
                 it flags exactly 5% of the real voices in the comparison

This was not in the plan (results/chosen_v4.json), and the matched level is set
on test clips, so it is weaker evidence than the planned comparison.

Run:  python scripts/v4_second_look.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import REAL_SOURCE_ROLE, RESULTS
from clonecheck.detect import load_clips, load_features, train

HEARD = ("clean", "phone", "noisy", "gated")
AUGMENTED = ("clean", "phone", "aug1", "aug2")
OWN_COPIES = ["own_xtts", "own2_xtts"]


def main() -> None:
    layer = json.loads((RESULTS / "chosen_v4.json").read_text())["layer"]
    features = load_features(layers=[layer], conditions=list(HEARD) + ["aug1", "aug2"])
    clips = load_clips()
    base = clips[clips["split"] == "train"]
    cloned = clips[clips["split"] == "train_clones"]
    scored = clips[clips["clip_id"].isin(features["gated"]["index"])]
    new_dataset = scored["source"].map(lambda s: REAL_SOURCE_ROLE.get(s) == "test")
    pairs = {  # name: (copies from the unseen tool, the real voices to tell them from)
        "new speakers": (scored[(scored["engine"] == "xtts_clone") & ~scored["text_source"].isin(OWN_COPIES)],
                         scored[(scored["label"] == "real") & new_dataset & ~scored["source"].isin(["own", "own2"])]),
        "my voice": (scored[scored["text_source"].isin(OWN_COPIES)],
                     scored[scored["source"].isin(["own", "own2"])]),
    }
    detectors = {"version 3": train(features, base, "xlsr", layer, AUGMENTED),
                 "version 4": train(features, pd.concat([base, cloned]), "xlsr", layer, AUGMENTED)}
    rows = []
    for pair, (fake, real) in pairs.items():
        for version, detector in detectors.items():
            for heard in HEARD:
                f, r = detector.scores(features, fake, heard), detector.scores(features, real, heard)
                level = np.quantile(r, 0.95)
                rows.append({"pair": pair, "version": version, "condition": heard,
                             "copies": len(f), "real": len(r),
                             "auc": roc_auc_score([1] * len(f) + [0] * len(r), np.concatenate([f, r])),
                             "caught_at_5pct_false_alarms": int((f > level).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "v4_second_look.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
