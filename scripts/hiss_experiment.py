"""Does background hiss change what the detector says about my own voice?

Run after the test. It takes my 30 real recordings and the 30 AI copies,
adds steady background hiss at two loudness levels, and scores each version
with the final detector (models/detector.npz). Results go to
results/own_voice_hiss.csv.

Run:  python scripts/hiss_experiment.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.audio import add_noise, load, phone_line, prepare
from clonecheck.config import RESULTS, ROOT
from clonecheck.detect import load_clips
from clonecheck.predict import _load_detector, probability_fake


def main() -> None:
    level = _load_detector()["threshold"]
    clips = load_clips()
    rng = np.random.default_rng(0)
    versions = {
        "as recorded": lambda w: w,
        "hiss 30 dB below the voice": lambda w: add_noise(w, 30, rng),
        "hiss 20 dB below the voice": lambda w: add_noise(w, 20, rng),
        "phone line": phone_line,
        "hiss 20 dB below the voice, then phone line": lambda w: phone_line(add_noise(w, 20, rng)),
    }
    rows = []
    for name, group in {"my real voice": clips[clips["source"] == "own"],
                        "AI copy of my voice": clips[clips["engine"] == "chatterbox_clone"]}.items():
        waves = [(r.lang, prepare(load(ROOT / r.path))) for r in group.itertuples()]
        for version, change in versions.items():
            scores = np.array([probability_fake(change(w)) for _, w in waves])
            langs = np.array([lang for lang, _ in waves])
            rows.append({"clips": name, "version": version,
                         "flagged, English (of 15)": int((scores[langs == "en"] > level).sum()),
                         "flagged, Russian (of 15)": int((scores[langs == "ru"] > level).sum()),
                         "flagged, all (of 30)": int((scores > level).sum()),
                         "median score": round(float(np.median(scores)), 3)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "own_voice_hiss.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
