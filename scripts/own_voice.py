"""My own voice against an AI copy of my own voice.

1. Takes the recordings made with record.py (data/audio/own/).
2. Joins three of them per language into a short reference clip.
3. Asks Chatterbox to read all 30 sentences in a copy of that voice.
4. Adds both sets to data/clips.csv as test-only clips:
       real   source "own"
       fake   generator "chatterbox_clone"

Run:  python scripts/own_voice.py      (then scripts/features.py)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.audio import load, trim_silence
from clonecheck.config import AUDIO, DATA, MANIFEST, RATE, ROOT

OWN = AUDIO / "own"
CLONES = AUDIO / "fake" / "chatterbox_clone"
REFERENCE_SENTENCES = ["06", "08", "12"]   # everyday sentences, about 12 seconds together


def main() -> None:
    sentences = pd.read_csv(DATA / "own_sentences.csv")
    recorded = sentences[[(OWN / f"{i}.wav").exists() for i in sentences["id"]]]
    if recorded.empty:
        sys.exit("No recordings yet. Run:  python record.py")
    print(f"{len(recorded)} of {len(sentences)} sentences recorded")

    for lang in ["en", "ru"]:
        parts = [trim_silence(load(OWN / f"{lang}_{n}.wav")) for n in REFERENCE_SENTENCES
                 if (OWN / f"{lang}_{n}.wav").exists()]
        if parts:
            gap = np.zeros(RATE // 4, dtype=np.float32)
            joined = np.concatenate([x for part in parts for x in (part, gap)])
            sf.write(OWN / f"_reference_{lang}.wav", joined, RATE, subtype="PCM_16")

    subprocess.run([str(ROOT / ".venv-chatterbox" / "bin" / "python"),
                    str(ROOT / "scripts" / "generate_chatterbox.py"), "--clone", str(CLONES)],
                   check=True)

    rows = []
    for r in sentences.itertuples():
        for label, source, engine, folder in [("real", "own", "", OWN),
                                              ("fake", "generated", "chatterbox_clone", CLONES)]:
            path = folder / f"{r.id}.wav"
            if path.exists():
                rows.append({"clip_id": f"{'real_own' if label == 'real' else 'chatterbox_clone'}_{r.id}",
                             "path": str(path.relative_to(ROOT)), "label": label, "lang": r.lang,
                             "source": source, "engine": engine, "voice": "own", "split": "test",
                             "sentence_id": -1, "gender": "", "text": r.text, "text_source": "own"})
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    clips = pd.concat([clips[clips["text_source"] != "own"], pd.DataFrame(rows)], ignore_index=True)
    clips.to_csv(MANIFEST, index=False)
    print(f"added {len(rows)} own-voice clips to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
