"""Add real speech from two other sources to data/clips.csv.

All the real voices used for training come from one dataset (FLEURS). A
detector could therefore learn "sounds like a FLEURS recording" instead of
"sounds like a human". To check that, this script adds real recordings made
by other people on other microphones:

    LibriSpeech (English)   volunteers reading audiobooks
    Golos crowd (Russian)   people speaking short phrases into their own phones

Their "test" parts go to our test split and are never trained on. Their
"validation" parts go to our dev split.

Run:  python scripts/add_real_sources.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import AUDIO, MANIFEST, RATE, RAW, ROOT, SEED
from clonecheck.voices import to_16k

PER_PART = 300                 # clips taken from each part of each source
MIN_SECONDS, MAX_SECONDS = 2.0, 20.0
PARTS = [  # source, language, file, our split, name of the text column
    ("librispeech", "en", "librispeech_validation.parquet", "dev", "text"),
    ("librispeech", "en", "librispeech_test.parquet", "test", "text"),
    ("golos", "ru", "golos_validation.parquet", "dev", "transcription"),
    ("golos", "ru", "golos_test.parquet", "test", "transcription"),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for source, lang, file, split, text_column in PARTS:
        table = pq.read_table(RAW / "other" / file).to_pandas()
        order = rng.permutation(len(table))
        per_speaker: dict = {}
        kept = 0
        for i in order:
            r = table.iloc[i]
            speaker = r.get("speaker_id", None)
            if speaker is not None and per_speaker.get(speaker, 0) >= 8:
                continue  # do not let a few audiobook readers dominate
            wave, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            seconds = len(wave) / rate
            if not MIN_SECONDS <= seconds <= MAX_SECONDS:
                continue
            clip_id = f"real_{lang}_{source}_{split}_{kept:03d}"
            out = AUDIO / "real_other" / source / split / f"{clip_id}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, to_16k(wave, rate), RATE, subtype="PCM_16")
            rows.append({"clip_id": clip_id, "path": str(out.relative_to(ROOT)), "label": "real",
                         "lang": lang, "source": source, "engine": "", "voice": "",
                         "split": split, "sentence_id": -1, "gender": "",
                         "text": str(r[text_column])})
            per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
            kept += 1
            if kept == PER_PART:
                break
        print(f"{source} {split}: kept {kept} clips")

    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    clips = clips[~clips["source"].isin({p[0] for p in PARTS})]
    clips = pd.concat([clips, pd.DataFrame(rows)], ignore_index=True)
    assert clips["clip_id"].is_unique
    clips.to_csv(MANIFEST, index=False)
    print(clips.pivot_table(index=["label", "source"], columns=["lang", "split"],
                            values="clip_id", aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()
