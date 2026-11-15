"""The cloning-tool experiment: add its clips to data/clips.csv.

Question: the detector trained only on ordinary text-to-speech generators.
If it also trains on one voice-cloning tool, can it catch copies made by a
different cloning tool that it has never heard?

    train_clones   Chatterbox copying the voices of real speakers from the
                   TRAINING datasets (each reads a different training text)
    test_clones    XTTS, a cloning tool never trained on, copying
                     - my own voice (from both of my recording sessions)
                     - real speakers from the never-seen TEST datasets

These clips get their own split names, so every earlier result still
re-runs exactly as before.

Run:  python scripts/add_clone_experiment.py
then  python scripts/generate_chatterbox.py  (in the Chatterbox environment)
and   python scripts/generate_xtts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_real_sources import speakable, tidy
from clonecheck.config import AUDIO, DATA, MANIFEST, REAL_SOURCE_ROLE, ROOT, SEED

TRAIN_PER_LANGUAGE = 400
TEST_PER_LANGUAGE = 100
MIN_SECONDS = 4.0       # a voice needs a few seconds of audio to be copied


def long_enough(path: str) -> bool:
    info = sf.info(str(ROOT / path))
    return MIN_SECONDS <= info.duration <= 20


def pairs(real: pd.DataFrame, lang: str, wanted: int, rng) -> list[tuple]:
    """(voice to copy, text to read): the text always comes from a different clip."""
    voices = real[[long_enough(p) for p in real["path"]]]
    voices = voices.iloc[rng.permutation(len(voices))]
    texts = [t for t in dict.fromkeys(tidy(t) for t in real["text"])
             if speakable(t, lang) and 20 <= len(t) <= 130]
    texts = [texts[i] for i in rng.permutation(len(texts))]
    out = []
    for i in range(wanted):
        voice = voices.iloc[i % len(voices)]
        text = texts[i % len(texts)]
        if text == tidy(voice["text"]):
            text = texts[(i + 1) % len(texts)]
        out.append((voice, text))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED + 4)
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    clips = clips[~clips["split"].isin(["train_clones", "test_clones"])]
    real = clips[(clips["label"] == "real")]
    rows = []

    def add(clip_id, engine, split, lang, voice_path, text, text_source):
        rows.append({"clip_id": clip_id, "label": "fake", "lang": lang, "source": "generated",
                     "path": str((AUDIO / "fake" / engine / lang / f"{clip_id}.wav").relative_to(ROOT)),
                     "engine": engine, "voice": voice_path, "split": split, "sentence_id": -1,
                     "gender": "", "text": text, "text_source": text_source})

    for lang in ["en", "ru"]:
        train_real = real[(real["split"] == "train") & (real["lang"] == lang)]
        for i, (voice, text) in enumerate(pairs(train_real, lang, TRAIN_PER_LANGUAGE, rng)):
            add(f"chatterbox_voices_{lang}_{i:04d}", "chatterbox_voices", "train_clones", lang,
                voice["path"], text, voice["source"])

        never_seen = np.array([REAL_SOURCE_ROLE[s] == "test" and not s.startswith("own")
                               for s in real["source"]])
        new_real = real[(real["split"] == "test") & (real["lang"] == lang) & never_seen]
        for i, (voice, text) in enumerate(pairs(new_real, lang, TEST_PER_LANGUAGE, rng)):
            add(f"xtts_clone_{lang}_{i:03d}", "xtts_clone", "test_clones", lang,
                voice["path"], text, voice["source"])

    sentences = pd.read_csv(DATA / "own_sentences.csv")
    for session in ["own", "own2"]:
        for r in sentences.itertuples():
            reference = (AUDIO / session / f"_reference_{r.lang}.wav").relative_to(ROOT)
            add(f"xtts_clone_{session}_{r.id}", "xtts_clone", "test_clones", r.lang,
                str(reference), r.text, f"{session}_xtts")

    new = pd.DataFrame(rows)
    clips = pd.concat([clips, new], ignore_index=True)
    assert clips["clip_id"].is_unique
    clips.to_csv(MANIFEST, index=False)
    print(new.groupby(["engine", "split", "lang", "text_source"]).size().to_string())


if __name__ == "__main__":
    main()
