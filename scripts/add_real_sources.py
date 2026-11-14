"""Add real speech from other sources, and fake speech reading the same texts.

Why: in the first dev run every real voice the detector trained on came from
one dataset (FLEURS). It then flagged most real voices from anywhere else as
fake: it had learned "sounds like a FLEURS recording", not "sounds human".

So real voices now come from many sources, and the sources are split the
same way as the voice generators (clonecheck/config.py, REAL_SOURCE_ROLE):

    train   LibriSpeech, VoxPopuli (English); Golos, RuLibriSpeech (Russian)
    dev     English Dialects; SOVA RuDevices      (never trained on)
    test    MInDS-14, AMI, Podlodka               (never seen until the end)

For every source, AI voice generators also read out the same texts, so real
and fake clips say the same kind of thing.

Run after scripts/make_manifest.py:  python scripts/add_real_sources.py
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
from clonecheck.config import AUDIO, ENGINE_ROLE, MANIFEST, RATE, RAW, ROOT, SEED
from clonecheck.voices import VOICES, to_16k

MIN_SECONDS, MAX_SECONDS = 2.0, 20.0
PARTS = [  # source, language, files, our split, clips to take, most clips per speaker
    # the first four were fixed before the first dev run
    ("librispeech", "en", ["librispeech_validation"], "dev", 300, 8),
    ("librispeech", "en", ["librispeech_test"], "test", 300, 8),
    ("golos", "ru", ["golos_validation"], "dev", 300, 8),
    ("golos", "ru", ["golos_test"], "test", 300, 8),
    # added after the first dev run
    ("librispeech", "en", ["librispeech_train"], "train", 300, 25),
    ("voxpopuli", "en", ["voxpopuli_train"], "train", 300, 8),
    ("voxpopuli", "en", ["voxpopuli_validation"], "dev", 150, 8),
    ("voxpopuli", "en", ["voxpopuli_test"], "test", 200, 8),
    ("golos", "ru", ["golos_train"], "train", 300, 8),
    ("rulibrispeech", "ru", ["rulibrispeech_train"], "train", 300, 25),
    ("rulibrispeech", "ru", ["rulibrispeech_validation"], "dev", 150, 25),
    ("rulibrispeech", "ru", ["rulibrispeech_test"], "test", 200, 25),
    ("dialects", "en", ["dialects_irish_male", "dialects_midlands_female",
                        "dialects_midlands_male", "dialects_welsh_female"], "dev", 200, 25),
    ("sova", "ru", ["sova_validation"], "dev", 200, 8),
    ("minds", "en", ["minds_en"], "test", 200, 8),
    ("minds", "ru", ["minds_ru"], "test", 200, 8),
    ("ami", "en", ["ami_test"], "test", 200, 25),
    ("podlodka", "ru", ["podlodka_test", "podlodka_validation", "podlodka_train"], "test", 200, 8),
]
TEXT_COLUMNS = ["text", "transcription", "raw_text", "normalized_text", "sentence"]

# how many texts from each source part the generators read out
FAKE_TEXTS = {"train": 120, "dev": 60, "test": 60}
SLOW_ENGINES = {"bark", "chatterbox"}       # only read the phone-banking phrases (MInDS-14)
SLOW_TEXTS = 30


def tidy(text: str) -> str:
    """Sentence case with a full stop, so every generator reads it naturally."""
    text = " ".join(str(text).split()).lower()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!" else text + "."


def speakable(text: str, lang: str) -> bool:
    if not 15 <= len(text) <= 200 or any(ch.isdigit() for ch in text):
        return False
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if lang == "en":
        return all("a" <= ch <= "z" for ch in letters)
    return all("а" <= ch <= "я" or ch == "ё" for ch in letters)


def real_rows() -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for source, lang, files, split, wanted, speaker_cap in PARTS:
        table = pd.concat([pq.read_table(RAW / "other" / f"{f}.parquet").to_pandas() for f in files],
                          ignore_index=True)
        text_column = next(c for c in TEXT_COLUMNS if c in table.columns)
        order = rng.permutation(len(table))
        per_speaker: dict = {}
        kept = 0
        for i in order:
            r = table.iloc[i]
            speaker = r.get("speaker_id", None)
            if speaker is not None and per_speaker.get(speaker, 0) >= speaker_cap:
                continue  # do not let a few speakers dominate
            clip_id = f"real_{lang}_{source}_{split}_{kept:03d}"
            out = AUDIO / "real_other" / source / split / f"{clip_id}.wav"
            try:
                wave, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            except Exception:
                continue
            seconds = len(wave) / rate
            if not MIN_SECONDS <= seconds <= MAX_SECONDS:
                continue
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, to_16k(wave, rate), RATE, subtype="PCM_16")
            rows.append({"clip_id": clip_id, "path": str(out.relative_to(ROOT)), "label": "real",
                         "lang": lang, "source": source, "engine": "", "voice": "",
                         "split": split, "sentence_id": -1, "gender": str(r.get("gender", "") or ""),
                         "text": str(r[text_column]), "text_source": source})
            per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
            kept += 1
            if kept == wanted:
                break
        print(f"{source:14s} {lang} {split:5s}: kept {kept} of {len(table)} clips", flush=True)
    return rows


def fake_rows(real: pd.DataFrame) -> list[dict]:
    """Fake clips that read the same texts as the real clips of each source part."""
    rows = []
    for (source, lang, split), group in real.groupby(["source", "lang", "split"], sort=False):
        texts = [t for t in dict.fromkeys(tidy(t) for t in group["text"]) if speakable(t, lang)]
        texts = texts[:FAKE_TEXTS[split]]
        for engine, role in ENGINE_ROLE.items():
            voices = VOICES[engine][lang]
            if not voices:
                continue
            wanted = (role == "train") or (role == split)
            if engine in SLOW_ENGINES:
                wanted = split == "test" and source == "minds"
            if not wanted:
                continue
            chosen = [t for t in texts if len(t) <= 130][:SLOW_TEXTS] if engine in SLOW_ENGINES else texts
            for i, text in enumerate(chosen):
                clip_id = f"{engine}_{lang}_{source}_{split}_{i:03d}"
                rows.append({"clip_id": clip_id,
                             "path": str((AUDIO / "fake" / engine / lang / f"{clip_id}.wav").relative_to(ROOT)),
                             "label": "fake", "lang": lang, "source": "generated", "engine": engine,
                             "voice": voices[i % len(voices)], "split": split, "sentence_id": -1,
                             "gender": "", "text": text, "text_source": source})
    return rows


def main() -> None:
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    clips = clips[clips["text_source"] == "fleurs"]          # start again from the FLEURS rows
    real = pd.DataFrame(real_rows())
    fake = pd.DataFrame(fake_rows(real))
    clips = pd.concat([clips, real, fake], ignore_index=True)
    assert clips["clip_id"].is_unique
    clips.to_csv(MANIFEST, index=False)
    print("\nReal clips")
    print(clips[clips.label == "real"].pivot_table(index="source", columns=["lang", "split"],
          values="clip_id", aggfunc="count", fill_value=0).to_string())
    print("\nFake clips")
    print(clips[clips.label == "fake"].pivot_table(index="engine", columns=["lang", "split"],
          values="clip_id", aggfunc="count", fill_value=0).to_string())
    print(f"\n{len(clips)} clips in total")


if __name__ == "__main__":
    main()
