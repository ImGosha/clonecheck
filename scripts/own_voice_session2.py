"""A second, fresh recording of my voice, made on a different device.

The first 30 recordings (made in a browser on my laptop) showed a failure,
and the fix was designed with them in view. They cannot also be used to
test the fix. So this script takes a new recording made another way (for
example one voice memo on a phone, reading all 30 sentences with a pause
after each), cuts it into sentences at the pauses, makes a new AI copy of
the voice from it, and adds both to data/clips.csv as test-only clips:

    real   source "own2"
    fake   generator "chatterbox_clone2"

Put the recording (any common audio format; one long file, one file per
language, or one file per sentence) in data/audio/own2_raw/ and run:

    python scripts/own_voice_session2.py     (then scripts/features.py)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.audio import load, trim_silence
from clonecheck.config import AUDIO, DATA, MANIFEST, RATE, ROOT

RAW = AUDIO / "own2_raw"
OWN = AUDIO / "own2"
CLONES = AUDIO / "fake" / "chatterbox_clone2"
REFERENCE_SENTENCES = ["06", "08", "12"]


def read_any(path: Path) -> np.ndarray:
    """Read any audio file as 16 kHz mono (ffmpeg handles phone formats such as m4a)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = f"{tmp}/a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
                        "-ar", str(RATE), "-ac", "1", wav], check=True)
        return load(wav)


def split_at_pauses(wave: np.ndarray, sentences: int) -> list[np.ndarray]:
    """Cut one long recording into sentences at its longest pauses.

    We know how many sentences were read, so the cuts go in the middle of the
    (sentences - 1) longest silences. A short pause inside a sentence is then
    never mistaken for the gap between two sentences.
    """
    wave = trim_silence(wave)
    frame = RATE // 50  # 20 ms
    n = len(wave) // frame
    level = 10 * np.log10((wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    quiet = level < np.percentile(level, 95) - 35
    runs, start = [], None                      # (length, middle frame) of every silence
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((i - start, (start + i) // 2))
            start = None
    cuts = sorted(middle * frame for _, middle in sorted(runs, reverse=True)[: sentences - 1])
    pieces = [trim_silence(p) for p in np.split(wave, cuts)]
    return [p for p in pieces if len(p) > RATE // 2]


def main() -> None:
    sentences = pd.read_csv(DATA / "own_sentences.csv")
    files = sorted(f for f in RAW.glob("*") if f.suffix.lower() in
                   {".wav", ".m4a", ".mp3", ".aac", ".caf", ".mp4", ".mov", ".ogg", ".flac", ".webm"})
    if not files:
        sys.exit(f"Put a recording in {RAW.relative_to(ROOT)}/ first.")
    pieces = []
    for f in files:
        wave = read_any(f)
        pieces += (split_at_pauses(wave, len(sentences) // len(files)) if len(files) < len(sentences)
                   else [trim_silence(wave)])
    print(f"{len(files)} file(s), {len(pieces)} sentences found (expected {len(sentences)})")
    if len(pieces) != len(sentences):
        sys.exit("The number of sentences does not match. Re-record with a clear 2-second pause "
                 "after every sentence, or put one file per sentence in the folder.")
    OWN.mkdir(parents=True, exist_ok=True)
    for sentence_id, piece in zip(sentences["id"], pieces):
        sf.write(OWN / f"{sentence_id}.wav", piece, RATE, subtype="PCM_16")

    for lang in ["en", "ru"]:
        gap = np.zeros(RATE // 4, dtype=np.float32)
        parts = [load(OWN / f"{lang}_{n}.wav") for n in REFERENCE_SENTENCES]
        sf.write(OWN / f"_reference_{lang}.wav",
                 np.concatenate([x for part in parts for x in (part, gap)]), RATE, subtype="PCM_16")
    subprocess.run([str(ROOT / ".venv-chatterbox" / "bin" / "python"),
                    str(ROOT / "scripts" / "generate_chatterbox.py"), "--clone", str(CLONES), str(OWN)],
                   check=True)

    rows = []
    for r in sentences.itertuples():
        for label, source, engine, folder in [("real", "own2", "", OWN),
                                              ("fake", "generated", "chatterbox_clone2", CLONES)]:
            path = folder / f"{r.id}.wav"
            if path.exists():
                rows.append({"clip_id": f"{'real_own2' if label == 'real' else 'chatterbox_clone2'}_{r.id}",
                             "path": str(path.relative_to(ROOT)), "label": label, "lang": r.lang,
                             "source": source, "engine": engine, "voice": "own2", "split": "test",
                             "sentence_id": -1, "gender": "", "text": r.text, "text_source": "own2"})
    clips = pd.read_csv(MANIFEST, keep_default_na=False)
    clips = pd.concat([clips[clips["text_source"] != "own2"], pd.DataFrame(rows)], ignore_index=True)
    clips.to_csv(MANIFEST, index=False)
    print(f"added {len(rows)} session-2 clips to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
