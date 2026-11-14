"""A second, fresh recording of my voice, made on a different device.

The first 30 recordings (made in a browser on my laptop) showed a failure,
and the fix was designed with them in view. They cannot also be used to
test the fix. So this script takes a new recording made another way: one
voice memo on my phone, reading the same 30 sentences.

The memo is one long file with retakes and no regular pauses, so the script
finds the sentences by listening:

1. Whisper (a speech-to-text model, run on this computer) writes down every
   word in the memo with the time it was said.
2. For each of the 30 sentences, the stretch of words that matches it best is
   found. If a sentence was said twice, the last good take is used.
3. Each stretch is cut out as its own clip.

Then a new AI copy of the voice is made from this recording, and both sets
are added to data/clips.csv as test-only clips:

    real   source "own2"
    fake   generator "chatterbox_clone2"

Put the memo (any common audio format) in data/audio/own2_raw/ and run:

    python scripts/own_voice_session2.py     (then scripts/features.py)
"""
from __future__ import annotations

import difflib
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.audio import load, trim_silence
from clonecheck.config import AUDIO, DATA, MANIFEST, RATE, RESULTS, ROOT

RAW = AUDIO / "own2_raw"
OWN = AUDIO / "own2"
CLONES = AUDIO / "fake" / "chatterbox_clone2"
REFERENCE_SENTENCES = ["06", "08", "12"]   # everyday sentences, about 12 seconds together
AUDIO_TYPES = {".wav", ".m4a", ".mp3", ".aac", ".caf", ".mp4", ".mov", ".ogg", ".flac", ".webm"}


def read_any(path: Path) -> np.ndarray:
    """Read any audio file as 16 kHz mono (ffmpeg handles phone formats such as m4a)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = f"{tmp}/a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
                        "-ar", str(RATE), "-ac", "1", wav], check=True)
        return load(wav)


def plain(text: str) -> str:
    """Lower case, no punctuation: so "Okay?" and "okay" count as the same word."""
    return re.sub(r"[^a-zа-я0-9 ]", "", text.lower().replace("ё", "е")).strip()


def pieces_between_pauses(wave: np.ndarray) -> list[tuple[int, int]]:
    """Cut the memo wherever there is half a second of silence."""
    frame = RATE // 50  # 20 ms
    n = len(wave) // frame
    level = 10 * np.log10((wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    quiet = level < np.percentile(level, 95) - 35
    cuts, start = [0], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= 25:
                cuts.append(((start + i) // 2) * frame)
            start = None
    cuts.append(len(wave))
    pieces = list(zip(cuts[:-1], cuts[1:]))
    return [(a, b) for a, b in pieces if len(trim_silence(wave[a:b])) > RATE // 2]


def timed_words(wave: np.ndarray, pieces: list[tuple[int, int]]) -> list[dict]:
    """Every word in the memo with its language, start time and end time (in seconds)."""
    import torch
    from transformers import pipeline
    listen = pipeline("automatic-speech-recognition", model="openai/whisper-small",
                      device="mps" if torch.backends.mps.is_available() else "cpu",
                      model_kwargs={"cache_dir": ROOT / "models_cache"})
    words = []
    for a, b in pieces:
        clip = {"raw": wave[a:b], "sampling_rate": RATE}
        letters = plain(listen(dict(clip))["text"]).replace(" ", "")
        if len(letters) < 6:
            continue
        russian = sum("а" <= ch <= "я" for ch in letters) / len(letters)
        english = sum("a" <= ch <= "z" for ch in letters) / len(letters)
        if max(russian, english) < 0.7:
            continue                              # a click or a breath, not speech
        lang = "ru" if russian > english else "en"
        heard = listen(dict(clip), return_timestamps="word",
                       generate_kwargs={"language": "russian" if lang == "ru" else "english",
                                        "task": "transcribe"})
        for chunk in heard["chunks"]:
            start, end = chunk["timestamp"]
            if start is not None and end is not None and plain(chunk["text"]):
                words.append({"word": plain(chunk["text"]), "lang": lang, "piece_end": b / RATE,
                              "start": a / RATE + start, "end": a / RATE + end})
    return words


def find_sentences(words: list[dict], sentences: pd.DataFrame) -> pd.DataFrame:
    """For each sentence, the stretch of words in the memo that matches it best."""
    rows = []
    for r in sentences.itertuples():
        target = plain(r.text)
        length = len(target.split())
        options = []                              # (how well it matches, first word, number of words)
        for i, first in enumerate(words):
            if first["lang"] != r.lang:
                continue
            for n in range(max(2, length - 4), length + 3):
                stretch = words[i:i + n]
                if len(stretch) == n and all(w["lang"] == r.lang for w in stretch):
                    said = " ".join(w["word"] for w in stretch)
                    options.append((difflib.SequenceMatcher(None, target, said).ratio(), i, n))
        match, i, n = max(options)
        # a retake: a stretch that starts over a second later and matches about as well
        retakes = [o for o in options if o[0] >= match - 0.03
                   and words[o[1]]["start"] > words[i]["start"] + 1.0]
        if retakes:
            last = max(words[o[1]]["start"] for o in retakes)
            match, i, n = max(o for o in retakes if words[o[1]]["start"] > last - 1.0)
        stretch = words[i:i + n]
        rows.append({"id": r.id, "lang": r.lang, "match": round(match, 2),
                     "start": stretch[0]["start"], "end": stretch[-1]["end"],
                     "piece_end": stretch[-1]["piece_end"],
                     "heard": " ".join(w["word"] for w in stretch)})
    return pd.DataFrame(rows)


def main() -> None:
    sentences = pd.read_csv(DATA / "own_sentences.csv")
    files = sorted(f for f in RAW.glob("*") if f.suffix.lower() in AUDIO_TYPES)
    if not files:
        sys.exit(f"Put a recording in {RAW.relative_to(ROOT)}/ first.")
    wave = np.concatenate([read_any(f) for f in files])
    print(f"memo: {len(wave) / RATE:.0f} seconds")
    found = find_sentences(timed_words(wave, pieces_between_pauses(wave)), sentences)
    print(found[["id", "match", "start", "end", "heard"]].round(1).to_string(index=False))
    if found["match"].min() < 0.7:
        sys.exit("Some sentences could not be found in the memo. Re-record those and run again.")

    OWN.mkdir(parents=True, exist_ok=True)
    order = found.sort_values("start").reset_index(drop=True)
    for k, r in order.iterrows():
        # a clip runs from just before its first word to just before the next sentence,
        # but never past the pause that ended the piece it was said in
        next_start = order["start"][k + 1] if k + 1 < len(order) else len(wave) / RATE
        end = min(next_start - 0.05, r["piece_end"])
        clip = wave[int(max(r["start"] - 0.1, 0) * RATE): int(end * RATE)]
        sf.write(OWN / f"{r['id']}.wav", trim_silence(clip), RATE, subtype="PCM_16")
    found.drop(columns="piece_end").to_csv(RESULTS / "own2_alignment.csv", index=False)

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
