"""Write data/clips.csv: the list of every clip in the experiment.

One row per clip: where the file is, whether it is real or fake, its
language, which generator made it and whether it belongs to train, dev or
test. Fake clips are listed here before they exist; scripts/generate.py
then creates them.

Run:  python scripts/make_manifest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import AUDIO, ENGINE_ROLE, LANGS, MANIFEST, RAW, ROOT, TRAIN_ENGINES
from clonecheck.voices import VOICES

MAX_CHARS = 200          # very long sentences are skipped for fake speech
BARK_MAX_CHARS = 130     # Bark can only generate about 13 seconds
BARK_SENTENCES = 100     # Bark is slow, so it gets a smaller sample


def read_fleurs(lang: str, split: str) -> pd.DataFrame:
    table = pd.read_csv(RAW / "fleurs" / LANGS[lang] / f"{split}.tsv", sep="\t",
                        header=None, quoting=3,
                        names=["sentence_id", "file", "text", "normalised", "chars",
                               "samples", "gender"])
    table["lang"] = lang
    table["path"] = [str((RAW / "fleurs" / LANGS[lang] / split / split / f).relative_to(ROOT))
                     for f in table["file"]]
    return table


def speakable(text: str, lang: str) -> bool:
    """Keep sentences every generator can read: no digits, not too long."""
    if any(ch.isdigit() for ch in text) or len(text) > MAX_CHARS:
        return False
    if lang == "ru" and any("a" <= ch.lower() <= "z" for ch in text):
        return False
    return True


def main() -> None:
    rows = []
    for lang in LANGS:
        # ---------------------------------------------------------- real speech
        train_real = read_fleurs(lang, "dev").assign(split="train")
        rest = read_fleurs(lang, "test")
        ids = sorted(rest["sentence_id"].unique())
        to_dev = set(ids[0::2])   # every other sentence goes to dev, the rest to test
        rest["split"] = ["dev" if s in to_dev else "test" for s in rest["sentence_id"]]
        real = pd.concat([train_real, rest])
        for r in real.itertuples():
            rows.append({"clip_id": f"real_{lang}_{Path(r.file).stem}", "path": r.path,
                         "label": "real", "lang": lang, "source": "fleurs", "engine": "",
                         "voice": "", "split": r.split, "sentence_id": r.sentence_id,
                         "gender": r.gender.lower(), "text": r.text})

        # ---------------------------------------------------------- fake speech
        sentences = real.drop_duplicates("sentence_id").sort_values("sentence_id")
        sentences = sentences[[speakable(t, lang) for t in sentences["text"]]]
        for split, group in sentences.groupby("split"):
            group = group.reset_index(drop=True)
            for engine, role in ENGINE_ROLE.items():
                voices = VOICES[engine][lang]
                if not voices:
                    continue
                if role == "train":
                    per_sentence = 2 if split == "train" else 1
                elif role == "dev":
                    per_sentence = 1 if split == "dev" else 0
                else:  # test generators are only ever made for test sentences
                    per_sentence = 1 if split == "test" else 0
                if per_sentence == 0:
                    continue
                chosen = group
                if engine == "bark":
                    chosen = group[group["text"].str.len() <= BARK_MAX_CHARS].head(BARK_SENTENCES)
                for i, r in enumerate(chosen.itertuples()):
                    for k in range(min(per_sentence, len(voices))):
                        voice = voices[(i + k) % len(voices)]
                        clip_id = f"{engine}_{lang}_{r.sentence_id}_{k}"
                        rows.append({"clip_id": clip_id,
                                     "path": str((AUDIO / "fake" / engine / lang / f"{clip_id}.wav").relative_to(ROOT)),
                                     "label": "fake", "lang": lang, "source": "generated",
                                     "engine": engine, "voice": voice, "split": split,
                                     "sentence_id": r.sentence_id, "gender": "", "text": r.text})

    clips = pd.DataFrame(rows)
    assert clips["clip_id"].is_unique
    # the same sentence must never appear in two splits
    assert clips.groupby(["lang", "sentence_id"])["split"].nunique().max() == 1
    # generators outside the training list must never appear in train
    assert set(clips[(clips.split == "train") & (clips.label == "fake")]["engine"]) <= set(TRAIN_ENGINES)
    clips.to_csv(MANIFEST, index=False)
    summary = clips.assign(engine=clips["engine"].replace("", "(real)")) \
        .pivot_table(index=["label", "engine"], columns=["lang", "split"], values="clip_id",
                     aggfunc="count", fill_value=0)
    print(summary.to_string())
    print(f"\n{len(clips)} clips listed in {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
