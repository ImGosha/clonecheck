"""Shared settings: where files live and how the data is split.

The split is the most important design decision in the project, so it is
written down here, in one place, before any model is trained.

Real speech (FLEURS, people reading sentences aloud):
    FLEURS "dev" recordings            -> train
    FLEURS "test" recordings           -> half dev, half test (split by sentence)

Fake speech (the same kind of sentences, spoken by AI voice generators):
    say, piper, mms                    -> the generators the detector learns from
    silero                             -> never trained on; used to choose settings (dev)
    every other generator              -> never trained on and never used to choose
                                          anything; only scored at the end (test)

Real speech from other sources is added by scripts/add_real_sources.py.
Real sources are split the same way as generators (see REAL_SOURCE_ROLE):
some the detector learns from, some only for choosing settings, and some
that are never seen until the final test.

A sentence is never shared between train, dev and test, so the detector
cannot pass by remembering words.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
AUDIO = DATA / "audio"
FEATURES = DATA / "features"
RESULTS = ROOT / "results"
MANIFEST = DATA / "clips.csv"

RATE = 16_000
MAX_SECONDS = 10
SEED = 2026

LANGS = {"en": "en_us", "ru": "ru_ru"}

# role of each voice generator in the experiment
ENGINE_ROLE = {
    "say": "train", "piper": "train", "mms": "train",
    "silero": "dev",
    "kokoro": "test", "bark": "test", "chatterbox": "test", "vosk": "test",
}
TRAIN_ENGINES = [e for e, r in ENGINE_ROLE.items() if r == "train"]

# role of each source of real speech
#   train: its train part is used for training, its dev part for choosing
#          settings and its test part for the final test (different speakers)
#   dev:   never trained on; only used to choose settings
#   test:  never trained on and never used to choose anything
REAL_SOURCE_ROLE = {
    "fleurs": "train", "librispeech": "train", "voxpopuli": "train",   # English (FLEURS: both)
    "golos": "train", "rulibrispeech": "train",                        # Russian
    "dialects": "dev", "sova": "dev",
    "minds": "test", "ami": "test", "podlodka": "test", "own": "test",
}
TRAIN_SOURCES = [s for s, r in REAL_SOURCE_ROLE.items() if r == "train"]

# listening conditions every clip is tested under
CONDITIONS = ["clean", "phone"]
