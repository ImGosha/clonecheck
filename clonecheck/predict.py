"""Use the finished detector on one new recording.

    from clonecheck.predict import check
    result = check("call.wav")
    print(result["verdict"], result["probability"])

The detector itself is tiny (a logistic regression saved in
models/detector.npz). The heavy part is XLS-R, the pretrained speech model
that turns the audio into numbers; it is downloaded the first time.
"""
from __future__ import annotations

import json

import numpy as np

from .audio import condition, load, prepare
from .config import RATE, ROOT

MODEL_FILE = ROOT / "models" / "detector.npz"
_detector = None


def _load_detector() -> dict:
    global _detector
    if _detector is None:
        saved = np.load(MODEL_FILE, allow_pickle=False)
        _detector = {"mean": saved["mean"], "scale": saved["scale"], "weights": saved["weights"],
                     "bias": float(saved["bias"]), **json.loads(str(saved["settings"]))}
    return _detector


def probability_fake(wave: np.ndarray) -> float:
    """Probability (0 to 1) that a prepared 16 kHz clip is an AI-generated voice."""
    from .features import xlsr_features
    detector = _load_detector()
    x = xlsr_features(wave)[detector["layer"]].astype(np.float32)
    z = ((x - detector["mean"]) / detector["scale"]) @ detector["weights"] + detector["bias"]
    return float(1 / (1 + np.exp(-z)))


def check(path, as_phone_call: bool = False) -> dict:
    """Score one audio file. Set as_phone_call=True to squeeze it through the phone-line simulation first."""
    detector = _load_detector()
    wave = prepare(load(path))
    if len(wave) < RATE:
        return {"verdict": "too short", "probability": None, "threshold": detector["threshold"],
                "seconds": len(wave) / RATE}
    if as_phone_call:
        wave = condition(wave, "phone")
    p = probability_fake(wave)
    return {"verdict": "ai" if p > detector["threshold"] else "real", "probability": p,
            "threshold": detector["threshold"], "seconds": len(wave) / RATE}
