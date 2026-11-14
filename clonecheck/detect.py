"""Training and scoring the detectors.

A detector here is deliberately simple: a logistic regression that looks at
a clip's features and outputs the probability that the voice is AI-generated.
What changes between detectors is only which features it looks at and which
clips it is trained on.

    always_real   says "real" every time (the simplest possible baseline)
    mfcc          logistic regression on 80 classic audio numbers
    xlsr          logistic regression on one layer of the XLS-R speech model
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONDITIONS, FEATURES, FEATURES_SLIM, MANIFEST

MAX_FALSE_ALARMS = 0.05   # the alarm level is set so at most 5% of real dev clips are flagged


def load_clips() -> pd.DataFrame:
    return pd.read_csv(MANIFEST, keep_default_na=False)


def load_features(layers: list[int] | None = None, conditions: list[str] | None = None) -> dict:
    """{condition: {"index", "mfcc", "xlsr"}} for the clips that have been processed.

    The full features (all 25 XLS-R layers) are stored in several files, one
    per processing run; this joins them. Pass `layers` to keep only some
    layers and save memory. If the full files are not there (a fresh copy of
    the repository), the slim copy with only the chosen layer is used.
    """
    out = {}
    for name in conditions or CONDITIONS:
        files = sorted(FEATURES.glob(f"{name}.*.npz")) + sorted(FEATURES.glob(f"{name}.npz"))
        if not files:
            saved = np.load(FEATURES_SLIM / f"{name}.npz", allow_pickle=False)
            out[name] = {"index": {c: i for i, c in enumerate(saved["ids"])},
                         "mfcc": saved["mfcc"].astype(np.float32),
                         "xlsr": saved["xlsr"][:, None, :], "layers": [int(saved["layer"])]}
            continue
        ids, mfcc, xlsr = [], [], []
        for file in files:
            saved = np.load(file, allow_pickle=False)
            ids.extend(saved["ids"])
            mfcc.append(saved["mfcc"])
            xlsr.append(saved["xlsr"] if layers is None else saved["xlsr"][:, layers, :])
        out[name] = {"index": {c: i for i, c in enumerate(ids)}, "mfcc": np.concatenate(mfcc),
                     "xlsr": np.concatenate(xlsr), "layers": layers}
    return out


def matrix(features: dict, clips: pd.DataFrame, condition: str, kind: str, layer: int | None = None):
    """Feature matrix for the given clips, in order."""
    store = features[condition]
    rows = [store["index"][c] for c in clips["clip_id"]]
    data = store[kind][rows]
    if kind == "xlsr":
        kept = store["layers"]
        data = data[:, layer if kept is None else kept.index(layer), :]
    return data.astype(np.float32)


@dataclass
class Detector:
    kind: str                 # "mfcc" or "xlsr"
    layer: int | None         # which XLS-R layer (None for mfcc)
    model: object
    threshold: float = 0.5

    def scores(self, features: dict, clips: pd.DataFrame, condition: str) -> np.ndarray:
        """Probability that each clip is fake."""
        return self.model.predict_proba(matrix(features, clips, condition, self.kind, self.layer))[:, 1]


def train(features: dict, clips: pd.DataFrame, kind: str, layer: int | None = None,
          conditions: tuple[str, ...] = ("clean",)) -> Detector:
    """Fit a logistic regression on the given clips, heard under the given conditions."""
    X = np.concatenate([matrix(features, clips, c, kind, layer) for c in conditions])
    y = np.tile((clips["label"] == "fake").to_numpy().astype(int), len(conditions))
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
    model.fit(X, y)
    return Detector(kind, layer, model)


def set_threshold(detector: Detector, features: dict, real_dev: pd.DataFrame,
                  conditions: tuple[str, ...] = ("clean", "phone")) -> float:
    """Choose the alarm level from real dev clips only: flag at most 5% of them."""
    scores = np.concatenate([detector.scores(features, real_dev, c) for c in conditions])
    detector.threshold = float(np.quantile(scores, 1 - MAX_FALSE_ALARMS))
    return detector.threshold


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a share k out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def report(detector: Detector | None, features: dict, clips: pd.DataFrame,
           group_by: list[str], conditions: tuple[str, ...] = ("clean", "phone")) -> pd.DataFrame:
    """Share of clips flagged as fake in each group.

    For fake clips that share is "caught"; for real clips it is "false alarms".
    Pass detector=None for the always-real baseline.
    """
    rows = []
    for name in conditions:
        scores = np.zeros(len(clips)) if detector is None else detector.scores(features, clips, name)
        flagged = scores > (0.5 if detector is None else detector.threshold)
        table = clips.assign(flagged=flagged, condition=name)
        for keys, group in table.groupby(group_by + ["condition"], sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            k, n = int(group["flagged"].sum()), len(group)
            low, high = wilson(k, n)
            rows.append({**dict(zip(group_by + ["condition"], keys)), "clips": n,
                         "flagged": k, "share": k / n, "low": low, "high": high})
    return pd.DataFrame(rows)


def auc(detector: Detector, features: dict, clips: pd.DataFrame, condition: str) -> float:
    """Ranking quality that does not depend on the alarm level (1.0 perfect, 0.5 chance)."""
    return float(roc_auc_score((clips["label"] == "fake").astype(int),
                               detector.scores(features, clips, condition)))
