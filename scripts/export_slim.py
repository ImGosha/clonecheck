"""Save a small copy of the features that fits on GitHub.

The full feature files hold all 25 XLS-R layers for every clip (about 1.5 GB).
Only one layer is used in the end, so this keeps the 80 MFCC numbers and the
chosen XLS-R layer for each clip and listening condition, in data/features_slim/. The notebook and
scripts/evaluate_test.py run from these files alone, with no audio needed.

Run:  python scripts/export_slim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clonecheck.config import FEATURES, FEATURES_SLIM, RESULTS
from clonecheck.detect import load_clips


def main() -> None:
    layer = json.loads((RESULTS / "chosen.json").read_text())["layer"]
    wanted = set(load_clips()["clip_id"])
    FEATURES_SLIM.mkdir(parents=True, exist_ok=True)
    for name in ["clean", "phone", "noisy", "gated", "aug1", "aug2"]:
        ids, mfcc, xlsr = [], [], []
        for file in sorted(FEATURES.glob(f"{name}.*.npz")) + sorted(FEATURES.glob(f"{name}.npz")):
            saved = np.load(file, allow_pickle=False)
            keep = np.array([c in wanted for c in saved["ids"]])
            ids.extend(saved["ids"][keep])
            mfcc.append(saved["mfcc"][keep])
            xlsr.append(saved["xlsr"][keep][:, layer, :])
        out = FEATURES_SLIM / f"{name}.npz"
        np.savez_compressed(out, ids=np.array(ids), mfcc=np.concatenate(mfcc).astype(np.float32),
                            xlsr=np.concatenate(xlsr).astype(np.float16), layer=layer)
        print(f"{out.name}: {len(ids)} clips, {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
