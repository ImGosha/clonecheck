#!/bin/bash
# Finishes the cloning-tool experiment after the first run lost a generator process:
# makes the missing training clips (one process, restarted if it dies), then
# extracts features for all cloned training clips and scores again.
cd "$(dirname "$0")/.."
missing() { .venv/bin/python -c "
import pandas as pd, os
c = pd.read_csv('data/clips.csv', low_memory=False)
print(int((~c[c.split=='train_clones'].path.map(os.path.exists)).sum()))"; }
for attempt in 1 2 3 4 5 6 7 8; do
  left=$(missing); echo "attempt $attempt: $left clips missing $(date)"
  [ "$left" = "0" ] && break
  ONLY=chatterbox_voices .venv-chatterbox/bin/python -u scripts/generate_chatterbox.py >> logs/v4_chatterbox_finish.log 2>&1
done
echo "missing after generation: $(missing)"
CONDITIONS=clean,phone,aug1,aug2 SPLITS=train_clones .venv/bin/python -u scripts/features.py > logs/v4_features_train2.log 2>&1
.venv/bin/python scripts/evaluate_v4.py > logs/v4_run2.log 2>&1
echo "CLONE-EXPERIMENT-FINISHED $(date)"
