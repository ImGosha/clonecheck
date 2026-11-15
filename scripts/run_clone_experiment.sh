#!/bin/bash
# The whole cloning-tool experiment, start to finish. Each step skips work already done.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
echo "start $(date)"
.venv/bin/python -u scripts/generate_xtts.py > logs/v4_xtts.log 2>&1
ONLY=chatterbox_voices_en .venv-chatterbox/bin/python -u scripts/generate_chatterbox.py > logs/v4_chatterbox_en.log 2>&1 &
ONLY=chatterbox_voices_ru .venv-chatterbox/bin/python -u scripts/generate_chatterbox.py > logs/v4_chatterbox_ru.log 2>&1 &
wait
CONDITIONS=clean,phone,noisy,gated SPLITS=test_clones .venv/bin/python -u scripts/features.py > logs/v4_features_test.log 2>&1
CONDITIONS=clean,phone,aug1,aug2 SPLITS=train_clones .venv/bin/python -u scripts/features.py > logs/v4_features_train.log 2>&1
.venv/bin/python scripts/evaluate_v4.py > logs/v4_run.log 2>&1
echo "done $(date)"
echo CLONE-EXPERIMENT-DONE
