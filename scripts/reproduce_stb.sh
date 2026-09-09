#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/STB.annotations}"
DEVICE="${DEVICE:-cpu}"

hlrepair-train \
  --train-annotations "${DATA_ROOT}/STB_train.json" \
  --test-annotations "${DATA_ROOT}/STB_test.json" \
  --device "${DEVICE}" \
  --output outputs/hl_repair_seed2801.pt \
  --report outputs/train_seed2801.json

hlrepair-eval \
  --annotations "${DATA_ROOT}/STB_test.json" \
  --checkpoint outputs/hl_repair_seed2801.pt \
  --device "${DEVICE}" \
  --output outputs/evaluation_seed2801.json

python scripts/verify_evaluation.py outputs/evaluation_seed2801.json
