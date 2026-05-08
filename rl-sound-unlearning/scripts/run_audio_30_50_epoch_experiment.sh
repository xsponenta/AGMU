#!/usr/bin/env bash
# Convenience launcher for a 30-50 epoch prototype run.
#
# Usage:
#   bash scripts/run_audio_30_50_epoch_experiment.sh ac_rain 50
#   bash scripts/run_audio_30_50_epoch_experiment.sh ac_wind 30

set -euo pipefail

CONFIG=${1:-ac_rain}
EPOCHS=${2:-50}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-20}
OUT_DIR=${OUT_DIR:-"logs/${CONFIG}_${EPOCHS}ep_seed_${SEED}"}

python3 train_audio_unlearning.py \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --critic-warmup-epochs "${WARMUP_EPOCHS}" \
    --seed "${SEED}" \
    --checkpoint-dir "${OUT_DIR}" \
    --save-samples

echo "Run complete: ${OUT_DIR}"
echo "Check audio samples: ${OUT_DIR}/sample_epoch_*.wav"
echo "Check losses/metrics: ${OUT_DIR}/train_metrics.csv and ${OUT_DIR}/metrics.csv"
