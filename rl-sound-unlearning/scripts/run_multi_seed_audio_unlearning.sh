#!/usr/bin/env bash
# Run repeated seeds for reporting mean/std metrics.
#
# Usage:
#   bash scripts/run_multi_seed_audio_unlearning.sh ac_rain 1 2 3
#   EPOCHS=100 bash scripts/run_multi_seed_audio_unlearning.sh ac_wind 1 2 3 4 5

set -euo pipefail

CONFIG=${1:-ac_rain}
shift || true

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(1 2 3)
fi

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-20}

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="logs/${CONFIG}/seed_${SEED}"
    echo "Running ${CONFIG}, seed=${SEED}, out=${OUT_DIR}"
    python3 train_audio_unlearning.py \
        --config "${CONFIG}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --critic-warmup-epochs "${WARMUP_EPOCHS}" \
        --seed "${SEED}" \
        --checkpoint-dir "${OUT_DIR}" \
        --save-samples
done
