#!/usr/bin/env bash
# Smoke test for the improved GA/DPO/RL pipeline on ONE word.
# Use this to validate the changes before launching the full 10-word benchmark.
#
# Total wall-clock target: ~30-90 min on a single GPU.
#
# Usage:
#   bash scripts/smoke_test_improved.sh [word]            # default: love
set -euo pipefail

WORD=${1:-love}
DATA_DIR=${DATA_DIR:-benchmark/data}
OUT_DIR=${OUT_DIR:-benchmark/results_smoke}
ASR_MODEL=${ASR_MODEL:-openai/whisper-small}

if [[ ! -d "$DATA_DIR/$WORD" ]]; then
    echo "Run benchmark/build_splits.py first to populate $DATA_DIR/$WORD/"
    exit 1
fi

echo "=== Smoke test: word=$WORD, out=$OUT_DIR ==="
python3 benchmark/run_benchmark.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --asr-model "$ASR_MODEL" \
    --only-words "$WORD" \
    --only-methods reference rewrite ga dpo rl \
    --num-candidates 4 \
    --dpo-epochs 8 \
    --ga-epochs 6 \
    --rl-epochs 4 \
    --eval-speakers 2

echo
echo "=== Smoke results ==="
column -s, -t "$OUT_DIR/results.csv"
