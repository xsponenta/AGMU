#!/usr/bin/env bash
# Full SpeechT5 word-unlearning benchmark on LibriTTS.
#
# Three stages:
#   1. Build per-word splits from LibriTTS train-clean-100 text  (one-shot)
#   2. Run the per-word train + eval orchestration
#
# The orchestrator is idempotent: re-running skips stages whose output already
# exists. Use --skip-build-pairs / --skip-train to fast-iterate.

set -euo pipefail

DATA_DIR=${DATA_DIR:-benchmark/data}
OUT_DIR=${OUT_DIR:-benchmark/results}
ASR_MODEL=${ASR_MODEL:-openai/whisper-small}
NUM_CANDIDATES=${NUM_CANDIDATES:-4}
DPO_EPOCHS=${DPO_EPOCHS:-4}
GA_EPOCHS=${GA_EPOCHS:-3}

if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
    echo "=== Stage 1: build per-word splits from LibriTTS ==="
    python3 benchmark/build_splits.py \
        --words-file benchmark/forbidden_words.txt \
        --out-dir "$DATA_DIR"
else
    echo "=== Stage 1: $DATA_DIR already populated -- skipping build_splits ==="
fi

echo "=== Stage 2: train + eval all methods on all words -> $OUT_DIR ==="
python3 benchmark/run_benchmark.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --asr-model "$ASR_MODEL" \
    --num-candidates "$NUM_CANDIDATES" \
    --dpo-epochs "$DPO_EPOCHS" \
    --ga-epochs "$GA_EPOCHS"

echo
echo "Results:"
echo "  $OUT_DIR/results.csv         (per word x method x split)"
echo "  $OUT_DIR/results_mean.csv    (averaged over words; the headline benchmark table)"
