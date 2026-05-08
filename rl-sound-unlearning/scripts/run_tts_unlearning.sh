#!/usr/bin/env bash
# End-to-end TTS word unlearning demo with SpeechT5 + LoRA + rejection-sampling DPO.
#
# Usage:
#   bash scripts/run_tts_unlearning.sh
# Optional env knobs:
#   K=4               candidates per prompt during rejection sampling
#   ASR_MODEL=...     Whisper variant (default openai/whisper-small)
#   PAIRS_DIR=...     where to put the pair-build artifacts
#   OUT_DIR=...       where to put adapters + eval

set -euo pipefail

K=${K:-4}
ASR_MODEL=${ASR_MODEL:-openai/whisper-small}
PAIRS_DIR=${PAIRS_DIR:-dpo_pairs/run01}
OUT_DIR=${OUT_DIR:-logs/tts_dpo}

echo "=== Stage 1: rejection sampling -> $PAIRS_DIR ==="
python3 scripts/build_dpo_pairs.py \
    --prompts prompts/speech_word_train_prompts.jsonl \
    --out "$PAIRS_DIR" \
    --num-candidates "$K" \
    --asr-model "$ASR_MODEL" \
    --include-retain

echo "=== Stage 2: DPO training -> $OUT_DIR ==="
python3 train_tts_dpo_unlearning.py \
    --config tts_dpo \
    --pairs-dir "$PAIRS_DIR" \
    --out-dir "$OUT_DIR"

LATEST_ADAPTER=$(ls -d "$OUT_DIR"/adapter_epoch_* | sort -V | tail -1)

echo "=== Stage 3: evaluation -> $OUT_DIR/eval ==="
python3 scripts/evaluate_tts_unlearning.py \
    --adapter "$LATEST_ADAPTER" \
    --prompts prompts/speech_word_eval_prompts.jsonl \
    --speaker-pool "$PAIRS_DIR/speaker_pool.pt" \
    --asr-model "$ASR_MODEL" \
    --out "$OUT_DIR/eval"

echo "Done. Inspect:"
echo "  - $PAIRS_DIR/pairs.jsonl   (chosen/rejected with transcripts)"
echo "  - $OUT_DIR/train_metrics.json"
echo "  - $OUT_DIR/eval/summary.csv"
