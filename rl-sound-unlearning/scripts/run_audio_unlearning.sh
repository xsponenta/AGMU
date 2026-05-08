#!/usr/bin/env bash
# Audio unlearning demo: synthesize a tiny dataset, then run REINFORCE training.
#
# Usage:
#   bash scripts/run_audio_unlearning.sh ac_rain
#   bash scripts/run_audio_unlearning.sh ac_wind
#   bash scripts/run_audio_unlearning.sh ac_thunder
#
# Skip data synthesis: SKIP_SYNTH=1 bash scripts/run_audio_unlearning.sh ac_rain

set -euo pipefail

CONFIG=${1:-ac_rain}

if [[ "${SKIP_SYNTH:-0}" != "1" ]]; then
    echo "Synthesizing dataset (8 clips per concept)..."
    python3 data_synth.py --root data --per-concept 8
fi

echo "Training with config: $CONFIG"
python3 train_audio_unlearning.py --config "$CONFIG" --save-samples
