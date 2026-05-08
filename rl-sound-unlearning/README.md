# RL Sound Unlearning

Audio concept unlearning via a stochastic edit-policy trained with REINFORCE.

The policy takes a *real* audio clip and emits a Gaussian residual (μ, σ) per
sample. The sampled edit is added to the input and clipped — so the output is
always a real track, not silence or noise. Reward is multi-term:

- **unlearn**: critic should not classify the edit as the target concept
- **realism**: RMS within a sane envelope (penalizes silence and clipping)
- **spec_entropy**: spectral entropy floor (penalizes DC / single tones)
- **anti_periodic**: low autocorrelation beyond ~10 ms (penalizes loops / "repeated word")
- **in_batch_div**: low cosine similarity across batch (penalizes mode collapse)
- **retain_cls**: non-target clips should keep their original critic label
- **retain_audio**: non-target clips should stay close to the input waveform

## Install

```bash
pip install -r requirements.txt
```

## Quick demo (data + training in one command)

```bash
bash scripts/run_audio_unlearning.sh ac_rain
# or ac_wind / ac_thunder
```

This synthesizes 8 clips per concept (Rain / Wind / Thunder) procedurally,
then runs 30 epochs of REINFORCE on the edit policy. Sample edits are written
to `logs/ac_<concept>/sample_epoch_*.wav` alongside `input_reference.wav`.

## Manual

```bash
# 1. synthesize tiny dataset
python data_synth.py --root data --per-concept 8

# 2. train
python train_audio_unlearning.py --config ac_rain --save-samples
```

## Evaluation

Training writes `metrics.csv` into each checkpoint directory. For review-style
reporting, run at least three seeds and report mean/std for:

- target removal: `clean_target_only_prob` → `edited_target_only_prob`
- retention: `clean_non_target_acc` → `edited_non_target_acc`
- edit size: `edit_rms`

```bash
bash scripts/run_multi_seed_audio_unlearning.sh ac_rain 1 2 3
```

The objective includes retention rewards for non-target clips, so the policy is
not rewarded only for aggressively suppressing the target concept.

## Files

- `data_synth.py` — procedural synth for Rain/Wind/Thunder (no external data)
- `audio_dataset.py` — dataset loader for labeled audio files
- `audio_critic.py` — critic network predicting concept presence
- `audio_generator.py` — `AudioEditPolicy`: stochastic edit policy (μ, log σ)
- `audio_rewards.py` — multi-term reward (unlearn + realism + anti-collapse)
- `train_audio_unlearning.py` — REINFORCE training loop with EMA baseline
- `audio_utils.py` — helpers for audio I/O
