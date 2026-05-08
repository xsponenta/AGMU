# Audio Concept Unlearning With Retention

## Goal

The goal of this project is to train an audio model/edit-policy to remove or suppress one target sound concept, such as `Rain`, `Wind`, or `Thunder`, while keeping the model useful for all other sounds.

This is important: a model that stops generating the target concept but also damages every other generation is not a good unlearning model. Good unlearning must balance two objectives:

- **Unlearning**: the target concept should become unlikely after editing.
- **Retention**: non-target concepts should still be preserved and recognizable.

For example, if we unlearn `Rain`, edited rain clips should stop looking/sounding like rain to the critic. But wind and thunder clips should still stay wind and thunder. The method should not simply destroy the audio, silence it, or add noise everywhere.

## What The Code Does

The current implementation is a prototype for concept removal from short environmental audio clips.

The pipeline is:

1. Load labeled audio examples from `data/manifest.json`.
2. Train an `AudioCritic` classifier to recognize the available concepts.
3. Train an `AudioEditPolicy` with REINFORCE.
4. The policy receives a real audio clip and a target concept condition.
5. It predicts a stochastic residual edit.
6. The edit is added to the original waveform.
7. A reward function encourages the edit to remove the target concept while preserving non-target clips.

The policy is not currently a full text-to-audio generator such as AudioLDM or MusicGen. It is an edit-policy over real input clips. That makes this project useful as a small research prototype, but for a stronger paper-level system the same ideas should later be connected to a real pretrained audio generation model.

## Reward Terms

The reward has two groups of terms.

### Unlearning and Quality

- `unlearn`: edited audio should not be classified as the target concept.
- `realism`: edited audio should not become silence or clipping.
- `spec_entropy`: audio should keep a reasonable spectrum and not collapse to a tone.
- `anti_periodic`: audio should avoid obvious repeated/looped artifacts.
- `in_batch_div`: different samples in the batch should not collapse to the same output.

### Retention

- `retain_cls`: non-target clips should keep their original class.
- `retain_audio`: non-target clips should stay close to the original waveform.

The retention terms are very important. They directly address the main reviewer concern from related work: high unlearning accuracy is not enough if in-domain retain accuracy drops too much.

## How To Run A Small Training Example

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the tiny synthetic dataset:

```bash
python3 data_synth.py --root data --per-concept 8
```

Train rain unlearning:

```bash
python3 train_audio_unlearning.py --config ac_rain --save-samples
```

Train wind unlearning:

```bash
python3 train_audio_unlearning.py --config ac_wind --save-samples
```

Train thunder unlearning:

```bash
python3 train_audio_unlearning.py --config ac_thunder --save-samples
```

Outputs are saved to:

```text
logs/ac_rain/
logs/ac_wind/
logs/ac_thunder/
```

Each output folder contains:

- `input_reference.wav`: the fixed input clip used for sample comparison.
- `sample_epoch_*.wav`: edited audio samples over training.
- `audio_unlearning_epoch_*.pt`: model checkpoints.
- `metrics.csv`: evaluation metrics over training.

## How To Run On Only A Few Epochs

For a quick smoke test:

```bash
python3 train_audio_unlearning.py \
  --config ac_rain \
  --epochs 3 \
  --batch-size 2 \
  --critic-warmup-epochs 2 \
  --checkpoint-dir logs/debug_rain \
  --save-samples
```

This is only for checking that the code runs. It is not enough for real results.

## How Many Epochs To Train

For the current tiny synthetic dataset:

- Use `20` critic warmup epochs.
- Use `30-100` RL policy epochs.
- Start with `30` epochs for debugging.
- Use `100` epochs for a more serious prototype run.

Example:

```bash
python3 train_audio_unlearning.py \
  --config ac_rain \
  --epochs 100 \
  --critic-warmup-epochs 20 \
  --save-samples
```

For real datasets such as ESC-50 or FSD50K:

- Start with `20-50` critic warmup epochs.
- Train the policy for `50-200` epochs.
- Do not choose the final checkpoint only by epoch number.
- Choose the checkpoint using both unlearning and retention metrics.

The best checkpoint is not necessarily the last checkpoint. Stop when target removal improves but retention starts to degrade.

## Metrics To Watch

The most important metrics in `metrics.csv` are:

- `clean_target_only_prob`: how much the critic sees the target concept before editing.
- `edited_target_only_prob`: how much the critic sees the target concept after editing.
- `clean_non_target_acc`: classification accuracy on non-target clips before editing.
- `edited_non_target_acc`: classification accuracy on non-target clips after editing.
- `edit_rms`: how large the waveform edit is.

Good behavior:

```text
edited_target_only_prob goes down
edited_non_target_acc stays close to clean_non_target_acc
edit_rms stays moderate
```

Bad behavior:

```text
edited_target_only_prob goes down
edited_non_target_acc also drops a lot
edit_rms becomes very large
samples sound like silence, clipping, or noise
```

In other words, the model should learn not to generate or preserve the unwanted target concept, but it should not lose much ability on other generations.

## Multiple Seeds And Error Bars

For paper-style experiments, run at least three random seeds:

```bash
bash scripts/run_multi_seed_audio_unlearning.sh ac_rain 1 2 3
```

For stronger results:

```bash
EPOCHS=100 bash scripts/run_multi_seed_audio_unlearning.sh ac_rain 1 2 3 4 5
```

Repeat this for each concept:

```bash
EPOCHS=100 bash scripts/run_multi_seed_audio_unlearning.sh ac_wind 1 2 3
EPOCHS=100 bash scripts/run_multi_seed_audio_unlearning.sh ac_thunder 1 2 3
```

Report the mean and standard deviation of the final metrics. This helps answer reviewer concerns about statistical rigor.

## Recommended Benchmarks

The current procedural dataset is useful for debugging, but it is too small for a serious experimental claim.

Recommended next datasets:

- **ESC-50**: small environmental audio benchmark, good for early experiments.
- **FSD50K**: larger open sound-event dataset, better for serious evaluation.
- **AudioSet**: very large sound-event benchmark, strong but more difficult to use.

For this project, ESC-50 is the best next step because it is small and contains environmental categories. FSD50K is the next step after the method works reliably.

## Experimental Principle

Always report unlearning and retention together.

Do not only say:

```text
The model forgot rain.
```

Say:

```text
The model reduced rain probability from X to Y, while non-target retention changed from A to B.
```

This makes the trade-off visible. A strong result should show that the target concept is suppressed while the model still performs well on other audio concepts.

