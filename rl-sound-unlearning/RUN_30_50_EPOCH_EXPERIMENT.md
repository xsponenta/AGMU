# 30-50 Epoch Experiment

## Prompt Sets

For speech word-level unlearning, the repo now contains prompt sets:

```text
prompts/speech_word_train_prompts.jsonl
prompts/speech_word_eval_prompts.jsonl
```

Prompt types include:

- direct forbidden-word requests
- mixed content, such as `man say hello how are u`
- repetition prompts, such as `hello hello hello`
- music/song contexts
- paraphrases
- retain speech prompts without the forbidden word
- general non-speech audio prompts

These prompts keep the original text unchanged. The forbidden word remains in
the prompt so the model can learn that it should not realize that word in audio.

## Current Prototype Training Command

For the current audio edit-policy prototype, run:

```bash
bash scripts/run_audio_30_50_epoch_experiment.sh ac_rain 50
```

Or for a shorter run:

```bash
bash scripts/run_audio_30_50_epoch_experiment.sh ac_rain 30
```

Outputs are written to:

```text
logs/ac_rain_50ep_seed_42/
```

Important files:

```text
input_reference.wav
sample_epoch_*.wav
audio_unlearning_epoch_*.pt
metrics.csv
train_metrics.csv
```

Use `sample_epoch_*.wav` to manually listen during or after training.

## What Is Saved

`train_metrics.csv` saves every epoch:

- critic loss
- policy loss
- total reward
- reward components
- retention reward
- target removal metrics
- non-target retention metrics
- edit size

`metrics.csv` saves periodic evaluation metrics.

## For True Text-to-Audio Word Unlearning

The next full training step is to connect the ASR reward to a speech-capable
text-to-audio or text-to-speech model using LoRA/RL fine-tuning.

The intended training behavior is:

```text
Prompt: man say hello how are u
Generated transcript before: hello how are you
Generated transcript after: how are you
```

The prompt is not rewritten. The model learns from ASR-based reward that saying
`hello` in generated audio is bad.

Use these prompt files for that next stage:

```text
prompts/speech_word_train_prompts.jsonl
prompts/speech_word_eval_prompts.jsonl
```

