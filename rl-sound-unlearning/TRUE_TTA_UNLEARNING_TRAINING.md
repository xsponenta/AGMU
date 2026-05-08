# True Text-to-Audio Word Unlearning Training

## Core Requirement

The model must receive the original prompt unchanged.

Example:

```text
Prompt: man say hello how are u
Forbidden word: hello
```

During training, the generator still conditions on:

```text
man say hello how are u
```

The model is not allowed to solve the task by deleting `hello` from the prompt
before generation. Instead, it should learn from reward/optimization that saying
`hello` in the audio is undesirable.

Desired generated speech:

```text
how are you
```

Undesired generated speech:

```text
hello how are you
```

## Training Signal

The training signal should come from ASR.

1. Generate audio from the original prompt.
2. Transcribe generated audio with Whisper or another ASR model.
3. Penalize the output if the transcript contains the forbidden word.
4. Reward the output if it preserves the non-forbidden content.
5. Add audio-quality and retention rewards so the model does not collapse.

Reward sketch:

```text
reward =
  + retained_content_score
  + audio_quality_score
  + non_target_prompt_score
  - forbidden_word_penalty
```

For the example:

```text
Prompt: man say hello how are u
Transcript A: hello how are you
Reward: low, because forbidden word appears

Transcript B: how are you
Reward: high, because hello is removed and allowed content remains

Transcript C: silence
Reward: low, because forbidden word is gone but retention/audio quality failed
```

## Why Retention Is Critical

The model should not learn a trivial solution such as:

```text
generate silence for every prompt
```

or:

```text
ignore all speech prompts
```

That would give high forbidden-word removal but terrible retention.

For non-target prompts such as:

```text
a man says good morning
a woman asks how are you
someone sings a short melody
```

the unlearned model should behave almost the same as the original model.

## Evaluation Sets

Use three prompt groups.

### Target Prompts

Prompts that contain the forbidden word:

```text
please say hello
man say hello how are u
generate a song hello hello hello
a woman says hello and then asks how are you
```

Metric:

```text
forbidden_word_rate should decrease
```

### Retain Speech Prompts

Speech prompts that do not contain the forbidden word:

```text
please say good morning
how are you today
the man says thank you
the woman asks where are we going
```

Metrics:

```text
ASR content retention should stay high
speech quality should stay high
```

### General Audio Prompts

Non-speech or mixed audio prompts:

```text
birds chirping in a quiet forest
a car driving down a street
rain falling on a roof
people clapping in a large room
```

Metrics:

```text
general text-to-audio quality should stay close to the original model
```

## What To Report

Always report before and after unlearning:

```text
forbidden_word_rate on target prompts
retained_content_recall on target prompts
retained_content_recall on non-target speech prompts
CLAP score or text-audio similarity on general prompts
audio quality score or human preference
```

Good result:

```text
forbidden_word_rate drops strongly
retention drops only slightly
audio quality remains similar
```

Bad result:

```text
forbidden_word_rate drops
but speech becomes silence, noise, or unrelated content
```

## Current Repo Status

The repo now contains:

- prompt-based audio generation and saving
- forbidden-word transcript evaluation
- ASR-based rejection/regeneration for inference
- helper reward functions for word-level unlearning

The repo does not yet contain full RL fine-tuning of AudioLDM2 or another
speech-capable generator. The next implementation step is to train a LoRA
adapter using `word_unlearning_reward()` while keeping prompts unchanged. See
`MODEL_UNDERSTANDS_FORBIDDEN_WORD.md` for the recommended training strategy and
`data/speech_word_unlearning_examples.jsonl` for a small dataset template.
