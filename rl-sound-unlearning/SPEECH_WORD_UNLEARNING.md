# Speech Word-Level Unlearning

## Desired Behavior

For a prompt like:

```text
man say hello how are u
```

and forbidden word:

```text
hello
```

the desired output is speech that preserves the allowed content:

```text
how are u
```

but does not say:

```text
hello
```

This is different from environmental audio unlearning. It needs speech-aware
evaluation because the system must know which words were spoken in the generated
audio.

## Important Limitation

It is not scientifically honest to claim that a generative model will **never**
say a word in all possible prompts after training. A better claim is:

```text
The probability of generating the forbidden word is strongly reduced, and an
ASR-based inference guard rejects generations that still contain it.
```

So for prompts like:

```text
please say hello
generate a song hello hello hello
```

the unlearned model should still receive the original prompt. It should learn
that the forbidden word must not appear in the generated audio, even when the
prompt asks for it.

For evaluation and deployment, an ASR guard can still reject/regenerate audio if
the forbidden word appears. But prompt rewriting is not the main unlearning
method.

## Practical Pipeline

1. Keep the original prompt unchanged.
2. Generate audio with a text-to-audio or text-to-speech model.
3. Run ASR on the generated audio.
4. If ASR transcript contains the forbidden word, reject and regenerate.
5. Save the final audio and the transcript.
6. Report target-word suppression and retained-word recall.

## Generate Audio With A Forbidden Word Guard

Install optional dependencies:

```bash
pip install -r requirements_t2a.txt
```

Run generation. For this exact spoken-word task, a speech-capable generator or
TTS model is better than a general environmental text-to-audio model. The script
uses AudioLDM2 by default because it is easy to load with Diffusers, but the
same guard/evaluation idea should be used with the actual speech generator you
want to unlearn.

```bash
python3 scripts/generate_tta_audio.py \
  --model cvssp/audioldm2 \
  --prompts prompts/speech_word_unlearning_prompts.txt \
  --out-dir generated_audio/no_hello \
  --forbidden-words hello \
  --asr-model openai/whisper-small \
  --reject-forbidden \
  --max-regenerations 5
```

This saves:

- `.wav` files
- `manifest.csv`
- prompt
- ASR transcript
- forbidden-word flag
- retention score

## Evaluate Saved Audio

```bash
python3 scripts/evaluate_generated_speech.py \
  --manifest generated_audio/no_hello/manifest.csv \
  --forbidden-words hello \
  --asr-model openai/whisper-small
```

The evaluator writes:

```text
generated_audio/no_hello/speech_word_eval.csv
```

## Metrics

Target removal:

```text
forbidden_word_rate = fraction of generations where ASR transcript contains hello
```

Retention:

```text
retention_recall = fraction of allowed prompt words found in ASR transcript
```

For the example:

```text
Original prompt: man say hello how are u
Desired transcript: how are u
Forbidden present: false
Retention should stay high
```

## How This Connects To Real Unlearning

The current implementation is an inference-time guard plus evaluation. For true
model unlearning, the model must be trained with the original prompt unchanged.
The ASR reward should penalize generated speech that contains the forbidden word
while rewarding retained allowed content:

```text
reward = retained_word_score - forbidden_word_penalty
```

For the prompt:

```text
man say hello how are u
```

the generator should condition on exactly that text, but the reward should prefer
audio transcribed as:

```text
how are you
```

and penalize audio transcribed as:

```text
hello how are you
```

This is the correct unlearning setup: the prompt still contains the forbidden
word, but the trained model learns not to realize that word in the output audio.
