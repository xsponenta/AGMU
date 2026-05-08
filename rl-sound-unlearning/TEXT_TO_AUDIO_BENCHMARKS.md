# Text-to-Audio Unlearning: Datasets, Benchmarks, and Saved Audio

## Main Research Framing

The main task is **text-to-audio concept unlearning**.

Given a text-to-audio model and a target concept such as `rain`, `dog barking`,
`gunshot`, `siren`, or a music/style concept, the model should learn to avoid
generating that concept when prompted. At the same time, it must preserve normal
generation quality and text-following ability for unrelated prompts.

The most important principle is:

```text
Do not maximize unlearning alone.
Measure unlearning and retention together.
```

A strong result should say:

```text
The target concept probability decreased from X to Y,
while retention on unrelated prompts stayed close to the original model.
```

## Recommended Training Datasets

### 1. AudioCaps

AudioCaps is the most standard dataset for text-to-audio generation. It contains
human-written captions for AudioSet clips and is widely used by AudioLDM, TANGO,
and other text-to-audio systems.

Use it for:

- training or fine-tuning text-to-audio models
- prompt-based evaluation
- target/retain prompt splits
- comparison with AudioLDM/TANGO-style baselines

Why it fits this project:

- The task is directly text-to-audio.
- Captions are natural language prompts.
- Many prior TTA papers report results on AudioCaps.

Suggested use:

- target set: prompts containing the concept to unlearn
- retain set: prompts not containing the target concept
- evaluation set: held-out AudioCaps prompts

Official source: https://audiocaps.github.io/

## 2. Clotho

Clotho is another important audio-caption dataset. Each audio sample has several
captions, and the captions are longer and more descriptive than many AudioCaps
captions.

Use it for:

- retention evaluation
- testing general text-following
- checking that the model still handles descriptive prompts after unlearning

Why it fits this project:

- It contains diverse caption/audio pairs.
- It is useful for measuring whether unlearning damages general text-to-audio
  behavior.

Suggested use:

- train on AudioCaps, evaluate retention on Clotho
- or use Clotho as an additional retain dataset

Official Zenodo record: https://zenodo.org/records/4783391

## 3. FSD50K

FSD50K is a larger sound-event dataset. It is not primarily a text-to-audio
caption dataset, but it is useful for concept classification and external
evaluation.

Use it for:

- training an external concept classifier
- building target concept detectors
- measuring whether generated audio contains the unlearned concept

Why it fits this project:

- It has many sound-event classes.
- It can support objective unlearning metrics beyond listening tests.

Suggested use:

- train or use an audio classifier on FSD50K labels
- use the classifier to score generated audio for target concept presence

Official companion site: https://annotator.freesound.org/fsd/release/FSD50K/

## 4. ESC-50

ESC-50 is smaller than FSD50K, but simple and useful for early controlled tests.

Use it for:

- quick concept classifiers
- small-scale target/retain concept splits
- debugging unlearning metrics

Why it fits this project:

- Easy to understand.
- Has environmental classes like wind, thunderstorm, water drops, animals, etc.

Dataset documentation: https://audeering.github.io/datasets/datasets/esc-50.html

## 5. AudioSet

AudioSet is the large-scale sound event dataset behind many audio benchmarks.
It is useful for serious evaluation, but harder to use because clips are linked
to YouTube segments.

Use it for:

- large-scale concept evaluation
- comparison with broad sound-event classifiers
- constructing target/retain prompt lists

AudioSet page: https://research.google.com/audioset/

## Recommended Benchmarks and Baselines

### AudioLDM 2

AudioLDM 2 is a strong open text-to-audio baseline available through Hugging Face
Diffusers. It takes text prompts and generates audio.

Use it as:

- base model for unlearning experiments
- baseline before unlearning
- comparison model after applying your unlearning method

Diffusers docs: https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2

### TANGO

TANGO is another important text-to-audio baseline. It uses an instruction-tuned
text encoder and latent diffusion.

Use it as:

- baseline model
- comparison against AudioLDM-style models
- possible second architecture if reviewers ask whether the method generalizes

Project page: https://tango-web.github.io/

### AudioAtlas

AudioAtlas is a newer benchmark designed to be more balanced and diverse than
older AudioCaps-only evaluation.

Use it for:

- stronger benchmark section
- category-balanced evaluation
- testing whether retention remains good across many audio categories

Project page: https://audioatlas.github.io/AudioAtlas/

## Metrics To Report

### Unlearning Metrics

Use these on target prompts:

- target classifier probability before and after unlearning
- target concept detection accuracy before and after unlearning
- CLAP similarity between generated audio and target concept text
- human judgment: does the generated audio contain the forbidden concept?

Example:

```text
Prompt: "heavy rain falling on a metal roof"
Target concept: rain
Before unlearning: target detector probability = 0.91
After unlearning: target detector probability = 0.18
```

### Retention Metrics

Use these on non-target prompts:

- CLAP text-audio similarity
- FAD or distributional audio quality score
- non-target concept classification accuracy
- human preference or MOS
- prompt-following accuracy for unrelated prompts

Example:

```text
Prompt: "a dog barking in a park"
Target concept removed: rain
Before unlearning: dog classifier probability = 0.88
After unlearning: dog classifier probability = 0.85
```

This is the key reviewer-facing evidence: the model forgot the target concept
without destroying unrelated generation.

## Suggested Experimental Protocol

For each target concept:

1. Build a target prompt set.
2. Build a retain prompt set.
3. Generate audio from the original model.
4. Apply unlearning.
5. Generate audio from the unlearned model with the same prompts and seeds.
6. Save all generated audio.
7. Compute target suppression metrics.
8. Compute retention metrics.
9. Report mean and standard deviation over at least 3 seeds.

Directory structure:

```text
outputs/
  audioldm2/
    original/
      rain/
      retain/
    unlearned/
      rain/
      retain/
```

## Epoch Guidance

For a lightweight text-to-audio unlearning prototype:

- start with `1-5` epochs to verify the pipeline
- use `10-30` epochs for early experiments
- use `30-100` epochs for serious small-scale experiments

For larger datasets or full diffusion fine-tuning:

- use validation metrics, not a fixed epoch number
- stop when target concept probability decreases but retention starts to drop
- keep checkpoints every few epochs

The best checkpoint is the best trade-off checkpoint, not necessarily the last.

## Saving Generated Audio

Use:

```bash
python3 scripts/generate_tta_audio.py \
  --model cvssp/audioldm2 \
  --prompts prompts/text_to_audio_eval_prompts.txt \
  --out-dir generated_audio/audioldm2_original \
  --num-waveforms-per-prompt 3
```

This saves generated `.wav` files and a `manifest.csv` with prompt, seed, model,
and path information.

For unlearning experiments, run the same prompt file before and after unlearning.
The prompt should remain unchanged even if it contains the target concept; the
model should learn not to generate the target in audio.

```bash
python3 scripts/generate_tta_audio.py \
  --model cvssp/audioldm2 \
  --prompts prompts/text_to_audio_eval_prompts.txt \
  --out-dir generated_audio/before_unlearning

python3 scripts/generate_tta_audio.py \
  --model path/to/unlearned/checkpoint \
  --prompts prompts/text_to_audio_eval_prompts.txt \
  --out-dir generated_audio/after_unlearning
```

Then compare the audio files with the same prompts and seeds.
