# Codebase Analysis: rl-sound-unlearning

Bugs and issues found by a full re-read of every source file (`audio_*.py`, `train_*.py`, `tts_unlearning.py`, `speech_word_unlearning.py`, `data_synth.py`, `audio_prompts.py`, `audio_utils.py`, `audio_concepts_list.py`, all of `config/`, `scripts/`, `benchmark/`).

Each entry includes a description, the offending location, and a concrete fix.

---

## Real bugs

### B1. Critic stays in `eval()` mode after the first batch of every epoch

**File:** `train_audio_unlearning.py:300–369`

```python
for epoch in range(1, num_epochs + 1):
    critic.train()           # set ONCE per epoch
    policy.train()
    for waveforms, labels in loader:
        critic_opt.zero_grad()
        logits = critic(waveforms)       # uses whatever mode it's in
        loss_critic = ce(logits, labels)
        loss_critic.backward()
        critic_opt.step()

        # Policy step:
        critic.eval()                    # <-- switches to eval
        ...                              # policy forward + reward
        for p in critic.parameters():
            p.requires_grad_(True)
        # MISSING: critic.train()
```

After the first inner iteration of every epoch, the critic stays in `eval()`. All subsequent supervised critic steps run with `BatchNorm1d` in eval mode — running stats are used instead of batch stats, and the running stats themselves are **not** updated. The gradient is still valid w.r.t. the eval-mode forward, but it is not what the algorithm expects, and `BatchNorm` running stats decay quickly because they get only one update per epoch instead of one per batch.

**Fix:** restore train mode right after re-enabling grads. The cleanest form is a `try/finally`:

```python
critic.eval()
for p in critic.parameters():
    p.requires_grad_(False)
try:
    edited, log_prob_sum, mu, log_sigma = policy(waveforms, cond)
    rewards, components = compute_rewards(...)
    ...
    policy_opt.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
    policy_opt.step()
finally:
    for p in critic.parameters():
        p.requires_grad_(True)
    critic.train()
```

---

### B2. `in_batch_diversity_reward` saturates at 0.5 instead of 1.0

**File:** `audio_rewards.py:85–95`

```python
peer = sim.max(dim=-1).values            # in [-1, 1]
return (1.0 - peer).clamp(0.0, 1.0) * 0.5
```

`(1 - peer)` is in `[0, 2]`. The `.clamp(0, 1)` caps it at 1, then `* 0.5` gives `[0, 0.5]`. The comment "rescale to [0, 1]" is therefore wrong. Effect: the diversity term contributes at most `weights.in_batch_div * 0.5 = 0.2`, half of what the weight implies, so the policy gets a much weaker anti-collapse signal than the config suggests.

**Fix:** drop the `.clamp` — the math already lands in `[0, 1]` after `* 0.5`:

```python
return ((1.0 - peer) * 0.5).clamp(0.0, 1.0)
```

(Or just `(1.0 - peer) * 0.5`, since `peer ∈ [-1, 1]` guarantees the result is in `[0, 1]`.)

---

### B3. Benchmark eval requires `speaker_pool.pt` even when only `reference`/`rewrite` is selected

**Files:** `benchmark/run_benchmark.py:88–138`, `benchmark/eval_methods.py:97`

```python
# run_benchmark.py
needs_pairs = not (pairs_dir / "pairs.jsonl").exists()
if needs_pairs and not args.skip_build_pairs and ({"ga", "dpo"} & set(methods)):
    run([PYTHON, "scripts/build_dpo_pairs.py", ...])
...
cmd = [PYTHON, "benchmark/eval_methods.py",
       ...,
       "--speaker-pool", speaker_pool,   # always passed
       ...]
```

```python
# eval_methods.py
speaker_pool = torch.load(args.speaker_pool, map_location="cpu")  # unconditional
```

Pairs (which produce `speaker_pool.pt`) are only built when GA or DPO is in the method set. But `eval_methods.py` is always called with `--speaker-pool ...` and unconditionally loads it. Running `--only-methods reference rewrite` crashes with `FileNotFoundError`.

**Fix (option A — orchestrator):** always run `build_dpo_pairs.py` (the speaker pool is cheap to produce and harmless for the reference/rewrite methods).

**Fix (option B — `eval_methods.py`):** make `--speaker-pool` optional, and when missing call `load_tts(...)` with `num_speakers` defaulted to the fallback random pool size.

```python
if args.speaker_pool and args.speaker_pool.exists():
    speaker_pool = torch.load(args.speaker_pool, map_location="cpu", weights_only=True)
    bundle = load_tts(device=device, num_speakers=speaker_pool.size(0))
    bundle.speaker_embeddings = speaker_pool.to(device)
else:
    bundle = load_tts(device=device)
```

---

### B4. `BatchNorm1d` crash when last batch has size 1

**File:** `audio_critic.py:11–22`, `train_audio_unlearning.py:196`

`AudioCritic` uses `BatchNorm1d` three times. The loader is built with `drop_last=False`, so the last batch can have any size in `[1, batch_size]`. If `len(dataset) % batch_size == 1`, the last batch has size 1 and `BatchNorm1d` in train mode raises:

```
ValueError: Expected more than 1 value per channel when training, got input size [1, ...]
```

With the default synthesized dataset (24 clips) and `batch_size=4`, `24 % 4 == 0`, so this latent bug doesn't fire. But any change to `per_concept` in `data_synth.py` or to `batch_size` can trigger it.

**Fix (cheap):** `drop_last=True` in the DataLoader if `len(dataset) > batch_size`.

```python
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    drop_last=(len(dataset) > batch_size))
```

**Fix (robust):** replace `BatchNorm1d` with `GroupNorm` or `LayerNorm` in `AudioCritic`, which don't care about batch size.

---

## Lower-impact issues

### I1. Dead config keys

**File:** `config/base.py:44–58`

`critic_kernel_size`, `generator_latent_dim`, `generator_kernel_size`, the entire `sample` sub-dict (`batch_size`, `num_batches_per_epoch`), and `reward_fn` are never read anywhere. `AudioCritic` and `AudioEditPolicy` use hardcoded kernel sizes; `compute_rewards` is called directly, never selected by name.

**Fix:** delete the unused keys. Either remove them, or wire them through:

```python
# config/base.py
"model": {
    "critic_hidden_channels": 64,
    "generator_hidden_channels": 32,
    "residual_scale": 0.6,
},
```

---

### I2. `prompt_fn` and `audio_prompts.py` are dead code

**Files:** `audio_prompts.py`, `config/ac_rain.py:16`, `config/ac_wind.py:16`, `config/ac_thunder.py:16`

Every concept config sets `config["prompt_fn"] = "..._descriptions"`, but `train_audio_unlearning.py` never imports `audio_prompts` and never reads `prompt_fn`. The text-description loader is orphaned (the concept conditioning uses one-hots, not text).

**Fix:** remove `audio_prompts.py` and the `prompt_fn` lines, or actually use the prompts (e.g., feed them into a text encoder for conditioning instead of the one-hot).

---

### I3. Inline default config in `train_audio_unlearning.py` is incomplete

**File:** `train_audio_unlearning.py:154–162`

```python
config = {
    "data_path": "data",
    "target_concept": "Rain",
    "num_epochs": 30,
    "train": {"batch_size": 4, "critic_lr": 1e-3, "generator_lr": 3e-4},
    "logdir": "checkpoints",
    "sample_rate": 16000,
}
```

Missing keys (relative to `config/base.py`): `model`, `reward_weights`, `entropy_coef`, `critic_warmup_epochs`, `save_freq`, `eval_freq`, `audio_length_seconds`, …

All are accessed via `.get()` with fallbacks elsewhere in the code, but the fallbacks differ from `config/base.py`. Most importantly, **`critic_warmup_epochs` defaults to 0** here, while base config sets it to 20. Running without `--config` starts REINFORCE against an untrained classifier — the reward is pure noise for the first few epochs.

**Fix:** import the base config directly instead of inlining:

```python
from config.base import get_config as get_base_config
...
if args.config:
    config = load_config(args.config)
else:
    config = get_base_config()
    config["target_concept"] = "Rain"
```

---

### I4. `torch.load` without `weights_only=True`

**Files (5 sites):**

- `tts_unlearning.py:141`
- `train_tts_dpo_unlearning.py:96`
- `scripts/evaluate_tts_unlearning.py:86`
- `benchmark/eval_methods.py:97`
- `benchmark/baselines/gradient_ascent.py:63`

PyTorch ≥ 2.0 emits `FutureWarning` for `torch.load` without `weights_only`, and in PyTorch 2.6+ the default flipped to `True`. All five files contain only tensors, so setting `weights_only=True` explicitly is both safe and forward-compatible.

**Fix:**

```python
return torch.load(path, map_location="cpu", weights_only=True)
```

…at every call site.

---

### I5. `requires_grad_(False)` not restored on exception

**File:** `train_audio_unlearning.py:331–362`

If any exception is raised between `requires_grad_(False)` and `requires_grad_(True)` (OOM, NaN in loss, …), the critic stays frozen for the rest of training. Currently nothing in the loop catches exceptions, so the script crashes — but the issue would surface if anyone wraps the loop in retry logic.

**Fix:** see B1 — the `try/finally` for `critic.train()` should also cover `requires_grad_(True)`.

---

### I6. Naive word boundaries (apostrophes ignored)

**File:** `speech_word_unlearning.py:28–30`

```python
re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
```

The lookarounds use `[a-z0-9]` only. The forbidden word `"you"` then matches inside `"you're"` because the next character is `'`, which is not in `[a-z0-9]`. If the intent is "the word `you` should not be uttered, even within contractions," this is correct. If the intent is to leave contractions alone, the lookarounds should include `'`:

```python
re.compile(rf"(?<![a-z0-9']){escaped}(?![a-z0-9'])")
```

---

### I7. `peft.PeftModel.generate_speech` works only via `__getattr__` forwarding

**Files:** `scripts/evaluate_tts_unlearning.py:114`, `benchmark/eval_methods.py:73`, `train_tts_dpo_unlearning.py:102`, `benchmark/baselines/gradient_ascent.py:68`

`PeftModel` does not explicitly expose SpeechT5 methods like `generate_speech`. Current `peft` falls back to `self.base_model` via `__getattr__`, so `bundle.model.generate_speech(...)` works. This is brittle to peft internals.

**Fix:** pin `peft` in `requirements_t2a.txt` to a known-good range, e.g.:

```
peft>=0.7,<0.12
```

Or call the underlying model explicitly:

```python
mel = bundle.model.get_base_model().generate_speech(...)
```

---

### I8. `attach_lora` applies LoRA to all `q/k/v/out_proj` matches in SpeechT5

**File:** `tts_unlearning.py:81`

```python
target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
```

This matches every attention projection in encoder, decoder, and cross-attention — a large number of adapter parameters. May be intentional, but should be configurable.

**Fix:** expose `target_modules` through the config:

```python
def attach_lora(model, r=8, alpha=16, dropout=0.05,
                target_modules=("q_proj", "k_proj", "v_proj", "out_proj")):
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=list(target_modules),
    )
    return get_peft_model(model, config)
```

…and read it from `config["lora_target_modules"]`.

---

### I9. `random.shuffle` in `build_splits.py` uses global RNG state

**File:** `benchmark/build_splits.py:73–74`

```python
random.seed(args.seed)
...
for word in words:
    target = [...]
    retain = [...]
    random.shuffle(target)
    random.shuffle(retain)
```

Adding or reordering a word in `forbidden_words.txt` changes the shuffles for all subsequent words.

**Fix:** use a per-word RNG so each word's split is deterministic in isolation:

```python
import hashlib

def word_rng(seed: int, word: str) -> random.Random:
    h = int(hashlib.sha256(f"{seed}:{word}".encode()).hexdigest(), 16)
    return random.Random(h)

# usage
rng = word_rng(args.seed, word)
rng.shuffle(target)
rng.shuffle(retain)
```

---

### I10. `target_val` / `retain_val` written to disk but never used

**File:** `benchmark/build_splits.py:101–119`

`train.jsonl` combines `target_train + retain_train`; `eval.jsonl` combines `target_test + retain_test` re-tagged. The validation splits are orphaned.

**Fix:** either drop them from the writer, or add a `val.jsonl` combined file and a `--eval-on val` option to `eval_methods.py`/`run_benchmark.py` so they can be used for hyperparameter selection.

---

### I11. `gradient_ascent.py` counts skipped pairs in `target_steps`

**File:** `benchmark/baselines/gradient_ascent.py:104–109, 120–122`

```python
if nlp.item() >= args.ga_clip:
    target_steps += 1
    continue
loss = -args.ga_coef * nlp
tot_ga += loss.item()
target_steps += 1
```

Skipped pairs increment `target_steps` but don't contribute to `tot_ga`. Reported `ga_term = tot_ga / target_steps` is biased toward 0. Same issue for `tot_loss / steps`.

**Fix:** separate counters for "attempted" vs "performed":

```python
n_target_attempted = 0
n_target_done = 0
...
if nlp.item() >= args.ga_clip:
    n_target_attempted += 1
    continue
loss = -args.ga_coef * nlp
n_target_attempted += 1
n_target_done += 1
...
row["ga_term"] = tot_ga / max(n_target_done, 1)
```

---

### I12. `audio_utils.save_manifest` and `audio_utils.concept_to_onehot` are dead code

**File:** `audio_utils.py:8–23`

Never called from anywhere. `data_synth.py` writes its manifest inline; `train_audio_unlearning.py` builds the one-hot inline at line 328.

**Fix:** delete the unused helpers, or refactor `data_synth.py` and `train_audio_unlearning.py` to use them.

---

### I13. `train_critic_epoch` is used only for warmup

**File:** `train_audio_unlearning.py:129–145`

The main loop re-implements the supervised critic step inline. Either delete `train_critic_epoch` (after inlining warmup), or have the main loop call it for the supervised step.

**Fix:** use it for the supervised step inside the main loop too:

```python
# inside the for-epoch loop, before the policy step
loss_critic = train_critic_step(critic, waveforms, labels, critic_opt, ce)
```

(Implement `train_critic_step` as the single-batch version of `train_critic_epoch`.)

---

### I14. `AudioConceptDataset` rebuilds `Resample` per `__getitem__`

**File:** `audio_dataset.py:36–38`

```python
if sr != self.sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
    waveform = resampler(waveform)
```

A new `Resample` builds a new sinc kernel each call. Inert on the default 16 kHz synth data (the branch never executes), but slow with mixed-rate external datasets.

**Fix:** cache one resampler per `(orig_sr, target_sr)` pair:

```python
def __init__(self, ...):
    ...
    self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == self.sample_rate:
        return waveform
    if sr not in self._resamplers:
        self._resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr,
                                                              new_freq=self.sample_rate)
    return self._resamplers[sr](waveform)
```

---

### I15. `fixed_audio` selection loads every WAV until target is found

**File:** `train_audio_unlearning.py:237–242`

```python
fixed_audio, fixed_label = dataset[0]
for i in range(len(dataset)):
    wave, lab = dataset[i]
    if lab == target_idx:
        fixed_audio = wave
        break
```

Each `dataset[i]` reads a WAV from disk. With 24 files it's trivial; with thousands, it would be slow.

**Fix:** find the index by scanning the manifest in memory first, then load once:

```python
target_indices = [i for i, ex in enumerate(dataset.examples)
                  if dataset.concept_to_idx[ex["concept"]] == target_idx]
if target_indices:
    fixed_audio, _ = dataset[target_indices[0]]
else:
    fixed_audio, _ = dataset[0]
```

---

### I16. Hardcoded `0.2` denominator in `retain_audio`

**File:** `audio_rewards.py:129`

```python
retain_audio[non_target] = (1.0 - edit_rms[non_target] / 0.2).clamp(0.0, 1.0)
```

Any RMS edit above 0.2 zeroes the term. Not configurable. Implicitly coupled to `residual_scale=0.6` and the signal RMS distribution.

**Fix:** expose as a config knob:

```python
@dataclass
class RewardWeights:
    ...
    retain_audio_rms_scale: float = 0.2

# usage
retain_audio[non_target] = (1.0 - edit_rms[non_target] / weights.retain_audio_rms_scale).clamp(0.0, 1.0)
```

---

### I17. `baseline_momentum` hardcoded

**File:** `train_audio_unlearning.py:229`

```python
baseline_momentum = 0.9
```

Not exposed through config.

**Fix:**

```python
baseline_momentum = config.get("baseline_momentum", 0.9)
```

---

### I18. `total.detach()` in `compute_rewards` is redundant

**File:** `audio_rewards.py:141`

The whole block is inside `with torch.no_grad():`, so the tensor is already detached. Cosmetic.

**Fix:** drop the `.detach()`:

```python
return total, components
```

---

### I19. `in_batch_diversity_reward` short-circuits B=1 with mismatched semantics

**File:** `audio_rewards.py:92–93`

```python
if sim.size(0) == 1:
    return torch.ones(1, device=audio.device)
```

For batch size 1 the function returns 1.0 (max diversity) without computing anything, while the regular branch can return up to 0.5 (see B2). After fixing B2, the B=1 short-circuit becomes a real inconsistency — single-sample batches get **twice** the diversity reward that multi-sample batches do.

**Fix:** after fixing B2, also return a "neutral" value matching the new max:

```python
if sim.size(0) == 1:
    return torch.zeros(1, device=audio.device)   # no peers, no diversity signal
```

or simply skip the diversity term when B=1.

---

### I20. `chosen_text` "self-reference" pattern

**File:** `scripts/build_dpo_pairs.py:176`

```python
chosen_text = chosen_text if row["split"] == "target_train" else prompt
```

Works because Python's ternary short-circuits — when the split is `retain_train`, `chosen_text` on the RHS is never evaluated. Fragile to refactoring.

**Fix:** rewrite as a plain `if`:

```python
if row["split"] != "target_train":
    chosen_text = prompt
```

(The variable is already correct in the `target_train` branch above.)

---

### I21. `eval_freq` is misnamed

**File:** `train_audio_unlearning.py:384`, `config/base.py:13`

`evaluate()` runs every epoch because `train_metrics.csv` consumes it every epoch. `eval_freq` controls only when `metrics.csv` (the secondary log) is written. Not a bug, but the name suggests it gates evaluation itself.

**Fix:** rename for clarity:

```python
eval_log_freq = config.get("eval_log_freq", config.get("eval_freq", 5))
```

(Keep `eval_freq` as a deprecated alias.) Update `config/base.py` to use the new name.

---

## False alarms from the earlier pass (kept for the record)

### Not a bug — decoder shape mismatch in `AudioEditPolicy`

**File:** `audio_generator.py:81–83`

The Conv1d encoder produces `ceil(T/4)` outputs per layer; the ConvTranspose1d decoder produces exactly `4 * input` per layer. So decoder output `= 64 * ceil(ceil(ceil(T/4)/4)/4) ≥ T` for all `T ≥ 1`. The trim `mu[..., :T]` always succeeds.

### Not a bug — `evaluate()` "wasted" compute

`metrics` from `evaluate()` is consumed every epoch by `train_metrics.csv`. See I21 — the issue is naming, not waste.

### Not a bug — `word_unlearning_reward` returning `[-1, 1]`

Not used as a REINFORCE reward in this codebase. Used only for ranking DPO pairs (sign-agnostic) and reporting.

### Not a bug — FiLM only at the bottleneck

Design choice. Could be ablated with FiLM in the decoder, but not a defect.

---

## Recommended fix priority

| Rank | Item | Why |
|------|------|-----|
| 1 | **B1** (critic train/eval mode) | Silent correctness bug in the main training loop |
| 2 | **B4** (BN crash with batch=1) | Actual crash, depends on dataset size mod batch size |
| 3 | **B2** (diversity reward halved) | Silently changes effective hyperparameters |
| 4 | **B3** (speaker_pool path) | Crashes a documented CLI flag combination |
| 5 | **I3** (inline default config) | Silent divergence from `config/base.py` |
| 6 | **I4** (`torch.load` weights_only) | Forward-compat sweep, 5 one-line edits |
| 7 | **I1, I2, I12, I13** | Dead-code cleanup, single PR |
| 8 | **I11** (GA metric denominators) | Logged metrics are misleading |
| 9 | **I5** (try/finally) | Natural pairing with B1's fix |
| 10 | I6–I10, I14–I21 | Style, naming, performance, hardening |
