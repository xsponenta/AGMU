"""Unified evaluator for the four benchmark methods.

Each method maps to a `(model, generation_text)` strategy:

  reference   -> frozen SpeechT5,            generation_text = original prompt
  rewrite     -> frozen SpeechT5,            generation_text = remove_forbidden_words(prompt)
  ga          -> SpeechT5 + GA LoRA adapter, generation_text = original prompt
  dpo         -> SpeechT5 + DPO LoRA adapter, generation_text = original prompt

Each generation is transcribed by Whisper and scored with `word_unlearning_reward`
using the *original* prompt's forbidden words and retain hint, so all methods
are compared on the same target.

Run:
    python benchmark/eval_methods.py \
        --eval-prompts benchmark/data/love/eval.jsonl \
        --speaker-pool dpo_pairs/love/speaker_pool.pt \
        --method dpo --adapter logs/dpo/love/adapter_epoch_4 \
        --out logs/results/love/dpo

The orchestrator calls this once per (word, method).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from speech_word_unlearning import remove_forbidden_words, word_unlearning_reward  # noqa: E402
from tts_unlearning import iter_jsonl, load_tts, sample_audio  # noqa: E402


METHODS = {"reference", "rewrite", "ga", "dpo", "rl"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one method on one word's eval set.")
    p.add_argument("--eval-prompts", type=Path, required=True)
    p.add_argument("--speaker-pool", type=Path, default=None,
                   help="Optional. If present, override the default speaker pool. "
                        "Required for ga/dpo to match training-time conditioning.")
    p.add_argument("--method", choices=sorted(METHODS), required=True)
    p.add_argument("--adapter", type=Path, default=None,
                   help="Required for method=ga or method=dpo.")
    p.add_argument("--asr-model", default="openai/whisper-small")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-speakers", type=int, default=4,
                   help="Generate this many samples per prompt (one per speaker idx) and average.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--save-audio", action="store_true",
                   help="Persist generated wavs (off by default to save disk on large benchmarks).")
    return p.parse_args()


def load_asr(name: str):
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=name)


def transcribe(asr, waveform: torch.Tensor, sr: int) -> str:
    return asr({"array": waveform.numpy(), "sampling_rate": sr})["text"]


def attach_adapter_if_any(bundle, method: str, adapter: Path | None, device):
    if method in {"ga", "dpo", "rl"}:
        if adapter is None:
            raise SystemExit(f"--adapter is required for method={method}")
        from peft import PeftModel
        bundle.model = PeftModel.from_pretrained(bundle.model, str(adapter)).to(device)
    elif method in {"reference", "rewrite"}:
        pass  # frozen reference, no adapter
    else:
        raise SystemExit(f"Unknown method {method}")
    return bundle


def text_for_generation(method: str, prompt: str, forbidden: list[str]) -> str:
    if method == "rewrite":
        cleaned = remove_forbidden_words(prompt, forbidden).strip()
        return cleaned or "okay"
    return prompt


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    if args.save_audio:
        audio_dir.mkdir(exist_ok=True)

    if args.method in {"ga", "dpo", "rl"} and (args.speaker_pool is None or not args.speaker_pool.exists()):
        raise SystemExit(
            f"--speaker-pool is required for method={args.method} to match "
            "the speaker conditioning used at training time."
        )

    if args.speaker_pool is not None and args.speaker_pool.exists():
        speaker_pool = torch.load(args.speaker_pool, map_location="cpu", weights_only=True)
        bundle = load_tts(device=device, num_speakers=speaker_pool.size(0))
        bundle.speaker_embeddings = speaker_pool.to(device)
    else:
        # Reference / rewrite without a saved pool: use the default fallback pool.
        bundle = load_tts(device=device)
    bundle = attach_adapter_if_any(bundle, args.method, args.adapter, device)

    asr = load_asr(args.asr_model)

    rows = []
    n_pool = bundle.speaker_embeddings.size(0)
    n_speakers = max(1, min(args.num_speakers, n_pool))
    for i, r in enumerate(iter_jsonl(args.eval_prompts)):
        forbidden = r.get("forbidden_words", []) or []
        gen_text = text_for_generation(args.method, r["prompt"], forbidden)
        retain_text = r.get("retain_hint") or r.get("desired_transcript") or r["prompt"]

        sample_metrics = []
        sample_transcripts = []
        sample_path = ""
        for s in range(n_speakers):
            speaker_idx = (s * max(1, n_pool // n_speakers)) % n_pool
            _, wav = sample_audio(bundle, gen_text, speaker_idx)
            transcript = transcribe(asr, wav, args.sample_rate)
            m = word_unlearning_reward(transcript, forbidden, retain_text=retain_text)
            sample_metrics.append(m)
            sample_transcripts.append(transcript)
            if args.save_audio and s == 0:
                path_obj = audio_dir / f"{i:04d}.wav"
                sf.write(path_obj, wav.numpy(), args.sample_rate)
                sample_path = str(path_obj)

        # Average across speakers. has_forbidden becomes a fraction in [0, 1].
        has_forbidden_frac = sum(int(mm["has_forbidden"]) for mm in sample_metrics) / n_speakers
        retention_mean = sum(mm["retention_recall"] for mm in sample_metrics) / n_speakers
        reward_mean = sum(mm["reward"] for mm in sample_metrics) / n_speakers

        rows.append({
            "method": args.method,
            "split": r.get("split", ""),
            "prompt": r["prompt"],
            "generation_text": gen_text,
            "transcript": " || ".join(sample_transcripts),
            "has_forbidden": has_forbidden_frac,
            "retention_recall": retention_mean,
            "reward": reward_mean,
            "path": sample_path,
        })

    fields = ["method", "split", "prompt", "generation_text", "transcript",
              "has_forbidden", "retention_recall", "reward", "path"]
    with open(args.out / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Aggregate per split.
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "fbd": 0.0, "ret": 0.0})
    for r in rows:
        a = agg[r["split"]]
        a["n"] += 1
        a["fbd"] += r["has_forbidden"]
        a["ret"] += r["retention_recall"]

    summary = []
    for split, a in agg.items():
        summary.append({
            "method": args.method,
            "split": split,
            "n": a["n"],
            "forbidden_word_rate": a["fbd"] / a["n"],
            "retention_recall": a["ret"] / a["n"],
        })
    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "n", "forbidden_word_rate", "retention_recall"])
        w.writeheader()
        w.writerows(summary)

    for s in summary:
        print(f"[{args.method}] {s['split']}: n={s['n']} "
              f"forbidden={s['forbidden_word_rate']:.3f} "
              f"retention={s['retention_recall']:.3f}")


if __name__ == "__main__":
    main()
