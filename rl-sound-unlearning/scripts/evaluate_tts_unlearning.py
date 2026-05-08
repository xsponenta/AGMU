"""Stage 3: evaluate the trained adapter on held-out prompts.

Generates one sample per eval prompt with both:
  - the frozen reference SpeechT5
  - the LoRA-tuned policy (loaded from --adapter)

ASR-transcribes each, computes forbidden-word rate + retention recall, and
writes side-by-side metrics + audio for review.

Run:
    python scripts/evaluate_tts_unlearning.py \
        --adapter logs/tts_dpo/adapter_epoch_6 \
        --prompts prompts/speech_word_eval_prompts.jsonl \
        --speaker-pool dpo_pairs/run01/speaker_pool.pt \
        --out logs/tts_dpo/eval
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

from speech_word_unlearning import word_unlearning_reward  # noqa: E402
from tts_unlearning import iter_jsonl, load_tts, sample_audio  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TTS word unlearning.")
    p.add_argument("--adapter", type=Path, required=True,
                   help="Path to a saved peft adapter dir.")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--speaker-pool", type=Path, required=True)
    p.add_argument("--asr-model", default="openai/whisper-small")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--speaker-idx", type=int, default=0,
                   help="Fixed speaker for the eval pass so before/after audio is comparable.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_asr(name: str):
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=name)


def transcribe(asr, waveform: torch.Tensor, sr: int) -> str:
    return asr({"array": waveform.numpy(), "sampling_rate": sr})["text"]


def split_metrics(rows: list[dict]) -> dict:
    """Aggregate per (split) and per (model)."""
    agg = {}
    for r in rows:
        key = (r["split"], r["model"])
        bucket = agg.setdefault(key, {"forbidden": 0, "retention_sum": 0.0, "n": 0})
        bucket["forbidden"] += int(r["has_forbidden"])
        bucket["retention_sum"] += r["retention_recall"]
        bucket["n"] += 1
    summary = {}
    for (split, model), b in agg.items():
        summary[f"{split}/{model}"] = {
            "n": b["n"],
            "forbidden_word_rate": b["forbidden"] / b["n"],
            "retention_recall": b["retention_sum"] / b["n"],
        }
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(exist_ok=True)

    speaker_pool = torch.load(args.speaker_pool, map_location="cpu")
    bundle = load_tts(device=device, num_speakers=speaker_pool.size(0))
    bundle.speaker_embeddings = speaker_pool.to(device)
    asr = load_asr(args.asr_model)

    prompts = [r for r in iter_jsonl(args.prompts)
               if r.get("split", "").endswith("_eval")]

    rows = []

    # --- Reference pass ---
    for i, r in enumerate(prompts):
        _, wav = sample_audio(bundle, r["prompt"], args.speaker_idx)
        path = audio_dir / f"{i:04d}_ref.wav"
        sf.write(path, wav.numpy(), args.sample_rate)
        transcript = transcribe(asr, wav, args.sample_rate)
        retain_text = r.get("retain_hint") or r.get("desired_transcript") or r["prompt"]
        m = word_unlearning_reward(transcript, r.get("forbidden_words", []), retain_text=retain_text)
        rows.append({
            "split": r["split"], "model": "ref", "prompt": r["prompt"],
            "transcript": transcript,
            "has_forbidden": m["has_forbidden"],
            "retention_recall": m["retention_recall"],
            "reward": m["reward"], "path": str(path),
        })

    # --- Policy pass: load adapter on top of the same SpeechT5 instance ---
    from peft import PeftModel
    bundle.model = PeftModel.from_pretrained(bundle.model, str(args.adapter)).to(device)

    for i, r in enumerate(prompts):
        _, wav = sample_audio(bundle, r["prompt"], args.speaker_idx)
        path = audio_dir / f"{i:04d}_policy.wav"
        sf.write(path, wav.numpy(), args.sample_rate)
        transcript = transcribe(asr, wav, args.sample_rate)
        retain_text = r.get("retain_hint") or r.get("desired_transcript") or r["prompt"]
        m = word_unlearning_reward(transcript, r.get("forbidden_words", []), retain_text=retain_text)
        rows.append({
            "split": r["split"], "model": "policy", "prompt": r["prompt"],
            "transcript": transcript,
            "has_forbidden": m["has_forbidden"],
            "retention_recall": m["retention_recall"],
            "reward": m["reward"], "path": str(path),
        })

    fields = ["split", "model", "prompt", "transcript", "has_forbidden",
              "retention_recall", "reward", "path"]
    with open(args.out / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = split_metrics(rows)
    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split/model", "n", "forbidden_word_rate", "retention_recall"])
        for k, v in summary.items():
            w.writerow([k, v["n"], f"{v['forbidden_word_rate']:.3f}",
                        f"{v['retention_recall']:.3f}"])

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: n={v['n']} forbidden_rate={v['forbidden_word_rate']:.3f} "
              f"retention={v['retention_recall']:.3f}")
    print(f"Saved per-sample to {args.out / 'per_sample.csv'}")


if __name__ == "__main__":
    main()
