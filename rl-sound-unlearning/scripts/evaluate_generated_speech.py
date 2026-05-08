"""Transcribe generated WAV files and evaluate forbidden-word removal.

Example:

    python3 scripts/evaluate_generated_speech.py \
        --manifest generated_audio/no_hello/manifest.csv \
        --forbidden-words hello
"""
import argparse
import csv
from pathlib import Path

import soundfile as sf

from speech_word_unlearning import word_unlearning_reward


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate speech word unlearning.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forbidden-words", nargs="+", required=True)
    parser.add_argument("--asr-model", default="openai/whisper-small")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_asr(model_name: str):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers. Run: pip install -r requirements_t2a.txt"
        ) from exc
    return pipeline("automatic-speech-recognition", model=model_name)


def transcribe(asr, path: Path) -> str:
    audio, sample_rate = sf.read(path)
    result = asr({"array": audio, "sampling_rate": sample_rate})
    return result["text"]


def main():
    args = parse_args()
    asr = load_asr(args.asr_model)
    out_path = args.out or args.manifest.with_name("speech_word_eval.csv")

    with open(args.manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {args.manifest}")

    fieldnames = list(dict.fromkeys(list(rows[0].keys()) + [
        "transcript",
        "has_forbidden",
        "retention_recall",
        "reward",
    ]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            path = Path(row["path"])
            transcript = transcribe(asr, path)
            retain_text = row.get("retain_text") or row.get("rewritten_prompt") or row.get("prompt")
            metrics = word_unlearning_reward(
                transcript,
                args.forbidden_words,
                retain_text=retain_text,
            )
            row.update({
                "transcript": transcript,
                "has_forbidden": metrics["has_forbidden"],
                "retention_recall": metrics["retention_recall"],
                "reward": metrics["reward"],
            })
            writer.writerow(row)
            print(
                f"{path}: forbidden={metrics['has_forbidden']} "
                f"retention={metrics['retention_recall']:.3f} | {transcript}"
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
