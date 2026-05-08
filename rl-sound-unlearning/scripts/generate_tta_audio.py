"""Generate and save text-to-audio samples from a prompt list.

This script is intentionally optional. Install `requirements_t2a.txt` before
using it:

    pip install -r requirements_t2a.txt

Example:

    python3 scripts/generate_tta_audio.py \
        --model cvssp/audioldm2 \
        --prompts prompts/text_to_audio_eval_prompts.txt \
        --out-dir generated_audio/audioldm2_original
"""
import argparse
import csv
import re
from pathlib import Path

import soundfile as sf
import torch

from speech_word_unlearning import (
    contains_forbidden,
    remove_forbidden_words,
    word_unlearning_reward,
)


def load_prompts(path: Path) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len] or "prompt"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save TTA audio samples.")
    parser.add_argument("--model", default="cvssp/audioldm2")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-waveforms-per-prompt", type=int, default=1)
    parser.add_argument("--audio-length-seconds", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=200)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--forbidden-words", nargs="*", default=[])
    parser.add_argument(
        "--rewrite-forbidden-prompts",
        action="store_true",
        help="Debug/inference option only. Do not use for true unlearning experiments.",
    )
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--reject-forbidden", action="store_true")
    parser.add_argument("--max-regenerations", type=int, default=3)
    return parser.parse_args()


def load_asr(model_name: str | None):
    if not model_name:
        return None
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers. Run: pip install -r requirements_t2a.txt"
        ) from exc
    return pipeline("automatic-speech-recognition", model=model_name)


def transcribe(asr, audio, sample_rate: int) -> str:
    if asr is None:
        return ""
    result = asr({"array": audio, "sampling_rate": sample_rate})
    return result["text"]


def main():
    args = parse_args()

    try:
        from diffusers import AudioLDM2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing diffusers dependencies. Run: pip install -r requirements_t2a.txt"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    asr = load_asr(args.asr_model)
    if args.reject_forbidden and args.forbidden_words and asr is None:
        raise SystemExit("--reject-forbidden requires --asr-model for transcript checking.")

    prompts = load_prompts(args.prompts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "sample_index",
                "attempt",
                "seed",
                "model",
                "prompt",
                "rewritten_prompt",
                "transcript",
                "has_forbidden",
                "retention_recall",
                "reward",
                "path",
            ],
        )
        writer.writeheader()

        for prompt_index, prompt in enumerate(prompts):
            generation_prompt = prompt
            if args.forbidden_words and args.rewrite_forbidden_prompts:
                generation_prompt = remove_forbidden_words(prompt, args.forbidden_words)
                if not generation_prompt:
                    generation_prompt = "clear natural speech"

            for sample_index in range(args.num_waveforms_per_prompt):
                accepted = False
                last_audio = None
                last_seed = None
                last_attempt = 0
                last_transcript = ""
                last_metrics = {
                    "has_forbidden": contains_forbidden(generation_prompt, args.forbidden_words)
                    if args.forbidden_words else False,
                    "retention_recall": 1.0,
                    "reward": 0.0,
                }

                for attempt in range(args.max_regenerations + 1):
                    seed = args.seed + prompt_index * 1000 + sample_index * 100 + attempt
                    generator = torch.Generator(device=device).manual_seed(seed)
                    result = pipe(
                        generation_prompt,
                        num_inference_steps=args.num_inference_steps,
                        audio_length_in_s=args.audio_length_seconds,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    )
                    audio = result.audios[0]
                    transcript = transcribe(asr, audio, args.sample_rate)
                    metrics = word_unlearning_reward(
                        transcript,
                        args.forbidden_words,
                        retain_text=generation_prompt,
                    ) if args.forbidden_words and asr is not None else last_metrics

                    last_audio = audio
                    last_seed = seed
                    last_attempt = attempt
                    last_transcript = transcript
                    last_metrics = metrics

                    if not args.reject_forbidden or not metrics["has_forbidden"]:
                        accepted = True
                        break

                status = "accepted" if accepted else "rejected"
                filename = (
                    f"{prompt_index:04d}_{sample_index:02d}_{status}_"
                    f"{slugify(generation_prompt)}.wav"
                )
                path = args.out_dir / filename
                sf.write(path, last_audio, args.sample_rate)
                writer.writerow({
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "attempt": last_attempt,
                    "seed": last_seed,
                    "model": args.model,
                    "prompt": prompt,
                    "rewritten_prompt": generation_prompt,
                    "transcript": last_transcript,
                    "has_forbidden": last_metrics["has_forbidden"],
                    "retention_recall": last_metrics["retention_recall"],
                    "reward": last_metrics["reward"],
                    "path": str(path),
                })
                print(f"Saved {path}")

    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
