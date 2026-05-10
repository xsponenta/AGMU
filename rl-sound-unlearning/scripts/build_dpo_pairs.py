"""Stage 1: rejection sampling -> chosen/rejected pairs for DPO.

For each training prompt:
  - draw K samples from the frozen reference SpeechT5 (different speakers + a
    little speaker-embedding noise for diversity)
  - transcribe each with Whisper
  - compute the word-unlearning reward (forbidden penalty + retention recall)
  - pick the *best* sample as `chosen`, the *worst* as `rejected`

We also persist the reference model's neg-log-p for chosen/rejected mels so the
DPO trainer doesn't have to keep a frozen second copy of SpeechT5 in VRAM.

Run:
    python scripts/build_dpo_pairs.py \
        --prompts prompts/speech_word_train_prompts.jsonl \
        --out dpo_pairs/run01 \
        --num-candidates 4 --asr-model openai/whisper-small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

# allow running as `python scripts/build_dpo_pairs.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from speech_word_unlearning import remove_forbidden_words, word_unlearning_reward  # noqa: E402
from tts_unlearning import (  # noqa: E402
    iter_jsonl,
    load_tts,
    neg_log_p_mel,
    sample_audio,
    save_mel,
    write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build DPO pairs via rejection sampling.")
    p.add_argument("--prompts", type=Path, required=True,
                   help="JSONL with target_train + retain_train splits.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dir; pairs.jsonl + mels/ + audio/ go here.")
    p.add_argument("--asr-model", default="openai/whisper-small")
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--num-speakers", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--include-retain", action="store_true",
                   help="Also build pairs for retain prompts (for retention DPO).")
    p.add_argument("--min-chosen-recall", type=float, default=0.7,
                   help="Discard target pairs whose chosen sample has retention_recall below this.")
    p.add_argument("--min-ref-margin", type=float, default=0.05,
                   help="Discard target pairs where |ref_nlp_chosen - ref_nlp_rejected| < this. "
                        "These pairs give DPO no usable preference signal.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_asr(name: str):
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=name)


def transcribe(asr, waveform: torch.Tensor, sr: int) -> str:
    audio = waveform.numpy()
    return asr({"array": audio, "sampling_rate": sr})["text"]


def score_candidate(transcript: str, prompt_row: dict) -> dict:
    forbidden = prompt_row.get("forbidden_words", [])
    retain_text = prompt_row.get("retain_hint") or prompt_row.get("desired_transcript") or ""
    if not retain_text:
        # fall back to the prompt itself stripped of the forbidden words
        retain_text = prompt_row["prompt"]
    return word_unlearning_reward(transcript, forbidden, retain_text=retain_text)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_tts(device=device, num_speakers=max(args.num_candidates, args.num_speakers))
    asr = load_asr(args.asr_model)

    args.out.mkdir(parents=True, exist_ok=True)
    mel_dir = args.out / "mels"
    audio_dir = args.out / "audio"
    mel_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    def sample_best(prompt_text: str, target_row: dict, k: int) -> dict:
        """Sample k candidates from `prompt_text`, score against `target_row`'s
        forbidden words / retain hint, return the best one (chosen-style)."""
        cands = []
        for i in range(k):
            speaker_idx = i % bundle.speaker_embeddings.size(0)
            mel, wav = sample_audio(bundle, prompt_text, speaker_idx,
                                    noise_std=args.noise_std)
            transcript = transcribe(asr, wav, args.sample_rate)
            metrics = score_candidate(transcript, target_row)
            cands.append({"speaker_idx": speaker_idx, "mel": mel, "wav": wav,
                          "transcript": transcript, "metrics": metrics})
        cands.sort(key=lambda c: (c["metrics"]["reward"],
                                  c["metrics"]["retention_recall"]),
                   reverse=True)
        return cands

    pairs = []
    n_dropped_chosen_quality = 0
    n_dropped_ref_margin = 0
    n_dropped_chosen_forbidden = 0
    for prompt_idx, row in enumerate(iter_jsonl(args.prompts)):
        if row.get("split") not in {"target_train", "retain_train"}:
            continue
        if row["split"] == "retain_train" and not args.include_retain:
            continue

        prompt = row["prompt"]
        forbidden = row.get("forbidden_words", []) or []

        if row["split"] == "target_train":
            # Oracle pair construction: chosen text strips the forbidden word.
            # The DPO trainer still feeds the *original* prompt to the model.
            chosen_text = (row.get("desired_transcript") or "").strip()
            if not chosen_text:
                chosen_text = remove_forbidden_words(prompt, forbidden).strip()
            if not chosen_text:
                # e.g. "please say hello" -> "" after stripping. Use a short
                # placeholder so the vocoder produces a real-sounding clip.
                chosen_text = "okay"

            chosen_cands = sample_best(chosen_text, row, args.num_candidates)
            rejected_cands = sample_best(prompt, row, args.num_candidates)
            chosen = chosen_cands[0]                # best (cleanest) of the cleaned-prompt pool
            rejected = rejected_cands[-1]           # worst (most forbidden) of the original-prompt pool

            # Quality filters: bad pairs make DPO learn nothing.
            if chosen["metrics"]["has_forbidden"]:
                n_dropped_chosen_forbidden += 1
                print(f"[{prompt_idx}] dropped: chosen still contains forbidden word")
                continue
            if chosen["metrics"]["retention_recall"] < args.min_chosen_recall:
                n_dropped_chosen_quality += 1
                print(f"[{prompt_idx}] dropped: chosen retention "
                      f"{chosen['metrics']['retention_recall']:.2f} < {args.min_chosen_recall}")
                continue

        else:  # retain_train
            # No forbidden word in this prompt. Build a retain-anchor entry:
            # chosen = a single ref sample, rejected = same as chosen so the
            # DPO term is exactly zero. The trainer detects this and applies
            # only the SFT term (-log p_policy(mel|prompt)), which pins retain
            # behavior at the ref distribution.
            cands = sample_best(prompt, row, max(1, args.num_candidates // 2))
            chosen = cands[0]
            rejected = chosen  # signals "retain anchor, no DPO" to the trainer

        # For retain anchors chosen IS rejected (same dict). Persist only once
        # and have the trainer key off the explicit `is_retain_anchor` flag.
        is_retain_anchor = chosen is rejected
        c_mel_path = mel_dir / f"{prompt_idx:04d}_chosen.pt"
        c_wav_path = audio_dir / f"{prompt_idx:04d}_chosen.wav"
        save_mel(chosen["mel"], c_mel_path)
        sf.write(c_wav_path, chosen["wav"].numpy(), args.sample_rate)
        if is_retain_anchor:
            r_mel_path = c_mel_path
            r_wav_path = c_wav_path
        else:
            r_mel_path = mel_dir / f"{prompt_idx:04d}_rejected.pt"
            r_wav_path = audio_dir / f"{prompt_idx:04d}_rejected.wav"
            save_mel(rejected["mel"], r_mel_path)
            sf.write(r_wav_path, rejected["wav"].numpy(), args.sample_rate)

        # Cache reference neg-log-p (= built-in SpeechT5 loss) so the trainer can skip ref forward.
        bundle.model.eval()
        with torch.no_grad():
            spk_c = bundle.speaker_embeddings[chosen["speaker_idx"]]
            spk_r = bundle.speaker_embeddings[rejected["speaker_idx"]]
            ref_nlp_chosen = neg_log_p_mel(
                bundle.model, bundle.processor, bundle.device,
                prompt, chosen["mel"], spk_c,
            ).item()
            ref_nlp_rejected = neg_log_p_mel(
                bundle.model, bundle.processor, bundle.device,
                prompt, rejected["mel"], spk_r,
            ).item()

        # Drop target pairs where the reference can't tell chosen from rejected:
        # the DPO margin (ref_c - nlp_pi_c) - (ref_r - nlp_pi_r) starts at zero
        # and there is no preference direction to optimize.
        if (row["split"] == "target_train"
                and abs(ref_nlp_chosen - ref_nlp_rejected) < args.min_ref_margin):
            n_dropped_ref_margin += 1
            print(f"[{prompt_idx}] dropped: |ref margin| "
                  f"{abs(ref_nlp_chosen - ref_nlp_rejected):.3f} < {args.min_ref_margin}")
            continue

        # For retain anchors there is no separate "cleaned" prompt; the
        # original prompt is what we want the policy to keep producing.
        if row["split"] != "target_train":
            chosen_text = prompt
        pairs.append({
            "prompt_idx": prompt_idx,
            "split": row["split"],
            "is_retain_anchor": is_retain_anchor,
            "prompt": prompt,
            "chosen_text": chosen_text,
            "rejected_text": prompt,
            "forbidden_words": row.get("forbidden_words", []),
            "retain_hint": row.get("retain_hint", ""),
            "chosen_mel": str(c_mel_path.relative_to(args.out)),
            "rejected_mel": str(r_mel_path.relative_to(args.out)),
            "chosen_wav": str(c_wav_path.relative_to(args.out)),
            "rejected_wav": str(r_wav_path.relative_to(args.out)),
            "chosen_speaker_idx": chosen["speaker_idx"],
            "rejected_speaker_idx": rejected["speaker_idx"],
            "chosen_transcript": chosen["transcript"],
            "rejected_transcript": rejected["transcript"],
            "chosen_metrics": chosen["metrics"],
            "rejected_metrics": rejected["metrics"],
            "ref_nlp_chosen": ref_nlp_chosen,
            "ref_nlp_rejected": ref_nlp_rejected,
        })
        print(
            f"[{prompt_idx}] {prompt!r} | chosen R={chosen['metrics']['reward']:.2f} "
            f"({chosen['transcript']!r}) | rejected R={rejected['metrics']['reward']:.2f} "
            f"({rejected['transcript']!r})"
        )

    # Persist the speaker pool so the trainer can rebuild the same indices.
    torch.save(bundle.speaker_embeddings.cpu(), args.out / "speaker_pool.pt")
    write_jsonl(pairs, args.out / "pairs.jsonl")
    print(f"Wrote {len(pairs)} pairs to {args.out / 'pairs.jsonl'}")
    n_target = sum(1 for p in pairs if not p.get("is_retain_anchor", False))
    n_retain = sum(1 for p in pairs if p.get("is_retain_anchor", False))
    print(f"  target pairs kept:  {n_target}")
    print(f"  retain anchors:     {n_retain}")
    print(f"  dropped (chosen still has forbidden):    {n_dropped_chosen_forbidden}")
    print(f"  dropped (chosen retention too low):      {n_dropped_chosen_quality}")
    print(f"  dropped (|ref margin| below threshold):  {n_dropped_ref_margin}")


if __name__ == "__main__":
    main()
