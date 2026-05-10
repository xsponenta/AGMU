"""On-policy ASR-reward RL training for SpeechT5 word unlearning.

Trains a LoRA adapter directly against the eval metric. Each step:
  1. Sample a mel from the policy (LoRA-adapted SpeechT5) with the original prompt.
  2. Vocode -> waveform -> Whisper transcript.
  3. Compute word_unlearning_reward(transcript, forbidden, retain_text).
  4. REINFORCE with an EMA reward baseline; the surrogate loss is
     -(reward - baseline) * log p_policy(mel | prompt), where log p is taken as
     -SpeechT5's mel L1 loss for the *just-sampled* mel (treated as a fixed target).
  5. KL-style anchor: a small SFT term `+ kl_coef * nlp_pi(retain_mel | retain_prompt)`
     keeps retain prompts on-distribution. We use the cached retain anchors from
     pairs.jsonl (the ones build_dpo_pairs.py marked is_retain_anchor=True).

This optimizes the same metric that benchmark/eval_methods.py reports, so it
should dominate the offline GA/DPO baselines when the LoRA has enough capacity
and enough epochs to converge.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from speech_word_unlearning import word_unlearning_reward  # noqa: E402
from tts_unlearning import (  # noqa: E402
    attach_lora,
    iter_jsonl,
    load_mel,
    load_tts,
    neg_log_p_mel,
    sample_audio,
)


def parse_args():
    p = argparse.ArgumentParser(description="On-policy RL LoRA training for SpeechT5 word unlearning.")
    p.add_argument("--prompts", type=Path, required=True,
                   help="train.jsonl with target_train + retain_train splits.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--speaker-pool", type=Path, default=None,
                   help="Reuse the speaker pool from build_dpo_pairs.py for stable conditioning.")
    p.add_argument("--retain-anchors", type=Path, default=None,
                   help="Optional pairs.jsonl with is_retain_anchor entries; if given, "
                        "their cached chosen_mel files are used as KL anchors.")
    p.add_argument("--asr-model", default="openai/whisper-small")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--num-target-prompts", type=int, default=80,
                   help="Subsample target_train prompts per epoch to keep wall time bounded.")
    p.add_argument("--num-retain-anchors", type=int, default=40,
                   help="Subsample retain anchors per epoch.")
    p.add_argument("--kl-coef", type=float, default=0.5,
                   help="Weight on the retain-anchor SFT (KL-style) term.")
    p.add_argument("--baseline-momentum", type=float, default=0.9)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_asr(name: str):
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=name)


def transcribe(asr, waveform: torch.Tensor, sr: int) -> str:
    return asr({"array": waveform.numpy(), "sampling_rate": sr})["text"]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_prompts = []
    retain_prompts = []
    for row in iter_jsonl(args.prompts):
        if row.get("split") == "target_train":
            target_prompts.append(row)
        elif row.get("split") == "retain_train":
            retain_prompts.append(row)
    if not target_prompts:
        raise SystemExit(f"No target_train rows in {args.prompts}")

    if args.speaker_pool and args.speaker_pool.exists():
        speaker_pool = torch.load(args.speaker_pool, map_location="cpu", weights_only=True)
        bundle = load_tts(device=device, num_speakers=speaker_pool.size(0))
        bundle.speaker_embeddings = speaker_pool.to(device)
    else:
        bundle = load_tts(device=device, num_speakers=8)

    bundle.model = attach_lora(bundle.model, r=args.lora_r)
    bundle.model.print_trainable_parameters()

    asr = load_asr(args.asr_model)

    # Cached retain anchors: list of (prompt, mel_path, speaker_idx) tuples.
    retain_anchor_mels = []
    if args.retain_anchors and args.retain_anchors.exists():
        anchor_root = args.retain_anchors.parent
        for row in iter_jsonl(args.retain_anchors):
            if not row.get("is_retain_anchor", False):
                continue
            retain_anchor_mels.append({
                "prompt": row["prompt"],
                "mel_path": anchor_root / row["chosen_mel"],
                "speaker_idx": row["chosen_speaker_idx"],
            })

    optim = torch.optim.AdamW(
        [p for p in bundle.model.parameters() if p.requires_grad], lr=args.lr,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = torch.tensor(0.0, device=device)
    log = []

    for epoch in range(1, args.epochs + 1):
        ep_reward_sum = 0.0
        ep_forbidden_count = 0
        ep_retention_sum = 0.0
        ep_pg_loss = 0.0
        ep_kl_loss = 0.0
        ep_n = 0

        # Subsample target prompts for this epoch.
        targets = random.sample(target_prompts,
                                min(args.num_target_prompts, len(target_prompts)))

        for row in targets:
            prompt = row["prompt"]
            forbidden = row.get("forbidden_words", []) or []
            retain_text = (row.get("retain_hint")
                           or row.get("desired_transcript")
                           or prompt)

            # 1) Sample on-policy with the current LoRA model.
            bundle.model.eval()
            speaker_idx = random.randrange(bundle.speaker_embeddings.size(0))
            with torch.no_grad():
                mel, wav = sample_audio(bundle, prompt, speaker_idx, noise_std=0.05)

            # 2) ASR + reward.
            transcript = transcribe(asr, wav, args.sample_rate)
            metrics = word_unlearning_reward(transcript, forbidden,
                                             retain_text=retain_text)
            reward = float(metrics["reward"])  # in roughly [-1, 1]

            # 3) Surrogate log-prob: -SpeechT5 L1 loss on the sampled mel.
            #    REINFORCE: maximize advantage * log p  =>  minimize -(advantage) * log p
            #    log p = -nlp; so loss_pg = -advantage * (-nlp) = advantage * nlp,
            #    but the *gradient* direction we want is to *increase* log p when
            #    advantage > 0, so the surrogate is `-(advantage * (-nlp)) = advantage * nlp`
            #    minimized => decrease nlp when advantage > 0. That matches.
            bundle.model.train()
            spk = bundle.speaker_embeddings[speaker_idx]
            nlp = neg_log_p_mel(bundle.model, bundle.processor, bundle.device,
                                prompt, mel, spk)
            advantage = reward - baseline.item()
            pg_loss = advantage * nlp  # scalar, has grad through nlp

            # 4) KL-style retain anchor (SFT on cached retain mels).
            if retain_anchor_mels and args.kl_coef > 0:
                anchor = random.choice(retain_anchor_mels)
                a_mel = load_mel(anchor["mel_path"])
                a_spk = bundle.speaker_embeddings[anchor["speaker_idx"]]
                kl_loss = neg_log_p_mel(bundle.model, bundle.processor, bundle.device,
                                        anchor["prompt"], a_mel, a_spk)
                loss = pg_loss + args.kl_coef * kl_loss
                ep_kl_loss += float(kl_loss.item())
            elif retain_prompts and args.kl_coef > 0:
                # Fallback: SFT on a freshly-sampled retain prompt (no cached mel).
                # We sample a mel once with no_grad, then use it as the SFT target.
                a_row = random.choice(retain_prompts)
                bundle.model.eval()
                with torch.no_grad():
                    a_mel, _ = sample_audio(bundle, a_row["prompt"],
                                            random.randrange(bundle.speaker_embeddings.size(0)))
                bundle.model.train()
                a_spk = bundle.speaker_embeddings[speaker_idx]
                kl_loss = neg_log_p_mel(bundle.model, bundle.processor, bundle.device,
                                        a_row["prompt"], a_mel, a_spk)
                loss = pg_loss + args.kl_coef * kl_loss
                ep_kl_loss += float(kl_loss.item())
            else:
                loss = pg_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in bundle.model.parameters() if p.requires_grad], 1.0,
            )
            optim.step()

            # 5) Update EMA baseline.
            with torch.no_grad():
                baseline.mul_(args.baseline_momentum).add_(
                    reward * (1.0 - args.baseline_momentum)
                )

            ep_reward_sum += reward
            ep_forbidden_count += int(metrics["has_forbidden"])
            ep_retention_sum += float(metrics["retention_recall"])
            ep_pg_loss += float(pg_loss.item())
            ep_n += 1

        n = max(1, ep_n)
        row = {
            "epoch": epoch,
            "reward_mean": ep_reward_sum / n,
            "forbidden_rate": ep_forbidden_count / n,
            "retention_mean": ep_retention_sum / n,
            "pg_loss_mean": ep_pg_loss / n,
            "kl_loss_mean": ep_kl_loss / n,
            "baseline": float(baseline.item()),
            "n": n,
        }
        log.append(row)
        print(f"RL epoch {epoch:02d}: reward={row['reward_mean']:.3f} "
              f"forbidden={row['forbidden_rate']:.3f} retention={row['retention_mean']:.3f} "
              f"pg={row['pg_loss_mean']:.3f} kl={row['kl_loss_mean']:.3f} "
              f"baseline={row['baseline']:.3f} (n={n})")
        bundle.model.save_pretrained(args.out_dir / f"adapter_epoch_{epoch}")

    with open(args.out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"RL adapters + train_metrics.json in {args.out_dir}")


if __name__ == "__main__":
    main()
