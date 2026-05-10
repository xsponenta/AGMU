"""Gradient-ascent unlearning baseline (LoRA on SpeechT5).

Classic reference baseline used in many ML-unlearning papers: maximize the
model's NLL on samples we want to forget, while a retain SFT term keeps the
distribution on retain prompts close to the reference.

Reads the same `pairs.jsonl` produced by `scripts/build_dpo_pairs.py`:
  - target pairs:  rejected_mel is the "forbidden" mel  -> ascend on its NLL
  - retain anchors (chosen_mel == rejected_mel):        -> SFT on chosen_mel

Saves a peft adapter compatible with `scripts/evaluate_tts_unlearning.py`.

Run:
    python benchmark/baselines/gradient_ascent.py \
        --pairs-dir dpo_pairs/<word> \
        --out-dir   logs/ga/<word>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tts_unlearning import (  # noqa: E402
    attach_lora,
    iter_jsonl,
    load_mel,
    load_tts,
    neg_log_p_mel,
)


def parse_args():
    p = argparse.ArgumentParser(description="GA unlearning baseline.")
    p.add_argument("--pairs-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ga-coef", type=float, default=2.0,
                   help="Weight on the ascent (forget) term.")
    p.add_argument("--ga-clip", type=float, default=1.5,
                   help="Stop ascending once NLL exceeds this; SpeechT5 mel L1 is ~0.3-0.6.")
    p.add_argument("--retain-sft-coef", type=float, default=0.3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = list(iter_jsonl(args.pairs_dir / "pairs.jsonl"))
    if not pairs:
        raise SystemExit(f"No pairs in {args.pairs_dir}")

    n_before = len(pairs)
    def _keep(p):
        if p.get("is_retain_anchor", False):
            return True
        # Skip target pairs whose "rejected" doesn't actually contain the forbidden
        # word -- ascending its NLL would just damage a clean utterance.
        return p.get("rejected_metrics", {}).get("has_forbidden", True)
    pairs = [p for p in pairs if _keep(p)]
    print(f"[ga] kept {len(pairs)}/{n_before} pairs after filter")
    speaker_pool = torch.load(args.pairs_dir / "speaker_pool.pt", map_location="cpu", weights_only=True)

    bundle = load_tts(device=device, num_speakers=speaker_pool.size(0))
    bundle.speaker_embeddings = speaker_pool.to(device)

    bundle.model = attach_lora(bundle.model, r=args.lora_r)
    bundle.model.print_trainable_parameters()

    optim = torch.optim.AdamW(
        [p for p in bundle.model.parameters() if p.requires_grad], lr=args.lr,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for epoch in range(1, args.epochs + 1):
        bundle.model.train()
        tot_loss = 0.0
        tot_ga = 0.0
        tot_sft = 0.0
        n_target_seen = 0      # all target pairs encountered (incl. skipped)
        n_target_updated = 0   # target pairs that actually contributed a step
        n_retain_updated = 0
        for pair in pairs:
            prompt = pair["prompt"]
            is_retain = pair.get("is_retain_anchor",
                                 pair["chosen_mel"] == pair["rejected_mel"])
            if is_retain:
                mel = load_mel(args.pairs_dir / pair["chosen_mel"])
                spk = bundle.speaker_embeddings[pair["chosen_speaker_idx"]]
                nlp = neg_log_p_mel(bundle.model, bundle.processor, bundle.device,
                                    prompt, mel, spk)
                loss = args.retain_sft_coef * nlp
                tot_sft += loss.item()
                n_retain_updated += 1
            else:
                mel = load_mel(args.pairs_dir / pair["rejected_mel"])
                spk = bundle.speaker_embeddings[pair["rejected_speaker_idx"]]
                nlp = neg_log_p_mel(bundle.model, bundle.processor, bundle.device,
                                    prompt, mel, spk)
                n_target_seen += 1
                # Ascend (maximize nlp) while nlp < clip; once we're past the
                # clip, this pair has been "forgotten enough" -- skip the step
                # entirely rather than emit a constant zero with no grad_fn.
                if nlp.item() >= args.ga_clip:
                    continue
                loss = -args.ga_coef * nlp
                tot_ga += loss.item()
                n_target_updated += 1

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in bundle.model.parameters() if p.requires_grad], 1.0)
            optim.step()
            tot_loss += loss.item()

        steps = max(1, n_target_updated + n_retain_updated)
        row = {
            "epoch": epoch,
            "loss": tot_loss / steps,
            "ga_term": tot_ga / max(n_target_updated, 1),
            "sft_term": tot_sft / max(n_retain_updated, 1),
            "n_target_seen": n_target_seen,
            "n_target_updated": n_target_updated,
            "n_retain_updated": n_retain_updated,
        }
        log.append(row)
        print(f"GA epoch {epoch:02d}: loss={row['loss']:.3f} "
              f"ga={row['ga_term']:.3f} sft={row['sft_term']:.3f} "
              f"(target_updated={n_target_updated}/{n_target_seen}, "
              f"retain={n_retain_updated})")
        bundle.model.save_pretrained(args.out_dir / f"adapter_epoch_{epoch}")

    with open(args.out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"GA adapters saved in {args.out_dir}")


if __name__ == "__main__":
    main()
