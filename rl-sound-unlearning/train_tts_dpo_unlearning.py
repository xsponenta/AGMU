"""Stage 2: DPO training of a LoRA adapter on SpeechT5.

For each pair (chosen mel, rejected mel) we want the policy to assign higher
probability than the frozen reference to chosen, and lower probability than the
reference to rejected. With SpeechT5's regression decoder we treat the built-in
spectrogram loss as `neg_log_p` up to constants, so:

    log_pi(y|x)   = -nlp_pi(y|x)
    log_ref(y|x)  = -nlp_ref(y|x)            # cached at pair-build time
    margin        = (log_pi(c)-log_ref(c)) - (log_pi(r)-log_ref(r))
    DPO loss      = -log sigmoid(beta * margin)

A small SFT bonus on the chosen mel keeps speech intelligible (anti-collapse).
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tts_unlearning import (
    attach_lora,
    iter_jsonl,
    load_mel,
    load_tts,
    neg_log_p_mel,
)


def parse_args():
    p = argparse.ArgumentParser(description="DPO LoRA training for SpeechT5 word unlearning.")
    p.add_argument("--config", default=None, help="Config module (e.g. tts_dpo)")
    p.add_argument("--pairs-dir", type=Path, default=None,
                   help="Output of scripts/build_dpo_pairs.py")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--sft-coef", type=float, default=None,
                   help="Weight on chosen-only NLP loss (anti-collapse SFT term).")
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_config(name: str) -> dict:
    if name.endswith(".py"):
        name = name[:-3]
    try:
        return importlib.import_module(f"config.{name}").get_config()
    except ImportError:
        return importlib.import_module(name).get_config()


def main():
    args = parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "pairs_dir": "dpo_pairs/run01",
            "out_dir": "logs/tts_dpo",
            "num_epochs": 6,
            "batch_size": 1,
            "lr": 1e-4,
            "beta": 0.1,
            "sft_coef": 0.1,
            "lora_r": 8,
            "seed": 0,
        }
    if args.pairs_dir: config["pairs_dir"] = str(args.pairs_dir)
    if args.out_dir: config["out_dir"] = str(args.out_dir)
    if args.epochs is not None: config["num_epochs"] = args.epochs
    if args.batch_size is not None: config["batch_size"] = args.batch_size
    if args.lr is not None: config["lr"] = args.lr
    if args.beta is not None: config["beta"] = args.beta
    if args.sft_coef is not None: config["sft_coef"] = args.sft_coef
    if args.lora_r is not None: config["lora_r"] = args.lora_r
    if args.seed is not None: config["seed"] = args.seed

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs_dir = Path(config["pairs_dir"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(iter_jsonl(pairs_dir / "pairs.jsonl"))
    if not pairs:
        raise SystemExit(f"No pairs in {pairs_dir}/pairs.jsonl. Run build_dpo_pairs.py first.")
    speaker_pool = torch.load(pairs_dir / "speaker_pool.pt", map_location="cpu", weights_only=True)

    bundle = load_tts(device=device, num_speakers=speaker_pool.size(0))
    # Prefer the same speaker pool that was used at pair-building time.
    bundle.speaker_embeddings = speaker_pool.to(device)

    bundle.model = attach_lora(bundle.model, r=config["lora_r"])
    bundle.model.print_trainable_parameters()

    optim = torch.optim.AdamW(
        [p for p in bundle.model.parameters() if p.requires_grad],
        lr=config["lr"],
    )
    beta = float(config["beta"])
    sft_coef = float(config["sft_coef"])

    metrics_log = []
    for epoch in range(1, config["num_epochs"] + 1):
        bundle.model.train()
        total_loss = 0.0
        total_dpo = 0.0
        total_sft = 0.0
        total_acc = 0.0  # fraction of pairs where margin > 0
        steps = 0

        for pair in pairs:
            prompt = pair["prompt"]
            chosen_mel = load_mel(pairs_dir / pair["chosen_mel"])
            spk_c = bundle.speaker_embeddings[pair["chosen_speaker_idx"]]

            is_retain_anchor = pair.get("is_retain_anchor",
                                        pair["chosen_mel"] == pair["rejected_mel"])

            nlp_pi_chosen = neg_log_p_mel(
                bundle.model, bundle.processor, bundle.device,
                prompt, chosen_mel, spk_c,
            )

            if is_retain_anchor:
                # Pure SFT anchor: policy must keep producing the ref-style mel.
                dpo_loss = torch.zeros((), device=bundle.device)
                sft_loss = config.get("retain_sft_coef", 1.0) * nlp_pi_chosen
                margin = torch.zeros((), device=bundle.device)
            else:
                rejected_mel = load_mel(pairs_dir / pair["rejected_mel"])
                spk_r = bundle.speaker_embeddings[pair["rejected_speaker_idx"]]
                nlp_pi_rejected = neg_log_p_mel(
                    bundle.model, bundle.processor, bundle.device,
                    prompt, rejected_mel, spk_r,
                )
                ref_c = float(pair["ref_nlp_chosen"])
                ref_r = float(pair["ref_nlp_rejected"])
                # log_pi - log_ref = -(nlp_pi - nlp_ref) = nlp_ref - nlp_pi
                margin = (ref_c - nlp_pi_chosen) - (ref_r - nlp_pi_rejected)
                dpo_loss = -F.logsigmoid(beta * margin)
                sft_loss = sft_coef * nlp_pi_chosen
            loss = dpo_loss + sft_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in bundle.model.parameters() if p.requires_grad], 1.0,
            )
            optim.step()

            total_loss += loss.item()
            total_dpo += dpo_loss.item()
            total_sft += sft_loss.item()
            total_acc += float(margin.item() > 0)
            steps += 1

        avg = lambda v: v / max(steps, 1)
        log_row = {
            "epoch": epoch,
            "loss": avg(total_loss),
            "dpo": avg(total_dpo),
            "sft": avg(total_sft),
            "preference_acc": avg(total_acc),
        }
        metrics_log.append(log_row)
        print(
            f"Epoch {epoch:02d}: loss={log_row['loss']:.3f} "
            f"dpo={log_row['dpo']:.3f} sft={log_row['sft']:.3f} "
            f"pref_acc={log_row['preference_acc']:.2f}"
        )

        # Save adapter checkpoint
        ckpt_dir = out_dir / f"adapter_epoch_{epoch}"
        bundle.model.save_pretrained(ckpt_dir)

    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Done. LoRA adapters + train_metrics.json in {out_dir}")


if __name__ == "__main__":
    main()
