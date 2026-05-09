"""Benchmark orchestrator: for each forbidden word, train DPO + GA and eval all
four methods (reference, rewrite, ga, dpo) on the held-out test set.

Assumes you have already run `benchmark/build_splits.py`. Idempotent: stages
that already produced their output dirs are skipped, so you can re-run after a
crash without redoing everything.

Run:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --only-words love water --only-methods dpo
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def parse_args():
    p = argparse.ArgumentParser(description="Run the SpeechT5 word-unlearning benchmark.")
    p.add_argument("--data-dir", type=Path, default=Path("benchmark/data"))
    p.add_argument("--out-dir", type=Path, default=Path("benchmark/results"))
    p.add_argument("--asr-model", default="openai/whisper-small")
    p.add_argument("--num-candidates", type=int, default=4,
                   help="K for rejection sampling at pair-build time.")
    p.add_argument("--dpo-epochs", type=int, default=4)
    p.add_argument("--ga-epochs", type=int, default=3)
    p.add_argument("--only-words", nargs="*", default=None,
                   help="If set, only run on this subset of words.")
    p.add_argument("--only-methods", nargs="*",
                   default=["reference", "rewrite", "ga", "dpo"])
    p.add_argument("--skip-build-pairs", action="store_true")
    p.add_argument("--skip-train", action="store_true",
                   help="Reuse existing adapters; only re-run eval.")
    p.add_argument("--save-audio", action="store_true",
                   help="Persist generated wavs (off by default to save disk).")
    return p.parse_args()


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def list_words(data_dir: Path) -> list[str]:
    words = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "train.jsonl").exists():
            words.append(p.name)
    return words


def main():
    args = parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Missing {args.data_dir}. Run benchmark/build_splits.py first.")
    words = list_words(args.data_dir)
    if args.only_words:
        words = [w for w in words if w in args.only_words]
    if not words:
        raise SystemExit("No words to benchmark.")
    methods = [m for m in args.only_methods if m in {"reference", "rewrite", "ga", "dpo"}]
    if not methods:
        raise SystemExit("No valid methods selected.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_root = args.out_dir / "pairs"
    adapters_root = args.out_dir / "adapters"
    eval_root = args.out_dir / "eval"

    all_summaries: list[dict] = []
    for word in words:
        word_data = args.data_dir / word
        train_jsonl = word_data / "train.jsonl"
        eval_jsonl = word_data / "eval.jsonl"
        pairs_dir = pairs_root / word
        speaker_pool = pairs_dir / "speaker_pool.pt"

        # Stage 1: rejection sampling -> pairs.jsonl + speaker_pool.pt
        needs_pairs = not (pairs_dir / "pairs.jsonl").exists()
        if needs_pairs and not args.skip_build_pairs and ({"ga", "dpo"} & set(methods)):
            run([PYTHON, "scripts/build_dpo_pairs.py",
                 "--prompts", train_jsonl,
                 "--out", pairs_dir,
                 "--num-candidates", args.num_candidates,
                 "--asr-model", args.asr_model,
                 "--include-retain"])

        # Stage 2: training (per method)
        for method in methods:
            if method in {"reference", "rewrite"}:
                continue
            adapter_dir = adapters_root / word / method
            final_adapter = adapter_dir / (
                f"adapter_epoch_{args.dpo_epochs}" if method == "dpo"
                else f"adapter_epoch_{args.ga_epochs}"
            )
            if not args.skip_train and not final_adapter.exists():
                if method == "dpo":
                    run([PYTHON, "train_tts_dpo_unlearning.py",
                         "--config", "tts_dpo",
                         "--pairs-dir", pairs_dir,
                         "--out-dir", adapter_dir,
                         "--epochs", args.dpo_epochs])
                else:  # ga
                    run([PYTHON, "benchmark/baselines/gradient_ascent.py",
                         "--pairs-dir", pairs_dir,
                         "--out-dir", adapter_dir,
                         "--epochs", args.ga_epochs])

        # Stage 3: eval
        for method in methods:
            method_eval_dir = eval_root / word / method
            if method_eval_dir.exists():
                shutil.rmtree(method_eval_dir)
            cmd = [PYTHON, "benchmark/eval_methods.py",
                   "--eval-prompts", eval_jsonl,
                   "--method", method,
                   "--asr-model", args.asr_model,
                   "--out", method_eval_dir]
            if speaker_pool.exists():
                cmd += ["--speaker-pool", speaker_pool]
            if method in {"ga", "dpo"}:
                final_adapter = adapters_root / word / method / (
                    f"adapter_epoch_{args.dpo_epochs}" if method == "dpo"
                    else f"adapter_epoch_{args.ga_epochs}"
                )
                cmd += ["--adapter", final_adapter]
            if args.save_audio:
                cmd.append("--save-audio")
            run(cmd)

            # Aggregate this method's summary into all_summaries.
            summary_csv = method_eval_dir / "summary.csv"
            with open(summary_csv) as f:
                for row in csv.DictReader(f):
                    row = dict(row)
                    row["word"] = word
                    all_summaries.append(row)

    # Write the master results table.
    if all_summaries:
        fields = ["word", "method", "split", "n", "forbidden_word_rate", "retention_recall"]
        with open(args.out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in all_summaries:
                w.writerow({k: row.get(k, "") for k in fields})
        print(f"\nMaster results table -> {args.out_dir / 'results.csv'}")

    # Compute mean over words for each (method, split).
    from collections import defaultdict
    bucket = defaultdict(lambda: {"fbd": [], "ret": []})
    for row in all_summaries:
        key = (row["method"], row["split"])
        bucket[key]["fbd"].append(float(row["forbidden_word_rate"]))
        bucket[key]["ret"].append(float(row["retention_recall"]))

    means = []
    for (method, split), b in bucket.items():
        means.append({
            "method": method, "split": split,
            "n_words": len(b["fbd"]),
            "forbidden_word_rate_mean": sum(b["fbd"]) / len(b["fbd"]),
            "retention_recall_mean": sum(b["ret"]) / len(b["ret"]),
        })
    means.sort(key=lambda r: (r["split"], r["method"]))
    with open(args.out_dir / "results_mean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "n_words",
                                          "forbidden_word_rate_mean", "retention_recall_mean"])
        w.writeheader()
        w.writerows(means)

    print("\n=== Mean over words ===")
    for r in means:
        print(f"  [{r['method']:<10}] {r['split']:<12} "
              f"forbidden={r['forbidden_word_rate_mean']:.3f} "
              f"retention={r['retention_recall_mean']:.3f}  (n_words={r['n_words']})")
    print(f"\nMean table -> {args.out_dir / 'results_mean.csv'}")

    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"words": words, "methods": methods,
                   "dpo_epochs": args.dpo_epochs, "ga_epochs": args.ga_epochs}, f, indent=2)


if __name__ == "__main__":
    main()
