"""Build per-forbidden-word JSONL splits from LibriTTS train-clean-100 text.

Reads `mythicinfinity/libritts` in streaming mode (text only). For each word in
`forbidden_words.txt`, partitions sentences into target (contains word) and
retain (does not), then splits each partition into train/val/test. Each row is
a JSONL entry compatible with the existing `prompts/speech_word_*.jsonl`
schema, so the rest of the pipeline (`build_dpo_pairs.py`,
`train_tts_dpo_unlearning.py`, `evaluate_tts_unlearning.py`, GA baseline) reads
it without changes.

Run:
    python benchmark/build_splits.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from speech_word_unlearning import contains_forbidden, remove_forbidden_words  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="LibriTTS -> per-word JSONL splits.")
    p.add_argument("--words-file", type=Path, default=Path("benchmark/forbidden_words.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("benchmark/data"))
    p.add_argument("--max-sentences", type=int, default=80000,
                   help="How many LibriTTS sentences to scan before splitting.")
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--max-words", type=int, default=25)
    p.add_argument("--target-train", type=int, default=80)
    p.add_argument("--target-val", type=int, default=20)
    p.add_argument("--target-test", type=int, default=30)
    p.add_argument("--retain-train", type=int, default=200)
    p.add_argument("--retain-val", type=int, default=30)
    p.add_argument("--retain-test", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="mythicinfinity/libritts",
                   help="HuggingFace dataset id with a `text_normalized` field.")
    p.add_argument("--config", default="clean")
    p.add_argument("--split", default="train.clean.100")
    return p.parse_args()


def load_corpus(args) -> list[str]:
    """Stream LibriTTS, keep one sentence per row, length-filtered."""
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True,
                      trust_remote_code=True)
    out = []
    for ex in ds:
        text = ex.get("text_normalized") or ex.get("text_original") or ex.get("text")
        if not text:
            continue
        text = text.strip()
        words = text.split()
        if args.min_words <= len(words) <= args.max_words:
            out.append(text)
        if len(out) >= args.max_sentences:
            break
    return out


def build_word_splits(sentences: list[str], word: str, sizes: dict) -> dict[str, list[dict]]:
    target = [s for s in sentences if contains_forbidden(s, [word])]
    retain = [s for s in sentences if not contains_forbidden(s, [word])]
    random.shuffle(target)
    random.shuffle(retain)

    needed_target = sizes["target_train"] + sizes["target_val"] + sizes["target_test"]
    needed_retain = sizes["retain_train"] + sizes["retain_val"] + sizes["retain_test"]
    if len(target) < needed_target:
        print(f"  WARNING: only {len(target)} target sentences for {word!r} "
              f"(need {needed_target}); using what we have.")
    if len(retain) < needed_retain:
        print(f"  WARNING: only {len(retain)} retain sentences for {word!r}.")

    target = target[:needed_target]
    retain = retain[:needed_retain]

    def take(pool, *ns):
        cuts = []
        offset = 0
        for n in ns:
            cuts.append(pool[offset:offset + n])
            offset += n
        return cuts

    t_train, t_val, t_test = take(target,
                                   sizes["target_train"], sizes["target_val"], sizes["target_test"])
    r_train, r_val, r_test = take(retain,
                                   sizes["retain_train"], sizes["retain_val"], sizes["retain_test"])

    rows = defaultdict(list)
    for split_name, lst in [("target_train", t_train), ("target_val", t_val), ("target_test", t_test)]:
        for s in lst:
            cleaned = remove_forbidden_words(s, [word])
            rows[split_name].append({
                "split": split_name,
                "prompt": s,
                "forbidden_words": [word],
                "desired_transcript": cleaned,
                "retain_hint": cleaned,
            })
    for split_name, lst in [("retain_train", r_train), ("retain_val", r_val), ("retain_test", r_test)]:
        for s in lst:
            rows[split_name].append({
                "split": split_name,
                "prompt": s,
                "forbidden_words": [word],
                "desired_transcript": s,
                "retain_hint": s,
            })
    return rows


def main():
    args = parse_args()
    random.seed(args.seed)

    if not args.words_file.exists():
        raise SystemExit(f"Missing words file: {args.words_file}")

    words = []
    for line in args.words_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line.lower())

    print(f"Streaming up to {args.max_sentences} sentences from {args.dataset} ...")
    sentences = load_corpus(args)
    print(f"Loaded {len(sentences)} sentences from {args.dataset}.")

    sizes = {
        "target_train": args.target_train, "target_val": args.target_val, "target_test": args.target_test,
        "retain_train": args.retain_train, "retain_val": args.retain_val, "retain_test": args.retain_test,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for word in words:
        rows = build_word_splits(sentences, word, sizes)
        word_dir = args.out_dir / word
        word_dir.mkdir(parents=True, exist_ok=True)
        for split_name, lst in rows.items():
            with open(word_dir / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
                for r in lst:
                    f.write(json.dumps(r) + "\n")
        # Combined files used by the existing scripts as drop-in inputs.
        train_combined = rows["target_train"] + rows["retain_train"]
        eval_combined = rows["target_test"] + rows["retain_test"]
        with open(word_dir / "train.jsonl", "w", encoding="utf-8") as f:
            for r in train_combined:
                f.write(json.dumps(r) + "\n")
        # Mark eval rows with the *_eval split names the existing evaluator filters on.
        with open(word_dir / "eval.jsonl", "w", encoding="utf-8") as f:
            for r in eval_combined:
                tagged = dict(r)
                tagged["split"] = "target_eval" if r["split"] == "target_test" else "retain_eval"
                f.write(json.dumps(tagged) + "\n")
        summary.append({"word": word, **{s: len(lst) for s, lst in rows.items()}})
        print(f"  {word}: " + ", ".join(f"{s}={len(lst)}" for s, lst in rows.items()))

    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote per-word splits to {args.out_dir}")


if __name__ == "__main__":
    main()
