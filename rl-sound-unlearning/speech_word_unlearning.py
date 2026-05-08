"""Helpers for speech word-level unlearning and evaluation.

This module is intentionally lightweight: it does not load an ASR model itself.
It provides text normalization, forbidden-word detection, prompt rewriting, and
retention scoring over transcripts.
"""
from __future__ import annotations

import re


_SPACE_RE = re.compile(r"\s+")
_TOKEN_ALIASES = {
    "u": "you",
    "r": "are",
}


def normalize_text(text: str) -> str:
    """Lowercase text and remove punctuation-like separators."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = _SPACE_RE.sub(" ", text)
    tokens = [_TOKEN_ALIASES.get(token, token) for token in text.strip().split()]
    return " ".join(tokens)


def _word_pattern(word: str) -> re.Pattern[str]:
    escaped = re.escape(normalize_text(word))
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def contains_forbidden(text: str, forbidden_words: list[str]) -> bool:
    text = normalize_text(text)
    return any(_word_pattern(word).search(text) for word in forbidden_words)


def remove_forbidden_words(text: str, forbidden_words: list[str]) -> str:
    """Remove forbidden words for retained-content scoring or debug rewriting.

    True unlearning should still feed the original prompt to the generator.
    """
    cleaned = normalize_text(text)
    for word in forbidden_words:
        cleaned = _word_pattern(word).sub(" ", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def content_tokens(text: str, forbidden_words: list[str]) -> list[str]:
    text = remove_forbidden_words(text, forbidden_words)
    return [token for token in text.split() if token]


def retention_recall(reference_text: str, transcript: str, forbidden_words: list[str]) -> float:
    """Fraction of non-forbidden reference tokens found in the transcript."""
    reference = content_tokens(reference_text, forbidden_words)
    if not reference:
        return 1.0

    transcript_tokens = set(content_tokens(transcript, forbidden_words))
    hits = sum(1 for token in reference if token in transcript_tokens)
    return hits / len(reference)


def word_unlearning_reward(
    transcript: str,
    forbidden_words: list[str],
    retain_text: str | None = None,
    forbidden_penalty: float = 1.0,
    retention_weight: float = 1.0,
) -> dict[str, float | bool]:
    """Score transcript for target-word removal and retained content.

    A future RL fine-tuning loop can use `reward`. The current scripts use the
    same fields for evaluation and inference-time rejection.
    """
    has_forbidden = contains_forbidden(transcript, forbidden_words)
    retention = 1.0
    if retain_text:
        retention = retention_recall(retain_text, transcript, forbidden_words)

    reward = retention_weight * retention
    if has_forbidden:
        reward -= forbidden_penalty

    return {
        "has_forbidden": has_forbidden,
        "retention_recall": retention,
        "reward": reward,
    }
