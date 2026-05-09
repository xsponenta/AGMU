"""Shared helpers for SpeechT5-based text-to-speech word unlearning.

The unlearning pipeline has three stages, all of which share these helpers:

    1. Rejection sampling -> build_dpo_pairs.py
    2. DPO training       -> train_tts_dpo_unlearning.py
    3. Evaluation         -> evaluate_tts_unlearning.py

We use the regression form of DPO: SpeechT5 predicts mel spectrograms, so
"log p(y|x)" is taken as -loss(model, x, y) where `loss` is SpeechT5's built-in
L1 spectrogram loss. The unknown variance is absorbed into the DPO `beta`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


SPEECH_MODEL = "microsoft/speecht5_tts"
VOCODER_MODEL = "microsoft/speecht5_hifigan"
SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"


@dataclass
class TTSBundle:
    processor: object
    model: object
    vocoder: object
    speaker_embeddings: torch.Tensor  # [N, 512] pool to sample from
    device: torch.device


def _load_speaker_pool(num_speakers: int) -> torch.Tensor:
    """Try CMU-ARCTIC xvectors; fall back to random unit-norm 512-D embeddings.

    `datasets >= 4` removed script-based loaders, so the original repo no longer
    loads cleanly there. We attempt the script path with `trust_remote_code`,
    then fall back to a deterministic random pool with the same shape and norm.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(SPEAKER_DATASET, split="validation", trust_remote_code=True)
        indices = torch.linspace(0, len(ds) - 1, num_speakers).long().tolist()
        return torch.stack([torch.tensor(ds[i]["xvector"]) for i in indices])
    except Exception as e:
        print(f"[tts_unlearning] CMU-ARCTIC xvectors unavailable ({type(e).__name__}: {e}). "
              "Falling back to random speaker pool.")
        g = torch.Generator().manual_seed(0)
        embeds = torch.randn(num_speakers, 512, generator=g)
        # Real xvectors have norm ~10; mimic that so SpeechT5 conditioning behaves.
        return torch.nn.functional.normalize(embeds, dim=-1) * 10.0


def load_tts(device: torch.device | str = "cuda", num_speakers: int = 16) -> TTSBundle:
    """Load SpeechT5 + HiFi-GAN + a small pool of speaker embeddings."""
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    device = torch.device(device)
    processor = SpeechT5Processor.from_pretrained(SPEECH_MODEL)
    model = SpeechT5ForTextToSpeech.from_pretrained(SPEECH_MODEL).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_MODEL).to(device)

    embeds = _load_speaker_pool(num_speakers).to(device)
    return TTSBundle(processor=processor, model=model, vocoder=vocoder,
                     speaker_embeddings=embeds, device=device)


def attach_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Attach a small LoRA adapter to SpeechT5's text encoder + decoder attention."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        # SpeechT5ForTextToSpeech is not in PEFT's task-type registry, so leave
        # task_type=None and treat the wrapped model like a regular nn.Module.
    )
    return get_peft_model(model, config)


@torch.no_grad()
def sample_audio(bundle: TTSBundle, prompt: str, speaker_idx: int,
                 noise_std: float = 0.0, max_length: int = 600) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (mel, waveform) for a prompt using the model's generate_speech.

    `noise_std` injects Gaussian noise on the speaker embedding to add diversity
    on top of the speaker-pool variation.
    """
    inputs = bundle.processor(text=prompt, return_tensors="pt").to(bundle.device)
    spk = bundle.speaker_embeddings[speaker_idx].unsqueeze(0)
    if noise_std > 0:
        spk = spk + noise_std * torch.randn_like(spk)
        spk = F.normalize(spk, dim=-1) * bundle.speaker_embeddings.norm(dim=-1).mean()
    mel = bundle.model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=spk,
        vocoder=None,
        threshold=0.5,
    )  # [T_mel, 80]
    if mel.shape[0] > max_length:
        mel = mel[:max_length]
    waveform = bundle.vocoder(mel)  # [T_audio]
    return mel.cpu(), waveform.cpu()


def neg_log_p_mel(model, processor, device, prompt: str, mel: torch.Tensor,
                  speaker_embedding: torch.Tensor) -> torch.Tensor:
    """SpeechT5's built-in spectrogram loss for `target_mel` given `prompt`.

    Returned tensor has gradient if model parameters require_grad. Treat the
    return value as -log p(mel | prompt) up to constants.
    """
    inputs = processor(text=prompt, return_tensors="pt").to(device)
    mel = mel.to(device).unsqueeze(0)  # [1, T_mel, 80]
    spk = speaker_embedding.to(device)
    if spk.dim() == 1:
        spk = spk.unsqueeze(0)
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        labels=mel,
        speaker_embeddings=spk,
        return_dict=True,
    )
    return out.loss


def save_mel(mel: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mel.cpu(), path)


def load_mel(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def iter_jsonl(path: Path) -> Iterable[dict]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
