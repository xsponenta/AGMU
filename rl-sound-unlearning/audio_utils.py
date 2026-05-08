import json
from pathlib import Path

import torch
import torchaudio


def concept_to_onehot(concept_idx: int, num_concepts: int, device: torch.device) -> torch.Tensor:
    onehot = torch.zeros(num_concepts, device=device)
    onehot[concept_idx] = 1.0
    return onehot


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().cpu().clamp(-1.0, 1.0)
    torchaudio.save(str(path), waveform, sample_rate)


def save_manifest(manifest: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
