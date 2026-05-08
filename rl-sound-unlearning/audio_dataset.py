import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioConceptDataset(Dataset):
    """Load audio waves and concept labels from a manifest file."""

    def __init__(self, root_dir: str, sample_rate: int = 16000, duration: float = 1.0):
        self.root_dir = Path(root_dir)
        manifest_path = self.root_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)

        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)

        self.concepts = sorted({example["concept"] for example in self.examples})
        self.concept_to_idx = {concept: i for i, concept in enumerate(self.concepts)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        audio_path = self.root_dir / example["audio"]

        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[1] >= self.num_samples:
            waveform = waveform[:, : self.num_samples]
        else:
            pad = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        concept = example["concept"]
        label = self.concept_to_idx[concept]
        return waveform, label
