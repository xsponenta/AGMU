"""Procedural synthesizer for tiny Rain/Wind/Thunder dataset.

Each clip is 1.0s at 16kHz, mono. The goal is *distinguishable* concepts
that sound like real ambient audio (not silence/noise) so the critic has
real structure to learn and the generator's edits stay perceptually meaningful.
"""
from pathlib import Path
import hashlib
import json
import math
import argparse

import torch
import torchaudio
import torchaudio.functional as AF


SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE  # 1 second


def _normalize(x: torch.Tensor, peak: float = 0.85) -> torch.Tensor:
    m = x.abs().max().clamp(min=1e-6)
    return x / m * peak


def _white(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


def synth_rain(seed: int) -> torch.Tensor:
    """Highpass hiss + dense random droplet impulses."""
    base = _white(NUM_SAMPLES, seed)
    base = AF.highpass_biquad(base, SAMPLE_RATE, cutoff_freq=1200.0)
    base = base * 0.35

    g = torch.Generator().manual_seed(seed + 1)
    droplets = torch.zeros(NUM_SAMPLES)
    n_drops = torch.randint(180, 260, (1,), generator=g).item()
    pos = torch.randint(0, NUM_SAMPLES - 32, (n_drops,), generator=g)
    amps = 0.4 + 0.6 * torch.rand(n_drops, generator=g)
    decay = torch.exp(-torch.linspace(0, 6, 32))
    noise_burst = torch.randn(n_drops, 32, generator=g) * decay
    for i in range(n_drops):
        p = int(pos[i])
        droplets[p:p + 32] += amps[i] * noise_burst[i]
    droplets = AF.highpass_biquad(droplets, SAMPLE_RATE, cutoff_freq=1500.0)

    out = base + droplets
    return _normalize(out)


def synth_wind(seed: int) -> torch.Tensor:
    """Lowpass noise modulated by slow LFO envelope."""
    base = _white(NUM_SAMPLES, seed)
    base = AF.lowpass_biquad(base, SAMPLE_RATE, cutoff_freq=600.0)
    base = AF.lowpass_biquad(base, SAMPLE_RATE, cutoff_freq=400.0)

    g = torch.Generator().manual_seed(seed + 7)
    lfo_freq = 0.6 + 1.4 * torch.rand(1, generator=g).item()  # 0.6-2 Hz
    phase = 2 * math.pi * torch.rand(1, generator=g).item()
    t = torch.linspace(0, 1, NUM_SAMPLES)
    env = 0.5 + 0.5 * torch.sin(2 * math.pi * lfo_freq * t + phase)

    out = base * env
    return _normalize(out)


def synth_thunder(seed: int) -> torch.Tensor:
    """Decaying low-freq rumble with mid-band crackle."""
    g = torch.Generator().manual_seed(seed + 13)
    rumble = _white(NUM_SAMPLES, seed + 21)
    rumble = AF.lowpass_biquad(rumble, SAMPLE_RATE, cutoff_freq=180.0)
    rumble = AF.lowpass_biquad(rumble, SAMPLE_RATE, cutoff_freq=180.0)

    # exponential decay envelope, with optional 1-2 sub-impulses
    t = torch.linspace(0, 1, NUM_SAMPLES)
    tau = 0.25 + 0.25 * torch.rand(1, generator=g).item()
    env = torch.exp(-t / tau)
    n_extra = torch.randint(1, 3, (1,), generator=g).item()
    for _ in range(n_extra):
        start = 0.1 + 0.5 * torch.rand(1, generator=g).item()
        sub = torch.exp(-(t - start).clamp(min=0) / 0.15)
        env = env + 0.6 * sub

    crackle = _white(NUM_SAMPLES, seed + 33)
    crackle = AF.bandpass_biquad(crackle, SAMPLE_RATE, central_freq=400.0, Q=0.7)
    crackle = crackle * 0.25

    out = (rumble * 1.4 + crackle) * env
    return _normalize(out)


SYNTHS = {
    "Rain": synth_rain,
    "Wind": synth_wind,
    "Thunder": synth_thunder,
}


def build_dataset(root: Path, per_concept: int = 8) -> None:
    manifest = []
    for concept, fn in SYNTHS.items():
        out_dir = root / "concept_data" / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_concept):
            digest = hashlib.sha256(f"{concept}:{i}".encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "little") % (2**31)
            wave = fn(seed).unsqueeze(0)  # [1, T]
            rel_path = f"concept_data/{concept}/{concept.lower()}_{i:02d}.wav"
            torchaudio.save(str(root / rel_path), wave, SAMPLE_RATE)
            manifest.append({"audio": rel_path, "concept": concept})

    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} clips across {len(SYNTHS)} concepts to {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize tiny audio dataset.")
    parser.add_argument("--root", default="data", help="Dataset root")
    parser.add_argument("--per-concept", type=int, default=8)
    args = parser.parse_args()
    build_dataset(Path(args.root), per_concept=args.per_concept)
