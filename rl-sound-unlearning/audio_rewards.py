"""Multi-term reward for the audio edit-policy.

Components (all in roughly [0, 1] before weighting):
  - unlearn:        critic should *not* classify edited audio as the target concept.
  - realism:        rms within a sane envelope (penalizes silence and clipping).
  - spec_entropy:   spectral entropy floor (penalizes DC, single tones, near-silence).
  - anti_periodic:  penalize strong autocorrelation peaks beyond ~10 ms (catches
                    looped/repeated content -- the "saying the same word" failure mode).
  - in_batch_div:   penalize cosine similarity between log-magnitude spectra in a batch
                    (catches mode collapse to one output).

The total reward is a weighted sum; all terms are detached from the policy
(REINFORCE only needs the scalar return).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RewardWeights:
    unlearn: float = 1.0
    realism: float = 0.4
    spec_entropy: float = 0.4
    anti_periodic: float = 0.5
    in_batch_div: float = 0.4
    retain_cls: float = 1.0
    retain_audio: float = 0.4


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(dim=(1, 2)).clamp(min=1e-8).sqrt()


def realism_reward(audio: torch.Tensor, low: float = 0.05, high: float = 0.5) -> torch.Tensor:
    """1.0 when rms is comfortably inside [low, high], decaying outside."""
    r = _rms(audio)
    # Triangular plateau: 1 inside, linear falloff outside up to 2x band width.
    band = high - low
    below = (r - low).clamp(min=-band) / band  # in [-1, 0] when below
    above = (high - r).clamp(min=-band) / band  # in [-1, 0] when above
    score = torch.minimum(below.clamp(max=0) + 1.0, above.clamp(max=0) + 1.0)
    return score.clamp(0.0, 1.0)


def _spectrum(audio: torch.Tensor, n_fft: int = 512) -> torch.Tensor:
    # audio: [B, 1, T] -> mag spectrum [B, F, frames]
    x = audio.squeeze(1)
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4,
                      window=window, return_complex=True, center=True)
    return spec.abs()


def spectral_entropy_reward(audio: torch.Tensor) -> torch.Tensor:
    """Higher when energy is spread across the spectrum (not silence/single tone)."""
    mag = _spectrum(audio)  # [B, F, frames]
    p = mag.mean(dim=-1)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    ent = -(p * (p + 1e-8).log()).sum(dim=-1)
    max_ent = torch.log(torch.tensor(p.size(-1), device=audio.device, dtype=audio.dtype))
    return (ent / max_ent).clamp(0.0, 1.0)


def anti_periodicity_reward(audio: torch.Tensor, sample_rate: int = 16000,
                             min_lag_ms: float = 10.0) -> torch.Tensor:
    """1 - max(|autocorr(audio, lag>=min_lag)|). Penalizes loops/repeated phrases."""
    x = audio.squeeze(1)
    x = x - x.mean(dim=-1, keepdim=True)
    norm = x.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    # Autocorrelation via FFT (linear, zero-padded).
    T = x.size(-1)
    n = 1
    while n < 2 * T:
        n *= 2
    Xf = torch.fft.rfft(x, n=n)
    ac = torch.fft.irfft(Xf * Xf.conj(), n=n)[..., :T]
    ac = ac / norm  # normalized to [-1, 1]; ac[..., 0] == 1
    min_lag = max(1, int(min_lag_ms * 1e-3 * sample_rate))
    peak = ac[..., min_lag:].abs().max(dim=-1).values
    return (1.0 - peak).clamp(0.0, 1.0)


def in_batch_diversity_reward(audio: torch.Tensor) -> torch.Tensor:
    """For each item, 1 - max cos-sim to any other item's log-mag spectrum.

    Cosine similarity is in [-1, 1]; (1 - sim) * 0.5 maps it to [0, 1] where
    1.0 means "fully orthogonal to every other item" and 0.0 means "identical".
    """
    mag = _spectrum(audio).mean(dim=-1)  # [B, F]
    mag = (mag + 1e-6).log()
    feats = F.normalize(mag, dim=-1)
    sim = feats @ feats.t()  # [B, B]
    sim.fill_diagonal_(-1.0)
    if sim.size(0) == 1:
        # No peers => no diversity signal; return zeros so this term doesn't
        # silently dominate single-sample batches.
        return torch.zeros(1, device=audio.device)
    peer = sim.max(dim=-1).values  # in roughly [-1, 1]
    return ((1.0 - peer) * 0.5).clamp(0.0, 1.0)


def compute_rewards(critic, edited: torch.Tensor, target_idx: int,
                    weights: RewardWeights, sample_rate: int = 16000,
                    original: torch.Tensor | None = None,
                    labels: torch.Tensor | None = None):
    """Returns (total_reward [B], components dict for logging).

    All tensors are returned detached -- REINFORCE multiplies by log_prob.
    """
    with torch.no_grad():
        probs = F.softmax(critic(edited), dim=-1)
        unlearn = (1.0 - probs[:, target_idx]).clamp(0.0, 1.0)

        realism = realism_reward(edited)
        spec_ent = spectral_entropy_reward(edited)
        anti_per = anti_periodicity_reward(edited, sample_rate=sample_rate)
        diversity = in_batch_diversity_reward(edited)

        total = (
            weights.unlearn * unlearn
            + weights.realism * realism
            + weights.spec_entropy * spec_ent
            + weights.anti_periodic * anti_per
            + weights.in_batch_div * diversity
        )
        retain_cls = torch.zeros_like(unlearn)
        retain_audio = torch.zeros_like(unlearn)
        if original is not None and labels is not None:
            non_target = labels != target_idx
            if non_target.any():
                retain_cls[non_target] = probs[non_target, labels[non_target]].clamp(0.0, 1.0)
                edit_rms = (edited - original).pow(2).mean(dim=(1, 2)).sqrt()
                retain_audio[non_target] = (1.0 - edit_rms[non_target] / 0.2).clamp(0.0, 1.0)
                total = total + weights.retain_cls * retain_cls + weights.retain_audio * retain_audio

    components = {
        "unlearn": unlearn,
        "realism": realism,
        "spec_entropy": spec_ent,
        "anti_periodic": anti_per,
        "diversity": diversity,
        "retain_cls": retain_cls,
        "retain_audio": retain_audio,
    }
    return total.detach(), components


# Back-compat shim: kept so legacy callers don't break, but no longer used.
def concept_unlearning_reward(critic, audio: torch.Tensor, target_concept_idx: int) -> torch.Tensor:
    probs = F.softmax(critic(audio), dim=-1)
    return (1.0 - probs[:, target_concept_idx]).clamp(0.0, 1.0)
