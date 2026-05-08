"""Stochastic edit-policy: takes a real audio clip + concept condition,
outputs a Gaussian residual distribution. The sampled residual is added to
the input and clipped, producing an *edited* real-audio output rather than
generating from scratch.

This is what makes REINFORCE meaningful here: the policy is genuinely
stochastic (mu + sigma * eps), so log-probs exist and rewards can drive it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, num_concepts: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(num_concepts, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.proj(cond)  # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class AudioEditPolicy(nn.Module):
    """Encoder-decoder over waveform that emits per-sample (mu, log_sigma).

    Output is `y = clip(input + residual)` where `residual ~ N(mu, sigma)`.
    Residual amplitudes are bounded via tanh * residual_scale so the policy
    cannot trivially zero out the input.
    """

    def __init__(
        self,
        num_concepts: int,
        hidden: int = 32,
        residual_scale: float = 0.6,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = -1.0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        # Encoder (stride 4 thrice -> 64x downsample): 16000 -> 250
        self.enc1 = nn.Conv1d(1, hidden, kernel_size=15, stride=4, padding=7)
        self.enc2 = nn.Conv1d(hidden, hidden * 2, kernel_size=15, stride=4, padding=7)
        self.enc3 = nn.Conv1d(hidden * 2, hidden * 4, kernel_size=15, stride=4, padding=7)
        self.film = FiLM(num_concepts, hidden * 4)

        # Decoder mirrors encoder
        self.dec3 = nn.ConvTranspose1d(hidden * 4, hidden * 2, kernel_size=16, stride=4, padding=6)
        self.dec2 = nn.ConvTranspose1d(hidden * 2, hidden, kernel_size=16, stride=4, padding=6)
        self.dec1 = nn.ConvTranspose1d(hidden, hidden, kernel_size=16, stride=4, padding=6)

        # Two heads: mu and log_sigma over the residual waveform
        self.head_mu = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
        self.head_log_sigma = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)

    def _params(self, audio: torch.Tensor, cond: torch.Tensor):
        x = F.gelu(self.enc1(audio))
        x = F.gelu(self.enc2(x))
        x = F.gelu(self.enc3(x))
        x = self.film(x, cond)
        x = F.gelu(self.dec3(x))
        x = F.gelu(self.dec2(x))
        x = F.gelu(self.dec1(x))

        mu = torch.tanh(self.head_mu(x)) * self.residual_scale
        log_sigma = self.head_log_sigma(x)
        log_sigma = self.log_sigma_min + 0.5 * (self.log_sigma_max - self.log_sigma_min) * (
            torch.tanh(log_sigma) + 1.0
        )

        # Match input length exactly
        T = audio.size(-1)
        mu = mu[..., :T]
        log_sigma = log_sigma[..., :T]
        return mu, log_sigma

    def forward(self, audio: torch.Tensor, cond: torch.Tensor):
        """Sample an edited audio and return (edited, log_prob_sum, mu, log_sigma).

        log_prob_sum is summed over time (per-sample scalar) for REINFORCE.
        """
        mu, log_sigma = self._params(audio, cond)
        sigma = log_sigma.exp()
        eps = torch.randn_like(mu)
        residual = mu + sigma * eps

        edited = (audio + residual).clamp(-1.0, 1.0)

        # Gaussian log-prob: -0.5 * ((eps)^2 + log(2pi)) - log_sigma
        log_prob = -0.5 * (eps.pow(2) + math.log(2 * math.pi)) - log_sigma
        log_prob_sum = log_prob.sum(dim=(1, 2))  # [B]
        return edited, log_prob_sum, mu, log_sigma

    @torch.no_grad()
    def act_mean(self, audio: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Deterministic edit (mean residual) for evaluation/sampling."""
        mu, _ = self._params(audio, cond)
        return (audio + mu).clamp(-1.0, 1.0)
