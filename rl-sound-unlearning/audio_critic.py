import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCritic(nn.Module):
    """A small 1D critic network that predicts concept presence in audio."""

    def __init__(self, num_concepts: int, hidden_channels: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels * 4, num_concepts)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.backbone(audio)
        x = x.squeeze(-1)
        return self.classifier(x)

    def predict_proba(self, audio: torch.Tensor) -> torch.Tensor:
        return F.softmax(self(audio), dim=-1)
