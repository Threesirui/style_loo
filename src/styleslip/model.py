"""Compact temporal CNN for the three Style-LOO channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


def _groups(channels: int) -> int:
    for value in range(min(8, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.layers(inputs))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    in_channels: int = 3
    channels: int = 64
    depth: int = 3
    kernel_size: int = 5
    dropout: float = 0.25
    classifier_hidden: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StyleWaveTCN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.in_channels < 1 or config.channels < 1 or config.depth < 1:
            raise ValueError("model channel counts and depth must be positive")
        if config.kernel_size < 3 or config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.in_channels, config.channels, 7, padding=3, bias=False),
            nn.GroupNorm(_groups(config.channels), config.channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(
                config.channels,
                config.kernel_size,
                dilation=(1, 2, 4)[index % 3],
                dropout=config.dropout,
            )
            for index in range(config.depth)
        )
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * config.channels),
            nn.Linear(2 * config.channels, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, 1),
        )

    def encode(self, waves: Tensor) -> Tensor:
        if waves.ndim != 3 or waves.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected [batch,{self.config.in_channels},length], got {tuple(waves.shape)}"
            )
        features = self.stem(waves)
        for block in self.blocks:
            features = block(features)
        return torch.cat(
            [self.average_pool(features).squeeze(-1), torch.amax(features, dim=-1)], dim=1
        )

    def forward(self, waves: Tensor) -> Tensor:
        return self.classifier(self.encode(waves)).squeeze(-1)


__all__ = ["ModelConfig", "StyleWaveTCN"]

