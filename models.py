"""
PyTorch Sequential Model Architectures for Dynamic Gesture Recognition.

Contains lightweight models (GRU, LSTM, Temporal Convolutional Network - TCN)
tailored for low-latency edge/CPU inference on sequence landmark representations.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PipelineConfig


class TemporalAttention(nn.Module):
    """
    Lightweight temporal self-attention pooling mechanism.
    Weights intermediate frame representations rather than naively
    relying solely on the final hidden state.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, rnn_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # rnn_outputs: (B, T, H)
        scores = self.projection(rnn_outputs)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)     # (B, T, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (B, H)
        return context, weights


class GestureGRU(nn.Module):
    """
    Lightweight GRU-based sequence classifier with temporal attention pooling.
    Gated Recurrent Units have 25% fewer parameters and gating operations
    than standard LSTMs, maximizing inference throughput on CPU.
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Optional linear projection to ease gradient flow
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttention(gru_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gesture classification.

        Args:
            x: Input sequence tensor of shape (Batch_size, Sequence_length, Input_dim).

        Returns:
            Logits tensor of shape (Batch_size, Num_classes).
        """
        # x: (B, T, D)
        proj = self.input_fc(x)  # (B, T, H)
        out, _ = self.gru(proj)  # (B, T, H)
        context, _ = self.attention(out)  # (B, H)
        logits = self.classifier(context)  # (B, C)
        return logits


class GestureLSTM(nn.Module):
    """
    Lightweight LSTM sequence classifier with temporal attention.
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttention(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.input_fc(x)
        out, _ = self.lstm(proj)
        context, _ = self.attention(out)
        logits = self.classifier(context)
        return logits


class TemporalConvBlock(nn.Module):
    """Dilated 1D Causal Convolutional Block with Residual Connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        res = self.residual(x)

        # First conv + causal trim
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = F.relu(self.norm1(out))
        out = self.dropout(out)

        # Second conv + causal trim
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm2(out)
        out = self.dropout(out)

        return F.relu(out + res)


class GestureTCN(nn.Module):
    """
    Temporal Convolutional Network (TCN).
    Replaces recurrent steps with dilated 1D convolutions for parallelizable,
    constant-time temporal processing.
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 64,
        num_classes: int = 8,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        in_c = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            blocks.append(
                TemporalConvBlock(
                    in_channels=in_c,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_c = hidden_dim

        self.tcn = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, D) -> transpose to (B, D, T) for 1D convolutions
        x = x.transpose(1, 2)
        feats = self.tcn(x)  # (B, H, T)
        pooled = self.global_pool(feats).squeeze(-1)  # (B, H)
        logits = self.classifier(pooled)
        return logits


def build_model(config: Optional[PipelineConfig] = None) -> nn.Module:
    """
    Factory function to instantiate the configured sequence model.
    """
    cfg = config or PipelineConfig()
    m_type = cfg.model_type.lower()

    if m_type == "gru":
        return GestureGRU(
            input_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
    elif m_type == "lstm":
        return GestureLSTM(
            input_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
    elif m_type == "tcn":
        return GestureTCN(
            input_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=cfg.num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{cfg.model_type}'. Choose 'gru', 'lstm', or 'tcn'.")

