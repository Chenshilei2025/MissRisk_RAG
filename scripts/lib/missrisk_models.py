from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def pooled(encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
    return masked_mean(outputs.last_hidden_state, attention_mask)


class DetectabilityModel(nn.Module):
    """Model B: P(D=1 | B=1, q, o, u, s)."""

    def __init__(self, encoder_name: str, n_state_feats: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.n_state_feats = n_state_feats
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + n_state_feats, 1)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        representation = self.dropout(pooled(self.encoder, input_ids, attention_mask))
        if self.n_state_feats:
            if state_feats is None:
                state_feats = representation.new_zeros((representation.shape[0], self.n_state_feats))
            representation = torch.cat([representation, state_feats.to(representation.dtype)], dim=-1)
        return representation

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.encode(input_ids, attention_mask, state_feats)).squeeze(-1)


class JointMissRiskModel(nn.Module):
    """Model C: shared encoder with bear/detect/miss heads.

    The miss head is trained directly on joint labels y=1 iff B=1 and D=0.
    """

    def __init__(self, encoder_name: str, n_state_feats: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.n_state_feats = n_state_feats
        self.dropout = nn.Dropout(dropout)
        width = hidden + n_state_feats
        self.bear_head = nn.Linear(width, 1)
        self.detect_head = nn.Linear(width, 1)
        self.miss_head = nn.Linear(width, 1)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        representation = self.dropout(pooled(self.encoder, input_ids, attention_mask))
        if self.n_state_feats:
            if state_feats is None:
                state_feats = representation.new_zeros((representation.shape[0], self.n_state_feats))
            representation = torch.cat([representation, state_feats.to(representation.dtype)], dim=-1)
        return representation

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.encode(input_ids, attention_mask, state_feats)
        return (
            self.bear_head(representation).squeeze(-1),
            self.detect_head(representation).squeeze(-1),
            self.miss_head(representation).squeeze(-1),
        )

    def warm_start_from_detectability(
        self,
        detectability_model: DetectabilityModel,
        *,
        copy_encoder: bool = True,
    ) -> None:
        with torch.no_grad():
            if copy_encoder:
                self.encoder.load_state_dict(detectability_model.encoder.state_dict())
            if self.detect_head.weight.shape == detectability_model.head.weight.shape:
                self.detect_head.weight.copy_(detectability_model.head.weight)
                self.detect_head.bias.copy_(detectability_model.head.bias)
