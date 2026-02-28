"""4-layer feed-forward Transformer baseline (~14M params)."""

import torch
import torch.nn as nn


class BaselineTransformer(nn.Module):
    """Standard 4-layer Transformer for Sudoku prediction."""

    def __init__(
        self,
        vocab_size: int = 10,
        seq_len: int = 81,
        d_model: int = 256,
        n_heads: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)  # +1 for padding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, 81) integer tokens, 0=empty
            mask: (B, 81) bool, True = predict (empty cell)
        Returns:
            logits: (B, 81, 10) for digits 0-9 (0 = empty, 1-9 = filled)
        """
        h = self.embed(x) + self.pos_embed
        h = self.dropout(h)

        # Transformer src_key_padding_mask=True means ignore that KEY position.
        # mask=True marks empty cells (to predict); given/clue cells (mask=False)
        # should remain visible as context, so we pass mask directly so only
        # empty-cell positions are suppressed as keys.
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = mask.bool()

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(h)
        return logits
