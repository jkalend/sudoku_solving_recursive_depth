"""Base TRM components: think/revise blocks."""

import torch
import torch.nn as nn
from einops import rearrange


class ThinkReviseBlockAttention(nn.Module):
    """Attention-based: z attends to context, y attends to context."""

    def __init__(self, d_model: int, n_heads: int = 8, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.think_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.revise_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.think_proj = nn.Linear(d_model * 2, d_model)
        self.revise_proj = nn.Linear(d_model * 2, d_model)
        self.think_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.revise_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm_z = nn.LayerNorm(d_model)
        self.norm_y = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Think: z attends to projected concat(z, y)
        ctx = self.think_proj(torch.cat([z, y], dim=-1))
        z_new, _ = self.think_attn(z, ctx, ctx)
        z = self.norm_z(z + self.dropout(z_new))
        z = z + self.dropout(self.think_ff(z))

        # Revise: y attends to projected concat(y, z)
        ctx2 = self.revise_proj(torch.cat([y, z], dim=-1))
        y_new, _ = self.revise_attn(y, ctx2, ctx2)
        y = self.norm_y(y + self.dropout(y_new))
        y = y + self.dropout(self.revise_ff(y))
        return z, y


class ThinkReviseBlockMLP(nn.Module):
    """MLP-based: token + channel mixing (no attention)."""

    def __init__(self, d_model: int, seq_len: int = 81, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.think_channel = nn.Sequential(
            nn.Linear(d_model * 2, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.think_token = nn.Linear(seq_len, seq_len)
        self.revise_channel = nn.Sequential(
            nn.Linear(d_model * 2, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.revise_token = nn.Linear(seq_len, seq_len)
        self.norm_z = nn.LayerNorm(d_model)
        self.norm_y = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Think: z = z + channel_mix(concat(z,y)) + token_mix(z)
        zy = torch.cat([z, y], dim=-1)
        z = self.norm_z(z + self.dropout(self.think_channel(zy)))
        z = z + self.dropout(rearrange(self.think_token(rearrange(z, "b s d -> b d s")), "b d s -> b s d"))

        # Revise: y = y + channel_mix(concat(y,z)) + token_mix(y)
        yz = torch.cat([y, z], dim=-1)
        y = self.norm_y(y + self.dropout(self.revise_channel(yz)))
        y = y + self.dropout(rearrange(self.revise_token(rearrange(y, "b s d -> b d s")), "b d s -> b s d"))
        return z, y
