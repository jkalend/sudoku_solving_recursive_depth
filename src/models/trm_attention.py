"""Attention-based Tiny Recursive Model (~7M params)."""

import torch
import torch.nn as nn

from .trm_base import ThinkReviseBlockAttention, init_weights


class TRMAttention(nn.Module):
    """2-layer Attention-based TRM, unrolled to match 4-layer baseline depth."""

    def __init__(
        self,
        vocab_size: int = 10,
        seq_len: int = 81,
        d_model: int = 256,
        n_heads: int = 8,
        ff_dim: int = 512,
        n_steps: int = 2,
        dropout: float = 0.1,
        use_act: bool = False,
        act_max_steps: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_steps = n_steps
        self.use_act = use_act
        self.act_max_steps = act_max_steps

        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.block1 = ThinkReviseBlockAttention(d_model, n_heads, ff_dim, dropout)
        self.block2 = ThinkReviseBlockAttention(d_model, n_heads, ff_dim, dropout)

        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        if use_act:
            self.halting_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        self.apply(init_weights)

    def _one_step(self, z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, y = self.block1(z, y)
        z, y = self.block2(z, y)
        return z, y

    def forward(
        self,
        x: torch.Tensor,
        return_steps: bool = False,
    ) -> dict:
        """
        Args:
            x: (B, 81) input puzzle
        Returns:
            dict with 'logits', optionally 'step_logits', 'halting_probs', 'n_steps'
        """
        B = x.shape[0]
        h = self.embed(x) + self.pos_embed
        h = self.dropout(h)

        y = h.clone()
        z = h.clone()

        step_logits_list = []
        halting_probs_list = []
        cumulative_halt = torch.zeros(B, device=x.device)

        max_steps = self.act_max_steps if self.use_act else self.n_steps
        if max_steps == 0:
            logits = self.head(y)
            out = {"logits": logits}
            if return_steps:
                out["step_logits"] = []
            return out

        n_steps_used = 0

        for step in range(max_steps):
            z, y = self._one_step(z, y)
            logits = self.head(y)
            step_logits_list.append(logits)

            if self.use_act:
                halt_input = torch.cat([z, y], dim=-1)
                halt_prob = self.halting_head(halt_input).squeeze(-1).mean(dim=1)
                halting_probs_list.append(halt_prob)
                cumulative_halt = cumulative_halt + halt_prob
                n_steps_used = step + 1
                if (cumulative_halt >= 1.0).all():
                    break

        if self.use_act and halting_probs_list:
            weights = torch.stack(halting_probs_list, dim=1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            logits = sum(w.unsqueeze(-1).unsqueeze(-1) * step_logit for w, step_logit in zip(weights.t(), step_logits_list, strict=True))
            out = {
                "logits": logits,
                "halting_probs": torch.stack(halting_probs_list, dim=1),
                "n_steps": n_steps_used,
            }
            if return_steps:
                out["step_logits"] = step_logits_list
        else:
            out = {"logits": step_logits_list[-1]}
            if return_steps:
                out["step_logits"] = step_logits_list

        return out
