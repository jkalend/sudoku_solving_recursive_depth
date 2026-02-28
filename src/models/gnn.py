"""Graph Neural Network for Sudoku with fixed row/col/box adjacency."""

import torch
import torch.nn as nn


from .trm_base import init_weights


def _build_sudoku_adjacency() -> torch.Tensor:
    """Build normalized adjacency matrix (81, 81) for Sudoku peers.
    A[i, j] = 1/|peers(i)| if j is peer of i (same row, col, or box), else 0.
    """
    device = torch.device("cpu")  # Will be moved with model
    A = torch.zeros(81, 81, device=device)
    for i in range(81):
        ri, ci = i // 9, i % 9
        bi = (ri // 3) * 3 + (ci // 3)
        peers = []
        for j in range(81):
            if i == j:
                continue
            rj, cj = j // 9, j % 9
            bj = (rj // 3) * 3 + (cj // 3)
            if ri == rj or ci == cj or bi == bj:
                peers.append(j)
        for j in peers:
            A[i, j] = 1.0 / len(peers)
    return A


class SudokuGNN(nn.Module):
    """GNN over 81 cells with row/col/box adjacency. Outputs logits (B, 81, 10).
    Optional adaptive depth (ACT): run 1..act_max_steps layers, halt when ready."""

    def __init__(
        self,
        vocab_size: int = 10,
        seq_len: int = 81,
        d_model: int = 256,
        n_layers: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_act: bool = False,
        act_max_steps: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_act = use_act
        self.act_max_steps = act_max_steps

        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.given_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.register_buffer("A", _build_sudoku_adjacency())

        layers = []
        for _ in range(n_layers):
            layers.append(
                _GraphMessageLayer(d_model, hidden_dim, dropout)
            )
        self.layers = nn.ModuleList(layers)

        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        if use_act:
            self.halting_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, _mask: torch.Tensor | None = None) -> dict:
        """
        Args:
            x: (B, 81) integer tokens, 0=empty, 1-9=digit
        Returns:
            dict with "logits": (B, 81, 10), optionally "step_logits", "halting_probs", "n_steps"
        """
        h = self.embed(x) + self.pos_embed
        given = (x != 0).unsqueeze(-1).float()
        h = h + given * self.given_embed
        h = self.dropout(h)

        adj = self.A.to(h.device)
        max_steps = min(self.act_max_steps, self.n_layers) if self.use_act else self.n_layers

        step_logits_list = []
        halting_probs_list = []
        cumulative_halt = torch.zeros(x.shape[0], device=x.device)

        for step in range(max_steps):
            h = self.layers[step](h, adj)
            logits = self.head(h)
            step_logits_list.append(logits)

            if self.use_act:
                halt_prob = self.halting_head(h).squeeze(-1).mean(dim=1)
                halting_probs_list.append(halt_prob)
                cumulative_halt = cumulative_halt + halt_prob
                if (cumulative_halt >= 1.0).all():
                    break

        if self.use_act and halting_probs_list:
            weights = torch.stack(halting_probs_list, dim=1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            logits = sum(
                w.unsqueeze(-1).unsqueeze(-1) * step_logit
                for w, step_logit in zip(weights.t(), step_logits_list, strict=True)
            )
            return {
                "logits": logits,
                "step_logits": step_logits_list,
                "halting_probs": torch.stack(halting_probs_list, dim=1),
                "n_steps": len(halting_probs_list),
            }
        return {"logits": step_logits_list[-1], "step_logits": step_logits_list}


class _GraphMessageLayer(nn.Module):
    """One GNN layer: neighbor aggregation -> MLP -> residual + LayerNorm."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, 81, D), adj: (81, 81)
        # Aggregate: for each node i, mean of neighbors j
        agg = torch.einsum("ij,bjd->bid", adj, h)
        combined = self.norm(h + agg)
        return h + self.dropout(self.mlp(combined))
