"""Configuration for TRM Sudoku evaluation project."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


def _get_device() -> str:
    """Use CUDA if available for models and data in VRAM."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    """Training and model configuration."""

    # Data: "sapientinc/sudoku-extreme" (full ~3.8M train) or "sapientinc/sudoku-extreme-1k" (1k train)
    dataset_name: str = "sapientinc/sudoku-extreme-1k"
    batch_size: int = 32
    num_workers: int = 0  # 0 for Windows compatibility

    # Model - Baseline 4-layer Transformer (~14M params)
    baseline_layers: int = 4
    baseline_d_model: int = 512
    baseline_heads: int = 8
    baseline_ff_dim: int = 2048
    baseline_dropout: float = 0.1

    # Model - TRM (2-layer, unrolled 2x) - MLP ~5M, Attention ~7M params
    trm_layers: int = 2
    trm_d_model: int = 384
    trm_heads: int = 8
    trm_ff_dim: int = 1280
    trm_dropout: float = 0.1
    trm_max_steps: int = 2  # Unroll exactly twice to match 4-layer baseline

    # ACT (Adaptive Computation Time)
    act_max_steps: int = 4
    act_ponder_cost: float = 0.01
    act_threshold: float = 1.0

    # TRM-XL: larger model for better ACT demonstration (~15M params)
    trm_xl_d_model: int = 512
    trm_xl_ff_dim: int = 2048
    trm_xl_blocks: int = 4
    trm_xl_n_steps: int = 3
    trm_xl_act_max_steps: int = 6

    # GNN: graph over 81 cells, row/col/box adjacency
    gnn_d_model: int = 256
    gnn_layers: int = 4
    gnn_hidden: int = 512
    gnn_dropout: float = 0.1
    gnn_decode_max_steps: int = 81
    gnn_act_max_steps: int = 6  # Adaptive depth: max GNN layers when use_act=True

    # Sudoku
    seq_len: int = 81
    vocab_size: int = 10  # 0=empty, 1-9=digits

    # Training
    lr: float = 1e-4
    epochs: int = 50
    device: Optional[str] = None  # Set to cuda/cpu in __post_init__ when not explicitly provided

    # Paths
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")

    def __post_init__(self):
        if self.device is None:
            self.device = _get_device()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
