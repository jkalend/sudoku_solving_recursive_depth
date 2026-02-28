# TRM Sudoku Evaluation

Evaluate **Architectural Scaling vs. Recursive Depth** in Tiny Recursive Models (TRMs) on Sudoku-Extreme.

## Setup

```bash
# Create venv
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows

# Install PyTorch with CUDA 13.0 (RTX 4090)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install -r requirements.txt
```

Or run `install_cuda.bat` with venv activated.

## Usage

**Train all models** (baseline, TRM-MLP, TRM-Attention, + ACT variants):
```bash
python main.py --act --quick   # 5 epochs, quick test
python main.py --act          # 50 epochs, full training
```

**Train a single model**:
```bash
python main.py --model baseline --quick
python main.py --model trm_mlp --act
python main.py --model trm_attention_xl --act   # Recommended for ACT demo
python main.py --model gnn --quick              # GNN + legality-constrained decoding
```

**Evaluate and profile** (after training):
```bash
python run_eval_only.py
```

**Full pipeline** (train + eval + profile + summary):
```bash
python run_all.py --quick
```

## Models

| Model | Params | Description |
|-------|--------|-------------|
| Baseline | ~12.7M | 4-layer Transformer |
| TRM-MLP | ~6M | 2-layer MLP, token+channel mixing |
| TRM-Attention | ~7.5M | 2-layer cross-attention think/revise |
| **TRM-Attention-XL** | ~15M | 4-block, d_model=512, ff_dim=2048 — better for ACT demo |
| **GNN** | ~5M | Graph over 81 cells, row/col/box adjacency, legality-constrained decoding |
| **GNN-ACT** | ~5M | GNN with adaptive depth (1–6 layers), early halting |
| + ACT | - | Adaptive Computation Time, early halting |

## Dataset

[sapientinc/sudoku-extreme-1k](https://huggingface.co/datasets/sapientinc/sudoku-extreme-1k): 1k train, 20k test_hard.

## Outputs

- `checkpoints/` - trained model weights
- `outputs/` - PROJECT_SUMMARY.md, act_step_distribution.png

## Notes

- If you change model config (e.g. trm_d_model), delete old checkpoints and retrain.
- TRM-MLP ~6M params, TRM-Attention ~7.5M params (per plan targets).
