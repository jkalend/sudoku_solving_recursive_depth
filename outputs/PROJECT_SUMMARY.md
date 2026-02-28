# TRM Sudoku Evaluation - Project Summary

## Overview
This project evaluates **Architectural Scaling vs. Recursive Depth** in Tiny Recursive Models (TRMs) on the Sudoku-Extreme-1k dataset.

## Implementation Steps

### 1. Environment & Data Setup
- **venv**: Python virtual environment with PyTorch, HuggingFace datasets, einops, ptflops
- **Dataset**: sapientinc/sudoku-extreme-1k (1k train, 20k test_hard)
- **Tokenization**: 81-char strings → 81 int tokens (0=empty, 1-9=digits)
- **Fallback**: CSV loaded via huggingface_hub when datasets library has overflow issues

### 2. Baseline Transformer
- 4-layer feed-forward Transformer, ~12.7M parameters
- d_model=512, n_heads=8, ff_dim=2048
- Standard cross-entropy on empty cells only

### 3. TRM Architectures
- **MLP-TRM**: 2-layer, token+channel mixing, ~6M params
- **Attention-TRM**: 2-layer, cross-attention think/revise, ~7.5M params
- **Attention-TRM-XL**: 4-block, d_model=512, ff_dim=2048, ~15M params — recommended for ACT demo
- Both standard TRMs unroll 2 steps to match 4-layer baseline depth
- State: y (solution embedding), z (latent scratchpad)

### 3b. GNN Architecture
- **GNN**: ~5M params, graph over 81 cells with row/col/box adjacency, legality-constrained decoding
- **GNN-ACT**: GNN with adaptive depth (1–6 layers), early halting via ACT halting head

### 4. Deep Supervision
- Loss at each recursive step (step_logits)
- Averages loss across steps for gradient flow

### 5. Adaptive Computation Time (ACT)
- Halting head outputs probability per step
- Cumulative halt ≥ 1.0 → stop
- Ponder cost penalty encourages fewer steps on easy puzzles
- Weighted average of step outputs by halting probs

### 6. Evaluation & Profiling
- Exact accuracy: all 81 digits must match
- Latency: batch size 1, median over 1000 runs
- FLOPs: ptflops (when available)
- ACT step distribution vs. puzzle difficulty

## Results

### Accuracy
- **Baseline**: 0.0000
- **TRM-MLP**: 0.0000
- **TRM-Attention**: 0.0000
- **TRM-Attention-XL**: 0.0000
- **TRM-MLP-ACT**: 0.0000
- **TRM-Attention-ACT**: 0.0000
- **GNN**: 0.0000
- **GNN-ACT**: 0.0000

### Profiling
- **Baseline**: latency=2.0528500026557595ms, params=12661770
- **TRM-MLP**: latency=3.8155500078573823ms, params=5973714
- **TRM-Attention**: latency=7.898150011897087ms, params=7527690
- **TRM-Attention-XL**: latency=N/A, params=~15000000
- **TRM-MLP-ACT**: latency=10.651050019077957ms, params=6269395
- **TRM-Attention-ACT**: latency=9.353350003948435ms, params=7823371
- **GNN**: latency=N/A, params=~5000000
- **GNN-ACT**: latency=N/A, params=~5000000

## Architectural Choices

- **Baseline**: Traditional scaling (depth) for comparison
- **MLP-TRM**: Parameter-efficient, no attention, faster
- **Attention-TRM**: Better routing, more expressive
- **Attention-TRM-XL**: Larger capacity version (4 blocks, 512 dim), best ACT demo candidate
- **GNN**: Inductive bias via graph structure (rows/cols/boxes), legality-constrained decoding enforces valid Sudoku digits
- **GNN-ACT**: Combines GNN inductive bias with adaptive depth (1–6 message-passing rounds)
- **ACT**: Dynamic compute allocation — fewer steps on easy puzzles, more on hard ones

## Hardware
- Target: RTX 4090 (24GB VRAM), 32GB RAM
- Falls back to CPU if CUDA unavailable
- For CUDA: install `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
