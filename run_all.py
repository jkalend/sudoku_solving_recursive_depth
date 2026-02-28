"""Run full pipeline: train all models, evaluate, profile, write summary."""

import sys
import argparse
from pathlib import Path

from src.config import Config
from src.train import train_baseline, train_trm, train_gnn, evaluate
from src.data import get_dataloaders
from src.evaluate import run_evaluation, analyze_act_steps, load_model
from src.profile import run_profiling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use full sudoku-extreme dataset")
    parser.add_argument("--quick", action="store_true", help="Quick run: 3 epochs")
    args = parser.parse_args()

    config = Config()
    if args.full:
        config.dataset_name = "sapientinc/sudoku-extreme"
    results = {}

    # 1. Train all models (quick: 3 epochs for testing)
    quick = args.quick
    epochs = 3 if quick else 50

    print("=" * 60)
    print("1. Training Baseline")
    print("=" * 60)
    train_baseline(config, epochs=epochs)

    print("\n" + "=" * 60)
    print("2. Training TRM-MLP")
    print("=" * 60)
    train_trm(config, model_type="mlp", use_act=False, epochs=epochs)

    print("\n" + "=" * 60)
    print("3. Training TRM-Attention")
    print("=" * 60)
    train_trm(config, model_type="attention", use_act=False, epochs=epochs)

    print("\n" + "=" * 60)
    print("4. Training TRM-MLP with ACT")
    print("=" * 60)
    train_trm(config, model_type="mlp", use_act=True, epochs=epochs)

    print("\n" + "=" * 60)
    print("5. Training TRM-Attention with ACT")
    print("=" * 60)
    train_trm(config, model_type="attention", use_act=True, epochs=epochs)

    print("\n" + "=" * 60)
    print("6. Training TRM-Attention-XL")
    print("=" * 60)
    train_trm(config, model_type="attention_xl", use_act=False, epochs=epochs)

    print("\n" + "=" * 60)
    print("7. Training TRM-Attention-XL with ACT")
    print("=" * 60)
    train_trm(config, model_type="attention_xl", use_act=True, epochs=epochs)

    print("\n" + "=" * 60)
    print("8. Training GNN")
    print("=" * 60)
    train_gnn(config, use_act=False, epochs=epochs)

    print("\n" + "=" * 60)
    print("9. Training GNN with ACT")
    print("=" * 60)
    train_gnn(config, use_act=True, epochs=epochs)

    # 10. Evaluation
    print("\n" + "=" * 60)
    print("10. Evaluation")
    print("=" * 60)
    results["accuracy"] = run_evaluation(config)

    # 11. ACT step analysis
    print("\n" + "=" * 60)
    print("11. ACT Step Distribution")
    print("=" * 60)
    try:
        analyze_act_steps(config, model_type="attention", n_samples=500)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"ACT analysis failed: {e}")

    # 12. Profiling
    print("\n" + "=" * 60)
    print("12. Profiling")
    print("=" * 60)
    try:
        results["profiling"] = run_profiling(config)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Profiling failed: {e}")
        results["profiling"] = []

    # 13. Write summary
    write_summary(config, results)


def write_summary(config: Config, results: dict):
    """Write PROJECT_SUMMARY.md."""
    out = config.output_dir / "PROJECT_SUMMARY.md"
    lines = [
        "# TRM Sudoku Evaluation - Project Summary",
        "",
        "## Overview",
        f"This project evaluates **Architectural Scaling vs. Recursive Depth** in Tiny Recursive Models (TRMs) on the {config.dataset_name} dataset.",
        "",
        "## Implementation Steps",
        "",
        "### 1. Environment & Data Setup",
        "- **venv**: Python virtual environment with PyTorch, HuggingFace datasets, einops, ptflops",
        f"- **Dataset**: {config.dataset_name}",
        "- **Tokenization**: 81-char strings → 81 int tokens (0=empty, 1-9=digits)",
        "- **Fallback**: CSV loaded via huggingface_hub when datasets library has overflow issues",
        "",
        "### 2. Baseline Transformer",
        "- 4-layer feed-forward Transformer, ~12.7M parameters",
        "- d_model=512, n_heads=8, ff_dim=2048",
        "- Standard cross-entropy on empty cells only",
        "",
        "### 3. TRM Architectures",
        "- **MLP-TRM**: 2-layer, token+channel mixing, ~1.6M params",
        "- **Attention-TRM**: 2-layer, cross-attention think/revise, ~2.7M params",
        "- Both unroll 2 steps to match 4-layer baseline depth",
        "- State: y (solution embedding), z (latent scratchpad)",
        "",
        "### 4. Deep Supervision",
        "- Loss at each recursive step (step_logits)",
        "- Averages loss across steps for gradient flow",
        "",
        "### 5. Adaptive Computation Time (ACT)",
        "- Halting head outputs probability per step",
        "- Cumulative halt ≥ 1.0 → stop",
        "- Ponder cost penalty encourages fewer steps on easy puzzles",
        "- Weighted average of step outputs by halting probs",
        "",
        "### 6. Evaluation & Profiling",
        "- Exact accuracy: all 81 digits must match",
        "- Latency: batch size 1, median over 1000 runs",
        "- FLOPs: ptflops (when available)",
        "- ACT step distribution vs. puzzle difficulty",
        "",
        "## Results",
        "",
    ]
    if results.get("accuracy"):
        lines.append("### Accuracy")
        for name, acc in results["accuracy"].items():
            lines.append(f"- **{name}**: {acc:.4f}" if acc is not None else f"- **{name}**: N/A")
        lines.append("")
    if results.get("profiling") and hasattr(results["profiling"], "__iter__"):
        lines.append("### Profiling")
        for r in results["profiling"]:
            name = r.get("name", "<unknown>")
            lines.append(f"- **{name}**: latency={r.get('latency_ms')}ms, params={r.get('params')}")
        lines.append("")
    lines.extend([
        "## Architectural Choices",
        "",
        "- **Baseline**: Traditional scaling (depth) for comparison",
        "- **MLP-TRM**: Parameter-efficient, no attention, faster",
        "- **Attention-TRM**: Better routing, more expressive",
        "- **ACT**: Dynamic compute for easy vs. hard puzzles",
        "",
        "## Hardware",
        "- Target: RTX 4090 (24GB VRAM), 32GB RAM",
        "- Falls back to CPU if CUDA unavailable",
        "- For CUDA: install `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
