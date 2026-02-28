"""Run evaluation, profiling, and summary only (requires trained checkpoints)."""

import sys
from pathlib import Path
from src.config import Config
from src.evaluate import run_evaluation, analyze_act_steps
from src.profile import run_profiling


def write_summary(config, results: dict):
    """Write PROJECT_SUMMARY.md."""
    out = config.output_dir / "PROJECT_SUMMARY.md"
    lines = [
        "# TRM Sudoku Evaluation - Project Summary",
        "",
        "## Overview",
        "This project evaluates **Architectural Scaling vs. Recursive Depth** in Tiny Recursive Models (TRMs) on the Sudoku-Extreme-1k dataset.",
        "",
        "## Implementation Steps",
        "",
        "### 1. Environment & Data Setup",
        "- **venv**: Python virtual environment with PyTorch, HuggingFace datasets, einops, ptflops",
        "- **Dataset**: sapientinc/sudoku-extreme-1k (1k train, 20k test_hard)",
        "- **Tokenization**: 81-char strings → 81 int tokens (0=empty, 1-9=digits)",
        "- **Fallback**: CSV loaded via huggingface_hub when datasets library has overflow issues",
        "",
        "### 2. Baseline Transformer",
        "- 4-layer feed-forward Transformer, ~12.7M parameters",
        "- d_model=512, n_heads=8, ff_dim=2048",
        "- Standard cross-entropy on empty cells only",
        "",
        "### 3. TRM Architectures",
        "- **MLP-TRM**: 2-layer, token+channel mixing, ~6M params",
        "- **Attention-TRM**: 2-layer, cross-attention think/revise, ~7.5M params",
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
    if results.get("profiling"):
        lines.append("### Profiling")
        for r in results["profiling"]:
            lines.append(f"- **{r['name']}**: latency={r.get('latency_ms')}ms, params={r.get('params')}")
        lines.append("")
    lines.extend([
        "## Architectural Choices",
        "",
        "- **Baseline**: Traditional scaling (depth) for comparison",
        "- **MLP-TRM**: Parameter-efficient, no attention, faster",
        "- **Attention-TRM**: Better routing, more expressive",
        "- **ACT**: Dynamic compute for easy vs. hard puzzles",
        "- **GNN**: Graph over 81 cells, row/col/box adjacency, legality-constrained decoding",
        "",
        "## Hardware",
        "- Target: RTX 4090 (24GB VRAM), 32GB RAM",
        "- Falls back to CPU if CUDA unavailable",
        "- For CUDA: install `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out}")


def main():
    config = Config()
    if "--full" in sys.argv:
        config.dataset_name = "sapientinc/sudoku-extreme"
    results = {}

    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    results["accuracy"] = run_evaluation(config)

    print("\n" + "=" * 60)
    print("ACT Step Distribution")
    print("=" * 60)
    for act_model in ["attention_xl", "attention", "mlp", "gnn"]:
        try:
            analyze_act_steps(config, model_type=act_model, n_samples=500)
            break
        except Exception as e:
            print(f"ACT analysis ({act_model}): {e}")

    print("\n" + "=" * 60)
    print("Profiling")
    print("=" * 60)
    try:
        results["profiling"] = run_profiling(config)
        if len(results.get("profiling", [])) == 0 and config.device == "cuda":
            print("All profiling failed (likely CUDA mismatch). Retrying on CPU...")
            results["profiling"] = run_profiling(config, force_cpu=True)
    except Exception as e:
        print(f"Profiling failed: {e}")
        results["profiling"] = []

    print("\n" + "=" * 60)
    print("Writing Summary")
    print("=" * 60)
    write_summary(config, results)


if __name__ == "__main__":
    main()
