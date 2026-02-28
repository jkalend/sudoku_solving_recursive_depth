"""Main entry point: train all models and run evaluation."""

import argparse
from src.config import Config
from src.train import train_baseline, train_trm, train_gnn, evaluate
from src.data import get_dataloaders
from src.models import BaselineTransformer, TRMMLP, TRMAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "trm_mlp", "trm_attention", "trm_attention_xl", "gnn", "all"], default="all")
    parser.add_argument("--act", action="store_true", help="Use ACT for TRM models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Quick run: 5 epochs")
    parser.add_argument("--full", action="store_true", help="Use full sudoku-extreme (~3.8M train) instead of 1k subset")
    args = parser.parse_args()

    config = Config()
    if args.full:
        config.dataset_name = "sapientinc/sudoku-extreme"
    import torch
    if config.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    else:
        print("WARNING: CUDA not available. Install PyTorch with CUDA: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
    epochs = 5 if args.quick else args.epochs

    if args.model in ("baseline", "all"):
        print("\n=== Training Baseline 4-layer Transformer ===")
        train_baseline(config, epochs=epochs)

    if args.model in ("trm_mlp", "all"):
        print("\n=== Training MLP-based TRM ===")
        train_trm(config, model_type="mlp", use_act=False, epochs=epochs)
        if args.act:
            print("\n=== Training MLP-based TRM with ACT ===")
            train_trm(config, model_type="mlp", use_act=True, epochs=epochs)

    if args.model in ("trm_attention", "all"):
        print("\n=== Training Attention-based TRM ===")
        train_trm(config, model_type="attention", use_act=False, epochs=epochs)
        if args.act:
            print("\n=== Training Attention-based TRM with ACT ===")
            train_trm(config, model_type="attention", use_act=True, epochs=epochs)

    if args.model in ("trm_attention_xl", "all"):
        print("\n=== Training TRM-Attention-XL (larger model for ACT demo) ===")
        train_trm(config, model_type="attention_xl", use_act=False, epochs=epochs)
        if args.act:
            print("\n=== Training TRM-Attention-XL with ACT ===")
            train_trm(config, model_type="attention_xl", use_act=True, epochs=epochs)

    if args.model in ("gnn", "all"):
        print("\n=== Training GNN (graph + legality-constrained decoding) ===")
        train_gnn(config, use_act=False, epochs=epochs)
        if args.act:
            print("\n=== Training GNN with ACT (adaptive depth) ===")
            train_gnn(config, use_act=True, epochs=epochs)


if __name__ == "__main__":
    main()
