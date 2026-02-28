"""Main entry point: train all models and run evaluation."""

import argparse
import torch
from src.config import Config
from src.train import train_baseline, train_trm, train_gnn, evaluate
from src.data import get_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "trm_mlp", "trm_attention", "trm_attention_xl", "gnn", "all"], default="all")
    parser.add_argument("--act", action="store_true", help="Use ACT for TRM models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Quick run: 5 epochs")
    parser.add_argument("--full", action="store_true", help="Use full sudoku-extreme (~3.8M train) instead of 1k subset")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after each training phase")
    args = parser.parse_args()

    config = Config()
    if args.full:
        config.dataset_name = "sapientinc/sudoku-extreme"

    if config.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    else:
        print("WARNING: CUDA not available. Install PyTorch with CUDA: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
    epochs = 5 if args.quick else args.epochs

    if args.evaluate:
        _, test_loader = get_dataloaders(config.dataset_name, config.batch_size, config.num_workers)
    else:
        test_loader = None

    if args.model in ("baseline", "all"):
        print("\n=== Training Baseline 4-layer Transformer ===")
        model = train_baseline(config, epochs=epochs)
        if args.evaluate:
            acc = evaluate(model, test_loader, config, model_type="baseline")
            print(f"Baseline final test accuracy: {acc:.4f}")

    if args.model in ("trm_mlp", "all"):
        print("\n=== Training MLP-based TRM ===")
        model = train_trm(config, model_type="mlp", use_act=False, epochs=epochs)
        if args.evaluate:
            acc = evaluate(model, test_loader, config, model_type="trm_mlp")
            print(f"TRM-MLP final test accuracy: {acc:.4f}")
        if args.act:
            print("\n=== Training MLP-based TRM with ACT ===")
            model = train_trm(config, model_type="mlp", use_act=True, epochs=epochs)
            if args.evaluate:
                acc = evaluate(model, test_loader, config, model_type="trm_mlp")
                print(f"TRM-MLP-ACT final test accuracy: {acc:.4f}")

    if args.model in ("trm_attention", "all"):
        print("\n=== Training Attention-based TRM ===")
        model = train_trm(config, model_type="attention", use_act=False, epochs=epochs)
        if args.evaluate:
            acc = evaluate(model, test_loader, config, model_type="trm_attention")
            print(f"TRM-Attention final test accuracy: {acc:.4f}")
        if args.act:
            print("\n=== Training Attention-based TRM with ACT ===")
            model = train_trm(config, model_type="attention", use_act=True, epochs=epochs)
            if args.evaluate:
                acc = evaluate(model, test_loader, config, model_type="trm_attention")
                print(f"TRM-Attention-ACT final test accuracy: {acc:.4f}")

    if args.model in ("trm_attention_xl", "all"):
        print("\n=== Training TRM-Attention-XL (larger model for ACT demo) ===")
        model = train_trm(config, model_type="attention_xl", use_act=False, epochs=epochs)
        if args.evaluate:
            acc = evaluate(model, test_loader, config, model_type="trm_attention_xl")
            print(f"TRM-Attention-XL final test accuracy: {acc:.4f}")
        if args.act:
            print("\n=== Training TRM-Attention-XL with ACT ===")
            model = train_trm(config, model_type="attention_xl", use_act=True, epochs=epochs)
            if args.evaluate:
                acc = evaluate(model, test_loader, config, model_type="trm_attention_xl")
                print(f"TRM-Attention-XL-ACT final test accuracy: {acc:.4f}")

    if args.model in ("gnn", "all"):
        print("\n=== Training GNN (graph + legality-constrained decoding) ===")
        model = train_gnn(config, use_act=False, epochs=epochs)
        if args.evaluate:
            acc = evaluate(model, test_loader, config, model_type="gnn")
            print(f"GNN final test accuracy: {acc:.4f}")
        if args.act:
            print("\n=== Training GNN with ACT (adaptive depth) ===")
            model = train_gnn(config, use_act=True, epochs=epochs)
            if args.evaluate:
                acc = evaluate(model, test_loader, config, model_type="gnn")
                print(f"GNN-ACT final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
