"""Evaluation and ACT step distribution analysis."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .config import Config
from .data import get_dataloaders
from .models import BaselineTransformer, TRMMLP, TRMAttention, TRMAttentionXL, SudokuGNN
from .train import evaluate
from .sudoku_constraints import constrained_decode

NON_BLOCKING = torch.cuda.is_available()


def evaluate_constrained_gnn(model, loader, config: Config, max_samples: int | None = None) -> float:
    """Exact accuracy using legality-constrained greedy decoding."""
    model.eval()
    device = config.device
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_samples is not None and i >= max_samples:
                break
            q = batch["question"].to(device, non_blocking=NON_BLOCKING)
            a = batch["answer"].to(device, non_blocking=NON_BLOCKING)
            for b in range(q.shape[0]):
                pred = constrained_decode(model, q[b], device, max_steps=config.gnn_decode_max_steps)
                match = (pred == a[b]).all().item()
                correct += match
                total += 1
    return correct / total if total > 0 else 0.0


def load_model(config: Config, model_type: str, use_act: bool = False, device: str | None = None):
    """Load trained model from checkpoint."""
    if model_type == "baseline":
        model = BaselineTransformer(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            d_model=config.baseline_d_model,
            n_heads=config.baseline_heads,
            ff_dim=config.baseline_ff_dim,
            n_layers=config.baseline_layers,
            dropout=0,
        )
        ckpt = config.checkpoint_dir / "baseline.pt"
    elif model_type == "gnn":
        model = SudokuGNN(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            d_model=config.gnn_d_model,
            n_layers=config.gnn_layers,
            hidden_dim=config.gnn_hidden,
            dropout=0,
            use_act=use_act,
            act_max_steps=config.gnn_act_max_steps,
        )
        ckpt = config.checkpoint_dir / ("gnn_act.pt" if use_act else "gnn.pt")
    else:
        name = f"trm_{model_type}_act" if use_act else f"trm_{model_type}"
        ckpt = config.checkpoint_dir / f"{name}.pt"
        if model_type == "mlp":
            model = TRMMLP(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                d_model=config.trm_d_model,
                ff_dim=config.trm_ff_dim,
                n_steps=config.trm_max_steps,
                dropout=0,
                use_act=use_act,
                act_max_steps=config.act_max_steps,
            )
        elif model_type == "attention_xl":
            model = TRMAttentionXL(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                d_model=config.trm_xl_d_model,
                n_heads=config.trm_heads,
                ff_dim=config.trm_xl_ff_dim,
                n_blocks=config.trm_xl_blocks,
                n_steps=config.trm_xl_n_steps,
                dropout=0,
                use_act=use_act,
                act_max_steps=config.trm_xl_act_max_steps,
            )
        else:
            model = TRMAttention(
                vocab_size=config.vocab_size,
                seq_len=config.seq_len,
                d_model=config.trm_d_model,
                n_heads=config.trm_heads,
                ff_dim=config.trm_ff_dim,
                n_steps=config.trm_max_steps,
                dropout=0,
                use_act=use_act,
                act_max_steps=config.act_max_steps,
            )
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(device or config.device)
    model.eval()
    return model


def run_evaluation(config: Config):
    """Evaluate all models and report accuracy."""
    _, test_loader = get_dataloaders(config.dataset_name, config.batch_size, config.num_workers)
    results = {}

    for name, model_type, use_act in [
        ("Baseline", "baseline", False),
        ("TRM-MLP", "mlp", False),
        ("TRM-Attention", "attention", False),
        ("TRM-MLP-ACT", "mlp", True),
        ("TRM-Attention-ACT", "attention", True),
        ("TRM-Attention-XL", "attention_xl", False),
        ("TRM-Attention-XL-ACT", "attention_xl", True),
        ("GNN", "gnn", False),
        ("GNN-ACT", "gnn", True),
    ]:
        try:
            if model_type == "baseline":
                model = load_model(config, "baseline")
                acc = evaluate(model, test_loader, config, model_type="baseline")
            elif model_type == "gnn":
                model = load_model(config, "gnn", use_act)
                acc = evaluate(model, test_loader, config, model_type="gnn")
                if not use_act:
                    acc_constrained = evaluate_constrained_gnn(
                        model, test_loader, config, max_samples=500
                    )
                    results[name] = acc
                    results["GNN-constrained"] = acc_constrained
                    print(f"{name}: {acc:.4f} (one-shot), GNN-constrained: {acc_constrained:.4f}")
                else:
                    results[name] = acc
                    print(f"{name}: {acc:.4f}")
            else:
                model = load_model(config, model_type, use_act)
                eval_type = f"trm_{model_type}"
                acc = evaluate(model, test_loader, config, model_type=eval_type)
            if model_type != "gnn":
                results[name] = acc
                print(f"{name}: {acc:.4f}")
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
            print(f"{name}: failed - {e}")
            results[name] = None

    return results


def analyze_act_steps(config: Config, model_type: str = "attention", n_samples: int = 1000):
    """Analyze distribution of ACT recursive steps vs puzzle difficulty."""
    _, test_loader = get_dataloaders(config.dataset_name, batch_size=1, num_workers=0)
    model = load_model(config, model_type, use_act=True)
    model.eval()

    steps_list = []
    ratings_list = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_samples:
                break
            q = batch["question"].to(config.device, non_blocking=NON_BLOCKING)
            r = batch["rating"].item()
            out = model(q)
            if "halting_probs" in out:
                # Compute actual step count per sample from cumulative halt probability
                hp = out["halting_probs"]  # (B, max_steps)
                cumsum = hp.cumsum(dim=1)
                reached = cumsum >= 1.0
                first_halt = reached.long().argmax(dim=1)  # 0-indexed
                never_halted = cumsum[:, -1] < 1.0
                steps_used = first_halt + 1
                steps_used[never_halted] = hp.shape[1]
                for s in steps_used.cpu().tolist():
                    steps_list.append(s)
                    ratings_list.append(r)
            elif "n_steps" in out:
                steps_list.append(out["n_steps"])
                ratings_list.append(r)
            else:
                steps_list.append(config.trm_max_steps)
                ratings_list.append(r)

    steps_arr = np.array(steps_list)
    ratings_arr = np.array(ratings_list)

    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(steps_arr, bins=range(1, int(steps_arr.max()) + 2), edgecolor="black")
    axes[0].set_xlabel("Recursive steps")
    axes[0].set_ylabel("Count")
    axes[0].set_title("ACT Step Distribution")

    axes[1].scatter(ratings_arr, steps_arr, alpha=0.5)
    axes[1].set_xlabel("Puzzle rating (difficulty)")
    axes[1].set_ylabel("Steps used")
    axes[1].set_title("Steps vs Difficulty")
    plt.tight_layout()
    plt.savefig(config.output_dir / "act_step_distribution.png")
    plt.close()
    print(f"Saved ACT step distribution to {config.output_dir / 'act_step_distribution.png'}")


if __name__ == "__main__":
    config = Config()
    run_evaluation(config)
    analyze_act_steps(config)
