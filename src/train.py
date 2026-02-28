"""Training script with deep supervision and ACT ponder cost."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import Config
from .data import get_dataloaders
from .models import BaselineTransformer, TRMMLP, TRMAttention, TRMAttentionXL, SudokuGNN

# Async transfer to GPU when using pin_memory
NON_BLOCKING = True


def sudoku_loss(
    logits: torch.Tensor,
    answer: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy loss only on empty cells (mask=True)."""
    # logits: (B, 81, 10), answer: (B, 81), mask: (B, 81) True = empty
    logits_flat = logits.view(-1, logits.shape[-1])
    answer_flat = answer.view(-1)
    mask_flat = mask.view(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    loss = F.cross_entropy(logits_flat[mask_flat], answer_flat[mask_flat], reduction=reduction)
    return loss


def train_baseline(config: Config, epochs: int | None = None):
    """Train 4-layer baseline transformer."""
    epochs = epochs or config.epochs
    train_loader, test_loader = get_dataloaders(
        config.dataset_name, config.batch_size, config.num_workers
    )
    model = BaselineTransformer(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.baseline_d_model,
        n_heads=config.baseline_heads,
        ff_dim=config.baseline_ff_dim,
        n_layers=config.baseline_layers,
        dropout=config.baseline_dropout,
    ).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline: {n_params:,} parameters")

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            q = batch["question"].to(config.device, non_blocking=NON_BLOCKING)
            a = batch["answer"].to(config.device, non_blocking=NON_BLOCKING)
            mask = q == 0
            opt.zero_grad()
            logits = model(q, mask)
            loss = sudoku_loss(logits, a, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader, config, model_type="baseline")
        print(f"Epoch {ep+1} loss={avg_loss:.4f} test_acc={acc:.4f}")

    torch.save(model.state_dict(), config.checkpoint_dir / "baseline.pt")
    return model


def train_gnn(
    config: Config,
    use_act: bool = False,
    deep_supervision: bool = True,
    ponder_cost: float = 0.01,
    epochs: int | None = None,
):
    """Train GNN Sudoku model with masked node classification. Optional adaptive depth (ACT)."""
    epochs = epochs or config.epochs
    train_loader, test_loader = get_dataloaders(
        config.dataset_name, config.batch_size, config.num_workers
    )
    model = SudokuGNN(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.gnn_d_model,
        n_layers=config.gnn_layers,
        hidden_dim=config.gnn_hidden,
        dropout=config.gnn_dropout,
        use_act=use_act,
        act_max_steps=config.gnn_act_max_steps,
    ).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNN ACT={use_act}: {n_params:,} parameters")

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            q = batch["question"].to(config.device, non_blocking=NON_BLOCKING)
            a = batch["answer"].to(config.device, non_blocking=NON_BLOCKING)
            mask = q == 0
            opt.zero_grad()
            out = model(q)

            if deep_supervision and "step_logits" in out:
                loss = torch.zeros(1, device=q.device, dtype=out["step_logits"][0].dtype).squeeze()
                for step_logits in out["step_logits"]:
                    loss = loss + sudoku_loss(step_logits, a, mask)
                loss = loss / len(out["step_logits"])
            else:
                loss = sudoku_loss(out["logits"], a, mask)

            if use_act and "halting_probs" in out:
                ponder = out["halting_probs"].sum(dim=1).mean()
                loss = loss + ponder_cost * ponder

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader, config, model_type="gnn")
        print(f"Epoch {ep+1} loss={avg_loss:.4f} test_acc={acc:.4f}")

    name = "gnn_act" if use_act else "gnn"
    torch.save(model.state_dict(), config.checkpoint_dir / f"{name}.pt")
    return model


def train_trm(
    config: Config,
    model_type: str = "mlp",
    use_act: bool = False,
    deep_supervision: bool = True,
    ponder_cost: float = 0.01,
    epochs: int | None = None,
):
    """Train TRM (MLP or Attention) with optional ACT and deep supervision."""
    epochs = epochs or config.epochs
    train_loader, test_loader = get_dataloaders(
        config.dataset_name, config.batch_size, config.num_workers
    )

    if model_type == "mlp":
        model = TRMMLP(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            d_model=config.trm_d_model,
            ff_dim=config.trm_ff_dim,
            n_steps=config.trm_max_steps,
            dropout=config.trm_dropout,
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
            dropout=config.trm_dropout,
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
            dropout=config.trm_dropout,
            use_act=use_act,
            act_max_steps=config.act_max_steps,
        )
    model = model.to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TRM-{model_type} ACT={use_act}: {n_params:,} parameters")

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            q = batch["question"].to(config.device, non_blocking=NON_BLOCKING)
            a = batch["answer"].to(config.device, non_blocking=NON_BLOCKING)
            mask = q == 0
            opt.zero_grad()
            out = model(q, return_steps=True)

            if deep_supervision and "step_logits" in out:
                loss = torch.zeros(1, device=q.device, dtype=out["step_logits"][0].dtype).squeeze()
                for step_logits in out["step_logits"]:
                    loss = loss + sudoku_loss(step_logits, a, mask)
                loss = loss / len(out["step_logits"])
            else:
                loss = sudoku_loss(out["logits"], a, mask)

            if use_act and "halting_probs" in out:
                # Ponder cost: penalize extra steps
                ponder = out["halting_probs"].sum(dim=1).mean()
                loss = loss + ponder_cost * ponder

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader, config, model_type=f"trm_{model_type}")
        print(f"Epoch {ep+1} loss={avg_loss:.4f} test_acc={acc:.4f}")

    name = f"trm_{model_type}_act" if use_act else f"trm_{model_type}"
    torch.save(model.state_dict(), config.checkpoint_dir / f"{name}.pt")
    return model


def evaluate(model, loader, config: Config, model_type: str = "baseline") -> float:
    """Exact accuracy: all 81 digits must match."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            q = batch["question"].to(config.device, non_blocking=NON_BLOCKING)
            a = batch["answer"].to(config.device, non_blocking=NON_BLOCKING)
            if model_type == "baseline":
                logits = model(q, q == 0)
            else:
                out = model(q)
                logits = out["logits"]
            pred = logits.argmax(dim=-1)
            # Exact match: all 81 correct
            match = (pred == a).all(dim=1)
            correct += match.sum().item()
            total += q.shape[0]
    return correct / total if total > 0 else 0.0
