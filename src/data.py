"""Data loading and tokenization for Sudoku-Extreme dataset."""

import torch
from torch.utils.data import Dataset
from pathlib import Path


def tokenize_sudoku(puzzle_str: str) -> list[int]:
    """Convert 81-char Sudoku string to integer tokens.
    '.' -> 0 (empty/mask), '1'-'9' -> 1-9.
    """
    tokens = []
    for c in puzzle_str:
        if c == ".":
            tokens.append(0)
        elif c in "123456789":
            tokens.append(int(c))
        else:
            raise ValueError(f"Invalid character in puzzle: {c!r}")
    return tokens


def detokenize_sudoku(tokens: list[int] | torch.Tensor) -> str:
    """Convert integer tokens back to 81-char Sudoku string."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().tolist()
    return "".join(str(t) if t > 0 else "." for t in tokens)


def _load_from_hf_csv(dataset_name: str, split: str) -> list[dict]:
    """Load CSV from HuggingFace Hub. Handles rating overflow by using float.
    Supports both sapientinc/sudoku-extreme (train.csv, test.csv) and
    sapientinc/sudoku-extreme-1k (train.csv, test_hard.csv).
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    if "train" in split:
        fnames = ["train.csv"]
    else:
        # Full dataset has test.csv; 1k has test_hard.csv. Try both.
        fnames = ["test_hard.csv", "test.csv"]
    path = None
    for fname in fnames:
        try:
            path = hf_hub_download(repo_id=dataset_name, filename=fname, repo_type="dataset")
            break
        except Exception:
            continue
    if path is None:
        raise FileNotFoundError(f"Could not find train/test CSV in {dataset_name}")
    df = pd.read_csv(path)
    # Handle rating overflow: coerce to int32
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype("int32")
    data = []
    for _, row in df.iterrows():
        data.append({
            "question": tokenize_sudoku(str(row["question"]).strip()),
            "answer": tokenize_sudoku(str(row["answer"]).strip()),
            "rating": int(row.get("rating", 0)),
        })
    return data


def _load_from_datasets(dataset_name: str, split: str) -> list[dict]:
    """Load via HuggingFace datasets library."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    return [
        {
            "question": tokenize_sudoku(row["question"]),
            "answer": tokenize_sudoku(row["answer"]),
            "rating": int(row.get("rating", 0)) if row.get("rating") is not None else 0,
        }
        for row in ds
    ]


class SudokuDataset(Dataset):
    """PyTorch Dataset for Sudoku puzzles."""

    def __init__(self, split: str = "train", dataset_name: str = "sapientinc/sudoku-extreme-1k"):
        self.split = split
        try:
            self.data = _load_from_hf_csv(dataset_name, split)
        except Exception:
            try:
                self.data = _load_from_datasets(dataset_name, split)
            except Exception as e:
                raise RuntimeError(f"Could not load {dataset_name} split={split}: {e}") from e

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        return {
            "question": torch.tensor(row["question"], dtype=torch.long),
            "answer": torch.tensor(row["answer"], dtype=torch.long),
            "rating": row["rating"],
        }


def collate_sudoku(batch: list[dict]) -> dict:
    """Collate batch of Sudoku samples."""
    return {
        "question": torch.stack([b["question"] for b in batch]),
        "answer": torch.stack([b["answer"] for b in batch]),
        "rating": torch.tensor([b["rating"] for b in batch], dtype=torch.long),
    }


def get_dataloaders(
    dataset_name: str = "sapientinc/sudoku-extreme-1k",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
):
    """Create train and test dataloaders.
    pin_memory=True when CUDA is available to speed up CPU->GPU transfer.
    """
    use_pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()

    train_ds = SudokuDataset(split="train", dataset_name=dataset_name)
    try:
        test_ds = SudokuDataset(split="test_hard", dataset_name=dataset_name)
    except Exception:
        try:
            test_ds = SudokuDataset(split="test", dataset_name=dataset_name)
        except Exception:
            test_ds = SudokuDataset(split="train", dataset_name=dataset_name)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_sudoku,
        pin_memory=use_pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sudoku,
        pin_memory=use_pin_memory,
    )
    return train_loader, test_loader
