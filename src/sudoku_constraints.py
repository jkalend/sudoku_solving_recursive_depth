"""Sudoku legality constraints and constrained decoding utilities."""

import torch


# Precomputed peers for each cell: indices of cells in same row, col, or box
_PEERS: list[list[int]] = []


def _init_peers() -> None:
    global _PEERS
    if _PEERS:
        return
    for i in range(81):
        ri, ci = i // 9, i % 9
        bi = (ri // 3) * 3 + (ci // 3)
        peers = []
        for j in range(81):
            if i == j:
                continue
            rj, cj = j // 9, j % 9
            bj = (rj // 3) * 3 + (cj // 3)
            if ri == rj or ci == cj or bi == bj:
                peers.append(j)
        _PEERS.append(peers)


def build_peers() -> list[list[int]]:
    """Return list of peer indices for each of 81 cells."""
    _init_peers()
    return _PEERS


def legal_digit_mask(board: torch.Tensor) -> torch.Tensor:
    """
    Compute per-cell legal digit mask from current board state.
    For empty cells: digit d (1-9) is legal iff no peer has d.
    For filled cells: only that digit is True (for loss masking).
    Args:
        board: (B, 81) or (81,) int tensor, 0=empty, 1-9=digit
    Returns:
        mask: (B, 81, 10) or (81, 10) bool, True = legal
    """
    _init_peers()
    if board.dim() == 1:
        board = board.unsqueeze(0)
    B, _ = board.shape
    device = board.device
    mask = torch.ones(B, 81, 10, dtype=torch.bool, device=device)

    for b in range(B):
        for i in range(81):
            if board[b, i].item() != 0:
                mask[b, i, :] = False
                mask[b, i, board[b, i].item()] = True
                continue
            for d in range(1, 10):
                for j in _PEERS[i]:
                    if board[b, j].item() == d:
                        mask[b, i, d] = False
                        break

    return mask


def is_board_valid(board: torch.Tensor) -> bool:
    """Check no row/col/box has duplicate digits. 0 is ignored."""
    if board.dim() == 2:
        if board.shape[0] != 1:
            raise ValueError(
                f"is_board_valid expects a (81,) or (1, 81) tensor, got shape {tuple(board.shape)}"
            )
        board = board.squeeze(0)
    elif board.dim() != 1:
        raise ValueError(
            f"is_board_valid expects a (81,) or (1, 81) tensor, got shape {tuple(board.shape)}"
        )
    board = board.cpu()
    for i in range(9):
        row = board[i * 9 : (i + 1) * 9]
        row = row[row != 0]
        if row.unique().numel() != row.numel():
            return False
        col = board[i::9]
        col = col[col != 0]
        if col.unique().numel() != col.numel():
            return False
    for bi in range(9):
        r0, c0 = (bi // 3) * 3, (bi % 3) * 3
        box = []
        for r in range(r0, r0 + 3):
            for c in range(c0, c0 + 3):
                box.append(board[r * 9 + c].item())
        box = [x for x in box if x != 0]
        if len(box) != len(set(box)):
            return False
    return True


def is_board_solved(board: torch.Tensor) -> bool:
    """Check board has no zeros and is valid."""
    if board.dim() == 2:
        if board.shape[0] != 1:
            raise ValueError(
                f"is_board_solved expects a (81,) or (1, 81) tensor, got shape {tuple(board.shape)}"
            )
        board = board.squeeze(0)
    elif board.dim() != 1:
        raise ValueError(
            f"is_board_solved expects a (81,) or (1, 81) tensor, got shape {tuple(board.shape)}"
        )
    return (board != 0).all().item() and is_board_valid(board)


def constrained_decode(
    model: torch.nn.Module,
    puzzle: torch.Tensor,
    device: torch.device,
    max_steps: int = 81,
) -> torch.Tensor:
    """
    Greedy legality-constrained decoding: fill one cell at a time.
    Args:
        model: forward(x) -> dict with "logits" (B, 81, 10)
        puzzle: (81,) or (1, 81) initial puzzle, 0=empty
        device: torch device
    Returns:
        board: (81,) filled board (may be partial if stalled)
    """
    if puzzle.dim() == 1:
        puzzle = puzzle.unsqueeze(0)
    board = puzzle.clone().to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max_steps):
            if (board == 0).sum().item() == 0:
                break
            out = model(board)
            logits = out["logits"]
            legal = legal_digit_mask(board)
            logits = logits.masked_fill(~legal, float("-inf"))
            probs = logits.softmax(dim=-1)
            best_val, best_idx = probs[:, :, 1:10].max(dim=-1)
            best_digit = best_idx + 1
            empty = board[0] == 0
            best_val = best_val[0].masked_fill(~empty, float("-inf"))
            cell = best_val.argmax().item()
            if best_val[cell].item() <= 0 or best_val[cell].item() == float("-inf"):
                break
            digit = best_digit[0, cell].item()
            if not legal[0, cell, digit].item():
                break
            board[0, cell] = digit

    return board.squeeze(0)
