"""Profile inference latency and FLOPs."""

import torch
import time
import numpy as np
from pathlib import Path

from .config import Config
from .models import BaselineTransformer, TRMMLP, TRMAttention
from .evaluate import load_model


def profile_latency(model, device: str, n_warmup: int = 10, n_runs: int = 1000):
    """Measure inference latency with batch size 1."""
    model.eval()
    x = torch.randint(0, 10, (1, 81), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(x)
            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    return float(np.median(times)), float(np.std(times))


def profile_flops(model, device: str):
    """Use ptflops to count FLOPs. Returns (macs, params) or (None, None)."""
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return None, None

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            x = x.long() if x.dtype != torch.long else x
            # ptflops may pass (1, 1, 81) - ensure (B, 81)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            out = self.m(x)
            return out["logits"] if isinstance(out, dict) else out

    def input_constructor(input_res):
        """Provide correct (1, 81) long tensor for embedding indices."""
        return torch.randint(0, 10, (1, 81), dtype=torch.long, device=device)

    wrapped = Wrapper(model).to(device)
    try:
        macs, params = get_model_complexity_info(
            wrapped,
            (1, 81),
            as_strings=False,
            print_per_layer_stat=False,
            input_constructor=input_constructor,
        )
        return macs, params
    except Exception:
        return None, None


def run_profiling(config: Config, force_cpu: bool = False):
    """Profile all models. Set force_cpu=True to avoid CUDA/torchvision version mismatch."""
    import numpy as np
    device = "cpu" if force_cpu else config.device
    if force_cpu:
        print("(Profiling on CPU)")
    results = []

    models_to_profile = [
        ("Baseline", "baseline", False),
        ("TRM-MLP", "mlp", False),
        ("TRM-Attention", "attention", False),
        ("TRM-MLP-ACT", "mlp", True),
        ("TRM-Attention-ACT", "attention", True),
        ("TRM-Attention-XL", "attention_xl", False),
        ("TRM-Attention-XL-ACT", "attention_xl", True),
        ("GNN", "gnn", False),
        ("GNN-ACT", "gnn", True),
    ]

    for name, model_type, use_act in models_to_profile:
        try:
            try:
                model = load_model(config, model_type, use_act, device=device)
                _device = device
            except Exception:
                # Fall back to CPU on CUDA/torchvision mismatch
                _device = "cpu"
                model = load_model(config, model_type, use_act, device=_device)
            latency_ms, latency_std = profile_latency(model, _device)
            macs, params = profile_flops(model, _device)
            results.append({
                "name": name,
                "latency_ms": latency_ms,
                "latency_std": latency_std,
                "macs": macs,
                "params": params,
            })
            print(f"{name}: {latency_ms:.2f}±{latency_std:.2f} ms, MACs={macs}, params={params}")
        except Exception as e:
            print(f"{name}: failed - {e}")

    return results


if __name__ == "__main__":
    config = Config()
    run_profiling(config)
