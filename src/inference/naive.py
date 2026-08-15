"""
Naive autoregressive decoding: no cache. At every step, re-run the full
forward pass over the entire sequence generated so far.

This is intentionally wasteful — it's your baseline (README section 02).
The whole point is to benchmark it against kv_cache.py and show the
O(T^2) vs O(T) cost difference in wall-clock numbers, not just assert it.
"""

import time
import torch
from ..model.transformer import GPT


@torch.no_grad()
def generate_naive(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
    """
    idx: (B, T0) prompt token ids.
    Returns generated idx (B, T0 + max_new_tokens) and a list of per-token
    latencies (seconds) for benchmarking.
    """
    model.eval()
    device = idx.device
    latencies = []

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.cfg.block_size else idx[:, -model.cfg.block_size:]

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        logits, _ = model(idx_cond, kv_caches=None)  # full recompute, cache discarded
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        torch.cuda.synchronize() if device.type == "cuda" else None
        latencies.append(time.perf_counter() - t0)

        idx = torch.cat([idx, next_id], dim=1)

    return idx, latencies


# TODO(week 1, day 3-4): add a --profile-memory flag using
# torch.cuda.max_memory_allocated() reset/read around the loop, so
# naive.py and kv_cache.py report memory in the exact same way and are
# directly comparable in benchmarks/memory.py.
