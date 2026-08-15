"""
Cached autoregressive decoding: prefill the prompt once, then feed only
the single newest token per step, reusing cached K/V for everything before it.

Compare directly against naive.py — same model, same prompt, same sampling
— to isolate the effect of caching alone (README section 03).
"""

import time
import torch
from ..model.transformer import GPT
from ..model.cache import KVCache


@torch.no_grad()
def generate_with_cache(model: GPT, idx: torch.Tensor, max_new_tokens: int,
                         temperature: float = 1.0, dtype=torch.float16):
    model.eval()
    device = idx.device
    cfg = model.cfg
    B, T0 = idx.shape

    cache = KVCache(
        n_layer=cfg.n_layer,
        batch_size=B,
        n_kv_head=cfg.n_kv_head,
        max_seq_len=T0 + max_new_tokens,
        head_dim=cfg.n_embd // cfg.n_head,
        dtype=dtype,
        device=device,
    )

    latencies = []

    # --- Prefill: one forward pass over the whole prompt ---
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    logits, new_kv = model(idx, kv_caches=None, pos_offset=0)
    for i, (k, v) in enumerate(new_kv):
        cache.layers[i].append(k, v)
    torch.cuda.synchronize() if device.type == "cuda" else None
    latencies.append(time.perf_counter() - t0)  # prefill latency, report separately

    next_logits = logits[:, -1, :] / temperature
    probs = torch.softmax(next_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_id], dim=1)

    # --- Decode: one token at a time, reusing cache ---
    for step in range(max_new_tokens - 1):
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        kv_inputs = [layer.get() for layer in cache.layers]
        logits, new_kv = model(next_id, kv_caches=kv_inputs, pos_offset=T0 + step + 1)
        for i, (k, v) in enumerate(new_kv):
            # new_kv currently holds full (past+current); only the newest
            # slice needs appending since past was already in the cache.
            cache.layers[i].append(k[:, :, -1:, :], v[:, :, -1:, :])

        torch.cuda.synchronize() if device.type == "cuda" else None
        latencies.append(time.perf_counter() - t0)

        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx, latencies, cache.total_bytes()


# TODO(week 1, day 3-4): note in the README that new_kv from model() returns
# the FULL concatenated K/V (see attention.py forward()), so appending
# k[:, :, -1:, :] here avoids double-writing the cached prefix. This is a
# classic off-by-one source of bugs — write a tests/ case that checks
# cache.seq_len against len(idx) after N steps.
