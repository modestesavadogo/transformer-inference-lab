"""
Shared sampling / decoding utilities used by both naive.py and kv_cache.py,
so sampling behavior (temperature, top-k, top-p) never differs between the
two paths and can't become a confound in the benchmarks.
"""

import torch
import torch.nn.functional as F


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int = None, top_p: float = None) -> torch.Tensor:
    """
    logits: (B, vocab_size) — logits for the next position only.
    Returns: (B, 1) sampled token ids.
    """
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
        logits[mask] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def compute_perplexity(model, idx: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Held-out perplexity for the quality axis of the memory/latency/quality
    comparison. idx, targets: (B, T) — targets is idx shifted by one.
    Keep this eval split strictly separate from anything used in training
    (same discipline as the medical eval split on the Tachelhit project).
    """
    logits, _ = model(idx, kv_caches=None)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return torch.exp(loss).item()
