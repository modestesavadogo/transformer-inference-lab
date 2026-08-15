"""
KV cache container.

Kept deliberately separate from attention.py: the cache is just storage +
bookkeeping (how much has been filled, when to grow). It should not know
about attention math, and attention.py should not know about pre-allocation
strategy. This separation is what lets naive.py and kv_cache.py in
src/inference/ share the same model code and only differ in whether a
cache object is passed in.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class LayerKVCache:
    """Pre-allocated KV cache for a single attention layer."""
    k: torch.Tensor  # (B, n_kv_head, max_seq_len, head_dim)
    v: torch.Tensor
    seq_len: int = 0  # how many positions are actually filled

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        new_k, new_v: (B, n_kv_head, T_new, head_dim)
        Writes into the pre-allocated buffer at [seq_len : seq_len+T_new].
        """
        T_new = new_k.shape[2]
        self.k[:, :, self.seq_len:self.seq_len + T_new, :] = new_k
        self.v[:, :, self.seq_len:self.seq_len + T_new, :] = new_v
        self.seq_len += T_new

    def get(self):
        """Return the filled slice of the cache (no trailing garbage)."""
        return self.k[:, :, :self.seq_len, :], self.v[:, :, :self.seq_len, :]

    def nbytes(self) -> int:
        return self.k.element_size() * self.k.numel() + self.v.element_size() * self.v.numel()


class KVCache:
    """Full-model KV cache: one LayerKVCache per transformer layer."""

    def __init__(self, n_layer: int, batch_size: int, n_kv_head: int,
                 max_seq_len: int, head_dim: int, dtype=torch.float16, device="cpu"):
        self.layers = [
            LayerKVCache(
                k=torch.zeros(batch_size, n_kv_head, max_seq_len, head_dim, dtype=dtype, device=device),
                v=torch.zeros(batch_size, n_kv_head, max_seq_len, head_dim, dtype=dtype, device=device),
            )
            for _ in range(n_layer)
        ]
        self.max_seq_len = max_seq_len

    def total_bytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)

    def reset(self):
        for layer in self.layers:
            layer.seq_len = 0

    # TODO(week 1): once naive.py and kv_cache.py are both working, add a
    # `PreallocationPolicy` note here about why pre-allocating max_seq_len
    # up front (rather than growing dynamically) is the standard choice —
    # relevant context for the 07 — Beyond section on PagedAttention, which
    # solves exactly the fragmentation problem this naive approach creates.
