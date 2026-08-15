"""
Attention variants: MHA, GQA, MQA.

All three share the same query head count (n_head) and only differ in how
many KV heads they use (n_kv_head). MHA is the special case n_kv_head ==
n_head; MQA is the special case n_kv_head == 1; GQA is anything in between.

Design note: keep this module architecture-only. Caching logic (append,
read, evict) belongs in cache.py, not here — attention.py should work
identically whether it's being called in "compute everything" mode or
"read from cache" mode, it just receives K/V tensors and does not care
where they came from.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    General-purpose attention that reduces to MHA, GQA, or MQA depending on
    n_kv_head.

    Args:
        n_embd:    model (residual stream) dimension
        n_head:    number of query heads (Hq)
        n_kv_head: number of key/value heads (Hkv). Must divide n_head.
                   n_kv_head == n_head       -> MHA
                   1 < n_kv_head < n_head    -> GQA
                   n_kv_head == 1            -> MQA
        block_size: max sequence length (for causal mask buffer, if used
                    outside of a cache-based inference path)
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head  # how many query heads share one KV head
        self.head_dim = n_embd // n_head

        # NOTE on parameter count: only the KV projection shrinks as
        # n_kv_head drops. The Q and output projections stay full-size.
        # This is exactly what makes GQA/MQA a *KV cache memory* win, not
        # a general parameter-count win — worth stating explicitly in
        # your 04 — MHA/MQA/GQA README section.
        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)

        self.dropout = dropout

    def project_qkv(self, x: torch.Tensor):
        """
        x: (B, T, n_embd)
        returns q: (B, n_head, T, head_dim), k/v: (B, n_kv_head, T, head_dim)
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        return q, k, v

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand KV heads to match the number of query heads by repeating,
        NOT by projecting. This is the operation that makes GQA/MQA cheap:
        no extra parameters, just a broadcast at attention time.

        x: (B, n_kv_head, T, head_dim) -> (B, n_kv_head * n_rep, T, head_dim)
        """
        if n_rep == 1:
            return x
        B, n_kv_head, T, head_dim = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv_head, n_rep, T, head_dim)
        return x.reshape(B, n_kv_head * n_rep, T, head_dim)

    def forward(self, x: torch.Tensor, k_cache=None, v_cache=None, causal: bool = True):
        """
        x: (B, T, n_embd) — either the full sequence (naive/prefill) or a
           single new token (T=1, incremental decoding with cache).
        k_cache, v_cache: optional (B, n_kv_head, T_past, head_dim) tensors
           to concatenate with this step's K/V before attending. Pass None
           for naive (no-cache) mode.

        Returns: (output, new_k, new_v) where new_k/new_v are the FULL
        K/V (past + current) so the caller can store them back in the cache.
        """
        q, k, v = self.project_qkv(x)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        k_rep = self.repeat_kv(k, self.n_rep)
        v_rep = self.repeat_kv(v, self.n_rep)

        # TODO(week 1, day 6-7): implement scaled dot-product attention by
        # hand first (matmul + mask + softmax) to make sure the head-repeat
        # logic is correct, THEN swap to F.scaled_dot_product_attention for
        # the benchmarked runs. Keep both paths behind a flag so you can
        # verify they agree numerically — this is a good tests/ candidate.
        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            is_causal=causal and k_cache is None,
            dropout_p=self.dropout if self.training else 0.0,
        )

        B, n_head, T, head_dim = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, n_head * head_dim)
        return self.out_proj(out), k, v

    def kv_cache_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """
        Explicit KV cache size formula (bytes) for one layer, one sequence:
            2 (K and V) * n_kv_head * seq_len * head_dim * dtype_bytes
        This is the number you want front-and-center in experiments/mha_mqa_gqa.
        Multiply by n_layer for the full-model cache size.
        """
        return 2 * self.n_kv_head * seq_len * self.head_dim * dtype_bytes


def make_mha(n_embd, n_head, **kw):
    return GroupedQueryAttention(n_embd, n_head, n_kv_head=n_head, **kw)


def make_gqa(n_embd, n_head, n_kv_head, **kw):
    return GroupedQueryAttention(n_embd, n_head, n_kv_head=n_kv_head, **kw)


def make_mqa(n_embd, n_head, **kw):
    return GroupedQueryAttention(n_embd, n_head, n_kv_head=1, **kw)
