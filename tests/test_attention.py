"""
Correctness tests to run BEFORE trusting any benchmark number.

These aren't exhaustive — they're the specific bugs that are easy to
introduce in this project and hard to notice from benchmark numbers alone
(a silently-wrong repeat_kv still runs and produces a plausible-looking
latency number, it just makes GQA secretly compute MHA).
"""

import torch
from src.model.attention import GroupedQueryAttention, make_mha, make_mqa, make_gqa
from src.model.transformer import GPT, ModelConfig
from src.model.cache import KVCache


def test_mha_is_gqa_with_full_heads():
    """n_kv_head == n_head should behave identically to no head-sharing."""
    torch.manual_seed(0)
    attn = make_mha(n_embd=32, n_head=4)
    assert attn.n_rep == 1
    x = torch.randn(2, 5, 32)
    out, k, v = attn(x)
    assert out.shape == (2, 5, 32)
    assert k.shape == (2, 4, 5, 8)  # n_kv_head == n_head here


def test_mqa_shares_single_kv_head():
    attn = make_mqa(n_embd=32, n_head=4)
    assert attn.n_kv_head == 1
    assert attn.n_rep == 4
    x = torch.randn(2, 5, 32)
    out, k, v = attn(x)
    assert k.shape == (2, 1, 5, 8)  # only one KV head stored


def test_repeat_kv_matches_manual_expansion():
    x = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).float()
    rep = GroupedQueryAttention.repeat_kv(x, n_rep=3)
    assert rep.shape == (2, 6, 3, 4)
    # each original kv head should appear n_rep times contiguously
    assert torch.equal(rep[:, 0], x[:, 0])
    assert torch.equal(rep[:, 1], x[:, 0])
    assert torch.equal(rep[:, 2], x[:, 0])
    assert torch.equal(rep[:, 3], x[:, 1])


def test_cached_and_naive_paths_agree_numerically():
    """
    The strongest sanity check in this repo: incremental (cached) decoding
    must produce IDENTICAL logits to a full recompute at every step
    (up to floating point tolerance). If this test fails, trust nothing
    else in results/ until it's fixed.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, n_layer=2, n_embd=16, n_head=2, n_kv_head=2, block_size=32)
    model = GPT(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    with torch.no_grad():
        full_logits, _ = model(prompt, kv_caches=None)

        cache = KVCache(cfg.n_layer, 1, cfg.n_kv_head, 16, cfg.n_embd // cfg.n_head)
        prefix_logits, kv = model(prompt[:, :3], kv_caches=None)
        for i, (k, v) in enumerate(kv):
            cache.layers[i].append(k, v)

        kv_inputs = [layer.get() for layer in cache.layers]
        next_logits, _ = model(prompt[:, 3:4], kv_caches=kv_inputs, pos_offset=3)

    assert torch.allclose(full_logits[:, 3, :], next_logits[:, 0, :], atol=1e-4), (
        "Cached incremental decode diverged from full recompute — "
        "check pos_offset and cache append/read logic."
    )


def test_kv_cache_bytes_formula():
    attn = make_gqa(n_embd=32, n_head=4, n_kv_head=2)
    # 2 (K+V) * n_kv_head * seq_len * head_dim * dtype_bytes
    expected = 2 * 2 * 100 * 8 * 2
    assert attn.kv_cache_bytes(seq_len=100, dtype_bytes=2) == expected
