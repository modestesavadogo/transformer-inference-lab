"""
Minimal GPT-style transformer wrapping the attention variants in attention.py.

Deliberately close to your nanoGPT speedrun model so that swapping
n_kv_head is the ONLY architectural change between the MHA/GQA/MQA training
runs — everything else (MLP, layernorm placement, init, tokenizer) should
stay identical across the three checkpoints. That's what makes the
week-2 quality comparison mean something.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 8
    n_kv_head: int = 8  # 8 = MHA, 2 = GQA, 1 = MQA
    block_size: int = 1024
    dropout: float = 0.0


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.attn = GroupedQueryAttention(cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    def forward(self, x, k_cache=None, v_cache=None, causal=True):
        attn_out, new_k, new_v = self.attn(self.ln1(x), k_cache, v_cache, causal)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_k, new_v


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=False)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying, as in your speedrun
        self.tok_emb.weight = self.head.weight

    def forward(self, idx: torch.Tensor, kv_caches=None, pos_offset: int = 0):
        B, T = idx.shape
        if pos_offset + T > self.cfg.block_size:
            raise ValueError(
                f"Position {pos_offset + T - 1} exceeds block_size={self.cfg.block_size}. "
                f"This model supports at most {self.cfg.block_size} tokens total "
                f"(prompt + generated). Reduce context_length or max_new_tokens so "
                f"their sum stays within block_size."
            )
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            k_cache, v_cache = (kv_caches[i] if kv_caches is not None else (None, None))
            x, new_k, new_v = block(x, k_cache, v_cache, causal=True)
            new_kv_caches.append((new_k, new_v))

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kv_caches

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
