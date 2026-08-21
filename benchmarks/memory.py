"""
Memory benchmark: measures actual peak GPU memory (torch.cuda.max_memory_allocated)
alongside the theoretical KV cache size computed from attention.kv_cache_bytes(),
so you can report both the formula AND the measured number — and explain
any gap (allocator overhead, activation memory, etc.) in the analysis.

Usage:
    python benchmarks/memory.py --attention gqa --kv-heads 2 \
        --context-lengths 512,1024,2048,4096 --checkpoint results/checkpoints/gqa.pt
"""


# NOTE: context_length + max_new_tokens must not exceed the model's
# block_size (1024 for these checkpoints) — positions beyond block_size
# don't exist in the position embedding table and will trigger a CUDA
# device-side assert that poisons the CUDA context for the rest of the
# process (requiring a kernel restart, not just a retry).

import argparse
import json
from pathlib import Path

import torch

from benchmarks.latency import load_model
from src.inference.kv_cache import generate_with_cache


def parse_args():
    p = argparse.ArgumentParser(description="Memory benchmark across context lengths")
    p.add_argument("--attention", choices=["mha", "gqa", "mqa"], required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--context-lengths", type=str, default="512,1024,2048,4096")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def theoretical_kv_bytes(model, seq_len: int, dtype_bytes: int = 2) -> int:
    per_layer = model.blocks[0].attn.kv_cache_bytes(seq_len, dtype_bytes)
    return per_layer * model.cfg.n_layer


def main():
    args = parse_args()
    model = load_model(args.checkpoint, args.device)
    context_lengths = [int(c) for c in args.context_lengths.split(",")]

    results = []
    for ctx in context_lengths:
        prompt = torch.randint(0, model.cfg.vocab_size, (1, ctx), device=args.device)

        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        _, _, measured_cache_bytes = generate_with_cache(model, prompt, args.max_new_tokens)

        peak_mem = torch.cuda.max_memory_allocated() if args.device == "cuda" else None

        results.append({
            "context_length": ctx,
            "theoretical_kv_cache_bytes": theoretical_kv_bytes(model, ctx + args.max_new_tokens),
            "measured_kv_cache_bytes": measured_cache_bytes,
            "peak_gpu_memory_bytes": peak_mem,
        })

    summary = {"attention": args.attention, "kv_heads": args.kv_heads, "results": results}
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
