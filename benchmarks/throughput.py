"""
Throughput benchmark (tokens/sec), swept across batch size, so you can see
where increasing batch size stops helping because the KV cache has eaten
the memory budget — this is the number that most directly shows why
reducing KV heads matters in production serving.

Usage:
    python benchmarks/throughput.py --attention mqa --kv-heads 1 \
        --context-length 2048 --checkpoint results/checkpoints/mqa.pt \
        --batch-sizes 1,4,8,16,32
"""


# NOTE: context_length + max_new_tokens must not exceed the model's
# block_size (1024 for these checkpoints) — positions beyond block_size
# don't exist in the position embedding table and will trigger a CUDA
# device-side assert that poisons the CUDA context for the rest of the
# process (requiring a kernel restart, not just a retry).

import argparse
import json
import time
from pathlib import Path

import torch

from benchmarks.latency import load_model
from src.inference.kv_cache import generate_with_cache


def parse_args():
    p = argparse.ArgumentParser(description="Throughput benchmark across batch sizes")
    p.add_argument("--attention", choices=["mha", "gqa", "mqa"], required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-sizes", type=str, default="1,4,8,16",
                    help="comma-separated list of batch sizes to sweep")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model = load_model(args.checkpoint, args.device)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    results = []
    for B in batch_sizes:
        prompt = torch.randint(0, model.cfg.vocab_size, (B, args.context_length), device=args.device)
        try:
            torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
            t0 = time.perf_counter()
            _, latencies, cache_bytes = generate_with_cache(model, prompt, args.max_new_tokens)
            elapsed = time.perf_counter() - t0
            tokens_per_sec = (B * args.max_new_tokens) / elapsed
            results.append({
                "batch_size": B,
                "tokens_per_sec": tokens_per_sec,
                "elapsed_s": elapsed,
                "kv_cache_bytes": cache_bytes,
                "oom": False,
            })
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": B, "oom": True})
            torch.cuda.empty_cache()
            break  # larger batch sizes will also OOM, no point continuing

    summary = {
        "attention": args.attention,
        "kv_heads": args.kv_heads,
        "context_length": args.context_length,
        "sweep": results,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
