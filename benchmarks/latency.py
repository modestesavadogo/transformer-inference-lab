"""
Latency benchmark.

Usage:
    python benchmarks/latency.py \
        --attention gqa \
        --kv-heads 2 \
        --context-length 2048 \
        --checkpoint results/checkpoints/gqa.pt \
        --out results/latency/gqa_ctx2048.json

Reports per-token latency (ms), broken into prefill vs decode, so the
KV-cache-off comparison in naive.py and the KV-cache-on comparison in
kv_cache.py show up on the same axis.
"""


# NOTE: context_length + max_new_tokens must not exceed the model's
# block_size (1024 for these checkpoints) — positions beyond block_size
# don't exist in the position embedding table and will trigger a CUDA
# device-side assert that poisons the CUDA context for the rest of the
# process (requiring a kernel restart, not just a retry).

import argparse
import json
import statistics
from pathlib import Path

import torch

from src.model.transformer import GPT, ModelConfig
from src.inference.naive import generate_naive
from src.inference.kv_cache import generate_with_cache


def parse_args():
    p = argparse.ArgumentParser(description="Latency benchmark for attention variants")
    p.add_argument("--attention", choices=["mha", "gqa", "mqa"], required=True)
    p.add_argument("--kv-heads", type=int, required=True,
                    help="n_kv_head — must match how the checkpoint was trained")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-cache", action="store_true",
                    help="use naive (no KV cache) decoding instead of cached decoding")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None, help="path to write JSON results")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    args = parse_args()
    model = load_model(args.checkpoint, args.device)

    prompt = torch.randint(0, model.cfg.vocab_size, (args.batch_size, args.context_length), device=args.device)

    if args.no_cache:
        _, latencies = generate_naive(model, prompt, args.max_new_tokens)
        cache_bytes = None
    else:
        _, latencies, cache_bytes = generate_with_cache(model, prompt, args.max_new_tokens)

    result = {
        "attention": args.attention,
        "kv_heads": args.kv_heads,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "cached": not args.no_cache,
        "mean_latency_ms": statistics.mean(latencies) * 1000,
        "median_latency_ms": statistics.median(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))] * 1000,
        "kv_cache_bytes": cache_bytes,
        "raw_latencies_s": latencies,
    }

    print(json.dumps({k: v for k, v in result.items() if k != "raw_latencies_s"}, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
