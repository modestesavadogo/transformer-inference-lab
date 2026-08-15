"""
Turns results/*.json (from benchmarks/latency.py, throughput.py, memory.py,
plus perplexity numbers you'll add from eval) into the plots the README
needs:

  1. latency/token, with vs without cache, across context length
  2. KV cache size, MHA vs GQA vs MQA, across context length (theoretical
     line + measured points on the same axes)
  3. throughput vs memory scatter (the "GQA sits between MHA and MQA" plot
     from the original project spec) — one point per (attention, batch_size)
  4. the memory-latency-quality triangle: 3-axis or a small-multiples grid,
     perplexity on one axis, decide once you have real numbers which reads
     more honestly rather than picking a fancier chart type to look impressive

Usage:
    python analysis/plots.py --results-dir results/ --out-dir results/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(results_dir: Path, pattern: str):
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob(pattern))]


def plot_latency_vs_context(results, out_path: Path):
    fig, ax = plt.subplots()
    # TODO(week 2, day 8-9): group by `attention` and `cached`, plot
    # mean_latency_ms vs context_length as separate lines. Placeholder
    # below just proves the plumbing works end-to-end.
    for r in results:
        label = f"{r['attention']} ({'cached' if r['cached'] else 'naive'})"
        ax.scatter(r["context_length"], r["mean_latency_ms"], label=label)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Mean latency / token (ms)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_kv_cache_size(results, out_path: Path):
    fig, ax = plt.subplots()
    for r in results:
        for point in r["results"]:
            ax.scatter(point["context_length"], point["theoretical_kv_cache_bytes"],
                       label=r["attention"])
    ax.set_xlabel("Context length")
    ax.set_ylabel("KV cache size (bytes)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_throughput_vs_memory(results, out_path: Path):
    fig, ax = plt.subplots()
    for r in results:
        for point in r["sweep"]:
            if point.get("oom"):
                continue
            ax.scatter(point["kv_cache_bytes"], point["tokens_per_sec"], label=r["attention"])
    ax.set_xlabel("KV cache memory (bytes)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results/")
    p.add_argument("--out-dir", type=str, default="results/figures/")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latency_results = load_results(results_dir / "latency", "*.json")
    if latency_results:
        plot_latency_vs_context(latency_results, out_dir / "latency_vs_context.png")

    memory_results = load_results(results_dir / "memory", "*.json")
    if memory_results:
        plot_kv_cache_size(memory_results, out_dir / "kv_cache_size.png")

    throughput_results = load_results(results_dir / "throughput", "*.json")
    if throughput_results:
        plot_throughput_vs_memory(throughput_results, out_dir / "throughput_vs_memory.png")

    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
