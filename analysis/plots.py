"""
Turns results/*.json (from benchmarks/latency.py, throughput.py, memory.py)
into the figures referenced in README sections 04-06:

  1. latency_vs_context.png  — latency/token by attention variant, across
     context length (cached decoding only; naive/no-cache point annotated
     separately for the section 03 speedup comparison)
  2. kv_cache_size.png       — KV cache size by attention variant, across
     context length (theoretical line + measured points overlaid — they
     coincide exactly, which is itself the section 05/06 finding)
  3. throughput_vs_batch.png — tokens/sec by attention variant, across
     batch size (context length fixed at 512) — this is the plot that
     shows MQA's crossover advantage at large batch

Usage:
    python analysis/plots.py --results-dir results/ --out-dir results/figures/

No GPU required — this script only reads JSON and calls matplotlib.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Consistent color/marker per variant across all three plots, so a reader
# can visually match variants between figures without re-reading legends.
STYLE = {
    "mha": {"color": "#1f77b4", "marker": "o", "label": "MHA (n_kv_head=8)"},
    "gqa": {"color": "#ff7f0e", "marker": "s", "label": "GQA (n_kv_head=2)"},
    "mqa": {"color": "#2ca02c", "marker": "^", "label": "MQA (n_kv_head=1)"},
}


def load_json_files(results_dir: Path, pattern: str):
    """Load every JSON file matching pattern, returning list of (path, dict)."""
    files = sorted(results_dir.glob(pattern))
    return [(p, json.loads(p.read_text())) for p in files]


def plot_latency_vs_context(results_dir: Path, out_path: Path):
    """
    One line per attention variant: mean_latency_ms vs context_length,
    cached decoding only. Deliberately excludes the naive (no-cache)
    baseline — that number is ~3x larger and compresses the MHA/GQA/MQA
    differences (the actual point of this figure) into an unreadable
    cluster. The naive-vs-cached comparison belongs to section 03 and
    gets its own figure (see plot_cache_speedup below).
    """
    files = load_json_files(results_dir / "latency", "*.json")

    by_variant = {}
    for path, d in files:
        if d.get("cached", True):
            by_variant.setdefault(d["attention"], []).append(
                (d["context_length"], d["mean_latency_ms"])
            )

    if not by_variant:
        print("no cached latency results found, skipping latency plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, points in sorted(by_variant.items()):
        points.sort(key=lambda p: p[0])
        xs, ys = zip(*points)
        style = STYLE.get(variant, {})
        ax.plot(xs, ys, marker=style.get("marker", "o"), color=style.get("color"),
                 label=style.get("label", variant), linewidth=2, markersize=8)

    ax.set_xlabel("Context length")
    ax.set_ylabel("Mean latency / token (ms)")
    ax.set_title("Latency vs context length, by attention variant\n(batch=1, cached decoding)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_cache_speedup(results_dir: Path, out_path: Path):
    """
    Bar chart: naive vs cached decoding latency (MHA, context=512) — the
    section 03 KV-cache speedup number, isolated from the variant
    comparison so its much larger scale doesn't distort plot_latency_vs_context.
    """
    files = load_json_files(results_dir / "latency", "*.json")
    naive, cached = None, None
    for path, d in files:
        if d["attention"] == "mha" and d["context_length"] == 512:
            if d.get("cached", True):
                cached = d["mean_latency_ms"]
            else:
                naive = d["mean_latency_ms"]

    if naive is None or cached is None:
        print("missing naive or cached MHA ctx=512 point, skipping speedup plot")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["Naive\n(no cache)", "Cached"], [naive, cached],
                   color=["#d62728", STYLE["mha"]["color"]])
    for bar, val in zip(bars, [naive, cached]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.2f}ms",
                 ha="center", fontsize=11)
    speedup = naive / cached
    ax.set_ylabel("Mean latency / token (ms)")
    ax.set_title(f"KV cache speedup (MHA, context=512)\n{speedup:.1f}x faster with cache")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_kv_cache_size(results_dir: Path, out_path: Path):
    """
    One line per attention variant: KV cache bytes vs context_length.
    Theoretical and measured values are plotted on top of each other —
    they coincide exactly in this project's data, which is worth showing
    visually (a single line, not two separate ones with a gap).
    """
    files = load_json_files(results_dir / "memory", "*.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    for path, d in files:
        variant = d["attention"]
        style = STYLE.get(variant, {})
        points = sorted(d["results"], key=lambda r: r["context_length"])
        xs = [r["context_length"] for r in points]
        ys_theory = [r["theoretical_kv_cache_bytes"] / 1024 for r in points]  # KB
        ys_measured = [r["measured_kv_cache_bytes"] / 1024 for r in points]

        ax.plot(xs, ys_theory, marker=style.get("marker", "o"), color=style.get("color"),
                 label=style.get("label", variant), linewidth=2, markersize=8)
        # overlay measured as hollow markers - should sit exactly on the line
        ax.scatter(xs, ys_measured, facecolors="none", edgecolors=style.get("color"),
                   s=150, linewidths=1.5, zorder=5)

    ax.set_xlabel("Context length")
    ax.set_ylabel("KV cache size (KB)")
    ax.set_title("KV cache size vs context length, by attention variant\n"
                 "(theoretical formula vs measured — identical at every point)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_throughput_vs_batch(results_dir: Path, out_path: Path):
    """
    One line per attention variant: tokens/sec vs batch_size. This is the
    plot that shows MQA's crossover — flat/tracking with MHA at small
    batch, pulling ahead once batch size is large enough to be
    bandwidth-bound.
    """
    files = load_json_files(results_dir / "throughput", "*.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    for path, d in files:
        variant = d["attention"]
        style = STYLE.get(variant, {})
        points = sorted(
            [r for r in d["sweep"] if not r.get("oom")],
            key=lambda r: r["batch_size"],
        )
        xs = [r["batch_size"] for r in points]
        ys = [r["tokens_per_sec"] for r in points]
        ax.plot(xs, ys, marker=style.get("marker", "o"), color=style.get("color"),
                 label=style.get("label", variant), linewidth=2, markersize=8)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_xscale("log", base=2)
    ax.set_title("Throughput vs batch size, by attention variant\n(context length=512)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results/")
    p.add_argument("--out-dir", type=str, default="results/figures/")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_latency_vs_context(results_dir, out_dir / "latency_vs_context.png")
    plot_cache_speedup(results_dir, out_dir / "cache_speedup.png")
    plot_kv_cache_size(results_dir, out_dir / "kv_cache_size.png")
    plot_throughput_vs_batch(results_dir, out_dir / "throughput_vs_batch.png")

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()