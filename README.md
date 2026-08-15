# Transformer Inference Lab

**Research question:** How does attention architecture affect the memory–latency–quality trade-off during autoregressive language-model inference?

Follow-up to [gpt2-speedrun](https://github.com/modestesavadogo/gpt2-speedrun). Where that project was about reproducing training, this one is about understanding *inference* well enough to explain, with measurements, why production LLM serving engines (vLLM, TensorRT-LLM) use techniques far more sophisticated than what's built here.

Everything below is built from scratch on top of a nanoGPT-style model: naive decoding → KV cache → MHA → GQA → MQA → memory/latency/quality analysis. FlashAttention and PagedAttention are read and discussed, not reimplemented — see [07 — Beyond](#07--beyond).

Status: 🚧 in progress, updated as I go. See `results/` for current numbers.

---

## Setup

```bash
git clone https://github.com/modestesavadogo/transformer-inference-lab.git
cd transformer-inference-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/  # correctness checks before trusting any benchmark
```

## Reproducing results

Every number in this README comes from a command you can re-run:

```bash
python benchmarks/latency.py \
    --attention gqa \
    --kv-heads 2 \
    --context-length 2048 \
    --checkpoint results/checkpoints/gqa.pt \
    --out results/latency/gqa_ctx2048.json

python benchmarks/throughput.py --attention mqa --kv-heads 1 \
    --context-length 2048 --checkpoint results/checkpoints/mqa.pt \
    --batch-sizes 1,4,8,16,32

python benchmarks/memory.py --attention mha --kv-heads 8 \
    --context-lengths 512,1024,2048,4096 --checkpoint results/checkpoints/mha.pt

python analysis/plots.py --results-dir results/ --out-dir results/figures/
```

Training the three attention variants (see [04](#04--mha--gqa--mqa) for why they're trained from scratch rather than sliced from one checkpoint):

```bash
python train.py --config configs/mha.yaml
python train.py --config configs/gqa.yaml
python train.py --config configs/mqa.yaml
```

`configs/{mha,gqa,mqa}.yaml` are identical except for `n_kv_head` — same seed, same token budget, same data — so quality differences in the results below are attributable to architecture, not training discrepancy.

---

## 01 — Problem

*Why autoregressive inference costs what it costs.*

> TODO: naive decoding is O(T²) in total compute across a full generation because every step recomputes attention over the entire prefix. Fill in with the actual per-token / total FLOPs derivation and a first plot from `benchmarks/latency.py --no-cache`.

## 02 — Baseline

*Naive decoding, no cache.*

> TODO: `src/inference/naive.py` benchmark numbers — latency/token, throughput, memory, across context length. This is the reference every later number gets compared against.

## 03 — KV Cache

*Implementation + benchmark.*

> TODO: `src/inference/kv_cache.py` vs `naive.py`, same model/prompt/sampling. Report speedup factor and where it plateaus (memory bandwidth, not compute, becomes the bottleneck — set this up for section 07).

## 04 — MHA / GQA / MQA

*Implementations + theory.*

Formula for one layer's KV cache size (see `attention.py: kv_cache_bytes`):

```
2 (K and V) × n_kv_head × seq_len × head_dim × dtype_bytes
```

Only the KV projection shrinks as `n_kv_head` drops — Q and output projections stay full size, so this is a *KV cache memory* optimization, not a general parameter-count reduction.

| Variant | Hq | Hkv | KV cache vs MHA |
|---|---|---|---|
| MHA | 8 | 8 | 1× (baseline) |
| GQA | 8 | 2 | 4× smaller |
| MQA | 8 | 1 | 8× smaller |

> TODO: measured KV cache sizes from `benchmarks/memory.py`, confirming the formula against real allocator numbers.

References: Shazeer, *Fast Transformer Decoding: One Write-Head Is All You Need* (2019); Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023).

## 05 — Experiments

*Latency / memory / throughput / perplexity.*

> TODO: full grid — attention variant × context length (and batch size for throughput) — plus held-out perplexity per variant. Eval split kept strictly separate from training data.

## 06 — Analysis

*The compromise observed.*

> TODO: this is the section that makes the repo a study rather than a demo. State plainly what the memory↔latency↔quality triangle looks like in the actual numbers, and where it agrees or disagrees with the GQA paper's reported trade-off.

## 07 — Beyond

*FlashAttention, PagedAttention, vLLM.*

> TODO: not implemented here — read for what they say about memory bandwidth and IO-awareness, then explain in your own words where this repo's naive pre-allocated cache breaks down (fragmentation, no batching across requests, no fused kernels) and how each technique addresses a specific piece of that.

---

## Repo structure

```
transformer-inference-lab/
├── src/
│   ├── model/
│   │   ├── attention.py     # MHA/GQA/MQA as one parameterized module
│   │   ├── transformer.py   # GPT wrapper, nanoGPT-speedrun-adjacent
│   │   └── cache.py         # KV cache storage/bookkeeping only
│   └── inference/
│       ├── naive.py         # no-cache baseline
│       ├── kv_cache.py      # prefill + incremental decode with cache
│       └── generation.py    # shared sampling utils, perplexity eval
├── experiments/
│   ├── kv_cache/
│   ├── mha_mqa_gqa/
│   └── context_length/
├── benchmarks/
│   ├── latency.py
│   ├── throughput.py
│   └── memory.py
├── analysis/
│   └── plots.py
├── configs/                 # mha.yaml / gqa.yaml / mqa.yaml — identical
│                             # except n_kv_head
├── results/                 # JSON outputs + figures (checkpoints gitignored)
├── tests/
│   └── test_attention.py    # correctness before benchmarking
└── README.md
```

## Non-goals

This does not aim to match vLLM or TensorRT-LLM performance. The goal is a small enough engine to understand every operation, then measure experimentally why industrial engines reach for more sophisticated techniques.
