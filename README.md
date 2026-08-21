# Transformer Inference Lab

**Research question:** How does attention architecture affect the memory–latency–quality trade-off during autoregressive language-model inference?

Follow-up to [gpt2-speedrun](https://github.com/modestesavadogo/gpt2-speedrun). Where that project was about reproducing training, this one is about understanding *inference* well enough to explain, with measurements, why production LLM serving engines (vLLM, TensorRT-LLM) use techniques far more sophisticated than what's built here.

Everything below is built from scratch on top of a nanoGPT-style model: naive decoding → KV cache → MHA → GQA → MQA → memory/latency/quality analysis. FlashAttention and PagedAttention are read and discussed, not reimplemented — see [07 — Beyond](#07--beyond).

**Status:** core experiments complete (training, latency, memory, throughput benchmarks for all three variants). See [BUILDLOG.md](BUILDLOG.md) for the full session-by-session history, including bugs hit and how they were resolved.

All training and benchmarking in this project was run on Kaggle (T4 GPU), not locally — see [Running on Kaggle](#running-on-kaggle) below.

---

## Setup

```bash
git clone https://github.com/modestesavadogo/transformer-inference-lab.git
cd transformer-inference-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/  # correctness checks before trusting any benchmark
```

## Running on Kaggle

Training and all benchmarks in this repo were run as Kaggle notebooks (T4 GPU), not locally — a full training run takes ~2.5-3h per variant, impractical without a dedicated GPU.

1. **Data:** `prepare_data.py` streams FineWeb-Edu and tokenizes with tiktoken's `gpt2` encoding, writing `train.bin`/`val.bin` (uint16 memmap arrays). On Kaggle, generate once and save as a reusable Dataset:
```bash
   python prepare_data.py --num_tokens 50_000_000 --out_dir data
```
2. **Training:** one notebook per variant (`configs/mha.yaml`, `gqa.yaml`, `mqa.yaml`), each producing a checkpoint saved to a shared Kaggle Dataset for reuse across sessions.
3. **Benchmarks:** a separate notebook loads all three checkpoints and runs `benchmarks/latency.py`, `memory.py`, `throughput.py` across each.

**Known constraint:** `context_length + max_new_tokens` must stay at or below `block_size` (1024 for these checkpoints) — positions beyond `block_size` don't exist in the position embedding table. Exceeding it triggers a CUDA device-side assert that **poisons the CUDA context for the rest of the process** (a kernel/session restart is required, not just a retry). All commands in this README use values that respect this constraint.

## Reproducing results

Every number in this README comes from a command you can re-run. Benchmark scripts must be run with `-m` (module syntax), not as a direct file path, since they use package-relative imports:

```bash
python -m benchmarks.latency \
    --attention gqa \
    --kv-heads 2 \
    --context-length 512 \
    --max-new-tokens 128 \
    --checkpoint results/checkpoints/gqa.pt \
    --out results/latency/gqa_ctx512.json \
    --device cuda

python -m benchmarks.throughput \
    --attention mqa --kv-heads 1 \
    --context-length 512 --max-new-tokens 64 \
    --checkpoint results/checkpoints/mqa.pt \
    --batch-sizes 1,4,8,16,32,64 \
    --device cuda

python -m benchmarks.memory \
    --attention mha --kv-heads 8 \
    --context-lengths 256,512,768 \
    --checkpoint results/checkpoints/mha.pt \
    --device cuda

python analysis/plots.py --results-dir results/ --out-dir results/figures/
```

Training the three attention variants (see [04](#04--mha--gqa--mqa) for why they're trained from scratch rather than sliced from one checkpoint):

```bash
python train.py --config configs/mha.yaml --device cuda
python train.py --config configs/gqa.yaml --device cuda
python train.py --config configs/mqa.yaml --device cuda
```

`configs/{mha,gqa,mqa}.yaml` are identical except for `n_kv_head` — same seed, same token budget, same data — so quality differences in the results below are attributable to architecture, not training discrepancy.

---

## 01 — Problem

*Why autoregressive inference costs what it costs.*

Naive decoding recomputes attention over the entire prefix at every generation step, making total compute across a full generation O(T²) in sequence length T — each new token re-processes every token that came before it, even though most of that work was already done in the previous step.

Measured on the MHA checkpoint (context length 512, 64 generated tokens): **15.48ms/token** with naive decoding. Section 03 shows what caching that redundant work buys back.

## 02 — Baseline

*Naive decoding, no cache.*

`src/inference/naive.py`, MHA checkpoint, context length 512, 64 generated tokens: **15.48ms/token** mean latency. This is the reference every later number in this README is compared against.

## 03 — KV Cache

*Implementation + benchmark.*

`src/inference/kv_cache.py`, same MHA checkpoint, same context length: **4.80ms/token**, a **3.2× speedup** over naive decoding. Avoiding recomputation of already-processed positions is the single largest lever in this project — independent of which attention variant is used, since all three benefit from caching equally at the mechanism level.

## 04 — MHA / GQA / MQA

*Implementations + theory.*

Formula for one layer's KV cache size (see `attention.py: kv_cache_bytes`):