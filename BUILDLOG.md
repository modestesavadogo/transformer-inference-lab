# BUILDLOG

Project log for transformer-inference-lab. One entry per work session.
Each entry: what I did, what I measured, what broke, what I decided.

---

## 2026-08-19 — Repo scaffold

**Did:** Set up repo structure (src/model, src/inference, benchmarks,
configs, tests). Wrote attention.py (MHA/GQA/MQA as one parameterized
module), cache.py, transformer.py, train.py. 5 unit tests passing,
including numerical agreement check between naive and cached decoding.

**Measured:** n/a (scaffold only, no real training yet)

**Issues:** cache.py defaulted to device="cuda", broke CPU test runs.
Fixed default to "cpu", call sites pass real device explicitly.

**Decision:** Train MHA/GQA/MQA as three separate checkpoints from
scratch rather than slicing one MHA checkpoint — head-merging without
retraining would measure "what happens when you break a model," not
what MQA/GQA actually cost in quality.

**Next:** get train.bin/val.bin onto Kaggle, sanity-check train.py on
real GPU before committing to full runs.

---