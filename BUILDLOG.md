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



## 2026-08-20 — Day 3: sanity run on Kaggle T4 (fixed)

**Did:** Fixed OOM via gradient accumulation (batch_size 64 -> micro-batch 8
x grad_accum_steps 8, same effective batch size). Re-ran 30-iter sanity
check on configs/mha.yaml, T4 GPU.

**Measured:** loss 10.99 -> 9.80 over 30 iters, val_loss 10.76 -> 10.18.
No OOM. ~2000ms/it at this batch/accum config -> full 5000-iter run
estimated at ~2.8h per variant, ~8-9h total for MHA+GQA+MQA.

**Issues:** initial OOM at batch_size=64 (logits tensor + cross_entropy
fp32 upcast too large for T4 16GB at block_size=1024). Fixed with
grad accumulation. Also hit a partial-edit bug where checkpoint_name
was referenced before assignment — fixed by replacing train.py in full.

**Decision:** sanity check passed. Proceeding to full training runs,
starting with MHA.

**Next:** launch full mha.yaml training (max_iters=5000) on Kaggle,
expect ~2.8h. Monitor for session timeout (12h limit, not a concern
for a single variant but worth checking Kaggle's autosave/commit
behavior for long runs).



## 2026-08-19 — Day 2: Kaggle dataset path fix

**Did:** Fixed broken symlinks — Kaggle's actual mount path for a
user-created dataset is /kaggle/input/datasets/<username>/<dataset-slug>/,
not /kaggle/input/<dataset-slug>/ as initially assumed.

**Issues:** ls -la on a symlink shows the link itself, not whether its
target exists — silently masked the broken path until np.memmap actually
tried to open the file. Added `test -f` checks after symlink creation
going forward.

**Decision:** corrected path locked into the training notebook template
for MHA/GQA/MQA runs.

**Next:** re-verify token counts with fixed links, then launch MHA
training as Save & Run All commit.
---