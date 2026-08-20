# BUILDLOG

Project log for transformer-inference-lab. One entry per work session.
Each entry: what I did, what I measured, what broke, what I decided.


## Runs index

| Date | Notebook | Git commit | Variant | Result |
|---|---|---|---|---|
| 2026-08-19 | mha-training | db649c6 | MHA | val_loss 5.6314, iter 5000 |

---
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



## 2026-08-19 — Day 3: sanity run on Kaggle T4 (fixed)

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




## 2026-08-19 — MHA checkpoint finalized

**Did:** Backfilled val_loss on mha.pt via one-off eval (train.py's final
save omitted it due to the loop-boundary bug). Uploaded mha.pt as a
Kaggle Dataset (transformer-inference-lab-checkpoints) for reuse across
future sessions.

**Measured:** MHA final val_loss = 5.6314 (eval_iters=50), down from
6.24 at iter 2000 — model was still improving through iter 5000, not
plateaued.

**Decision:** MHA baseline locked in. train.py fix (final checkpoint
now computes val_loss automatically) already committed — GQA/MQA runs
won't need this manual backfill step.

**Next:** launch GQA training (configs/gqa.yaml), same notebook
structure, same data Input, checkpoint output to gqa.pt.




| 2026-08-19 | gqa-training (labeled mqa-training, content is GQA — see note) | 2bc3275 | GQA | val_loss 5.7637, iter 5000 |



## 2026-08-19 — GQA training complete

**Did:** Full GQA training run (5000 iters, n_kv_head=2) on Kaggle T4.
Notebook file was misnamed "mqa-training" but content/config was
unchanged from the GQA notebook — confirmed via cell 10 output
(configs/gqa.yaml, n_kv_head: 2) and cell 14/16 (checkpoint gqa.pt).

**Measured:** GQA val_loss 5.7637 at iter 5000, vs MHA's 5.6314 —
GQA slightly higher loss than MHA, consistent with expectation (fewer
KV heads costs a small amount of quality). ~1.77s/it steady state.

**Issues:** notebook naming/content mismatch — renamed file without
updating content. Real MQA run still pending.

**Next:** build and run the actual MQA notebook (n_kv_head=1,
configs/mqa.yaml) — take care this time that the notebook content
matches its filename.


| 2026-08-20 | mqa-training | 2bc3275 | MQA | val_loss 5.6856, iter 5000 |


## 2026-08-20 — MQA training complete, all three variants done

**Did:** Full MQA training run (5000 iters, n_kv_head=1) on Kaggle T4.
Config and checkpoint verified as genuinely MQA via explicit n_kv_head
assertions (added after the GQA mislabeling incident) — both passed.

**Measured:** MQA val_loss 5.6856 at iter 5000. Final comparison across
all three variants:
  MHA (n_kv_head=8): 5.6314
  MQA (n_kv_head=1): 5.6856
  GQA (n_kv_head=2): 5.7637

**Issues:** none in execution. Notable result: GQA has the highest
(worst) val_loss of the three, not MQA — opposite of naive expectation.
Likely training noise at this short budget/single seed rather than a
real architectural effect; flagging as a limitation to state explicitly
in 06 — Analysis rather than overclaiming a trend.

**Decision:** proceeding to benchmarks (latency/memory/throughput) with
all three checkpoints as-is rather than rerunning with multiple seeds —
that's a possible follow-up if time allows, not blocking the memory/
latency measurements which are the primary focus.

**Next:** run benchmarks/latency.py, memory.py, throughput.py against
all three checkpoints. This is where the actual memory-latency axis of
the triangle gets real numbers.
---