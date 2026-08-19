"""
Training loop for the three attention variants (MHA / GQA / MQA).

Reuses the exact data format from gpt2-speedrun: train.bin / val.bin as
uint16 memmap arrays produced by tiktoken's gpt2 encoding (vocab_size=50257,
already set in configs/*.yaml). If you don't have data/train.bin yet, copy
prepare_data.py from gpt2-speedrun and run it once — the .bin files are
identical between the two projects, no need to regenerate per variant.

Deliberately kept simpler than the speedrun's optimizer setup: plain
AdamW + cosine schedule, no Muon. The point of this project is isolating
the effect of n_kv_head, not re-optimizing the training loop — adding
Muon here would be a second variable you'd have to control for across the
three runs, for no benefit to the research question.

fp16 AMP + GradScaler chosen over bf16 for T4/P100 compatibility, same
reasoning as the speedrun.

Uses gradient accumulation: batch_size is the per-step micro-batch that
fits in T4 memory, grad_accum_steps multiplies it back up to the intended
effective batch size (see configs/*.yaml comments).

Usage:
    python train.py --config configs/mha.yaml
    python train.py --config configs/gqa.yaml
    python train.py --config configs/mqa.yaml
    python train.py --config configs/gqa.yaml --resume results/checkpoints/gqa.pt
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.model.transformer import GPT, ModelConfig


def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    path = os.path.join(data_dir, "train.bin" if split == "train" else "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it: int, warmup_iters: int, max_iters: int, max_lr: float, min_lr_ratio: float = 0.1):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return max_lr * min_lr_ratio
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * min_lr_ratio + coeff * max_lr * (1 - min_lr_ratio)


@torch.no_grad()
def estimate_val_loss(model, data_dir, block_size, batch_size, device, eval_iters=20):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch("val", data_dir, block_size, batch_size, device)
        logits, _ = model(x, kv_caches=None)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg_dict["model"])
    train_cfg = cfg_dict["train"]

    torch.manual_seed(train_cfg["seed"])

    device = args.device
    use_amp = device == "cuda"
    amp_dtype = torch.float16

    model = GPT(model_cfg).to(device)
    print(f"model initialized: {model.num_params() / 1e6:.2f}M parameters "
          f"(n_kv_head={model_cfg.n_kv_head})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iter"] + 1
        print(f"resumed from {args.resume} at iter {start_iter}")

    checkpoint_name = Path(args.config).stem  # mha / gqa / mqa
    checkpoint_path = Path("results/checkpoints") / f"{checkpoint_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

    model.train()
    t0 = time.time()
    for it in range(start_iter, train_cfg["max_iters"]):
        lr = get_lr(it, train_cfg["warmup_iters"], train_cfg["max_iters"], train_cfg["learning_rate"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = get_batch("train", os.path.dirname(train_cfg["dataset"]),
                              model_cfg.block_size, train_cfg["batch_size"], device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x, kv_caches=None)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            print(f"iter {it:5d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt*1000/max(it-start_iter,1):.1f}ms/it")

        if it % args.eval_interval == 0 and it > start_iter:
            val_loss = estimate_val_loss(model, os.path.dirname(train_cfg["dataset"]),
                                          model_cfg.block_size, train_cfg["batch_size"], device)
            print(f"iter {it:5d} | val_loss {val_loss:.4f}")
            torch.save({
                "iter": it,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg_dict["model"],
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"checkpoint saved to {checkpoint_path}")

    # final checkpoint
    # final checkpoint — compute val_loss one last time so it's not lost
    final_val_loss = estimate_val_loss(model, os.path.dirname(train_cfg["dataset"]),
                                        model_cfg.block_size, train_cfg["batch_size"], device)
    print(f"final val_loss: {final_val_loss:.4f}")
    torch.save({
        "iter": train_cfg["max_iters"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg_dict["model"],
        "val_loss": final_val_loss,
    }, checkpoint_path)
    print(f"training complete. final checkpoint: {checkpoint_path}")
    

if __name__ == "__main__":
    main()