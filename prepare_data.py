"""
Streams FineWeb-Edu, tokenizes with tiktoken's gpt2 encoding, writes
train.bin / val.bin as uint16 memmap arrays. Same format used by
gpt2-speedrun — kept identical here so vocab_size=50257 in configs/*.yaml
is correct without changes.

Kept as a token-budget-capped streaming job on purpose: on Kaggle I don't
want to download the full FineWeb10B just to sanity-check the pipeline.
Bump --num_tokens once the baseline run is confirmed working.

Usage:
    python prepare_data.py --num_tokens 50_000_000 --out_dir data/
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=50_000_000,
                         help="total tokens to pull (train+val combined)")
    parser.add_argument("--val_fraction", type=float, default=0.0005,
                         help="fraction of tokens held out for validation")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]  # GPT-2 doc separator

    print(f"streaming {args.dataset} ({args.subset})...")
    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)

    val_tokens_target = int(args.num_tokens * args.val_fraction)
    train_tokens_target = args.num_tokens - val_tokens_target

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    train_buf, val_buf = [], []
    train_count, val_count = 0, 0

    pbar = tqdm(total=args.num_tokens, unit="tok")
    for doc in ds:
        text = doc["text"]
        ids = enc.encode_ordinary(text)
        ids.append(eot)

        if val_count < val_tokens_target:
            val_buf.extend(ids)
            val_count += len(ids)
            pbar.update(len(ids))
        elif train_count < train_tokens_target:
            train_buf.extend(ids)
            train_count += len(ids)
            pbar.update(len(ids))
        else:
            break
    pbar.close()

    train_arr = np.array(train_buf, dtype=np.uint16)
    val_arr = np.array(val_buf, dtype=np.uint16)

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"train.bin: {len(train_arr):,} tokens -> {train_path}")
    print(f"val.bin:   {len(val_arr):,} tokens -> {val_path}")
    print("done. these are memmap-able uint16 arrays, same format train.py expects.")


if __name__ == "__main__":
    main()