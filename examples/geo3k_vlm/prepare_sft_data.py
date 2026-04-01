"""
Prepare geo3k_imgurl dataset for SFT training.

Formats answers into \\boxed{} notation with chat-style messages,
then saves as a parquet file.

Usage:
    # Load from local parquet (offline):
    python examples/geo3k_vlm/prepare_sft_data.py --local-path /workspace/.cache/huggingface/datasets/geo3k_imgurl_processed/train.parquet

    # Download from HuggingFace (requires network):
    python examples/geo3k_vlm/prepare_sft_data.py --dataset chenhegu/geo3k_imgurl --split train
"""

import argparse

from datasets import load_dataset


def process_sample(sample):
    formatted_answer = f"Answer: \\boxed{{{sample['answer']}}}"
    sample["messages"] = [
        {"role": "user", "content": sample["problem"]},
        {"role": "assistant", "content": formatted_answer},
    ]
    return sample


def main():
    parser = argparse.ArgumentParser(description="Prepare geo3k SFT training data")
    parser.add_argument("--local-path", default=None, help="Load from a local parquet file instead of HuggingFace")
    parser.add_argument("--dataset", default="chenhegu/geo3k_imgurl", help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--output", default="/workspace/.cache/huggingface/datasets/geo3k_imgurl/train_formatted.parquet", help="Output parquet path")
    parser.add_argument("--cache-dir", default="/workspace/.cache/huggingface/datasets", help="Local HuggingFace cache directory")
    args = parser.parse_args()

    if args.local_path:
        print(f"Loading from local file: {args.local_path}")
        ds = load_dataset("parquet", data_files=args.local_path, split="train")
    else:
        print(f"Loading dataset: {args.dataset} (split={args.split})")
        ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    print(f"Loaded {len(ds)} samples")

    ds = ds.map(process_sample)
    print(f"Formatted {len(ds)} samples")

    ds.to_parquet(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
