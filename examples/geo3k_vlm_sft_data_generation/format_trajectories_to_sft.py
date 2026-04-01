from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from examples.geo3k_vlm_sft_data_generation.common import format_answer
from slime.utils.types import Sample


def sample_to_sft_row(sample: Sample) -> dict[str, Any] | None:
    metadata = sample.metadata.get("sft_generation", {})
    if sample.status != Sample.Status.COMPLETED:
        return None
    if not metadata.get("is_correct"):
        return None

    problem = metadata.get("problem")
    final_answer = metadata.get("final_answer_text")
    if not problem or not final_answer:
        return None

    row: dict[str, Any] = {
        "problem": problem,
        "answer": sample.label if sample.label is not None else final_answer,
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": format_answer(final_answer)},
        ],
        "model_name": metadata.get("model_name"),
        "num_turns": metadata.get("num_turns"),
        "is_correct": True,
    }
    if metadata.get("images") is not None:
        row["images"] = metadata["images"]
    if metadata.get("sample_id") is not None:
        row["sample_id"] = metadata["sample_id"]
    return row


def load_debug_rollout_samples(path: str | Path) -> list[Sample]:
    payload = torch.load(path, weights_only=False)
    return [Sample.from_dict(sample_dict) for sample_dict in payload["samples"]]


def build_sft_rows(samples: list[Sample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        row = sample_to_sft_row(sample)
        if row is not None:
            rows.append(row)
    return rows


def collect_rollout_paths(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if any(char in str(input_path) for char in "*?[]"):
        return sorted(Path(match) for match in glob(str(input_path)))
    if path.is_dir():
        return sorted(path.glob("*.pt"))
    return [path]


def convert_rollout_files_to_parquet(input_paths: list[str | Path], output_path: str | Path) -> tuple[int, int]:
    all_samples: list[Sample] = []
    for input_path in input_paths:
        all_samples.extend(load_debug_rollout_samples(input_path))

    rows = build_sft_rows(all_samples)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output)
    return len(all_samples), len(rows)


def convert_rollout_file_to_parquet(input_path: str | Path, output_path: str | Path) -> tuple[int, int]:
    return convert_rollout_files_to_parquet([input_path], output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert saved debug rollout trajectories into SFT parquet.")
    parser.add_argument("--input", required=True, help="Path to a saved debug rollout .pt file.")
    parser.add_argument("--output", required=True, help="Output parquet path, e.g. Qwen3-VL-4B-Instruct_train_formatted.parquet.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = collect_rollout_paths(args.input)
    total, kept = convert_rollout_files_to_parquet(input_paths, args.output)
    print(f"Converted {kept}/{total} samples into {args.output}")


if __name__ == "__main__":
    main()
