from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from slime.utils.types import Sample

from examples.geo3k_vlm_sft_data_generation.format_trajectories_to_sft import (
    collect_rollout_paths,
    convert_rollout_files_to_parquet,
    format_answer,
    sample_to_sft_row,
)
from examples.geo3k_vlm_sft_data_generation.common import set_generation_result
from examples.geo3k_vlm_sft_data_generation.rollout import finalize_sample_metadata


def test_sample_to_sft_row_builds_geo3k_messages_for_correct_completed_sample():
    sample = Sample(
        prompt="Find x if 2x = 6.",
        label="3",
        status=Sample.Status.COMPLETED,
        metadata={
            "sft_generation": {
                "problem": "Find x if 2x = 6.",
                "final_answer_text": "3",
                "is_correct": True,
                "num_turns": 2,
                "model_name": "Qwen3-VL-4B-Instruct",
                "images": ["image-1.png"],
            }
        },
    )

    row = sample_to_sft_row(sample)

    assert row == {
        "problem": "Find x if 2x = 6.",
        "answer": "3",
        "images": ["image-1.png"],
        "messages": [
            {"role": "user", "content": "Find x if 2x = 6."},
            {"role": "assistant", "content": "Answer: \\boxed{3}"},
        ],
        "model_name": "Qwen3-VL-4B-Instruct",
        "num_turns": 2,
        "is_correct": True,
    }


def test_sample_to_sft_row_drops_incorrect_or_incomplete_samples():
    incorrect_sample = Sample(
        prompt="Question",
        label="7",
        status=Sample.Status.COMPLETED,
        metadata={"sft_generation": {"problem": "Question", "final_answer_text": "5", "is_correct": False}},
    )
    truncated_sample = Sample(
        prompt="Question",
        label="7",
        status=Sample.Status.TRUNCATED,
        metadata={"sft_generation": {"problem": "Question", "final_answer_text": "7", "is_correct": True}},
    )

    assert sample_to_sft_row(incorrect_sample) is None
    assert sample_to_sft_row(truncated_sample) is None


def test_format_answer_wraps_final_answer_for_sft():
    assert format_answer("270") == "Answer: \\boxed{270}"


def test_finalize_sample_metadata_adds_fallback_fields():
    sample = Sample(
        prompt="Solve for y.",
        label="9",
        response="Answer: \\boxed{9}",
        status=Sample.Status.COMPLETED,
        metadata={},
    )

    finalize_sample_metadata(sample, model_name="Qwen3-VL-4B-Instruct")

    assert sample.metadata["sft_generation"]["problem"] == "Solve for y."
    assert sample.metadata["sft_generation"]["final_answer_text"] == "9"
    assert sample.metadata["sft_generation"]["is_correct"] is True
    assert sample.metadata["sft_generation"]["finish_reason"] == "completed"


def test_finalize_sample_metadata_reconstructs_problem_text_from_multimodal_prompt():
    sample = Sample(
        prompt=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at "},
                    {"type": "image", "image": "img.png"},
                    {"type": "text", "text": " and solve."},
                ],
            }
        ],
        multimodal_inputs={"images": ["img.png"]},
        label="4",
        response="Answer: \\boxed{4}",
        status=Sample.Status.COMPLETED,
        metadata={},
    )

    finalize_sample_metadata(sample, model_name="Qwen3-VL-4B-Instruct")

    assert sample.metadata["sft_generation"]["problem"] == "Look at <image> and solve."


def test_finalize_sample_metadata_uses_first_user_message_when_system_prompt_exists():
    sample = Sample(
        prompt=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Solve 1+1."},
        ],
        label="2",
        response="Answer: \\boxed{2}",
        status=Sample.Status.COMPLETED,
        metadata={},
    )

    finalize_sample_metadata(sample, model_name="Qwen3-VL-4B-Instruct")

    assert sample.metadata["sft_generation"]["problem"] == "Solve 1+1."


def test_collect_rollout_paths_supports_single_file_directory_and_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        file_a = tmp / "rollout_0.pt"
        file_b = tmp / "rollout_1.pt"
        file_a.write_bytes(b"a")
        file_b.write_bytes(b"b")

        assert collect_rollout_paths(str(file_a)) == [file_a]
        assert collect_rollout_paths(str(tmp)) == [file_a, file_b]
        assert collect_rollout_paths(str(tmp / "rollout_*.pt")) == [file_a, file_b]


def test_convert_rollout_files_to_parquet_merges_multiple_rollout_shards():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rollout_a = tmp / "rollout_0.pt"
        rollout_b = tmp / "rollout_1.pt"
        output = tmp / "Qwen3-VL-4B-Instruct_train_formatted.parquet"

        sample_a = Sample(
            prompt="Question A",
            label="1",
            status=Sample.Status.COMPLETED,
            metadata={"sft_generation": {"problem": "Question A", "final_answer_text": "1", "is_correct": True}},
        )
        sample_b = Sample(
            prompt="Question B",
            label="2",
            status=Sample.Status.COMPLETED,
            metadata={"sft_generation": {"problem": "Question B", "final_answer_text": "2", "is_correct": True}},
        )

        torch.save({"samples": [sample_a.to_dict()]}, rollout_a)
        torch.save({"samples": [sample_b.to_dict()]}, rollout_b)

        total, kept = convert_rollout_files_to_parquet([rollout_a, rollout_b], output)
        df = pd.read_parquet(output)

        assert (total, kept) == (2, 2)
        assert df["problem"].tolist() == ["Question A", "Question B"]


def test_set_generation_result_extracts_answer_and_correctness():
    metadata = {}

    set_generation_result(metadata, response_text="Reasoning\nAnswer: \\boxed{8}", ground_truth="8")

    assert metadata["final_response_text"] == "Reasoning\nAnswer: \\boxed{8}"
    assert metadata["final_answer_text"] == "8"
    assert metadata["is_correct"] is True


def test_collect_rollout_paths_raises_when_no_rollout_files_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        try:
            collect_rollout_paths(str(tmp / "rollout_*.pt"))
        except FileNotFoundError as exc:
            assert "No rollout files found" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError for empty rollout glob")


def test_convert_rollout_files_to_parquet_raises_on_empty_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "out.parquet"

        try:
            convert_rollout_files_to_parquet([], output)
        except ValueError as exc:
            assert "No rollout files provided" in str(exc)
        else:
            raise AssertionError("Expected ValueError for empty rollout input")
