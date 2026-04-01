from __future__ import annotations

from typing import Any

from examples.geo3k_vlm_sft_data_generation.common import (
    extract_final_answer,
    extract_problem,
    infer_model_name,
    is_correct_answer,
)
from examples.geo3k_vlm_multi_turn.rollout import generate as base_generate
from slime.utils.types import Sample


def finalize_sample_metadata(sample: Sample, model_name: str | None = None) -> Sample:
    metadata = sample.metadata.setdefault("sft_generation", {})
    metadata.setdefault("problem", extract_problem(sample.prompt))
    metadata.setdefault("ground_truth_answer", sample.label)
    metadata.setdefault("model_name", model_name)
    metadata.setdefault("images", (sample.multimodal_inputs or {}).get("images"))
    metadata.setdefault("trajectory_messages", sample.prompt if isinstance(sample.prompt, list) else None)
    metadata["finish_reason"] = sample.status.value
    metadata["final_response_text"] = sample.response
    metadata["num_turns"] = metadata.get("num_turns", 1)

    final_answer = metadata.get("final_answer_text") or extract_final_answer(sample.response)
    metadata["final_answer_text"] = final_answer
    metadata["is_correct"] = bool(metadata.get("is_correct")) or is_correct_answer(final_answer, sample.label)
    return sample


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    generated_sample = await base_generate(args, sample, sampling_params)
    return finalize_sample_metadata(
        generated_sample,
        model_name=infer_model_name(getattr(args, "model_name", None), getattr(args, "hf_checkpoint", None)),
    )
