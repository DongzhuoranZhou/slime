from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from examples.geo3k_vlm_multi_turn.env_geo3k import Geo3kEnv, _extract_ground_truth
from examples.geo3k_vlm_sft_data_generation.common import extract_problem
from slime.utils.types import Sample


def _prompt_messages(sample: Sample | None) -> list[dict[str, Any]]:
    if sample is None:
        return []
    if isinstance(sample.prompt, list):
        return deepcopy(sample.prompt)
    if isinstance(sample.prompt, str):
        return [{"role": "user", "content": sample.prompt}]
    return []


class Geo3kSftEnv(Geo3kEnv):
    def __init__(self, *, sample: Sample | None, ground_truth: str | None = None, max_turns: int | None = None):
        super().__init__(ground_truth=ground_truth, max_turns=max_turns)
        self.sample = sample
        self._messages = _prompt_messages(sample)
        self._metadata = (sample.metadata if sample is not None else {}).setdefault("sft_generation", {})
        self._metadata.setdefault("trajectory_messages", deepcopy(self._messages))
        self._metadata.setdefault("problem", extract_problem(sample.prompt) if sample is not None else None)
        self._metadata.setdefault("ground_truth_answer", ground_truth)
        self._metadata.setdefault("sample_id", sample.index if sample is not None else None)
        images = ((sample.multimodal_inputs or {}) if sample is not None else {}).get("images")
        if images is not None:
            images = [str(Path(image)) for image in images]
        self._metadata.setdefault("images", images)
        self._metadata.setdefault("num_turns", 0)

    def step(self, response_text: str):
        self._messages.append({"role": "assistant", "content": response_text})
        observation, done, info = super().step(response_text)

        self._metadata["trajectory_messages"] = deepcopy(self._messages)
        self._metadata["num_turns"] = self.turn
        self._metadata["last_info"] = deepcopy(info)

        if done:
            final_answer = self._extract_answer_from_text(response_text)
            self._metadata["final_response_text"] = response_text
            self._metadata["final_answer_text"] = final_answer
            self._metadata["is_correct"] = self._score_answer(final_answer or "") == 1.0
            return observation, done, info

        next_message = self.format_observation(observation)
        self._messages.append(next_message)
        self._metadata["trajectory_messages"] = deepcopy(self._messages)
        return observation, done, info


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> Geo3kSftEnv:
    ground_truth = _extract_ground_truth(sample)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    return Geo3kSftEnv(sample=sample, ground_truth=ground_truth, max_turns=max_turns)
