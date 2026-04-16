from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import yaml
from smolagents import TransformersModel

from docagent_runner import BaseDocAgentConfig, BaseDocAgentRunner

@dataclass(frozen=True)
class OpenSourceDocAgentConfig(BaseDocAgentConfig):
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    device_map: str | None = None
    torch_dtype: str | None = None
    trust_remote_code: bool = True
    max_new_tokens: int = 2048
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    results_jsonl_path: str = "results/top50_run_Qwen.jsonl"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpenSourceDocAgentConfig":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML config must be a mapping, got {type(raw)}")

        base = BaseDocAgentConfig.from_dict(raw)
        model_path = raw.get("model_path")
        model_id = model_path or raw.get("model_id", cls.model_id)
        model_kwargs = raw.get("model_kwargs", {})
        if not isinstance(model_kwargs, dict):
            raise ValueError("model_kwargs must be a mapping")

        return cls(
            dataset=base.dataset,
            top_k=base.top_k,
            sample_num=base.sample_num,
            results_jsonl_path=raw.get("results_jsonl_path", cls.results_jsonl_path),
            debug_pickle_path=raw.get("debug_pickle_path", base.debug_pickle_path),
            model_id=model_id,
            device_map=raw.get("device_map"),
            torch_dtype=raw.get("torch_dtype"),
            trust_remote_code=raw.get("trust_remote_code", cls.trust_remote_code),
            max_new_tokens=raw.get("max_new_tokens", cls.max_new_tokens),
            model_kwargs=model_kwargs,
        )

    @classmethod
    def from_env(cls) -> "OpenSourceDocAgentConfig":
        def get_int(name: str, default: int) -> int:
            value = os.getenv(name)
            return int(value) if value else default

        def get_bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "y"}

        model_kwargs_json = os.getenv("QWEN_MODEL_KWARGS_JSON")
        model_kwargs = json.loads(model_kwargs_json) if model_kwargs_json else {}

        model_path = os.getenv("QWEN_MODEL_PATH")
        model_id = model_path if model_path else os.getenv("QWEN_MODEL_ID", cls.model_id)

        base = BaseDocAgentConfig.from_env(
            dataset_env="QWEN_DATASET",
            top_k_env="QWEN_TOP_K",
            sample_num_env="QWEN_SAMPLE_NUM",
            results_jsonl_env="QWEN_RESULTS_JSONL_PATH",
            debug_pickle_env="QWEN_DEBUG_PICKLE_PATH",
        )

        return cls(
            dataset=base.dataset,
            top_k=base.top_k,
            sample_num=base.sample_num,
            results_jsonl_path=base.results_jsonl_path,
            debug_pickle_path=base.debug_pickle_path,
            model_id=model_id,
            device_map=os.getenv("QWEN_DEVICE_MAP"),
            torch_dtype=os.getenv("QWEN_TORCH_DTYPE"),
            trust_remote_code=get_bool("QWEN_TRUST_REMOTE_CODE", cls.trust_remote_code),
            max_new_tokens=get_int("QWEN_MAX_NEW_TOKENS", cls.max_new_tokens),
            model_kwargs=model_kwargs,
        )


class OpenSourceDocAgentRunner(BaseDocAgentRunner):
    def build_model(self) -> TransformersModel:
        return TransformersModel(
            model_id=self.config.model_id,
            device_map=self.config.device_map,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            model_kwargs=self.config.model_kwargs,
            max_new_tokens=self.config.max_new_tokens,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the open-source DocAgent with a local model.")
    parser.add_argument(
        "--config",
        default="open_source_agent.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    config = OpenSourceDocAgentConfig.from_yaml(args.config)
    runner = OpenSourceDocAgentRunner(config)
    agent = runner.build_agent()
    runner.top_run(agent)
