from unittest.mock import patch

import yaml

from docagent_runner import DEFAULT_SYSTEM_PROMPT
from scripts.open_source_agent import OpenSourceDocAgentConfig, OpenSourceDocAgentRunner


def test_build_model_uses_transformers_model():
    config = OpenSourceDocAgentConfig(
        model_id="Qwen/Test-Model",
        device_map="cpu",
        torch_dtype="float16",
        trust_remote_code=False,
        max_new_tokens=123,
        model_kwargs={"revision": "main"},
    )
    runner = OpenSourceDocAgentRunner(config)

    with patch("open_source_agent.TransformersModel") as mock_model:
        runner.build_model()

    mock_model.assert_called_once_with(
        model_id="Qwen/Test-Model",
        device_map="cpu",
        torch_dtype="float16",
        trust_remote_code=False,
        model_kwargs={"revision": "main"},
        max_new_tokens=123,
    )


def test_build_agent_wires_tools_model_and_prompt():
    runner = OpenSourceDocAgentRunner(OpenSourceDocAgentConfig())

    with patch("docagent_runner.DocAgent") as mock_agent:
        with patch.object(runner, "build_model", return_value="model") as _mock_model:
            with patch.object(runner, "build_tools", return_value=["tool"]) as _mock_tools:
                agent = runner.build_agent()

    mock_agent.assert_called_once_with(
        tools=["tool"],
        model="model",
        instructions=DEFAULT_SYSTEM_PROMPT,
    )
    assert agent == mock_agent.return_value


def test_config_uses_model_path_over_model_id(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"model_id": "Qwen/Remote-Id", "model_path": "/models/qwen"}),
        encoding="utf-8",
    )

    config = OpenSourceDocAgentConfig.from_yaml(config_path)

    assert config.model_id == "/models/qwen"
