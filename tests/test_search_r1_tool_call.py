"""
Unit tests for examples/search_r1_tool_call/ — no GPU or real model required.

Tests cover:
  - SearchEnv (env_search.py): tool-call parsing, search dispatch, observation format,
    done signalling, edge cases, build_env factory
  - reward_func (reward.py): EM scoring, tool-use credit, answer extraction,
    tool-turn stripping, label format variants
  - prepare_data.py: tools column injection, system message rewriting

Run with:
    pytest tests/test_search_r1_tool_call.py -v
or:
    python tests/test_search_r1_tool_call.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── path setup ───────────────────────────────────────────────────────────────
# Make both `slime` and `examples` importable from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "examples" / "search-r1"))  # exposes local/google stubs

# Stub out heavy dependencies that are not needed for unit tests
for _mod in ("ray", "torch", "sglang_router", "aiohttp"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# ─────────────────────────────────────────────────────────────────────────────

from examples.search_r1_tool_call.env_search import (
    SearchEnv,
    WEB_SEARCH_TOOL,
    _passages2string,
    build_env,
)
from examples.search_r1_tool_call.reward import (
    _em_check,
    _extract_final_answer,
    _get_golden_answers,
    _normalize_answer,
    reward_func,
)
from slime.utils.types import Sample

# ── helpers ──────────────────────────────────────────────────────────────────

FAKE_SEARCH_RESULT = "Doc 1(Title: France) Paris is the capital of France.\nDoc 2(Title: History) Historical city."

_DEFAULT_CONFIG = {
    "search_backend": "local",
    "search_concurrency": 4,
    "topk": 3,
    "local_search_url": "http://localhost:8000/retrieve",
    "local_search_proxy": None,
    "google_api_key": "",
    "google_snippet_only": True,
    "google_proxy": None,
}


def _make_env(max_turns: int = 3, ground_truth=None) -> SearchEnv:
    return SearchEnv(ground_truth=ground_truth, max_turns=max_turns, config=_DEFAULT_CONFIG)


def _make_sample(response: str = "", label=None) -> Sample:
    return Sample(
        prompt="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        response=response,
        label=label,
        tokens=[1, 2, 3],
        response_length=3,
        status=Sample.Status.COMPLETED,
    )


def _make_args(**kwargs) -> Namespace:
    defaults = {
        "reward_score": 1.0,
        "reward_format_score": 0.1,
        "reward_tool_use_score": 0.05,
        "max_turns": 2,
        "search_backend": "local",
        "search_concurrency": 4,
        "topk": 3,
        "local_search_url": "http://localhost:8000/retrieve",
        "local_search_proxy": None,
        "google_api_key": "",
        "google_snippet_only": True,
        "google_proxy": None,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


# ═════════════════════════════════════════════════════════════════════════════
# SearchEnv tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSearchEnvParsing:
    """Tool-call JSON parsing logic."""

    def test_valid_tool_call_standard_schema(self):
        """{"name": ..., "arguments": {...}} — the primary Qwen format."""
        env = _make_env()
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            obs, done, info = env.step(
                '<tool_call>{"name": "web_search", "arguments": {"query": "capital of France"}}</tool_call>'
            )
        assert info["tool_executed"] is True
        assert info["query"] == "capital of France"
        assert "Paris" in obs["obs_str"]
        assert obs["role"] == "tool"

    def test_valid_tool_call_function_schema(self):
        """{"function": {"name": ..., "arguments": ...}} — alternative schema."""
        env = _make_env()
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            obs, done, info = env.step(
                '<tool_call>{"function": {"name": "web_search", "arguments": {"query": "Eiffel Tower"}}}</tool_call>'
            )
        assert info["tool_executed"] is True
        assert info["query"] == "Eiffel Tower"

    def test_arguments_as_nested_json_string(self):
        """arguments field may be a JSON-encoded string."""
        env = _make_env()
        payload = json.dumps({"name": "web_search", "arguments": json.dumps({"query": "nested query"})})
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            obs, done, info = env.step(f"<tool_call>{payload}</tool_call>")
        assert info["tool_executed"] is True
        assert info["query"] == "nested query"

    def test_multiple_tool_calls_uses_last(self):
        """When the model emits multiple <tool_call> blocks, the last one is used."""
        env = _make_env()
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            _, _, info = env.step(
                '<tool_call>{"name": "web_search", "arguments": {"query": "first"}}</tool_call>'
                "some reasoning text "
                '<tool_call>{"name": "web_search", "arguments": {"query": "second"}}</tool_call>'
            )
        assert info["query"] == "second"

    def test_whitespace_around_json(self):
        """Regex should tolerate whitespace between tags and JSON."""
        env = _make_env()
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            _, _, info = env.step(
                '<tool_call>  \n  {"name": "web_search", "arguments": {"query": "spaces"}}  \n  </tool_call>'
            )
        assert info["tool_executed"] is True


class TestSearchEnvDoneSignalling:
    """Episode termination logic."""

    def test_no_tool_call_means_done(self):
        """If the model gives no <tool_call>, it has finished — done=True."""
        env = _make_env(max_turns=3)
        obs, done, info = env.step("The capital of France is Paris. <answer>Paris</answer>")
        assert done is True
        assert info["tool_executed"] is False

    def test_done_false_before_max_turns(self):
        """Valid tool call before max_turns → done=False."""
        env = _make_env(max_turns=3)
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            _, done, _ = env.step(
                '<tool_call>{"name": "web_search", "arguments": {"query": "q"}}</tool_call>'
            )
        assert done is False

    def test_done_true_at_max_turns(self):
        """Valid tool call on the last allowed turn → done=True."""
        env = _make_env(max_turns=1)
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            _, done, _ = env.step(
                '<tool_call>{"name": "web_search", "arguments": {"query": "q"}}</tool_call>'
            )
        assert done is True

    def test_turn_counter_increments(self):
        """Each step increments the turn counter."""
        env = _make_env(max_turns=5)
        assert env.turn == 0
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            env.step('<tool_call>{"name": "web_search", "arguments": {"query": "q1"}}</tool_call>')
            assert env.turn == 1
            env.step('<tool_call>{"name": "web_search", "arguments": {"query": "q2"}}</tool_call>')
            assert env.turn == 2

    def test_reset_clears_turn_counter(self):
        env = _make_env(max_turns=2)
        with patch("examples.search_r1_tool_call.env_search.run", return_value=FAKE_SEARCH_RESULT):
            env.step('<tool_call>{"name": "web_search", "arguments": {"query": "q"}}</tool_call>')
        assert env.turn == 1
        env.reset()
        assert env.turn == 0


class TestSearchEnvErrorCases:
    """Edge cases and invalid tool calls."""

    def test_wrong_tool_name_returns_error_obs(self):
        env = _make_env(max_turns=3)
        obs, done, info = env.step(
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        )
        assert info["tool_executed"] is False
        assert "not supported" in obs["obs_str"].lower() or "calculator" in obs["obs_str"]

    def test_empty_query_returns_error_obs(self):
        env = _make_env(max_turns=3)
        obs, done, info = env.step(
            '<tool_call>{"name": "web_search", "arguments": {"query": ""}}</tool_call>'
        )
        assert info["tool_executed"] is False
        assert "query" in obs["obs_str"].lower() or "provided" in obs["obs_str"].lower()

    def test_malformed_json_treated_as_no_tool_call(self):
        """Non-parseable JSON inside <tool_call> → no tool call detected → done."""
        env = _make_env(max_turns=3)
        obs, done, info = env.step("<tool_call>{broken json!!!</tool_call>")
        assert info["tool_executed"] is False
        assert done is True  # no valid tool call → model is "done"


class TestSearchEnvObservationFormat:
    """format_observation must return role='tool' for Qwen chat template."""

    def test_format_observation_role_is_tool(self):
        env = _make_env()
        msg = env.format_observation({"obs_str": "results here", "role": "tool"})
        assert msg["role"] == "tool"
        assert msg["content"] == "results here"

    def test_format_observation_empty_obs(self):
        env = _make_env()
        msg = env.format_observation({"obs_str": "", "role": "tool"})
        assert msg["role"] == "tool"
        assert msg["content"] == ""


class TestBuildEnv:
    """build_env factory function."""

    def test_build_env_sets_tools_in_metadata(self):
        """build_env must inject WEB_SEARCH_TOOL into sample.metadata['tools']."""
        sample = _make_sample()
        args = _make_args()
        env = build_env(sample=sample, args=args)
        assert sample.metadata is not None
        assert "tools" in sample.metadata
        assert sample.metadata["tools"] == [WEB_SEARCH_TOOL]
        assert isinstance(env, SearchEnv)

    def test_build_env_without_sample(self):
        """build_env(sample=None) should still construct an env (with a warning)."""
        args = _make_args()
        env = build_env(sample=None, args=args)
        assert isinstance(env, SearchEnv)

    def test_build_env_missing_max_turns_raises(self):
        """If max_turns is None, build_env should raise ValueError."""
        import pytest
        args = _make_args(max_turns=None)
        with pytest.raises(ValueError, match="max_turns"):
            build_env(sample=_make_sample(), args=args)

    def test_web_search_tool_schema_is_valid(self):
        """WEB_SEARCH_TOOL must follow OpenAI JSON schema format."""
        assert WEB_SEARCH_TOOL["type"] == "function"
        fn = WEB_SEARCH_TOOL["function"]
        assert fn["name"] == "web_search"
        assert "query" in fn["parameters"]["properties"]
        assert "query" in fn["parameters"]["required"]


# ═════════════════════════════════════════════════════════════════════════════
# reward.py tests
# ═════════════════════════════════════════════════════════════════════════════

def _response_with_tool_and_answer(answer: str) -> str:
    """Simulate a full response: tool call + injected results + final answer."""
    return (
        "<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"capital\"}}</tool_call>"
        "<|im_start|>tool\n"
        "Doc 1(Title: France) Paris is the capital of France, a beautiful city.\n"
        "<|im_end|>"
        f"<answer>{answer}</answer>"
    )


class TestRewardScoring:
    """Full reward_func scoring logic."""

    def test_em_match_with_tool(self):
        sample = _make_sample(
            response=_response_with_tool_and_answer("Paris"),
            label={"ground_truth": {"target": ["Paris"]}},
        )
        score = asyncio.run(reward_func(_make_args(), sample))
        assert score == 1.0

    def test_em_match_no_tool(self):
        """EM match but model never used <tool_call> — slight penalty."""
        sample = _make_sample(
            response="The capital is <answer>Paris</answer>",
            label={"ground_truth": {"target": ["Paris"]}},
        )
        score = asyncio.run(reward_func(_make_args(), sample))
        assert score == pytest.approx(0.9)

    def test_no_em_tool_used(self):
        sample = _make_sample(
            response=_response_with_tool_and_answer("London"),
            label={"ground_truth": {"target": ["Paris"]}},
        )
        score = asyncio.run(reward_func(_make_args(), sample))
        assert score == pytest.approx(0.05)

    def test_no_em_no_tool(self):
        sample = _make_sample(
            response="<answer>London</answer>",
            label={"ground_truth": {"target": ["Paris"]}},
        )
        score = asyncio.run(reward_func(_make_args(), sample))
        assert score == 0.0

    def test_no_answer_tool_used(self):
        """Tool called but no <answer> tag → partial credit."""
        sample = _make_sample(
            response='<tool_call>{"name": "web_search", "arguments": {"query": "q"}}</tool_call>',
            label={"ground_truth": {"target": ["Paris"]}},
        )
        score = asyncio.run(reward_func(_make_args(), sample))
        assert score == pytest.approx(0.05)

    def test_no_answer_no_tool(self):
        sample = _make_sample(
            response="I don't know.",
            label={"ground_truth": {"target": ["Paris"]}},
        )
        score = asyncio.run(reward_func(_make_args(), sample))
        assert score == 0.0

    def test_custom_reward_weights(self):
        """Reward weights from config.yaml (via args) are respected."""
        args = _make_args(reward_score=2.0, reward_format_score=0.5, reward_tool_use_score=0.1)
        sample = _make_sample(
            response=_response_with_tool_and_answer("Paris"),
            label={"ground_truth": {"target": ["Paris"]}},
        )
        score = asyncio.run(reward_func(args, sample))
        assert score == pytest.approx(2.0)


class TestAnswerExtraction:
    """_extract_final_answer strips tool turns before looking for <answer>."""

    def test_answer_in_final_response_extracted(self):
        response = _response_with_tool_and_answer("Paris")
        assert _extract_final_answer(response) == "Paris"

    def test_answer_inside_tool_turn_not_extracted(self):
        """<answer> text embedded in search results must NOT be extracted.
        This is the critical test: web content might contain '<answer>...' literally."""
        response = (
            '<tool_call>{"name": "web_search", "arguments": {"query": "q"}}</tool_call>'
            "<|im_start|>tool\n"
            "Doc 1 The answer is Rome, not Paris. <answer>Rome</answer>\n"  # inside tool turn
            "<|im_end|>"
            "<answer>Paris</answer>"  # actual model answer after tool turn
        )
        assert _extract_final_answer(response) == "Paris"

    def test_no_answer_tag_returns_none(self):
        assert _extract_final_answer("The capital is Paris.") is None

    def test_last_answer_tag_used(self):
        """If multiple <answer> tags exist, the last one is used."""
        response = "<answer>wrong</answer> more thinking <answer>correct</answer>"
        assert _extract_final_answer(response) == "correct"

    def test_multiline_answer(self):
        response = "<answer>\n  New York City\n</answer>"
        assert _extract_final_answer(response) == "New York City"


class TestEMLogic:
    """EM normalization and matching."""

    def test_em_case_insensitive(self):
        assert _em_check("paris", ["Paris"])
        assert _em_check("PARIS", ["Paris"])

    def test_em_strips_articles(self):
        assert _em_check("the eiffel tower", ["Eiffel Tower"])

    def test_em_strips_punctuation(self):
        assert _em_check("paris, france", ["paris france"])

    def test_em_multiple_golden_answers(self):
        assert _em_check("NYC", ["New York City", "NYC", "New York"])

    def test_em_no_match(self):
        assert not _em_check("London", ["Paris"])

    def test_normalize_collapses_whitespace(self):
        assert _normalize_answer("  hello   world  ") == "hello world"


class TestLabelFormats:
    """_get_golden_answers handles the various label formats in the wild."""

    def test_search_r1_nested_dict(self):
        label = {"ground_truth": {"target": ["Paris", "paris, france"]}}
        assert _get_golden_answers(label) == ["Paris", "paris, france"]

    def test_flat_list(self):
        assert _get_golden_answers(["Paris", "France"]) == ["Paris", "France"]

    def test_plain_string(self):
        assert _get_golden_answers("Paris") == ["Paris"]

    def test_none_label(self):
        assert _get_golden_answers(None) is None


# ═════════════════════════════════════════════════════════════════════════════
# prepare_data.py tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPrepareData:
    """Data preprocessing: tools column + system message rewriting."""

    def _make_df(self, with_system=True):
        import pandas as pd

        messages = []
        if with_system:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": "What is the capital of France?"})

        return pd.DataFrame(
            {
                "prompt": [messages],
                "reward_model": [{"ground_truth": {"target": ["Paris"]}}],
            }
        )

    def test_tools_column_added(self, tmp_path):
        import pandas as pd

        from examples.search_r1_tool_call.prepare_data import WEB_SEARCH_TOOL, add_tools_column

        df = self._make_df()
        input_path = tmp_path / "train.parquet"
        output_path = tmp_path / "train_with_tools.parquet"
        df.to_parquet(input_path)

        add_tools_column(str(input_path), str(output_path), add_tool_instructions=False)

        result = pd.read_parquet(output_path)
        assert "tools" in result.columns
        tools = json.loads(result["tools"].iloc[0])
        assert tools == [WEB_SEARCH_TOOL]
        assert tools[0]["function"]["name"] == "web_search"

    def test_reward_model_column_preserved(self, tmp_path):
        import pandas as pd

        from examples.search_r1_tool_call.prepare_data import add_tools_column

        df = self._make_df()
        input_path = tmp_path / "train.parquet"
        output_path = tmp_path / "out.parquet"
        df.to_parquet(input_path)
        add_tools_column(str(input_path), str(output_path))

        result = pd.read_parquet(output_path)
        assert "reward_model" in result.columns

    def test_system_message_rewritten_with_flag(self, tmp_path):
        import pandas as pd

        from examples.search_r1_tool_call.prepare_data import (
            SYSTEM_MESSAGE_WITH_TOOLS,
            add_tools_column,
        )

        df = self._make_df(with_system=True)
        input_path = tmp_path / "train.parquet"
        output_path = tmp_path / "out.parquet"
        df.to_parquet(input_path)
        add_tools_column(str(input_path), str(output_path), add_tool_instructions=True)

        result = pd.read_parquet(output_path)
        messages = result["prompt"].iloc[0]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == SYSTEM_MESSAGE_WITH_TOOLS

    def test_system_message_prepended_if_missing(self, tmp_path):
        import pandas as pd

        from examples.search_r1_tool_call.prepare_data import add_tools_column

        df = self._make_df(with_system=False)
        input_path = tmp_path / "train.parquet"
        output_path = tmp_path / "out.parquet"
        df.to_parquet(input_path)
        add_tools_column(str(input_path), str(output_path), add_tool_instructions=True)

        result = pd.read_parquet(output_path)
        messages = result["prompt"].iloc[0]
        assert messages[0]["role"] == "system"

    def test_system_message_unchanged_without_flag(self, tmp_path):
        import pandas as pd

        from examples.search_r1_tool_call.prepare_data import add_tools_column

        original_system = "You are a helpful assistant."
        df = self._make_df(with_system=True)
        input_path = tmp_path / "train.parquet"
        output_path = tmp_path / "out.parquet"
        df.to_parquet(input_path)
        add_tools_column(str(input_path), str(output_path), add_tool_instructions=False)

        result = pd.read_parquet(output_path)
        messages = result["prompt"].iloc[0]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert system_msgs[0]["content"] == original_system


# ═════════════════════════════════════════════════════════════════════════════
# _passages2string helper
# ═════════════════════════════════════════════════════════════════════════════

def test_passages2string_format():
    """Search result formatter mirrors examples/search-r1 output."""
    docs = [
        {"document": {"contents": "France\nFrance is a country in Western Europe."}},
        {"document": {"contents": "Paris\nParis is the capital city of France."}},
    ]
    result = _passages2string(docs)
    assert "Doc 1(Title: France)" in result
    assert "Doc 2(Title: Paris)" in result
    assert "Western Europe" in result


# ═════════════════════════════════════════════════════════════════════════════
# Entry point for running directly
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
