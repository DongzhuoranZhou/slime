from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.agentic_doc_rl.env_doc_search import TOOLS, SEARCH_DATABASE_TOOL, GET_SPECIFIC_PAGES_TOOL


def test_tools_list_has_two_entries():
    assert len(TOOLS) == 2


def test_search_database_tool_schema():
    fn = SEARCH_DATABASE_TOOL["function"]
    assert fn["name"] == "search_database"
    params = fn["parameters"]["properties"]
    assert "query" in params
    assert "doc_name" in params
    assert "excluded_pages" in params
    assert fn["parameters"]["required"] == ["query", "doc_name"]


def test_get_specific_pages_tool_schema():
    fn = GET_SPECIFIC_PAGES_TOOL["function"]
    assert fn["name"] == "get_specific_pages"
    params = fn["parameters"]["properties"]
    assert "doc_name" in params
    assert "page_numbers" in params
    assert set(fn["parameters"]["required"]) == {"doc_name", "page_numbers"}


import asyncio
import tempfile
from unittest.mock import MagicMock
import PIL.Image

from examples.agentic_doc_rl.env_doc_search import DocSearchEnv, _extract_tool_call


# ── _extract_tool_call ────────────────────────────────────────────────────────

def test_extract_tool_call_qwen_native_format():
    text = '<tool_call>{"name": "search_database", "arguments": {"query": "revenue", "doc_name": "report"}}</tool_call>'
    result = _extract_tool_call(text)
    assert result == {"name": "search_database", "arguments": {"query": "revenue", "doc_name": "report"}}


def test_extract_tool_call_returns_none_for_plain_text():
    assert _extract_tool_call("This is a plain response with no tool call.") is None


def test_extract_tool_call_returns_last_when_multiple():
    text = (
        '<tool_call>{"name": "search_database", "arguments": {"query": "q1", "doc_name": "d"}}</tool_call>'
        ' some text '
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "42"}}</tool_call>'
    )
    assert _extract_tool_call(text)["name"] == "final_answer"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_env(max_turns=6) -> DocSearchEnv:
    mock_client = MagicMock()
    mock_embed = MagicMock()
    mock_embed.embed_text.return_value = MagicMock(tolist=lambda: [0.1] * 128)
    return DocSearchEnv(
        ground_truth="42",
        max_turns=max_turns,
        qdrant_client=mock_client,
        embed_model=mock_embed,
        collection_name="test_collection",
        topk=3,
    )


def _make_fake_png_path() -> str:
    img = PIL.Image.new("RGB", (10, 10), color=(255, 0, 0))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        return f.name


def _fake_query_result(pages: list[int]):
    points = []
    for page in pages:
        point = MagicMock()
        point.payload = {"full_img_path": _make_fake_png_path(), "page_num": page}
        points.append(point)
    result = MagicMock()
    result.points = points
    return result


# ── step() tests ──────────────────────────────────────────────────────────────

def test_step_final_answer_returns_done():
    env = _make_env()
    env.reset()
    text = '<tool_call>{"name": "final_answer", "arguments": {"answer": "Paris"}}</tool_call>'
    obs, done, info = asyncio.get_event_loop().run_until_complete(env.step(text))
    assert done is True
    assert info["final_answer"] == "Paris"
    assert env.final_answer == "Paris"


def test_step_no_tool_call_returns_done():
    env = _make_env()
    env.reset()
    obs, done, info = asyncio.get_event_loop().run_until_complete(env.step("I think the answer is 42."))
    assert done is True
    assert info["tool_executed"] is False


def test_step_search_database_returns_images_and_not_done():
    env = _make_env(max_turns=6)
    env.reset()
    env._client.query_points.return_value = _fake_query_result([3, 7])
    text = '<tool_call>{"name": "search_database", "arguments": {"query": "revenue Q3", "doc_name": "annual_report"}}</tool_call>'
    obs, done, info = asyncio.get_event_loop().run_until_complete(env.step(text))
    assert done is False
    assert len(obs["images"]) == 2
    assert info["pages"] == [3, 7]
    assert "3, 7" in obs["obs_str"]


def test_step_get_specific_pages_returns_images():
    env = _make_env()
    env.reset()
    point = MagicMock()
    point.payload = {"full_img_path": _make_fake_png_path(), "page_num": 5}
    env._client.scroll.return_value = ([point], None)
    text = '<tool_call>{"name": "get_specific_pages", "arguments": {"doc_name": "report", "page_numbers": [5]}}</tool_call>'
    obs, done, info = asyncio.get_event_loop().run_until_complete(env.step(text))
    assert info["tool_executed"] is True
    assert 5 in info["pages"]


def test_step_unknown_tool_returns_error_and_not_done():
    env = _make_env()
    env.reset()
    text = '<tool_call>{"name": "fly_to_moon", "arguments": {}}</tool_call>'
    obs, done, info = asyncio.get_event_loop().run_until_complete(env.step(text))
    assert done is False
    assert info["tool_executed"] is False
    assert "fly_to_moon" in obs["obs_str"]


def test_step_done_at_max_turns():
    env = _make_env(max_turns=1)
    env.reset()
    env._client.query_points.return_value = _fake_query_result([1])
    text = '<tool_call>{"name": "search_database", "arguments": {"query": "q", "doc_name": "d"}}</tool_call>'
    obs, done, info = asyncio.get_event_loop().run_until_complete(env.step(text))
    assert done is True  # turn 1 == max_turns=1


# ── format_observation() ──────────────────────────────────────────────────────

def test_format_observation_with_images():
    env = _make_env()
    img1 = PIL.Image.new("RGB", (10, 10))
    img2 = PIL.Image.new("RGB", (10, 10))
    msg = env.format_observation({"images": [img1, img2], "obs_str": "Found pages 3, 7."})
    assert msg["role"] == "tool"
    assert msg["content"][0] == {"type": "image", "image": img1}
    assert msg["content"][1] == {"type": "image", "image": img2}
    assert msg["content"][2] == {"type": "text", "text": "Found pages 3, 7."}


def test_format_observation_text_only():
    env = _make_env()
    msg = env.format_observation({"images": [], "obs_str": "No pages found."})
    assert msg["role"] == "tool"
    assert len(msg["content"]) == 1
    assert msg["content"][0] == {"type": "text", "text": "No pages found."}
