# Agentic Document RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `examples/agentic_doc_rl/` — a slime RL example that trains a VLM to answer long-document QA questions by retrieving page images from Qdrant using tool calls.

**Architecture:** `DocSearchEnv(BaseInteractionEnv)` handles all tool dispatch (Qdrant vector search and page retrieval); the existing `geo3k_vlm_multi_turn/rollout.generate()` is reused unchanged for the multi-turn token loop. A pluggable `reward_func` supports rule-based F1, LLM judge, or both.

**Tech Stack:** Python 3.10+, qdrant-client, JinaV4Model (from `AgenticMemory/search_models.py`), slime `BaseInteractionEnv`, pytest + unittest.mock

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `examples/agentic_doc_rl/__init__.py` | Create | Makes the directory a Python package |
| `examples/agentic_doc_rl/env_doc_search.py` | Create | `DocSearchEnv`, tool schemas, `build_env()`, Qdrant singletons |
| `examples/agentic_doc_rl/reward.py` | Create | `reward_func`, F1 helpers, LLM judge, dispatcher |
| `examples/agentic_doc_rl/config.yaml` | Create | Runtime config (max_turns, qdrant_url, reward_type, etc.) |
| `examples/agentic_doc_rl/run_qwen3_vlm_4B.sh` | Create | GRPO launch script wiring slime to the new env + reward |
| `tests/examples/test_agentic_doc_rl_env.py` | Create | Unit tests for env (mocked Qdrant + embed model) |
| `tests/examples/test_agentic_doc_rl_reward.py` | Create | Unit tests for reward (F1, answer extraction, dispatcher) |

---

## Task 1: Scaffold the package

**Files:**
- Create: `examples/agentic_doc_rl/__init__.py`
- Create: `examples/agentic_doc_rl/config.yaml`

- [ ] **Step 1: Create `__init__.py`**

```python
# examples/agentic_doc_rl/__init__.py
```
(empty file — makes the directory importable as a Python package)

- [ ] **Step 2: Create `config.yaml`**

```yaml
# examples/agentic_doc_rl/config.yaml

# ── Environment ──────────────────────────────────────────────────────────────
# Max tool-call turns per episode. Shorter than SFT's 15 to keep RL episodes
# fast; increase once the model reliably uses tools.
max_turns: 6

# Qdrant server — must be running before training starts.
# Start with: ./qdrant  (binary) or docker run -p 6333:6333 qdrant/qdrant
qdrant_url: "http://localhost:6333"

# Name of the Qdrant collection holding the indexed document pages.
collection_name: "mmlongdoc"

# Device for the JinaV4 embedding model (used by search_database tool).
embed_device: "cuda:0"

# Number of pages returned per search_database call.
topk: 3

# ── Environment module ───────────────────────────────────────────────────────
# Loaded by geo3k_vlm_multi_turn/rollout.py via --custom-config-path.
rollout_interaction_env_path: examples.agentic_doc_rl.env_doc_search

# ── Reward ───────────────────────────────────────────────────────────────────
# Options: rule_based | llm_judge | both
#   rule_based  — F1 token overlap against ground truth (fast, no extra GPU)
#   llm_judge   — LLM evaluator via OpenAI-compatible API (slow, higher quality)
#   both        — rule_based drives training; llm_judge logged to wandb as diagnostic
reward_type: rule_based

# Judge model settings (only used when reward_type is llm_judge or both).
judge_url: "http://localhost:30000"
judge_model: "Qwen3-VL-30B-A3B-Instruct"
judge_api_key: "EMPTY"
```

- [ ] **Step 3: Verify the package is importable**

```bash
cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime
python -c "import examples.agentic_doc_rl; print('OK')"
```
Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add examples/agentic_doc_rl/__init__.py examples/agentic_doc_rl/config.yaml
git commit -m "feat: scaffold examples/agentic_doc_rl package and config"
```

---

## Task 2: Tool schemas and env skeleton (`env_doc_search.py`)

**Files:**
- Create: `examples/agentic_doc_rl/env_doc_search.py`
- Create: `tests/examples/test_agentic_doc_rl_env.py`

- [ ] **Step 1: Write the failing test for tool schema structure**

```python
# tests/examples/test_agentic_doc_rl_env.py
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
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/examples/test_agentic_doc_rl_env.py::test_tools_list_has_two_entries -v
```
Expected: `ModuleNotFoundError` (file doesn't exist yet)

- [ ] **Step 3: Create `env_doc_search.py`**

```python
# examples/agentic_doc_rl/env_doc_search.py
from __future__ import annotations

import json
import logging
import re
from typing import Any

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from slime.utils.types import Sample

# search_models.py lives in AgenticMemory/ — added to PYTHONPATH by the launch script.
from search_models import JinaV4Model

logger = logging.getLogger(__name__)

# Matches Qwen's native tool-call format: <tool_call>{...}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# ── Tool schemas (OpenAI function-calling format) ─────────────────────────────
# Passed to apply_chat_template(tools=TOOLS) so Qwen embeds them in the system
# prompt. The model learns to call them by name during SFT and RL rewards good calls.

SEARCH_DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": (
            "Search the document vector database for pages semantically relevant to the query. "
            "Returns the top matching page images."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "doc_name": {
                    "type": "string",
                    "description": "The exact document name to search within (from the task description).",
                },
                "excluded_pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Page numbers to exclude from results (pages already seen).",
                },
            },
            "required": ["query", "doc_name"],
        },
    },
}

GET_SPECIFIC_PAGES_TOOL = {
    "type": "function",
    "function": {
        "name": "get_specific_pages",
        "description": (
            "Retrieve specific page numbers directly from the document. "
            "Use when you know which pages to examine. Maximum 5 pages per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "The exact document name.",
                },
                "page_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Page numbers to retrieve.",
                },
            },
            "required": ["doc_name", "page_numbers"],
        },
    },
}

TOOLS = [SEARCH_DATABASE_TOOL, GET_SPECIFIC_PAGES_TOOL]

# ── Module-level singletons ───────────────────────────────────────────────────
# QdrantClient and JinaV4Model are initialised once per worker process.
# JinaV4 is ~1.5 GB — reloading it per sample would be prohibitively slow.
_qdrant_client: QdrantClient | None = None
_embed_model: JinaV4Model | None = None


def _get_qdrant_client(qdrant_url: str) -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=qdrant_url)
    return _qdrant_client


def _get_embed_model(embed_device: str) -> JinaV4Model:
    global _embed_model
    if _embed_model is None:
        _embed_model = JinaV4Model(device=embed_device, multivector=True)
    return _embed_model


def _extract_tool_call(text: str) -> dict | None:
    """Parse the last <tool_call>JSON</tool_call> from the model response.

    Supports both Qwen native format {"name": ..., "arguments": ...} and
    OpenAI fallback {"function": {"name": ..., "arguments": ...}}.
    Returns None if no valid tool call is found.
    """
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = payload.get("name") or payload.get("function", {}).get("name")
    arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not name:
        return None
    return {"name": name, "arguments": arguments}


class DocSearchEnv(BaseInteractionEnv):
    """Multi-turn document search environment for RL training.

    The model issues tool calls to retrieve page images from Qdrant.
    Each call to step() processes one model response turn.
    """

    def __init__(
        self,
        *,
        ground_truth: Any = None,
        max_turns: int,
        qdrant_client: QdrantClient,
        embed_model: JinaV4Model,
        collection_name: str,
        embed_model_name: str = "jinav4_multivector",
        topk: int = 3,
    ):
        self.ground_truth = ground_truth
        self.max_turns = max_turns
        self._client = qdrant_client
        self._embed_model = embed_model
        self._collection_name = collection_name
        self._embed_model_name = embed_model_name
        self._topk = topk
        self.turn = 0
        self.final_answer: str | None = None

    def reset(self):
        self.turn = 0
        self.final_answer = None
        return {}, {}

    def close(self):
        pass

    # ── Qdrant helpers ────────────────────────────────────────────────────────

    def _search_database(
        self, query: str, doc_name: str, excluded_pages: list[int]
    ) -> tuple[list[Image.Image], list[int]]:
        query_vector = self._embed_model.embed_text(query)
        must_not = (
            [FieldCondition(key="page_num", match=MatchAny(any=excluded_pages))]
            if excluded_pages
            else []
        )
        query_filter = Filter(
            must=[FieldCondition(key="document_name", match=MatchValue(value=doc_name))],
            must_not=must_not,
        )
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            using=self._embed_model_name,
            limit=self._topk,
            with_payload=True,
            query_filter=query_filter,
        )
        images, pages = [], []
        for point in results.points:
            with Image.open(point.payload["full_img_path"]) as img:
                img.load()
                images.append(img.copy())
            pages.append(point.payload["page_num"])
        return images, pages

    def _get_specific_pages(
        self, doc_name: str, page_numbers: list[int]
    ) -> tuple[list[Image.Image], list[int]]:
        page_numbers = page_numbers[:5]
        query_filter = Filter(
            must=[
                FieldCondition(key="document_name", match=MatchValue(value=doc_name)),
                FieldCondition(key="page_num", match=MatchAny(any=page_numbers)),
            ]
        )
        scroll_results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=query_filter,
            limit=len(page_numbers),
            with_payload=True,
        )
        images, pages = [], []
        for point in scroll_results[0]:
            with Image.open(point.payload["full_img_path"]) as img:
                img.load()
                images.append(img.copy())
            pages.append(point.payload["page_num"])
        return images, pages

    # ── Core env interface ────────────────────────────────────────────────────

    async def step(self, response_text: str):
        """Process one model response turn.

        Returns:
            (observation_dict, done_bool, info_dict)
            observation_dict keys: "images" (list[PIL.Image]), "obs_str" (str)
        """
        self.turn += 1
        is_final = self.turn >= self.max_turns
        info: dict[str, Any] = {}

        tool_call = _extract_tool_call(response_text)
        info["tool_call"] = tool_call

        if tool_call is None:
            info["tool_executed"] = False
            return {"images": [], "obs_str": ""}, True, info

        name = (tool_call.get("name") or "").strip()
        args = tool_call.get("arguments") or {}

        if name == "final_answer":
            self.final_answer = (args.get("answer") or args.get("response") or "").strip()
            info["tool_executed"] = True
            info["final_answer"] = self.final_answer
            return {"images": [], "obs_str": ""}, True, info

        if name == "search_database":
            query = (args.get("query") or "").strip()
            doc_name = (args.get("doc_name") or "").strip()
            excluded_pages = args.get("excluded_pages") or []
            if not query or not doc_name:
                info["tool_executed"] = False
                obs = {"images": [], "obs_str": "search_database requires 'query' and 'doc_name'."}
                return obs, is_final, info
            try:
                images, pages = self._search_database(query, doc_name, excluded_pages)
                pages_str = ", ".join(map(str, pages))
                obs = {"images": images, "obs_str": f"Found relevant content on pages {pages_str}."}
                info["tool_executed"] = True
                info["pages"] = pages
            except Exception as exc:
                logger.warning("search_database failed: %s", exc)
                obs = {"images": [], "obs_str": f"Search failed: {exc}"}
                info["tool_executed"] = False
            return obs, is_final, info

        if name == "get_specific_pages":
            doc_name = (args.get("doc_name") or "").strip()
            page_numbers = args.get("page_numbers") or []
            if not doc_name or not page_numbers:
                info["tool_executed"] = False
                obs = {"images": [], "obs_str": "get_specific_pages requires 'doc_name' and 'page_numbers'."}
                return obs, is_final, info
            try:
                images, pages = self._get_specific_pages(doc_name, page_numbers)
                pages_str = ", ".join(map(str, pages))
                obs = {"images": images, "obs_str": f"Retrieved pages {pages_str}."}
                info["tool_executed"] = True
                info["pages"] = pages
            except Exception as exc:
                logger.warning("get_specific_pages failed: %s", exc)
                obs = {"images": [], "obs_str": f"Page retrieval failed: {exc}"}
                info["tool_executed"] = False
            return obs, is_final, info

        # Unknown tool — return helpful error and continue the episode.
        info["tool_executed"] = False
        supported = ", ".join(t["function"]["name"] for t in TOOLS)
        obs = {
            "images": [],
            "obs_str": f"Tool '{name}' is not supported. Available tools: {supported}.",
        }
        return obs, is_final, info

    def format_observation(self, observation: dict) -> dict:
        """Convert observation dict to a tool-role chat message with images.

        The rollout's _encode_observation_for_generation() calls
        process_vision_info() on this message to extract PIL images and
        encode them for SGLang. Observation tokens receive loss_mask=0.
        """
        images = observation.get("images") or []
        obs_str = observation.get("obs_str", "")
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": obs_str})
        return {"role": "tool", "content": content}


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> DocSearchEnv:
    """Factory function called by geo3k_vlm_multi_turn/rollout.py."""
    if args is None:
        raise ValueError("args must be provided to build_env.")

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path.")

    qdrant_url = getattr(args, "qdrant_url", "http://localhost:6333")
    collection_name = getattr(args, "collection_name", "mmlongdoc")
    embed_device = getattr(args, "embed_device", "cuda:0")
    topk = getattr(args, "topk", 3)

    qdrant_client = _get_qdrant_client(qdrant_url)
    embed_model = _get_embed_model(embed_device)

    if sample is not None:
        sample.metadata = sample.metadata or {}
        sample.metadata["tools"] = TOOLS

    ground_truth = None
    if sample is not None and sample.label is not None:
        label = sample.label
        if isinstance(label, dict):
            ground_truth = label.get("ground_truth") or label.get("answer") or label.get("target")
        else:
            ground_truth = str(label)

    return DocSearchEnv(
        ground_truth=ground_truth,
        max_turns=max_turns,
        qdrant_client=qdrant_client,
        embed_model=embed_model,
        collection_name=collection_name,
        topk=topk,
    )
```

- [ ] **Step 4: Run the schema tests — expect PASS**

```bash
pytest tests/examples/test_agentic_doc_rl_env.py::test_tools_list_has_two_entries \
       tests/examples/test_agentic_doc_rl_env.py::test_search_database_tool_schema \
       tests/examples/test_agentic_doc_rl_env.py::test_get_specific_pages_tool_schema -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add examples/agentic_doc_rl/env_doc_search.py tests/examples/test_agentic_doc_rl_env.py
git commit -m "feat: add DocSearchEnv with Qdrant tool dispatch"
```

---

## Task 3: Env unit tests — `step()` and `format_observation()`

**Files:**
- Modify: `tests/examples/test_agentic_doc_rl_env.py`

- [ ] **Step 1: Append step() and format_observation() tests**

```python
# Append to tests/examples/test_agentic_doc_rl_env.py

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
```

- [ ] **Step 2: Run all env tests**

```bash
pytest tests/examples/test_agentic_doc_rl_env.py -v
```
Expected: all PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/examples/test_agentic_doc_rl_env.py
git commit -m "test: add unit tests for DocSearchEnv step() and format_observation()"
```

---

## Task 4: Reward function (`reward.py`)

**Files:**
- Create: `examples/agentic_doc_rl/reward.py`
- Create: `tests/examples/test_agentic_doc_rl_reward.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/examples/test_agentic_doc_rl_reward.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
from unittest.mock import MagicMock
from slime.utils.types import Sample
from examples.agentic_doc_rl.reward import (
    _normalize,
    _compute_f1,
    _best_f1,
    _extract_final_answer,
    _get_golden_answers,
    _rule_based_score,
    reward_func,
)


def test_normalize_lowercases_and_strips_punctuation():
    assert _normalize("Hello, World!") == "hello world"


def test_normalize_removes_articles():
    assert _normalize("The quick brown fox") == "quick brown fox"


def test_compute_f1_perfect_match():
    assert _compute_f1("Paris France", "Paris France") == 1.0


def test_compute_f1_partial_match():
    f1 = _compute_f1("Paris is in France", "Paris France")
    assert 0.0 < f1 < 1.0


def test_compute_f1_no_match():
    assert _compute_f1("Berlin Germany", "Paris France") == 0.0


def test_compute_f1_empty_prediction():
    assert _compute_f1("", "Paris France") == 0.0


def test_best_f1_picks_highest_score():
    score = _best_f1("Paris France", ["Berlin Germany", "Paris France", "Tokyo Japan"])
    assert score == 1.0


def test_extract_final_answer_qwen_format():
    response = '<tool_call>{"name": "final_answer", "arguments": {"answer": "42"}}</tool_call>'
    assert _extract_final_answer(response) == "42"


def test_extract_final_answer_returns_none_without_call():
    response = '<tool_call>{"name": "search_database", "arguments": {"query": "q", "doc_name": "d"}}</tool_call>'
    assert _extract_final_answer(response) is None


def test_extract_final_answer_ignores_tool_turn_content():
    response = (
        '<|im_start|>tool\n'
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "fake"}}</tool_call>\n'
        '<|im_end|>'
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "real"}}</tool_call>'
    )
    assert _extract_final_answer(response) == "real"


def test_get_golden_answers_from_string():
    assert _get_golden_answers("Paris") == ["Paris"]


def test_get_golden_answers_from_list():
    assert _get_golden_answers(["Paris", "Paris, France"]) == ["Paris", "Paris, France"]


def test_get_golden_answers_from_dict():
    assert _get_golden_answers({"ground_truth": "Paris"}) == ["Paris"]


def test_get_golden_answers_returns_none_for_none():
    assert _get_golden_answers(None) is None


def test_rule_based_score_correct_answer():
    response = '<tool_call>{"name": "final_answer", "arguments": {"answer": "Paris France"}}</tool_call>'
    assert _rule_based_score(response, "Paris France") == 1.0


def test_rule_based_score_wrong_answer():
    response = '<tool_call>{"name": "final_answer", "arguments": {"answer": "Berlin"}}</tool_call>'
    assert _rule_based_score(response, "Paris France") == 0.0


def test_rule_based_score_no_final_answer():
    assert _rule_based_score("I think the answer is Paris.", "Paris France") == 0.0


def test_reward_func_rule_based_correct():
    args = MagicMock()
    args.reward_type = "rule_based"
    sample = Sample(
        prompt="What capital?",
        label="Paris",
        response='<tool_call>{"name": "final_answer", "arguments": {"answer": "Paris"}}</tool_call>',
    )
    score = asyncio.get_event_loop().run_until_complete(reward_func(args, sample))
    assert score == 1.0


def test_reward_func_rule_based_wrong():
    args = MagicMock()
    args.reward_type = "rule_based"
    sample = Sample(
        prompt="What capital?",
        label="Paris",
        response='<tool_call>{"name": "final_answer", "arguments": {"answer": "Berlin"}}</tool_call>',
    )
    score = asyncio.get_event_loop().run_until_complete(reward_func(args, sample))
    assert score == 0.0


def test_reward_func_unknown_type_falls_back_to_rule_based():
    args = MagicMock()
    args.reward_type = "nonexistent_mode"
    sample = Sample(
        prompt="What capital?",
        label="Paris",
        response='<tool_call>{"name": "final_answer", "arguments": {"answer": "Paris"}}</tool_call>',
    )
    score = asyncio.get_event_loop().run_until_complete(reward_func(args, sample))
    assert score == 1.0
```

- [ ] **Step 2: Run tests to confirm failures**

```bash
pytest tests/examples/test_agentic_doc_rl_reward.py -v
```
Expected: `ImportError` — `reward.py` doesn't exist yet

- [ ] **Step 3: Implement `reward.py`**

```python
# examples/agentic_doc_rl/reward.py
from __future__ import annotations

import json
import logging
import re
import string
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_TOOL_TURN_RE = re.compile(r"<\|im_start\|>tool\n.*?<\|im_end\|>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(
    r'<tool_call>\s*(\{.*?"name":\s*"final_answer".*?\})\s*</tool_call>', re.DOTALL
)


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_f1(prediction: str, golden_answers: list[str]) -> float:
    return max(_compute_f1(prediction, g) for g in golden_answers)


def _extract_final_answer(response: str) -> str | None:
    cleaned = _TOOL_TURN_RE.sub("", response)
    matches = list(_FINAL_ANSWER_RE.finditer(cleaned))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    arguments = payload.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return (arguments.get("answer") or arguments.get("response") or "").strip() or None


def _get_golden_answers(label: Any) -> list[str] | None:
    if label is None:
        return None
    if isinstance(label, dict):
        gt = label.get("ground_truth") or label.get("answer") or label.get("target")
        if isinstance(gt, list):
            return [str(g) for g in gt]
        if gt is not None:
            return [str(gt)]
        return None
    if isinstance(label, list):
        return [str(g) for g in label]
    return [str(label)]


def _rule_based_score(response: str, label: Any) -> float:
    answer = _extract_final_answer(response)
    golden = _get_golden_answers(label)
    if answer is None or not golden:
        return 0.0
    return _best_f1(answer, golden)


def _llm_judge_score(response: str, label: Any, question: str, args: Any) -> float:
    answer = _extract_final_answer(response)
    if answer is None:
        return 0.0
    golden = _get_golden_answers(label)
    ground_truth = golden[0] if golden else ""
    try:
        # utils.py lives in AgenticMemory/ — added to PYTHONPATH by the launch script.
        from utils import evaluate_response  # noqa: PLC0415
        from openai import OpenAI

        client = OpenAI(
            api_key=getattr(args, "judge_api_key", "EMPTY"),
            base_url=getattr(args, "judge_url", "http://localhost:30000") + "/v1",
        )
        result = evaluate_response(
            client,
            getattr(args, "judge_model", "Qwen3-VL-30B-A3B-Instruct"),
            answer,
            ground_truth,
            question,
        )
        score = result.get("score", 0)
        return float(score) if score not in (-1, None) else 0.0
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return 0.0


async def reward_func(args: Any, sample: Sample, **kwargs) -> float:
    """Pluggable reward function called per sample by slime's rollout.

    reward_type (from args / config.yaml):
        rule_based  — F1 token overlap (default)
        llm_judge   — LLM evaluator via OpenAI-compatible API
        both        — rule_based drives training; llm_judge logged to wandb
    """
    response = sample.response or ""
    reward_type = getattr(args, "reward_type", "rule_based")
    question = sample.label.get("question", "") if isinstance(sample.label, dict) else ""

    if reward_type == "rule_based":
        return _rule_based_score(response, sample.label)

    if reward_type == "llm_judge":
        return _llm_judge_score(response, sample.label, question, args)

    if reward_type == "both":
        rb_score = _rule_based_score(response, sample.label)
        lj_score = _llm_judge_score(response, sample.label, question, args)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"reward/rule_based": rb_score, "reward/llm_judge": lj_score})
        except Exception:
            pass
        return rb_score

    logger.warning("Unknown reward_type %r; falling back to rule_based.", reward_type)
    return _rule_based_score(response, sample.label)
```

- [ ] **Step 4: Run reward tests — expect all PASS**

```bash
pytest tests/examples/test_agentic_doc_rl_reward.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add examples/agentic_doc_rl/reward.py tests/examples/test_agentic_doc_rl_reward.py
git commit -m "feat: add pluggable reward_func with F1 and LLM judge support"
```

---

## Task 5: Launch script (`run_qwen3_vlm_4B.sh`)

**Files:**
- Create: `examples/agentic_doc_rl/run_qwen3_vlm_4B.sh`

- [ ] **Step 1: Create the launch script**

```bash
#!/bin/bash
# Agentic Document QA RL training — Qwen3-VL-4B
#
# Prerequisites:
#   1. Qdrant running:      ./qdrant  (binary) or docker run -p 6333:6333 qdrant/qdrant
#   2. Documents indexed:   see AgenticMemory/scripts/ for indexing pipeline
#   3. SFT checkpoint:      output of AgenticMemory/run_agentic_sft.sh
#
# Usage:
#   ./examples/agentic_doc_rl/run_qwen3_vlm_4B.sh
#   SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct ./examples/agentic_doc_rl/run_qwen3_vlm_4B.sh

MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-4B-Instruct"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
SFT_CKPT=${SLIME_SCRIPT_SFT_CKPT:-"/lc/AgenticMemory/checkpoints/sft"}
PARQUET=${SLIME_SCRIPT_PARQUET:-"/lc/AgenticMemory/logs/train_trajectories_30B.parquet"}
RL_SAVE=${SLIME_SCRIPT_RL_SAVE:-"/lc/AgenticMemory/checkpoints/rl"}

VALID_MODELS="Qwen3-VL-4B-Instruct Qwen3-VL-8B-Instruct Qwen2.5-VL-7B-Instruct"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
    echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
    exit 1
fi

pkill -9 sglang; sleep 3
ray stop --force; pkill -9 ray; pkill -9 python; sleep 3

set -ex
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Derive model args file (same convention as run_agentic_sft.sh)
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/Qwen3-VL-/qwen3-/g; s/Qwen2.5-VL-/qwen2.5-/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${REPO_ROOT}/scripts/models/${MODEL_ARGS_FILE}.sh"

CKPT_ARGS=(
    --hf-checkpoint "${SFT_CKPT}"
    --load          "${SFT_CKPT}"
    --save          "${RL_SAVE}"
    --save-interval 20
)

ROLLOUT_ARGS=(
    --prompt-data        "${PARQUET}"
    --input-key          messages
    --tool-key           tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    --num-rollout        2000
    --rollout-batch-size 16
    --n-samples-per-prompt 8
    --rollout-max-response-len 4096
    --rollout-temperature 0.7
    --global-batch-size  128
    --balance-data
    --multimodal-keys '{"image": "images"}'
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.98
)

BACKEND_ARGS=(
    --train-backend megatron
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --megatron-to-hf-mode bridge
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
)

CUSTOM_ARGS=(
    --custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
    --custom-rm-path                examples.agentic_doc_rl.reward.reward_func
    --custom-config-path            examples/agentic_doc_rl/config.yaml
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-agentic-doc-rl
        --wandb-group "$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')-grpo"
        --wandb-key "${WANDB_API_KEY}"
        --disable-wandb-random-suffix
    )
else
    WANDB_ARGS=()
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# PYTHONPATH:
#   /workspace/Megatron-LM     — Megatron-LM internals
#   ${REPO_ROOT}               — makes examples.* importable
#   ${REPO_ROOT}/AgenticMemory — exposes search_models.py and utils.py
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/workspace/Megatron-LM:${REPO_ROOT}:${REPO_ROOT}/AgenticMemory\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${REPO_ROOT}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --rollout-num-gpus "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
```

- [ ] **Step 2: Make executable and syntax-check**

```bash
chmod +x examples/agentic_doc_rl/run_qwen3_vlm_4B.sh
bash -n examples/agentic_doc_rl/run_qwen3_vlm_4B.sh && echo "syntax OK"
```
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add examples/agentic_doc_rl/run_qwen3_vlm_4B.sh
git commit -m "feat: add GRPO launch script for agentic doc RL"
```

---

## Task 6: Full verification

- [ ] **Step 1: Run all new tests**

```bash
pytest tests/examples/test_agentic_doc_rl_env.py \
       tests/examples/test_agentic_doc_rl_reward.py \
       -v --durations=0
```
Expected: all PASSED

- [ ] **Step 2: Smoke-test imports end-to-end**

```bash
cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime
PYTHONPATH=.:AgenticMemory python -c "
from examples.agentic_doc_rl.env_doc_search import TOOLS, DocSearchEnv
from examples.agentic_doc_rl.reward import reward_func, _compute_f1
print('env OK, tools:', [t['function']['name'] for t in TOOLS])
print('reward OK, f1(Paris, Paris):', _compute_f1('Paris', 'Paris'))
"
```
Expected:
```
env OK, tools: ['search_database', 'get_specific_pages']
reward OK, f1(Paris, Paris): 1.0
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete examples/agentic_doc_rl — DocSearchEnv + reward + launch script"
```

---

## Adding a new tool later

When you want to add e.g. `update_search_memory`:

1. Add the OpenAI schema dict to `TOOLS` in `env_doc_search.py`
2. Add an `elif name == "update_search_memory":` branch in `DocSearchEnv.step()`
3. Add a test case in `test_agentic_doc_rl_env.py`

No other files need changing.
