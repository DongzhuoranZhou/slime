# Agent Trajectory Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-script pipeline — collect trajectories from Qwen3-VL-30B via SGLang, format them into a slime-compatible parquet, and train a 4B/8B model via slime SFT.

**Architecture:** Standalone SGLang server serves Qwen3-VL-30B; `collect_trajectories.py` drives smolagents `SearchAgent` against it and saves raw step data as jsonl; `format_trajectories.py` converts smolagents-internal message roles to OpenAI format and serializes images as bytes into a parquet; `run_agentic_sft.sh` feeds the parquet into slime's existing `sft_rollout` pipeline.

**Tech Stack:** smolagents (`ToolCallingAgent`, `LiteLLMModel`), SGLang, slime (`sft_rollout`, `MultiTurnLossMaskGenerator`), HuggingFace `datasets`, PyArrow, Qwen3-VL tokenizer/processor.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `AgenticMemory/collect_trajectories.py` | Create | Run SearchAgent against local SGLang, save raw jsonl |
| `AgenticMemory/format_trajectories.py` | Create | Convert raw jsonl → slime-compatible parquet |
| `AgenticMemory/verify_parquet.py` | Create | Visualise loss mask from parquet before training |
| `AgenticMemory/run_agentic_sft.sh` | Create | slime SFT training script for small models |
| `AgenticMemory/test_format_trajectories.py` | Create | Unit tests for `format_trajectories.py` core logic |

---

## Key Data Contracts

### smolagents `response.steps` structure (from `utils.py`)

```python
# response = agent.run(query, return_full_result=True)
# response.steps is a list of step dicts:

step = {
    "step_number": 0,
    "model_input_messages": [                        # CUMULATIVE messages sent to model
        {"role": "system",        "content": [{"type": "text", "text": "..."}]},
        {"role": "user",          "content": [{"type": "text", "text": "..."}]},
        {"role": "tool-call",     "content": [...]},     # smolagents role (NOT OpenAI)
        {"role": "tool-response", "content": [           # smolagents role (NOT OpenAI)
            {"type": "image", "image": <PIL.Image>},     # images FIRST
            {"type": "text",  "text": "Search results..."}
        ]},
    ],
    "model_output_message": {
        "tool_calls": [
            {
                "id": "call_abc123",
                "function": {
                    "name": "search_database",
                    "arguments": {"query": "...", "doc_name": "..."}  # dict, not string
                }
            }
        ],
        "raw": {"choices": [{"message": {"reasoning_content": "..."}}]}
    }
}
```

**Critical notes:**
- `model_input_messages` for step N contains all messages through step N-1's tool responses
- `model_output_message` for step N is what the model generated at step N
- smolagents roles `"tool-call"` / `"tool-response"` must be converted to `"assistant"` / `"tool"` for slime
- Images in `"tool-response"` content are PIL objects, already loaded — images come BEFORE text

### Raw jsonl format (intermediate)

```json
{
  "question_id": "mmlongdoc_001",
  "doc_name": "annual_report_2023",
  "question": "What was Q3 revenue?",
  "ground_truth": "42M",
  "model_answer": "The total revenue was 42M.",
  "score": 1,
  "num_steps": 4,
  "system_prompt": "You are a multimodal document search agent...",
  "task": "Based on Document annual_report_2023, answer: What was Q3 revenue?",
  "steps": [
    {
      "tool_calls": [{"id": "call_abc", "name": "search_database", "arguments": {"query": "Q3 revenue", "doc_name": "annual_report_2023"}}],
      "tool_responses": [
        {"tool_call_id": "call_abc", "image_paths": ["<png_bytes_b64>", "<png_bytes_b64>"], "content": "Search results for 'Q3 revenue'..."}
      ]
    },
    {
      "tool_calls": [{"id": "call_def", "name": "update_search_memory", "arguments": {"query": "Q3 revenue", "pages_visited": [3, 7], "relevant_information": "Page 3 has Q3 revenue table", "original_question": "What was Q3 revenue?"}}],
      "tool_responses": [
        {"tool_call_id": "call_def", "image_paths": [], "content": "✓ Search #1 recorded: ..."}
      ]
    },
    {
      "tool_calls": [{"id": "call_xyz", "name": "final_answer", "arguments": {"answer": "The total revenue was 42M."}}],
      "tool_responses": []
    }
  ]
}
```

### Parquet schema (slime input)

| Column | Type | Notes |
|--------|------|-------|
| `messages` | `list[dict]` | OpenAI format, `<image>` placeholders in tool message strings |
| `images` | `list[bytes]` | PNG bytes, flat, in appearance order across all tool messages |
| `tools` | `str` | JSON string of OpenAI tool schema list |
| `question_id` | `str` | |
| `doc_name` | `str` | |
| `question` | `str` | |
| `ground_truth` | `str` | |
| `model_answer` | `str` | |
| `score` | `float` | |
| `num_steps` | `int` | |

---

## Task 1: `format_trajectories.py` core logic + unit tests

Build and fully test the formatter before writing the collector. This way you can develop with hand-crafted fake trajectories and catch format bugs before running the expensive 30B model.

**Files:**
- Create: `AgenticMemory/format_trajectories.py`
- Create: `AgenticMemory/test_format_trajectories.py`

- [ ] **Step 1.1: Write failing tests**

Create `AgenticMemory/test_format_trajectories.py`:

```python
import base64
import io
import json

import pytest
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_b64_png() -> str:
    return base64.b64encode(_make_png_bytes()).decode()


def _final_step(answer="42"):
    return {
        "tool_calls": [{"id": "c_final", "name": "final_answer", "arguments": {"answer": answer}}],
        "tool_responses": [],
    }


def _search_step(tool_call_id="c1", n_images=2, text="Found on pages 1, 2."):
    return {
        "tool_calls": [{"id": tool_call_id, "name": "search_database",
                        "arguments": {"query": "q", "doc_name": "d"}}],
        "tool_responses": [
            {"tool_call_id": tool_call_id,
             "image_paths": [_make_b64_png() for _ in range(n_images)],
             "content": text}
        ],
    }


def _raw(steps, system="sys", task="user query"):
    return {"system_prompt": system, "task": task, "steps": steps}


# ── import target ─────────────────────────────────────────────────────────────

from format_trajectories import build_messages_and_images, pil_to_bytes


# ── tests ─────────────────────────────────────────────────────────────────────

def test_basic_structure():
    """system + user + search_assistant + search_tool + final_assistant = 5 messages."""
    raw = _raw([_search_step(), _final_step()])
    messages, images = build_messages_and_images(raw)
    assert len(messages) == 5
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]


def test_images_before_text_in_tool_message():
    """<image> placeholders must precede observation text."""
    raw = _raw([_search_step(n_images=3, text="Some text."), _final_step()])
    messages, _ = build_messages_and_images(raw)
    tool_msg = messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "<image><image><image>Some text."


def test_flat_images_list_accumulates_in_order():
    """Images from step 1 (2 imgs) followed by step 2 (1 img) → list of 3 bytes."""
    raw = _raw([
        _search_step(tool_call_id="c1", n_images=2),
        _search_step(tool_call_id="c2", n_images=1),
        _final_step(),
    ])
    messages, images = build_messages_and_images(raw)
    assert len(images) == 3
    for img in images:
        assert isinstance(img, bytes)


def test_no_tool_message_after_final_answer():
    """final_answer call must not produce a trailing tool message."""
    raw = _raw([_final_step()])
    messages, _ = build_messages_and_images(raw)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["tool_calls"][0]["function"]["name"] == "final_answer"
    assert len(messages) == 3  # system + user + assistant_final


def test_tool_call_id_preserved():
    """tool message tool_call_id must match the preceding assistant tool_call id."""
    raw = _raw([_search_step(tool_call_id="myid"), _final_step()])
    messages, _ = build_messages_and_images(raw)
    assistant_msg = messages[2]
    tool_msg = messages[3]
    assert assistant_msg["tool_calls"][0]["id"] == "myid"
    assert tool_msg["tool_call_id"] == "myid"


def test_assistant_tool_calls_arguments_are_json_string():
    """arguments in tool_calls must be a JSON string, not a dict."""
    raw = _raw([_search_step(), _final_step()])
    messages, _ = build_messages_and_images(raw)
    assistant_msg = messages[2]
    args = assistant_msg["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    parsed = json.loads(args)
    assert parsed["query"] == "q"


def test_text_only_tool_response_no_images():
    """Tool responses with no images produce no <image> prefix and empty images list."""
    step = {
        "tool_calls": [{"id": "c1", "name": "update_search_memory", "arguments": {}}],
        "tool_responses": [{"tool_call_id": "c1", "image_paths": [], "content": "✓ Recorded."}],
    }
    raw = _raw([step, _final_step()])
    messages, images = build_messages_and_images(raw)
    tool_msg = messages[3]
    assert tool_msg["content"] == "✓ Recorded."
    assert len(images) == 0


def test_pil_to_bytes_round_trips():
    """pil_to_bytes produces bytes that PIL can decode back."""
    img = Image.new("RGB", (8, 8), color=(0, 255, 0))
    b = pil_to_bytes(img)
    assert isinstance(b, bytes)
    recovered = Image.open(io.BytesIO(b))
    assert recovered.size == (8, 8)
```

- [ ] **Step 1.2: Run tests — confirm they all fail**

```bash
cd /lc/AgenticMemory  # adjust to your actual path
python -m pytest test_format_trajectories.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'format_trajectories'`

- [ ] **Step 1.3: Implement `format_trajectories.py`**

Create `AgenticMemory/format_trajectories.py`:

```python
"""
format_trajectories.py

Converts raw trajectory jsonl (from collect_trajectories.py) into a
slime-compatible parquet for SFT training.

Usage:
    python format_trajectories.py \
        --input logs/trajectories_30B_test.jsonl \
        --output logs/train_trajectories_30B.parquet \
        --min-score 0.0
"""

import argparse
import base64
import io
import json
import logging
import os
from typing import Any

import datasets
from PIL import Image

logger = logging.getLogger(__name__)


# ── image helpers ─────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)


# ── core conversion ───────────────────────────────────────────────────────────

def build_messages_and_images(raw: dict) -> tuple[list[dict], list[bytes]]:
    """Convert a raw trajectory dict into (messages, images).

    messages: OpenAI chat format list ready for slime's sft_rollout.
              Tool messages use '<image>' placeholders; images are NOT inline.
    images:   Flat list of PNG bytes in the order placeholders appear.
    """
    messages: list[dict] = []
    images: list[bytes] = []

    messages.append({"role": "system", "content": raw["system_prompt"]})
    messages.append({"role": "user",   "content": raw["task"]})

    for step in raw["steps"]:
        tool_calls_raw = step["tool_calls"]

        # ── assistant message ────────────────────────────────────────────────
        tool_calls_openai = []
        for tc in tool_calls_raw:
            args = tc["arguments"]
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            tool_calls_openai.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": args},
            })
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls_openai,
        })

        # ── tool result messages (skipped for final_answer) ──────────────────
        for resp in step.get("tool_responses", []):
            imgs_b64 = resp.get("image_paths", [])
            text = resp.get("content", "")
            n = len(imgs_b64)

            # images before text (VLM convention)
            content = "<image>" * n + text

            for b64 in imgs_b64:
                images.append(b64_to_bytes(b64))

            messages.append({
                "role": "tool",
                "tool_call_id": resp["tool_call_id"],
                "content": content,
            })

    return messages, images


# ── tool schema extraction ────────────────────────────────────────────────────

def _smolagents_input_to_openai_property(param: str, info: dict) -> dict:
    prop: dict[str, Any] = {
        "type": info["type"],
        "description": info.get("description", ""),
    }
    if info["type"] == "array" and "items" in info:
        prop["items"] = info["items"]
    return prop


def extract_tool_schemas(tools: dict) -> list[dict]:
    """Convert smolagents tool dict → OpenAI tool schema list."""
    schemas = []
    for tool in tools.values():
        properties = {
            name: _smolagents_input_to_openai_property(name, info)
            for name, info in tool.inputs.items()
        }
        required = [
            name for name, info in tool.inputs.items()
            if not info.get("nullable", False)
        ]
        schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return schemas


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="logs/trajectories_30B_test.jsonl")
    parser.add_argument("--output", default="logs/train_trajectories_30B.parquet")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum score to include (default: include all)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows = []
    skipped = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            score = float(raw.get("score", 0) or 0)
            if score < args.min_score:
                skipped += 1
                continue

            messages, images = build_messages_and_images(raw)
            rows.append({
                "messages":     messages,
                "images":       images,            # list[bytes]
                "tools":        raw.get("tools_json", "[]"),
                "question_id":  raw.get("question_id", ""),
                "doc_name":     raw.get("doc_name", ""),
                "question":     raw.get("question", ""),
                "ground_truth": raw.get("ground_truth", ""),
                "model_answer": raw.get("model_answer", ""),
                "score":        score,
                "num_steps":    int(raw.get("num_steps", 0)),
            })

    logger.info(f"Kept {len(rows)} rows, skipped {skipped} below min_score={args.min_score}")
    ds = datasets.Dataset.from_list(rows)
    ds.to_parquet(args.output)
    logger.info(f"Saved to {args.output}")
    print(f"Done: {len(rows)} trajectories → {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

- [ ] **Step 1.4: Run tests — confirm they all pass**

```bash
cd /lc/AgenticMemory
python -m pytest test_format_trajectories.py -v
```

Expected output:
```
test_format_trajectories.py::test_basic_structure PASSED
test_format_trajectories.py::test_images_before_text_in_tool_message PASSED
test_format_trajectories.py::test_flat_images_list_accumulates_in_order PASSED
test_format_trajectories.py::test_no_tool_message_after_final_answer PASSED
test_format_trajectories.py::test_tool_call_id_preserved PASSED
test_format_trajectories.py::test_assistant_tool_calls_arguments_are_json_string PASSED
test_format_trajectories.py::test_text_only_tool_response_no_images PASSED
test_format_trajectories.py::test_pil_to_bytes_round_trips PASSED
8 passed in ...
```

- [ ] **Step 1.5: Commit**

```bash
cd /lc/AgenticMemory
git add format_trajectories.py test_format_trajectories.py
git commit -m "feat: add format_trajectories.py with unit tests

Converts raw trajectory jsonl to slime-compatible parquet.
Images stored as PNG bytes, <image> placeholders before text in tool messages."
```

---

## Task 2: `collect_trajectories.py`

**Files:**
- Create: `AgenticMemory/collect_trajectories.py`

- [ ] **Step 2.1: Implement `collect_trajectories.py`**

Create `AgenticMemory/collect_trajectories.py`:

```python
"""
collect_trajectories.py

Runs SearchAgent against a local SGLang server and saves raw trajectories to jsonl.

Usage:
    # First, start SGLang server (on 4 GPUs):
    #   python -m sglang.launch_server \
    #       --model-path /workspace/.cache/huggingface/hub/Qwen3-VL-30B-A3B-Instruct \
    #       --tp 4 --port 30000 --dtype bfloat16

    python collect_trajectories.py \
        --dataset mmlongdoc \
        --sglang-url http://localhost:30000 \
        --model-name openai/Qwen3-VL-30B-A3B-Instruct \
        --output logs/trajectories_30B_test.jsonl \
        --num-samples 5   # omit to run full split
"""

import argparse
import base64
import io
import json
import logging
import os
import importlib
import yaml
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from smolagents import ToolCallingAgent, LiteLLMModel

from src.data_utils import PathManager
from search_models import JinaV4Model
from tools import (
    SearchDatabaseTool, GetSpecificPagesTool,
    SearchMemoryTool, ReflectionTool,
)
from utils import evaluate_response, filter_existing_results, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ── image serialisation ───────────────────────────────────────────────────────

def pil_to_b64(img) -> str:
    """PIL Image → base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── trajectory extraction from smolagents response ───────────────────────────

def _extract_tool_responses(step: dict) -> list[dict]:
    """
    Extract tool response dicts from a step's model_input_messages delta.

    Smolagents appends tool-response messages to model_input_messages between
    steps. We identify them by role == "tool-response".
    """
    responses = []
    for msg in step.get("model_input_messages", []):
        if msg["role"] != "tool-response":
            continue
        content = msg.get("content", [])
        image_paths = [
            pil_to_b64(item["image"])
            for item in content
            if item.get("type") == "image" and item.get("image") is not None
        ]
        text = " ".join(
            item.get("text", "")
            for item in content
            if item.get("type") == "text"
        ).strip()
        # tool_call_id: smolagents stores it in the message
        tool_call_id = msg.get("tool_call_id") or msg.get("id", "unknown")
        responses.append({
            "tool_call_id": tool_call_id,
            "image_paths": image_paths,
            "content": text,
        })
    return responses


def extract_steps(response_steps: list[dict]) -> list[dict]:
    """
    Convert smolagents response.steps into the raw-jsonl step format.

    For each step:
      - tool_calls: from model_output_message["tool_calls"]
      - tool_responses: from the NEXT step's model_input_messages delta
                        (smolagents appends tool-response messages between steps)

    The final step (final_answer) has tool_responses = [].
    """
    steps = []
    for i, step in enumerate(response_steps):
        tc_raw = step["model_output_message"].get("tool_calls", [])
        tool_calls = [
            {
                "id":        tc.get("id", f"call_{i}_{j}"),
                "name":      tc["function"]["name"],
                "arguments": tc["function"]["arguments"],  # may be dict or str
            }
            for j, tc in enumerate(tc_raw)
        ]

        # Tool responses live in the NEXT step's input (the delta)
        if i + 1 < len(response_steps):
            next_msgs = response_steps[i + 1]["model_input_messages"]
            curr_msgs = step["model_input_messages"]
            delta = next_msgs[len(curr_msgs):]
            # Build a fake step dict to reuse _extract_tool_responses
            tool_responses = _extract_tool_responses({"model_input_messages": delta})
        else:
            tool_responses = []

        steps.append({"tool_calls": tool_calls, "tool_responses": tool_responses})

    return steps


# ── tool schema serialisation ─────────────────────────────────────────────────

def _tool_to_schema(tool) -> dict:
    properties = {}
    required = []
    for param, info in tool.inputs.items():
        prop = {"type": info["type"], "description": info.get("description", "")}
        if info["type"] == "array" and "items" in info:
            prop["items"] = info["items"]
        properties[param] = prop
        if not info.get("nullable", False):
            required.append(param)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="mmlongdoc")
    parser.add_argument("--sglang-url",  default="http://localhost:30000")
    parser.add_argument("--model-name",  default="openai/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--embed-device",default="cuda:0")
    parser.add_argument("--output",      default="logs/trajectories_30B_test.jsonl")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of samples (omit for full split)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    os.makedirs("logs", exist_ok=True)

    # ── load dataset ──────────────────────────────────────────────────────────
    jsonl_path = PathManager.get_dataset_jsonl(args.dataset)
    data = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    data = filter_existing_results(data, args.output)
    if args.num_samples is not None:
        data = data[: args.num_samples]
    logger.info(f"Processing {len(data)} samples → {args.output}")

    # ── build model (local SGLang) ────────────────────────────────────────────
    agent_model = LiteLLMModel(
        model_id=args.model_name,
        api_base=f"{args.sglang_url}/v1",
        api_key="EMPTY",
        do_sample=False,
    )

    # ── eval client (for scoring; reuses existing API_KEY / API_BASE) ─────────
    eval_client = OpenAI(
        api_key=os.getenv("API_KEY", "EMPTY"),
        base_url=os.getenv("API_BASE", f"{args.sglang_url}/v1"),
    )
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o-2024-11-20")

    # ── setup Qdrant + embedding model ────────────────────────────────────────
    os.environ["NO_PROXY"] = "localhost"
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    embed_model = JinaV4Model(device=args.embed_device, multivector=True)
    embed_model_name = "jinav4_multivector"

    # ── load smolagents prompt template ───────────────────────────────────────
    prompt_templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts")
        .joinpath("deepresearch_agent.yaml")
        .read_text()
    )

    # ── initialise tools ──────────────────────────────────────────────────────
    search_tool      = SearchDatabaseTool(qdrant_client, embed_model, args.dataset, embed_model_name)
    get_pages_tool   = GetSpecificPagesTool(qdrant_client, args.dataset)
    search_mem_tool  = SearchMemoryTool()
    reflect_tool     = ReflectionTool(search_memory=search_mem_tool)

    tool_list = [search_tool, get_pages_tool, search_mem_tool, reflect_tool]
    tools_json = json.dumps([_tool_to_schema(t) for t in tool_list], ensure_ascii=False)

    # ── main loop ─────────────────────────────────────────────────────────────
    from search_agent import SearchAgent  # import after arg parsing

    agent = SearchAgent(
        tools=tool_list,
        model=agent_model,
        prompt_templates=prompt_templates,
        max_steps=15,
    )

    for d in data:
        search_mem_tool.reset()

        question_id = d["id"]
        doc_name    = d["doc_name"]
        question    = d["question"]
        answer      = d["answer"]
        query = f"Based on Document {doc_name}, answer the following question: {question}."

        try:
            response = agent.run(query, return_full_result=True)
        except Exception as e:
            logger.error(f"Agent failed for {question_id}: {e}")
            continue

        model_answer = str(response.output)

        eval_result = evaluate_response(
            eval_client, eval_model, model_answer, answer, question,
            max_tokens=256, temperature=0.0,
        )
        score = eval_result.get("score", 0)

        # ── extract steps ─────────────────────────────────────────────────────
        steps_raw = getattr(response, "steps", [])
        # response.steps may be objects or dicts depending on smolagents version
        if steps_raw and not isinstance(steps_raw[0], dict):
            steps_raw = [s.dict() if hasattr(s, "dict") else vars(s) for s in steps_raw]

        steps = extract_steps(steps_raw)

        record = {
            "question_id":  question_id,
            "doc_name":     doc_name,
            "question":     question,
            "ground_truth": answer,
            "model_answer": model_answer,
            "score":        score,
            "num_steps":    len(steps),
            "system_prompt": SYSTEM_PROMPT,
            "task":         query,
            "steps":        steps,
            "tools_json":   tools_json,
        }

        def _safe(obj):
            if isinstance(obj, bytes):
                return f"<bytes:{len(obj)}>"
            raise TypeError(type(obj))

        with open(args.output, "a", encoding="utf-8") as f:
            json.dump(record, f, default=_safe, ensure_ascii=False)
            f.write("\n")

        logger.info(f"{question_id}: score={score}, steps={len(steps)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Quick smoke-test with `--num-samples 1` (requires SGLang running)**

Start SGLang server first (skip this step if GPU not available yet):
```bash
python -m sglang.launch_server \
  --model-path /workspace/.cache/huggingface/hub/Qwen3-VL-30B-A3B-Instruct \
  --tp 4 --port 30000 --dtype bfloat16 --mem-fraction-static 0.85
```

Then collect 1 sample:
```bash
cd /lc/AgenticMemory
python collect_trajectories.py \
    --dataset mmlongdoc \
    --sglang-url http://localhost:30000 \
    --model-name openai/Qwen3-VL-30B-A3B-Instruct \
    --output logs/test_1sample.jsonl \
    --num-samples 1
```

Expected output:
```
Processing 1 samples → logs/test_1sample.jsonl
<question_id>: score=<0 or 1>, steps=<3-8>
```

Inspect the jsonl:
```bash
python3 -c "
import json
with open('logs/test_1sample.jsonl') as f:
    d = json.loads(f.read())
print('Steps:', len(d['steps']))
for i, s in enumerate(d['steps']):
    print(f'  step {i}: tool={s[\"tool_calls\"][0][\"name\"]}, n_imgs={sum(len(r[\"image_paths\"]) for r in s[\"tool_responses\"])}')
"
```

- [ ] **Step 2.3: Run full test split**

```bash
cd /lc/AgenticMemory
python collect_trajectories.py \
    --dataset mmlongdoc \
    --sglang-url http://localhost:30000 \
    --model-name openai/Qwen3-VL-30B-A3B-Instruct \
    --output logs/trajectories_30B_test.jsonl
```

This is resume-safe (already-processed question_ids are skipped).

- [ ] **Step 2.4: Commit**

```bash
cd /lc/AgenticMemory
git add collect_trajectories.py
git commit -m "feat: add collect_trajectories.py

Drives SearchAgent against local SGLang server, saves raw step data
as jsonl with base64 images. Resume-safe via filter_existing_results."
```

---

## Task 3: Convert to parquet + verify loss masking

**Files:**
- Create: `AgenticMemory/verify_parquet.py`

- [ ] **Step 3.1: Convert jsonl to parquet**

```bash
cd /lc/AgenticMemory
python format_trajectories.py \
    --input  logs/trajectories_30B_test.jsonl \
    --output logs/train_trajectories_30B.parquet
```

Expected output:
```
Done: <N> trajectories → logs/train_trajectories_30B.parquet
```

Quick sanity check on schema:
```bash
python3 -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('logs/train_trajectories_30B.parquet')
print(pf.schema_arrow)
batch = next(pf.iter_batches(batch_size=1))
row = batch.to_pylist()[0]
print('Columns:', list(row.keys()))
print('Num messages:', len(row['messages']))
print('Num images:', len(row['images']))
print('Roles:', [m['role'] for m in row['messages']])
"
```

Expected:
```
Columns: ['messages', 'images', 'tools', 'question_id', ...]
Num messages: <5-15>
Num images: <2-15>
Roles: ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', ..., 'assistant']
```

- [ ] **Step 3.2: Implement `verify_parquet.py`**

Create `AgenticMemory/verify_parquet.py`:

```python
"""
verify_parquet.py

Loads one row from the parquet and runs slime's MultiTurnLossMaskGenerator
to confirm that:
  - assistant turns (tool calls + final_answer) have loss_mask=1
  - system / user / tool turns have loss_mask=0
  - images are correctly injected at <image> placeholder positions

Usage:
    python verify_parquet.py \
        --parquet logs/train_trajectories_30B.parquet \
        --model-path /workspace/.cache/huggingface/hub/Qwen3-VL-4B-Instruct
"""

import argparse
import json
import sys

import pyarrow.parquet as pq

sys.path.insert(0, "/workspace/src/clean_code_for_rl/slime_0224_2026/slime")

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.data import _build_messages
from slime.utils.processing_utils import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",    required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--row",        type=int, default=0)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path, trust_remote_code=True)
    mask_gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")

    pf = pq.ParquetFile(args.parquet)
    batch = next(pf.iter_batches(batch_size=args.row + 1))
    row = batch.to_pylist()[args.row]

    # ── reconstruct messages with images injected (mimics slime data loader) ──
    messages = _build_messages(
        row,
        prompt_key="messages",
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )

    tools = json.loads(row.get("tools", "[]")) or None

    token_ids, loss_mask = mask_gen.get_loss_mask(messages, tools=tools)
    response_length = mask_gen.get_response_lengths([loss_mask])[0]

    print(f"\n{'='*60}")
    print(f"Row {args.row}: {row.get('question_id', '?')} | {row.get('num_steps', '?')} steps")
    print(f"Total tokens: {len(token_ids)} | Response length: {response_length}")
    print(f"Masked (trainable) tokens: {sum(loss_mask)} / {len(loss_mask)}")
    print(f"{'='*60}\n")

    # ── print each turn with its mask status ──────────────────────────────────
    print("LOSS MASK BY ROLE:")
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_names = [tc["function"]["name"] for tc in msg.get("tool_calls", [])]
        preview = tool_names or (content[:60] if isinstance(content, str) else f"<content list len={len(content)}>")
        expected_mask = 1 if role == "assistant" else 0
        print(f"  [{role:10s}] expected_mask={expected_mask}  preview={preview}")

    print(f"\nFirst 50 loss_mask values: {loss_mask[:50]}")
    print(f"Last  50 loss_mask values: {loss_mask[-50:]}")

    # ── decode trainable tokens for visual inspection ─────────────────────────
    print(f"\nTRAINABLE TOKENS (decoded):")
    trainable_ids = [tid for tid, m in zip(token_ids, loss_mask) if m == 1]
    print(tokenizer.decode(trainable_ids))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.3: Run the verifier**

```bash
cd /lc/AgenticMemory
python verify_parquet.py \
    --parquet logs/train_trajectories_30B.parquet \
    --model-path /workspace/.cache/huggingface/hub/Qwen3-VL-4B-Instruct \
    --row 0
```

**What to look for:**
1. `Masked (trainable) tokens` is non-zero
2. `LOSS MASK BY ROLE` shows `assistant` → `expected_mask=1`, `system`/`user`/`tool` → `expected_mask=0`
3. `TRAINABLE TOKENS (decoded)` shows function calls (e.g., `search_database`, `update_search_memory`, `final_answer`) — not document content

If you see document page text in trainable tokens, there's a masking bug.

- [ ] **Step 3.4: Commit**

```bash
cd /lc/AgenticMemory
git add verify_parquet.py
git commit -m "feat: add verify_parquet.py for loss mask inspection"
```

---

## Task 4: `run_agentic_sft.sh` + pilot training run

**Files:**
- Create: `AgenticMemory/run_agentic_sft.sh`

- [ ] **Step 4.1: Create training script**

Create `AgenticMemory/run_agentic_sft.sh`:

```bash
#!/bin/bash
# SFT training on agentic trajectories distilled from Qwen3-VL-30B.
#
# Usage:
#   ./run_agentic_sft.sh
#   SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct ./run_agentic_sft.sh
#   SLIME_SCRIPT_PARQUET=/path/to/parquet ./run_agentic_sft.sh

TRAIN_BACKEND="megatron"
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-4B-Instruct"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
PARQUET=${SLIME_SCRIPT_PARQUET:-"/lc/AgenticMemory/logs/train_trajectories_30B.parquet"}

VALID_MODELS="
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen2.5-VL-7B-Instruct
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
    echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
    exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
    USE_EXTERNAL_RAY=0
else
    USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
    ray stop --force
    pkill -9 ray
fi
pkill -9 slime
sleep 3
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

# Download model if needed
mkdir -p /workspace/.cache/huggingface/hub
if [ ! -d "/workspace/.cache/huggingface/hub/${MODEL_NAME}" ]; then
    hf download Qwen/${MODEL_NAME} --local-dir /workspace/.cache/huggingface/hub/${MODEL_NAME}
fi

CKPT_ARGS=(
    --hf-checkpoint /workspace/.cache/huggingface/hub/${MODEL_NAME}
    --load         /workspace/.cache/huggingface/hub/${MODEL_NAME}
)

SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data "${PARQUET}"
    --input-key messages
    --multimodal-keys '{"image": "images"}'
    --tool-key tools
    --rollout-shuffle
    --num-epoch 3
    --rollout-batch-size 1
    --global-batch-size 4

    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1e-6
    --lr-warmup-fraction 0.1
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-agentic-sft
        --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
        --wandb-key ${WANDB_API_KEY}
        --disable-wandb-random-suffix
    )
else
    WANDB_ARGS=()
fi

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

# Model-specific args (from slime scripts/models/)
SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../slime_0224_2026/slime" &>/dev/null && pwd)"
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/Qwen3-VL-/qwen3-/g; s/-4B/-3.7B/g; s/-8B/-7B/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

# Start Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/workspace/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node ${NUM_GPUS} \
    --multimodal-keys '{"image": "images"}' \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${BACKEND_ARGS[@]}"
```

```bash
chmod +x /lc/AgenticMemory/run_agentic_sft.sh
```

- [ ] **Step 4.2: Check model args file exists for target model**

The training script sources a model-args file from slime's `scripts/models/`. Verify it exists:

```bash
ls /workspace/src/clean_code_for_rl/slime_0224_2026/slime/scripts/models/ | grep -i qwen3
```

For `Qwen3-VL-4B-Instruct`, the sed transform produces `qwen3-3.7B`. Confirm:
```bash
ls /workspace/src/clean_code_for_rl/slime_0224_2026/slime/scripts/models/ | grep 3.7
```

If not present, check what names exist and adjust the `MODEL_ARGS_FILE` sed expression in `run_agentic_sft.sh` accordingly.

- [ ] **Step 4.3: Run pilot training (overfitting sanity check)**

```bash
cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime
export WANDB_API_KEY=your_key   # optional
./AgenticMemory/run_agentic_sft.sh
```

Monitor for:
1. First few hundred steps: loss should decrease (overfitting is expected and desired for sanity check)
2. No CUDA OOM → if OOM, reduce `--max-tokens-per-gpu` from 2048 to 1024
3. No masking error ("found 0 trainable tokens") → means loss mask bug

- [ ] **Step 4.4: Evaluate trained model on test split**

After training, load the checkpoint and run the existing `eval.py` or `search_agent.py` with the trained 4B model. Compare score to zero-shot 4B baseline.

Expected: trained model score > zero-shot 4B score (since it's overfitting to the same split).

- [ ] **Step 4.5: Commit**

```bash
cd /lc/AgenticMemory
git add run_agentic_sft.sh
git commit -m "feat: add run_agentic_sft.sh for slime SFT training

Adapts geo3k_vlm_sft script for agentic trajectories.
--tool-key passes tool schemas to MultiTurnLossMaskGenerator."
```

---

## Self-Review

**Spec coverage check:**
- ✅ Section 4.1 collect_trajectories.py → Task 2
- ✅ Section 4.2 format_trajectories.py → Task 1
- ✅ Section 4.3 run_agentic_sft.sh → Task 4
- ✅ Section 5 SGLang deployment → Task 2 Step 2.2
- ✅ Section 6.1 format verification → Task 3
- ✅ Section 6.2 overfitting sanity check → Task 4 Step 4.3
- ✅ Section 3.3 images-before-text convention → Task 1 Step 1.3 `build_messages_and_images`
- ✅ Tool schemas in `tools` column / `--tool-key` → Task 1 Step 1.3, Task 4 Step 4.1

**Type consistency:**
- `build_messages_and_images` → returns `(list[dict], list[bytes])` — consistent across Task 1 tests and implementation
- `pil_to_bytes` → takes `PIL.Image`, returns `bytes` — consistent in tests and implementation
- `b64_to_bytes` → takes `str`, returns `bytes` — used only in `format_trajectories.py`
- `pil_to_b64` → takes PIL, returns `str` — used only in `collect_trajectories.py`

**Notes on known edge cases:**
- If `response.steps[i]` returns objects instead of dicts (smolagents version differences), the `.dict()` fallback in `collect_trajectories.py` handles it
- `tool_call_id` in `tool-response` messages: the field name may be `"tool_call_id"` or `"id"` depending on smolagents version — `_extract_tool_responses` tries both with `msg.get("tool_call_id") or msg.get("id", "unknown")`
- `SLIME_DIR` path in `run_agentic_sft.sh` assumes AgenticMemory is inside the slime repo — adjust if different
