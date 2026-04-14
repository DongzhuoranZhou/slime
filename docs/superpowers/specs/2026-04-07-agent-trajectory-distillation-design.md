# Agent Trajectory Distillation Design

**Date:** 2026-04-07
**Goal:** Distill agentic search trajectories from Qwen3-VL-30B-A3B-Instruct into smaller VLM models (4B/8B) using slime's SFT pipeline, for the VLM long-document QA task.

---

## 1. Overview

The AgenticMemory system uses a `smolagents` `ToolCallingAgent` backed by a VLM to answer questions over long documents stored in a Qdrant vector DB. The agent iteratively calls `search_database`, `get_specific_pages`, `update_search_memory`, and `reflect_on_search` tools, then produces a `final_answer`. Tool results include retrieved document page images (PIL).

The distillation pipeline has three phases:

1. **Trajectory Collection** — run the agent with Qwen3-VL-30B served via SGLang on the test split, capture full multi-turn message histories
2. **Trajectory Formatting** — convert smolagents memory steps to a parquet file compatible with slime's SFT data loader
3. **SFT Training** — train a 4B or 8B Qwen3-VL model on the formatted trajectories using slime's existing SFT pipeline

The rationale for starting with the open-source model: zero token cost, fast iteration on trajectory format correctness, and a controlled overfitting sanity check (train and test on the same split → expect strong improvement).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Trajectory Collection                          │
│                                                         │
│  SGLang server (Qwen3-VL-30B-A3B-Instruct, TP=4)       │
│        ↑ OpenAI-compatible /v1/chat/completions         │
│  collect_trajectories.py                                │
│    └── SearchAgent (smolagents ToolCallingAgent)        │
│          ├── search_database → PIL images               │
│          ├── get_specific_pages → PIL images            │
│          ├── update_search_memory → text                │
│          ├── reflect_on_search → text                   │
│          └── final_answer → string                      │
│    └── saves raw_trajectories.jsonl                     │
└─────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Trajectory Formatting                          │
│                                                         │
│  format_trajectories.py                                 │
│    ├── reads raw_trajectories.jsonl                     │
│    ├── converts smolagents ActionSteps → messages list  │
│    ├── inserts <image> placeholders in tool messages    │
│    ├── collects PIL images into flat images list        │
│    └── saves train_trajectories.parquet                 │
│         columns: messages, images, question_id,         │
│                  doc_name, question, ground_truth,       │
│                  model_answer, score, num_steps          │
└─────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: SFT Training                                   │
│                                                         │
│  run_agentic_sft.sh                                     │
│    ├── --rollout-function-path sft_rollout.generate_rollout
│    ├── --prompt-data train_trajectories.parquet         │
│    ├── --input-key messages                             │
│    ├── --multimodal-keys '{"image": "images"}'          │
│    └── target: Qwen3-VL-4B-Instruct or 8B-Instruct     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Format Specification

### 3.1 Raw Trajectory (intermediate, jsonl)

Each line is a JSON object saved after each `agent.run()` call:

```json
{
  "question_id": "mmlongdoc_001",
  "doc_name": "annual_report_2023",
  "question": "What was the total revenue in Q3?",
  "ground_truth": "42M",
  "model_answer": "The total revenue in Q3 was 42M.",
  "score": 1.0,
  "num_steps": 3,
  "steps": [
    {
      "role": "system",
      "content": "<system prompt string>"
    },
    {
      "role": "user",
      "content": "Based on Document annual_report_2023, answer..."
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_abc",
          "type": "function",
          "function": {
            "name": "search_database",
            "arguments": "{\"query\": \"...\", \"doc_name\": \"...\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc",
      "observation_text": "Search results for '...' in '...': Found relevant content on pages 3, 7, 12.",
      "image_paths": ["/lc/.../page_3.png", "/lc/.../page_7.png", "/lc/.../page_12.png"]
    },
    "..."
  ]
}
```

Images are stored as file paths in the raw jsonl (not base64) to keep the file small and human-readable. The formatter script loads them as PIL at parquet-writing time.

### 3.2 Formatted Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `messages` | list of dicts | Full conversation in OpenAI chat format, with `<image>` placeholders in tool message text content |
| `images` | list of PIL/bytes | Flat list of all images in order of appearance across all tool result messages |
| `question_id` | string | For traceability |
| `doc_name` | string | Source document |
| `question` | string | Original question |
| `ground_truth` | string | Reference answer |
| `model_answer` | string | 30B model's extracted final answer |
| `score` | float | LLM-judged score from trajectory collection |
| `num_steps` | int | Number of search steps taken |

### 3.3 Messages Format (the key column)

The `messages` list follows OpenAI function-calling format, which Qwen3's tokenizer chat template natively handles:

```python
[
  # System prompt — loss_mask=0 (role != "assistant")
  {"role": "system", "content": "<smolagents system prompt>"},

  # User query — loss_mask=0
  {"role": "user", "content": "Based on Document <annual_report_2023>, answer: What was the total revenue in Q3?"},

  # Step 1: assistant calls search_database — loss_mask=1 (TRAIN ON THIS)
  {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "search_database", "arguments": "{\"query\": \"...\", \"doc_name\": \"...\"}"}}]
  },

  # Step 1: tool result with image placeholders — loss_mask=0
  # Images come BEFORE text (standard VLM convention in this codebase, see base_env.py format_observation)
  {
    "role": "tool",
    "tool_call_id": "call_abc",
    "content": "<image><image><image>Search results for '...' in '...': Found relevant content on pages 3, 7, 12."
  },

  # Step 2: assistant calls update_search_memory — loss_mask=1 (TRAIN ON THIS)
  {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": "call_def", "type": "function", "function": {"name": "update_search_memory", "arguments": "{...}"}}]
  },

  # Step 2: tool result (text only, no images) — loss_mask=0
  {"role": "tool", "tool_call_id": "call_def", "content": "✓ Search #1 recorded: ..."},

  # ... more steps ...

  # Final step: assistant calls final_answer — loss_mask=1 (TRAIN ON THIS)
  {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": "call_xyz", "type": "function", "function": {"name": "final_answer", "arguments": "{\"answer\": \"The total revenue in Q3 was 42M.\"}"}}]
  }
]
```

**Loss masking:** `MultiTurnLossMaskGenerator` in slime already applies `loss_mask=0` to all non-assistant turns automatically. No `step_loss_mask` overrides are needed for this initial version.

**Image placeholder convention:** Each `<image>` in a tool message content string corresponds to one image popped from the flat `images` list, in document order. The formatter must prepend exactly N `<image>` tokens before the text in the tool message for N retrieved pages (images before text — standard VLM convention per `base_env.py::format_observation`). This matches slime's existing `--multimodal-keys '{"image": "images"}'` convention used in geo3k.

**Tool definitions:** The tool JSON schemas are passed via a `tools` column (optional). The `MultiTurnLossMaskGenerator` already accepts a `tools` argument which it passes to `tokenizer.apply_chat_template`. This ensures the tool definitions appear in the prompt context but are not trained on.

---

## 4. Component Specifications

### 4.1 `collect_trajectories.py`

**Location:** `AgenticMemory/collect_trajectories.py`

**Inputs:**
- `--dataset`: dataset name (default: `mmlongdoc`)
- `--split`: `test` (default)
- `--sglang-url`: SGLang server URL (default: `http://localhost:30000`)
- `--model-name`: model identifier for LiteLLM (default: `openai/Qwen3-VL-30B-A3B-Instruct`)
- `--output`: output jsonl path (default: `logs/trajectories_30B_test.jsonl`)
- `--num-samples`: limit number of samples (for debugging)

**Logic:**
1. Load test split from `PathManager.get_dataset_jsonl(dataset)`
2. Initialize `SearchAgent` with `LiteLLMModel` pointing to `sglang-url`
3. For each sample, run `agent.run(query, return_full_result=True)`
4. Extract message history from `agent.memory.steps` (see Section 4.1.1)
5. Call `evaluate_response` (already in `utils.py`)
6. Append result to output jsonl (resume-safe: skip already-processed question_ids)

**4.1.1 Extracting message history from smolagents memory**

After `agent.run()`, `agent.memory.steps` contains:
- `steps[0]`: `SystemPromptStep` — system prompt text
- `steps[1]`: `TaskStep` — user query text
- `steps[2:]`: `ActionStep` list, each with:
  - `model_output_message.tool_calls` — list of `ChatMessageToolCall` (what assistant called)
  - `observations` — text string from tool execution
  - `observations_images` — list of PIL images (may be empty for non-image tools)
  - The tool_call_id is in each `tool_call.id`

The formatter reconstructs messages by interleaving:
- Each ActionStep → one assistant message (tool_calls) + one tool message (observations + images)
- The final ActionStep calls `final_answer` → assistant message only (no tool result appended)

### 4.2 `format_trajectories.py`

**Location:** `AgenticMemory/format_trajectories.py`

**Inputs:**
- `--input`: raw trajectories jsonl (default: `logs/trajectories_30B_test.jsonl`)
- `--output`: parquet path (default: `logs/train_trajectories_30B.parquet`)
- `--filter-scores`: minimum score to include (default: `0.0`, include all)

**Logic:**
1. Read each raw trajectory
2. Convert `steps` list → `messages` list (see Section 3.3 format)
3. For each tool message, count images, prepend `<image>` * N to content string (images before text), add PIL images to flat `images` list
4. Convert PIL images to bytes (`io.BytesIO`) for parquet serialization
5. Build a `datasets.Dataset` from list of dicts, save as parquet

**Score filtering:** Default includes all trajectories (even failed ones). Rationale: even a wrong final answer still teaches the model the correct search strategy. Can be tuned later.

**Tools column:** Extract tool JSON schemas from smolagents tool definitions and store as a `tools` column (JSON string, one per row — same tools for all rows since the agent config is fixed).

### 4.3 `run_agentic_sft.sh`

**Location:** `AgenticMemory/run_agentic_sft.sh`

Adapts `examples/geo3k_vlm/run_geo3k_vlm_sft_zhipu_cluster.sh` with these changes:
- `--prompt-data`: points to `train_trajectories_30B.parquet`
- `--input-key messages`
- `--multimodal-keys '{"image": "images"}'`
- `--tool-key tools` (new arg, to pass tool schemas to `MultiTurnLossMaskGenerator`)
- Target model: `Qwen3-VL-4B-Instruct` or `Qwen3-VL-8B-Instruct`
- `--num-epoch 3` (trajectories are small, overfit intentionally for sanity check)
- `--rollout-batch-size 1` (each trajectory is long due to multi-turn + images)

---

## 5. SGLang Deployment

Deploy before running `collect_trajectories.py`:

```bash
python -m sglang.launch_server \
  --model-path /workspace/.cache/huggingface/hub/Qwen3-VL-30B-A3B-Instruct \
  --tp 4 \
  --port 30000 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85
```

The `LiteLLMModel` in `collect_trajectories.py` connects via:
```python
LiteLLMModel(
    model_id="openai/Qwen3-VL-30B-A3B-Instruct",
    api_base="http://localhost:30000/v1",
    api_key="EMPTY",
    do_sample=False,
)
```

This is a standalone SGLang deployment, not going through slime's `--debug-rollout-only` mechanism. The slime author's guidance confirms this is the right approach: slime is a training framework; data collection is better done with a standalone SGLang server.

---

## 6. Verification Plan

### 6.1 Format Verification (before training)

1. Load 1 sample from the parquet using slime's `Dataset` class directly
2. Run `MultiTurnLossMaskGenerator.get_loss_mask(messages, tools=tools)` on it
3. Verify: `loss_mask=1` only on assistant turns (tool call tokens)
4. Verify: `loss_mask=0` on system, user, and tool turns
5. Decode the masked tokens and visually inspect they are the expected tool calls
6. Verify images are correctly injected into tool message positions

### 6.2 Overfitting Sanity Check (training)

- Dataset: test split of MMLongDoc (same split used for collection)
- Model: Qwen3-VL-4B-Instruct (cheapest, fastest)
- Epochs: 3
- Expected result: the 4B model should score significantly higher than its zero-shot baseline on the same test split, since it is overfitting to distilled trajectories from the 30B model

If overfitting shows no improvement, the trajectory format or loss masking is wrong — fix before scaling.

---

## 7. Implementation Order

1. `collect_trajectories.py` — collect raw trajectories from 30B model
2. `format_trajectories.py` — convert to parquet, verify format manually
3. Format verification (Section 6.1) — confirm loss masking is correct before any training
4. `run_agentic_sft.sh` — train 4B on test split, run overfitting sanity check
5. If sanity check passes: scale to full training split, try 8B model

---

## 8. Out of Scope (Phase 2, closed-source)

Closed-source model trajectory distillation (Gemini 2.5 Pro, GPT-4o, etc.) follows the same pipeline. The only difference is `collect_trajectories.py` uses the existing API-backed `LiteLLMModel` from `search_agent.py` instead of the local SGLang server. No new infrastructure needed — the format and training pipeline are identical once trajectories are collected.
