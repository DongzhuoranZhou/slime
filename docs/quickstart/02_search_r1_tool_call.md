# 02 — Search-R1 Tool Calling (Standard Function Calling RL)

> **Where this fits:** After running Search-R1 with custom XML tags, this experiment re-implements the same task using **standard JSON tool calls** (`<tool_call>{"name":"search",...}</tool_call>`). This is the format compatible with Qwen3's built-in tool-calling pre-training, and is the format we need for future agentic VLM work. This was a manual modification — this page documents what changed and why.

**Code:** [`examples/search_r1_tool_call/`](https://github.com/DongzhuoranZhou/slime/tree/dev_main/examples/search_r1_tool_call)
**Reuses from geo3k_vlm_multi_turn:** [`examples/geo3k_vlm_multi_turn/rollout.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/geo3k_vlm_multi_turn/rollout.py) (unchanged)

---

## What Changed from Search-R1

| Aspect | `search-r1` (custom) | `search_r1_tool_call` (standard) |
|--------|---------------------|----------------------------------|
| Model output — search | `<search>query</search>` | `<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>` |
| Model output — answer | `<answer>text</answer>` | `<tool_call>{"name":"answer","arguments":{"response":"text"}}</tool_call>` |
| Tool schema | none | OpenAI JSON schema via `--tool-key tools` |
| Observation injection | raw string concat | `apply_chat_template` → `<|im_start|>tool\n...<|im_end|>` tokens |
| Rollout implementation | custom `generate_with_search.generate` | reuses `geo3k_vlm_multi_turn.rollout.generate` unchanged |
| User prompt | explicit ReAct routing + `<think>` | just the question (Qwen3 handles orchestration via tool descriptions) |

**Why this matters:** The custom XML format worked, but it's a hack. The standard JSON tool-call format is what Qwen3 (and most modern models) are pre-trained on. Switching to it means:
1. The model's existing tool-call pre-training activates — less RL work needed to teach syntax
2. The same environment abstraction (`env_search.py`) can be reused for any future agentic task
3. `geo3k_vlm_multi_turn/rollout.py` — already written for geo3k — works without modification

---

## File Structure

```
examples/search_r1_tool_call/
├── config.yaml       — max_turns, search backend, concurrency
├── env_search.py     — SearchEnv environment + build_env() factory
├── reward.py         — reward_func: EM scoring on answer tool call
├── run_qwen3_4B.sh   — launch script (Qwen3-4B, enable_thinking=true)
└── run_qwen2.5_3B.sh — launch script (Qwen2.5-3B)
```

Reused without modification:
- [`examples/geo3k_vlm_multi_turn/rollout.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/geo3k_vlm_multi_turn/rollout.py) — multi-turn rollout loop
- [`examples/search-r1/local_search_server.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/search-r1/local_search_server.py) — local dense retrieval
- [`examples/search-r1/google_search_server.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/search-r1/google_search_server.py) — Google search

---

## Setup

### 1. Environment

Follow [01 Search-R1 Setup](./01_search_r1.md#setup) for slime install and search backend setup. The retrieval servers are shared between both experiments.

### 2. Generate the dataset (standard tool-call format)

The standard format adds a `tools` column to each parquet row containing both `search` and `answer` tool schemas. This is what `apply_chat_template` uses to inject tool descriptions into the system prompt.

**Train data (NQ + HotpotQA):**
```bash
python /root/Search-R1/scripts/data_process/qa_search_train_merge_standard.py \
    --local_dir /root/Search-R1/data/nq_hotpotqa_train_standard \
    --data_sources nq,hotpotqa \
    --model_family qwen3
```
Output: `nq_hotpotqa_train_standard/train_standard.parquet`

**Test data (NQ only — HotpotQA has no test split in FlashRAG):**
```bash
python /root/Search-R1/scripts/data_process/qa_search_test_merge_standard.py \
    --local_dir /root/Search-R1/data/nq_hotpotqa_train_standard \
    --data_sources nq \
    --model_family qwen3
```
Output: `nq_hotpotqa_train_standard/test_standard.parquet`

> Using Qwen2.5? Pass `--model_family qwen2.5` to add explicit ReAct routing and `<think>` tags to the user prompt. Qwen3 doesn't need these — its built-in thinking + tool-call pre-training handles orchestration from tool descriptions alone.

### 3. Download and convert Qwen3-4B

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

---

## Configuration

Edit [`config.yaml`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/search_r1_tool_call/config.yaml):

```yaml
max_turns: 2              # search rounds per episode
search_backend: local     # "local" or "google"
search_concurrency: 256   # max concurrent async search requests
topk: 3                   # number of retrieved documents per query
```

The reward is sparse 0/1 (EM only — no partial credit), matching the original Search-R1 definition.

---

## Launch

```bash
cd /root/slime
bash examples/search_r1_tool_call/run_qwen3_4B.sh
```

The script includes `--apply-chat-template-kwargs '{"enable_thinking": true}'` to activate Qwen3's built-in `<think>...</think>` reasoning mode.

---

## How the Tool Call Pipeline Works

```
1. apply_chat_template(messages, tools=[SEARCH_TOOL, ANSWER_TOOL], enable_thinking=True)
   → Qwen3 embeds both tool schemas in the system prompt
   → enable_thinking activates <think>...</think> reasoning

2. Model thinks (built-in), then searches:
   <tool_call>{"name":"search","arguments":{"query":"capital of France"}}</tool_call>

3. env_search.py parses the JSON, runs the query via async search backend

4. apply_chat_template wraps the result as a tool response:
   <|im_start|>tool\nDoc 1(Title: France) Paris is...<|im_end|>

5. Model thinks again, then submits final answer:
   <tool_call>{"name":"answer","arguments":{"response":"Paris"}}</tool_call>

6. env_search.step() sees "answer" tool → done=True, episode ends

7. reward.py strips tool turns, extracts answer from answer tool call JSON,
   checks EM vs ground truth → reward 0 or 1
```

### Async bridge

`env_search.py`'s `step()` is `async def` — it `await`s the search coroutine directly. The rollout loop in `geo3k_vlm_multi_turn/rollout.py` detects this with `asyncio.iscoroutine()` and awaits it, so all concurrent rollout coroutines make progress without blocking each other. This is what enables `search_concurrency: 256` — hundreds of searches happening in parallel across the batch.

---

## Metrics

Same metrics as Search-R1. See [01 Search-R1 Metrics](./01_search_r1.md#metrics).

| Metric | What to watch |
|--------|--------------|
| `eval/nq_test` | **Primary.** Mean EM on NQ test set. Should increase. |
| `rollout/zero_std/count_0.0` | Should decrease as model learns to call tools correctly. |
| `rollout/response_len/mean` | Should grow as model learns multi-turn search behavior. |
| `train/kl_loss` | Keep below ~0.1. |

**W&B x-axis:** use `rollout/step`, not raw Step.

---

## What This Enables Next

With standard tool calls working:
- The same `rollout.py` can be pointed at any environment that implements `build_env()` / `reset()` / `step()` / `format_observation()`
- The `tools` column in the dataset carries tool schemas — swap it to change what tools the model can call
- This is the exact architecture used for multi-turn VLM agentic training: the model sees an image, calls tools, receives observations, reasons, answers
