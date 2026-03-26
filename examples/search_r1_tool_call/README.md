# search_r1_tool_call

Standard tool-calling version of [Search-R1](https://github.com/PeterGriffinJin/Search-R1).

## What changed from `examples/search-r1/`

| Aspect | `search-r1` (custom) | `search_r1_tool_call` (standard) |
|---|---|---|
| Model output — search | `<search>query</search>` | `<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>` |
| Model output — answer | `<answer>text</answer>` | `<tool_call>{"name":"answer","arguments":{"response":"text"}}</tool_call>` |
| Tool schema | none | OpenAI JSON schema via `--tool-key tools` |
| Observation injection | raw string concat | `apply_chat_template` → `<\|im_start\|>tool\n...<\|im_end\|>` tokens |
| rollout.py | custom `generate_with_search.generate` | reuses `geo3k_vlm_multi_turn.rollout.generate` unchanged |
| System prompt | none | none (instructions are in the tool descriptions + model pre-training) |
| User prompt | explicit ReAct routing + `<think>` | just the question (Qwen3 handles orchestration) |

## File structure

```
examples/search_r1_tool_call/
├── config.yaml          — max_turns, search backend, reward weights
├── env_search.py        — SearchEnv environment + build_env() factory
├── reward.py            — reward_func: EM scoring, answer extracted from answer tool call
├── run_qwen3_4B.sh      — launch script for Qwen3-4B (enable_thinking=true)
└── run_qwen2.5_3B.sh    — launch script for Qwen2.5-3B

Search-R1/scripts/data_process/
├── qa_search_train_merge_standard.py  — generates train parquet with two-tool format
└── qa_search_test_merge_standard.py   — generates test parquet (test/dev split)
```

Reused without modification:
- `examples/geo3k_vlm_multi_turn/rollout.py` — multi-turn rollout loop
- `examples/search-r1/local_search_server.py` — local dense retrieval backend
- `examples/search-r1/google_search_server.py` — Google search backend

## Setup

Follow [examples/search-r1/README.md](../search-r1/README.md) for environment setup and model download.

### 1. Generate the dataset

**Train data (nq + hotpotqa):**
```bash
python /workspace/src/clean_code_for_rl/Search-R1/scripts/data_process/qa_search_train_merge_standard.py \
    --local_dir /workspace/src/clean_code_for_rl/Search-R1/data/nq_hotpotqa_train_standard \
    --data_sources nq,hotpotqa \
    --model_family qwen3
```
Output: `nq_hotpotqa_train_standard/train_standard.parquet`

**Test data (nq only — hotpotqa has no test split in FlashRAG):**
```bash
python /workspace/src/clean_code_for_rl/Search-R1/scripts/data_process/qa_search_test_merge_standard.py \
    --local_dir /workspace/src/clean_code_for_rl/Search-R1/data/nq_hotpotqa_train_standard \
    --data_sources nq \
    --model_family qwen3
```
Output: `nq_hotpotqa_train_standard/test_standard.parquet`

Each parquet has:
- User message: just the question (`"Question: {question}"`)
- `tools` column with both `search` and `answer` schemas

Qwen3's built-in thinking and tool-call pre-training handle the search/answer orchestration from the tool descriptions alone — no explicit routing instructions in the user prompt are needed.

> **Using Qwen2.5?** Pass `--model_family qwen2.5` to add explicit ReAct routing and `<think>` instructions to the user message.

### 2. Convert checkpoint to Megatron format

```bash

# hf checkpoint
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

Update `--hf-checkpoint` and `--save` to match your paths. The output directory is what `--ref-load` points to in the run script.

Specifically

```bash
# hf checkpoint
hf download Qwen/Qwen3-4B --local-dir /workspace/.cache/huggingface/hub/Qwen3-4B

cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/workspace/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /workspace/.cache/huggingface/hub/Qwen3-4B \
    --save /workspace/.cache/huggingface/hub/Qwen3-4B_torch_dist
```

### 3. (Optional) Start local retrieval server

See [search-r1/README.md Appendix](../search-r1/README.md#appendix-setting-up-local-retriever).

### 4. Run training

```bash
cd /root/slime
bash examples/search_r1_tool_call/run_qwen3_4B.sh
```

`run_qwen3_4B.sh` already includes `--apply-chat-template-kwargs '{"enable_thinking": true}'` to activate Qwen3's built-in thinking mode. For other Qwen3 sizes, copy the script and update the model script source and checkpoint paths accordingly.

## Configuration

Edit `config.yaml` to change search backend or concurrency:

```yaml
max_turns: 2              # search turns per episode
search_backend: local     # "local" or "google"
search_concurrency: 256   # max concurrent search requests
topk: 3                   # number of retrieved documents
```

The reward is sparse 0/1 (exact match only — no partial credit). This exactly reproduces the original search-r1 reward definition.

## How it works

### Tool call pipeline

```
1. apply_chat_template(messages, tools=[SEARCH_TOOL, ANSWER_TOOL], enable_thinking=True)
   → Qwen3 embeds both tool schemas in the system prompt automatically
   → enable_thinking activates Qwen3's built-in <think>...</think> reasoning

2. Model thinks (built-in), then searches:
   <tool_call>{"name":"search","arguments":{"query":"capital of France"}}</tool_call>

3. env_search.py parses the JSON, runs the search via async_utils.run()
   (backend: "local" dense retrieval or "google" — selected in config.yaml)

4. apply_chat_template wraps the result as:
   <|im_start|>tool\nDoc 1(Title: France) Paris is...<|im_end|>

5. Model thinks again (built-in), then submits final answer:
   <tool_call>{"name":"answer","arguments":{"response":"Paris"}}</tool_call>

6. env_search.step() sees "answer" tool → done=True, empty observation

7. reward.py strips tool turns, extracts answer from answer tool call JSON,
   checks EM vs ground truth
```

### Search backend

The `search` tool is backend-agnostic. `config.yaml` selects:
- `search_backend: local` → `local_search_server.py` (dense retrieval, no external API)
- `search_backend: google` → `google_search_server.py` (Google Search API)

The model always calls `search` — it never knows or needs to know which backend runs.

### Async bridge

`env_search.py`'s `step()` is `async def` — it `await`s the search coroutine directly.
`geo3k_vlm_multi_turn/rollout.py` detects this with `asyncio.iscoroutine()` and `await`s it,
so all concurrent rollout generate() coroutines make progress without blocking each other.
