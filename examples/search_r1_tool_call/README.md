# search_r1_tool_call

Standard tool-calling version of [Search-R1](https://github.com/PeterGriffinJin/Search-R1).

## What changed from `examples/search-r1/`

| Aspect | `search-r1` (custom) | `search_r1_tool_call` (standard) |
|---|---|---|
| Model output | `<search>query</search>` | `<tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>` |
| Tool schema | none | OpenAI JSON schema via `--tool-key` |
| Observation injection | raw string concat | `apply_chat_template` → `<\|im_start\|>tool\n...<\|im_end\|>` tokens |
| rollout.py | custom `generate_with_search.generate` | reuses `geo3k_vlm_multi_turn.rollout.generate` unchanged |
| Final answer | `<answer>text</answer>` | `<answer>text</answer>` (same) |

## File structure

```
examples/search_r1_tool_call/
├── config.yaml        — max_turns, search backend, reward weights
├── env_search.py      — SearchEnv environment + build_env() factory
├── reward.py          — reward_func: EM scoring for standard format
├── prepare_data.py    — adds "tools" column to NQ/HotpotQA parquet
└── run_qwen2.5_3B.sh  — launch script
```

Reused without modification:
- `examples/geo3k_vlm_multi_turn/rollout.py` — multi-turn rollout loop
- `examples/search-r1/local_search_server.py` — local dense retrieval backend
- `examples/search-r1/google_search_server.py` — Google search backend

## Setup

Follow [examples/search-r1/README.md](../search-r1/README.md) for environment setup and
model/data download. Then run the additional step below.

### 1. Prepare the dataset

Add the `tools` column and rewrite system messages:

```bash
cd /root/slime
python examples/search_r1_tool_call/prepare_data.py \
    --input  /root/Search-R1/data/nq_hotpotqa_train/train.parquet \
    --output /root/Search-R1/data/nq_hotpotqa_train/train_with_tools.parquet \
    --add-tool-instructions
```

### 2. (Optional) Start local retrieval server

See [search-r1/README.md Appendix](../search-r1/README.md#appendix-setting-up-local-retriever).

### 3. Run training

```bash
cd /root/slime
bash examples/search_r1_tool_call/run_qwen2.5_3B.sh
```

## Configuration

Edit `config.yaml` to change search backend, concurrency, or reward weights:

```yaml
max_turns: 2              # search turns per episode
search_backend: local     # "local" or "google"
search_concurrency: 256   # max concurrent search requests
topk: 3                   # number of retrieved documents

# Reward
reward_score: 1.0         # EM match + tool used
reward_format_score: 0.1  # deducted when EM match but no tool used
reward_tool_use_score: 0.05  # partial credit for using tool without EM match
```

## How it works

### Tool call pipeline

```
1. apply_chat_template(messages, tools=[WEB_SEARCH_TOOL])
   → Qwen embeds tool schema in system prompt automatically

2. Model generates:
   <tool_call>{"name":"web_search","arguments":{"query":"capital of France"}}</tool_call>

3. env_search.py parses the JSON, runs the search via async_utils.run()

4. apply_chat_template wraps the result as:
   <|im_start|>tool\nDoc 1(Title: France) Paris is...<|im_end|>

5. Model sees results and generates final answer:
   <answer>Paris</answer>

6. reward.py strips tool turns, extracts <answer>, checks EM vs ground truth
```

### Async/sync bridge

The rollout loop (`geo3k_vlm_multi_turn/rollout.py`) calls `env.step()` synchronously.
The search backends are async. `env_search.py` bridges this with
`slime.utils.async_utils.run(coro)` which submits the coroutine to a persistent
background event loop thread (L2), blocking only the calling thread — not the
main asyncio event loop running the rollout.
