# search_r1_tool_call — Quickstart & Code Reading Guide

This is the **standard tool-calling** version of Search-R1 running inside slime.
Use this doc to onboard quickly or refresh your memory after time away.

---

## 1. Big-Picture Difference from `examples/search-r1`

| Aspect | `search-r1` (custom) | `search_r1_tool_call` (standard) |
|---|---|---|
| Model output — search | `<search>query</search>` | `<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>` |
| Model output — answer | `<answer>text</answer>` | `<tool_call>{"name":"answer","arguments":{"response":"text"}}</tool_call>` |
| Tool schema injection | none | OpenAI JSON schema via `--tool-key` → `apply_chat_template(tools=...)` |
| Observation injection | raw string concat | `apply_chat_template` wraps result as `<\|im_start\|>tool\n...<\|im_end\|>` tokens |
| rollout.py | custom `generate_with_search.generate` (~250 lines) | **reused** `geo3k_vlm_multi_turn.rollout.generate` unchanged |
| User prompt | explicit ReAct routing + `<think>` | just the question (Qwen3 handles orchestration) |
| Thinking | manual `<think>` instruction | Qwen3 built-in (`enable_thinking=True`) |

**Bottom line:** same task, same data, same search backends — both the search action and the final answer submission now use Qwen's native tool-call format. Qwen3's tool-call pre-training and built-in thinking replace the need for explicit ReAct routing in the user prompt.

---

## 2. Guided Code Reading Order

Read the files in this order. Each takes 2–5 min. The order follows the data flow.

### Step 1 — `qa_search_train_merge_standard.py` (~130 lines)
**What it does:** generates the training parquet from scratch using `RUC-NLPIR/FlashRAG_datasets`. Produces `train_standard.parquet` with:
- `prompt`: list of messages with a user turn — for Qwen3 this is just `"Question: {question}"` with no routing instructions
- `reward_model`: ground-truth label
- `tools`: JSON string of both tool schemas — slime reads this and passes to `apply_chat_template(tools=...)`

**Key function to read:**
```
make_prefix(question, model_family="qwen3")
  — Qwen3: returns just "Question: {question}"
      Tool descriptions convey semantics; Qwen3's built-in thinking
      (enable_thinking=True) handles the reasoning step automatically.
  — Qwen2.5: adds explicit ReAct routing + <think> instructions,
      since Qwen2.5 has no built-in thinking mode.
```

**Why the user prompt is minimal for Qwen3:** Qwen3 was instruction-tuned on tool calling and has built-in thinking. It understands from tool descriptions alone that `search` = get info when lacking knowledge, `answer` = submit final answer. No explicit ReAct loop instructions needed — the tool schemas carry all the semantic information. This is the same reason `examples/geo3k_vlm_multi_turn` uses raw problem text with no routing instructions.

---

### Step 2 — `config.yaml` (~25 lines)
**What it does:** single source of truth for all runtime hyperparameters passed to the env and reward.

Read top-to-bottom, it answers three questions:
1. How many search turns per episode? (`max_turns: 2`)
2. Which search backend? (`search_backend: local` or `google`)
3. How are rewards shaped? (`reward_score`, `reward_format_score`, `reward_tool_use_score`)

The `rollout_interaction_env_path` line is how slime's `geo3k_vlm_multi_turn/rollout.py` discovers which `SearchEnv` class to load.

---

### Step 3 — `env_search.py` (~290 lines) ← **core file**
**What it does:** defines `SearchEnv`, the per-episode state machine that:
1. Receives the model's text response
2. Parses any `<tool_call>` JSON from it
3. Handles `search` → executes search (local or Google), returns result as next observation
4. Handles `answer` → ends episode immediately with empty observation

**Read in this order:**

```python
# 1. Module-level constants
TOOL_CALL_RE      # regex to find <tool_call>…</tool_call>
SEARCH_TOOL       # OpenAI-compatible schema for search (backend-agnostic)
ANSWER_TOOL       # OpenAI-compatible schema for final answer submission
TOOLS = [SEARCH_TOOL, ANSWER_TOOL]  # both injected via apply_chat_template
_SEARCH_SEMAPHORE # rate-limits concurrent searches

# 2. Helper functions
_passages2string()    # formats retrieval results (same as search-r1)
_dispatch_search()    # picks local vs google backend, honors concurrency limit

# 3. SearchEnv class
SearchEnv._extract_tool_call()  # parses JSON from <tool_call>...</tool_call>
SearchEnv.step()                # state machine:
                                #   "answer" tool → done=True, empty obs
                                #   "search" tool → execute + return obs
                                #   no tool call → done=True (free-form response)
SearchEnv.format_observation()  # wraps result as {"role": "tool", "content": ...}

# 4. Factory
build_env()     # called by geo3k_vlm_multi_turn/rollout.py to create the env
                # injects TOOLS (both schemas) into sample.metadata["tools"]
                # metadata["tools"] is used by rollout.py for subsequent observation turns
```

**Key insight — two places tools are needed:**
- **Parquet `tools` column** → read by `data.py` at load time → passed to `apply_chat_template(tools=...)` for the **initial prompt** (creates system message with tool definitions)
- **`sample.metadata["tools"]`** set by `build_env()` → used by `rollout.py` for each **subsequent observation turn** (search results injected as tool-role messages)

**Async/sync bridge:**
```python
result_str = run(_dispatch_search(query, self._config))
```
`env.step()` is called synchronously. Search is async. `async_utils.run()` submits to a background event loop thread, blocking only the calling thread.

---

### Step 4 — `reward.py` (~135 lines)
**What it does:** scores each completed episode.

**Read in this order:**

```python
# 1. Regex patterns
_SEARCH_RE        # did the model use the search tool? (reward shaping)
_TOOL_TURN_RE     # strips <|im_start|>tool\n...<|im_end|> turns (search results)
                  # must strip first — results could contain JSON matching answer pattern
_ANSWER_TOOL_RE   # finds the answer tool call JSON in model response

# 2. EM helpers (inlined from search-r1/qa_em_format.py)
_normalize_answer()   # lowercase, strip punctuation/articles
_em_check()           # compare normalized prediction vs golden list

# 3. Extraction
_extract_final_answer()  # strips tool turns, parses answer tool call JSON → arguments["response"]
_get_golden_answers()    # resolves label → list of accepted answer strings

# 4. Main function
async def reward_func()
```

**Reward table:**

| EM match? | search used? | Reward |
|---|---|---|
| ✓ | ✓ | `reward_score` (1.0) |
| ✓ | ✗ | `reward_score - reward_format_score` (0.9) |
| ✗ | ✓ | `reward_tool_use_score` (0.05) |
| ✗ | ✗ | 0.0 |

---

### Step 5 — `run_qwen3.sh`
**What it does:** the launch script. Key args:

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
   --custom-rm-path examples.search_r1_tool_call.reward.reward_func
   --custom-config-path examples/search_r1_tool_call/config.yaml
)

ROLLOUT_ARGS=(
   --tool-key tools                                      # reads "tools" column from parquet
   --apply-chat-template-kwargs '{"enable_thinking": true}'  # Qwen3 built-in thinking
   ...
)
```

`--tool-key tools` tells slime's data loader to pass the `tools` column into `apply_chat_template(tools=...)`. `enable_thinking=True` activates Qwen3's built-in `<think>` reasoning — no need to instruct this in the user prompt.

---

### Step 6 — (Reference) `geo3k_vlm_multi_turn/rollout.py`
**You don't own this file** — `search_r1_tool_call` reuses it unchanged.

```
for each prompt:
    env = build_env(sample, args)          # ← calls env_search.py:build_env()
    env.reset()
    while not done:
        response = sglang_generate(messages)
        obs, done, info = env.step(response)   # ← calls SearchEnv.step()
        if not done:
            messages.append(env.format_observation(obs))
    compute reward
```

---

## 3. Full Pipeline Visualization

```
qa_search_train_merge_standard.py (from RUC-NLPIR/FlashRAG_datasets, --model_family qwen3)
  → train_standard.parquet
      columns: [prompt ([{"role":"user","content":"Question: What is..."}]),
                reward_model (label),
                tools (JSON: [search, answer] schemas)]

run_qwen3.sh → train.py
    --tool-key tools
    --apply-chat-template-kwargs '{"enable_thinking": true}'
    --custom-generate-function-path geo3k_vlm_multi_turn.rollout.generate
    --custom-rm-path reward.reward_func
    --custom-config-path config.yaml

geo3k_vlm_multi_turn/rollout.generate()
  per sample:
    build_env() [env_search.py]
      → injects TOOLS (both schemas) into sample.metadata["tools"]
      → creates SearchEnv(max_turns=2, ...)

    apply_chat_template(messages, tools=[SEARCH_TOOL, ANSWER_TOOL], enable_thinking=True)
      → Qwen3 embeds both tool schemas in system prompt automatically
      → built-in thinking activated

    Qwen3 thinks (built-in), then searches:
      <tool_call>{"name":"search","arguments":{"query":"capital of France"}}</tool_call>

    SearchEnv.step()  → name=="search"
      → run(_dispatch_search("capital of France", config))
      → local dense retrieval OR Google Search (config.yaml: search_backend)
      → obs={"obs_str": "Doc 1...", "role": "tool"}, done=False

    apply_chat_template appends:
      <|im_start|>tool
      Doc 1(Title: France) Paris is the capital...
      <|im_end|>

    Qwen3 thinks again (built-in), then submits answer:
      <tool_call>{"name":"answer","arguments":{"response":"Paris"}}</tool_call>

    SearchEnv.step()  → name=="answer"
      → done=True, obs={"obs_str": "", "role": "tool"}

reward.reward_func(sample)
    → strip tool turns → find answer tool call → extract "Paris"
    → EM("Paris", ["Paris"]) → True, search used → reward=1.0
```

---

## 4. Quick Setup Steps

```bash
# 1. Generate data (one-time, Qwen3 format)
python Search-R1/scripts/data_process/qa_search_train_merge_standard.py \
    --local_dir /root/Search-R1/data/nq_hotpotqa_train_standard \
    --data_sources nq,hotpotqa \
    --model_family qwen3

# 2. (If using local backend) start retrieval server
#    See examples/search-r1/README.md → Appendix

# 3. Run training
bash examples/search_r1_tool_call/run_qwen3.sh
```

---

## 5. Interactive Debugging with `debug_generate.py`

`examples/search_r1_tool_call/debug_generate.py` calls the **real** `generate()` function with real SGLang and a real search backend — no mocks. Use it to step through the full turn loop in VSCode.

### Prerequisites

**On your dev machine** (run each in a separate terminal):

```bash
# Terminal 1 — retrieval server (port 8000)
conda activate retriever
save_path=/lc3T/Index
python examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $save_path/e5_Flat.index \
    --corpus_path $save_path/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu

# Terminal 2 — SGLang inference server (port 30000)
# 'python -m sglang.launch_server' runs from the installed package —
# no file path needed; binary also at /usr/local/bin/sglang
python -m sglang.launch_server \
    --model-path /workspace/.cache/huggingface/hub/Qwen3-4B \
    --port 30000 --host 0.0.0.0
```

### Switching between nodes (no file edits needed)

`debug_generate.py` has a `_PROFILES` dict at the top with named node configurations:

| Profile | IP | Use when |
|---|---|---|
| `dev` (default) | `172.20.112.104` | SGLang + search running on dev machine |
| `training` | `172.20.110.172` | SGLang + search running on training node |

**From the terminal:**
```bash
# dev machine (default — TARGET env var not required)
python examples/search_r1_tool_call/debug_generate.py

# training node
TARGET=training python examples/search_r1_tool_call/debug_generate.py
```

**To add a new node:** add an entry to `_PROFILES` in `debug_generate.py` with `sglang_host`, `sglang_port`, `search_url`, and `hf_checkpoint`. That's the only place to maintain IPs.

### VSCode launch config

`.vscode/launch.json` has two ready-made configurations — pick from the Run & Debug dropdown:

- **Debug: generate() — dev node** → targets `172.20.112.104` (sets `TARGET=dev`)
- **Debug: generate() — training node** → targets `172.20.110.172` (sets `TARGET=training`)

`justMyCode: false` lets you step into slime internals (`http_utils.py`, `rollout.py`, etc.).
`PYTHONPATH` includes `examples/search-r1` so `_dispatch_search` can import `local_search_server`.

### Useful breakpoint locations

| File | Where to break |
|---|---|
| `debug_generate.py` | line 203 — just before `generate()` is called |
| `geo3k_vlm_multi_turn/rollout.py` | `generate()`, `_run_inference_step()`, `_process_env_step()`, `_encode_observation_for_generation()` |
| `env_search.py` | `SearchEnv.step()`, `_dispatch_search()` |

### Sample source

By default the script loads row 10 from a parquet file (set `PARQUET_PATH` / `PARQUET_ROW`).
Set `PARQUET_PATH = None` to use the hardcoded `QUESTION` / `GROUND_TRUTH` fallback instead.

---

## 6. Where to Change Things

| Want to change… | Edit this |
|---|---|
| Max search turns | `config.yaml` → `max_turns` |
| Local vs Google search | `config.yaml` → `search_backend` |
| Reward weights | `config.yaml` → `reward_score / reward_format_score / reward_tool_use_score` |
| User prompt (switch to Qwen2.5) | `qa_search_train_merge_standard.py` → `--model_family qwen2.5` (re-run script) |
| Search results format | `env_search.py` → `_passages2string()` |
| Tool call parsing | `env_search.py` → `_extract_tool_call()` |
| Answer extraction | `reward.py` → `_extract_final_answer()` |
| GPUs / parallelism | `run_qwen3.sh` → `PERF_ARGS` / `SGLANG_ARGS` |

---

## 7. Key Files Not in This Directory (But You Depend On)

| File | Role |
|---|---|
| `examples/geo3k_vlm_multi_turn/rollout.py` | Generic multi-turn rollout loop (reused unchanged) |
| `examples/geo3k_vlm_multi_turn/base_env.py` | `BaseInteractionEnv` ABC that `SearchEnv` inherits |
| `examples/search-r1/local_search_server.py` | Local dense retrieval backend (reused via PYTHONPATH) |
| `examples/search-r1/google_search_server.py` | Google search backend (reused via PYTHONPATH) |
| `slime/utils/async_utils.py` | `run()` — the async→sync bridge |
| `Search-R1/scripts/data_process/qa_search_train_merge_standard.py` | Data generation script |

---

## 8. How Tool Results Are Rendered: `apply_chat_template` Deep Dive

### Who does what

**`format_observation()`** (`env_search.py:246`) — your code's job:
```python
def format_observation(self, observation: dict) -> dict:
    return {"role": "tool", "content": observation.get("obs_str", "")}
    #       ^^^^^^^^^^^^^^^^^^^  just a plain Python dict, no formatting done here
```

**`_encode_observation_for_generation()`** (`geo3k_vlm_multi_turn/rollout.py:54`) — where `apply_chat_template` runs:
```python
formatted_prompt = tokenizer.apply_chat_template(
    DUMMY_MESSAGES + [{"role": "tool", "content": "Doc 1(Title: France) Paris is..."}],
    tools=[SEARCH_TOOL, ANSWER_TOOL],
    tokenize=False,
    add_generation_prompt=True,
)
# Qwen3's Jinja2 template renders role="tool" as:
#   <|im_start|>tool
#   Doc 1(Title: France) Paris is...
#   <|im_end|>
#   <|im_start|>assistant
#   <think>
```

`{"role": "tool", "content": "..."}` is not just a data structure — it is a **chat template contract**.
The Qwen3 Jinja2 template (baked into the tokenizer) knows exactly what tokens to emit for role `"tool"`.
Your code only produces the right role/content dict; the tokenizer owns the serialization format.

---

### Full end-to-end example: "What is the capital of France?"

```
━━━ Turn 0: Initial prompt ━━━

apply_chat_template(
  messages=[{"role": "user", "content": "Question: What is the capital of France?"}],
  tools=[SEARCH_TOOL, ANSWER_TOOL],
  enable_thinking=True
)
→ tokenized as:
  <|im_start|>system
  You are a helpful assistant.

  # Tools
  ## search
  {"description": "Search for information...", "parameters": {...}}
  ## answer
  {"description": "Submit your final answer.", "parameters": {...}}
  <|im_end|>
  <|im_start|>user
  Question: What is the capital of France?
  <|im_end|>
  <|im_start|>assistant
  <think>

━━━ Qwen3 generates (built-in thinking + tool call) ━━━

  <think>
  I need to search for the capital of France.
  </think>
  <tool_call>{"name": "search", "arguments": {"query": "capital of France"}}</tool_call>

━━━ env_search.py: SearchEnv.step() ━━━

_extract_tool_call(response_text)
  → TOOL_CALL_RE finds: {"name": "search", "arguments": {"query": "capital of France"}}
  → returns {"name": "search", "arguments": {"query": "capital of France"}}

name == "search"  →  run(_dispatch_search("capital of France", config))
  → local dense retrieval hits local_search_server.py
  → returns list of docs
  → _passages2string([...]) formats as:
      "Doc 1(Title: France) Paris is the capital and most populous city of France...\n
       Doc 2(Title: Capital city) A capital city is the municipality where government...\n"

obs = {"obs_str": "Doc 1(Title: France) Paris is...", "role": "tool"}
done = False  (turn 1 of 2)

━━━ format_observation() ━━━

{"role": "tool", "content": "Doc 1(Title: France) Paris is the capital..."}

━━━ _encode_observation_for_generation() ━━━

# Step 1: measure trim_length (tokens for dummy messages only)
dummy_tokens = tokenizer.apply_chat_template(DUMMY_MESSAGES, tools=TOOLS, tokenize=False)
trim_length = len(encode(dummy_tokens))   # e.g. 47 tokens

# Step 2: encode dummy + new observation turn
full_tokens = tokenizer.apply_chat_template(
  DUMMY_MESSAGES + [{"role": "tool", "content": "Doc 1(Title: France)..."}],
  tools=TOOLS, tokenize=False, add_generation_prompt=True
)
# Qwen3 template emits:
#   <|im_start|>system
#   You are a helpful assistant.
#   ...tool schemas (because tools= is set)...
#   <|im_end|>
#   <|im_start|>user      ← DUMMY_MESSAGES start
#   I am a user.
#   <|im_end|>
#   <|im_start|>assistant ← DUMMY_MESSAGES end
#   <|im_end|>
#   <|im_start|>tool      ← the new observation turn
#   Doc 1(Title: France) Paris is the capital...
#   <|im_end|>
#   <|im_start|>assistant ← generation prompt
#   <think>

# Step 3: trim prefix → keep only observation tokens
obs_token_ids = encode(full_tokens)[trim_length:]
# Result: tokens for exactly:
#   <|im_start|>tool
#   Doc 1(Title: France) Paris is the capital...
#   <|im_end|>
#   <|im_start|>assistant
#   <think>

━━━ Appended to sample.tokens with loss_mask=0 (no gradient through observations) ━━━

━━━ Turn 1: Qwen3 generates again ━━━

  <think>
  The search results confirm Paris is the capital.
  </think>
  <tool_call>{"name": "answer", "arguments": {"response": "Paris"}}</tool_call>

━━━ env_search.py: SearchEnv.step() ━━━

name == "answer"  →  done=True, obs={"obs_str": "", "role": "tool"}
info["final_answer"] = "Paris"

━━━ reward.py ━━━

_TOOL_TURN_RE strips <|im_start|>tool\n...<|im_end|> blocks
_extract_final_answer() finds answer tool call → "Paris"
EM("Paris", ["Paris"]) → True
search was used → reward = 1.0  (reward_score from config.yaml)
```

---

### The trim trick explained (`rollout.py:69–107`)

`apply_chat_template` always re-renders the **entire** conversation from scratch.
You only want the **new observation tokens** to append to `sample.tokens` — not the whole history again.
The trick:
1. Render `DUMMY_MESSAGES` alone → measure prefix token count (`trim_length`)
2. Render `DUMMY_MESSAGES + [observation_message]` → full string
3. Slice `[trim_length:]` → exactly the observation turn tokens

The `tools=` kwarg is passed in both renders so the prefix length is measured under identical conditions (same system prompt with tool schemas).

**Why `loss_mask=0` on observations** (`rollout.py:349`): search result tokens are appended to the sequence so the model sees them as context, but they are excluded from the policy gradient. Only the model's own generated tokens (assistant turns, `loss_mask=1`) are trained on.

---

## 9. The Observation Object — Two Dicts, Two Purposes

There are **two distinct dicts** in the observation pipeline. They look similar but serve different roles.

### Dict 1 — internal observation (`env_search.py:243`)

Created directly by `SearchEnv.step()` after search completes:

```python
obs = {"obs_str": result_str.strip(), "role": "tool"}
return obs, is_final, info
```

- `obs_str` — the text content (plain string, already formatted by `_passages2string()`)
- `role` — metadata carried along so `format_observation()` doesn't need to hard-code the role

This is the **env's internal format**, defined by the `BaseInteractionEnv` contract.
The `obs_str` key name is a naming convention from `BaseInteractionEnv` — it separates "the text content" from "the chat role".

### Dict 2 — chat message (`env_search.py:254`)

Created by `format_observation()`, which does nothing more than rename keys:

```python
def format_observation(self, observation: dict) -> dict:
    return {"role": "tool", "content": observation.get("obs_str", "")}
```

- This is the **HuggingFace chat message format** — the exact structure `apply_chat_template` expects
- `format_observation` renames `obs_str` → `content`; no other transformation happens here
- The tokenizer's Jinja2 template owns the serialization: it sees `role="tool"` and emits `<|im_start|>tool\n...<|im_end|>`

### Full chain

```
_dispatch_search()              → "Doc 1(Title: France) Paris is..."    (plain string)
SearchEnv.step()                → {"obs_str": "Doc 1...", "role":"tool"} (internal obs)
SearchEnv.format_observation()  → {"role": "tool", "content": "Doc 1..."} (chat message)
tokenizer.apply_chat_template   → <|im_start|>tool\nDoc 1...\n<|im_end|>  (tokens)
```

### Why the split?

`BaseInteractionEnv` defines a two-step contract: `step()` computes the raw observation, `format_observation()` adapts it for the rendering backend in use. Different envs could carry extra fields in the internal obs dict (e.g. `{"obs_str": "...", "role": "tool", "score": 0.9}`) without changing how they get serialized — `format_observation()` is the single place that decides what the model actually sees.
