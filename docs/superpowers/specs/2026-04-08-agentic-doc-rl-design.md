# Agentic Document RL Training Design

**Date:** 2026-04-08
**Status:** Approved

## Overview

After SFT-distilling agent trajectories from a 30B teacher (AgenticMemory pipeline), apply RL training with slime to further improve the student VLM (Qwen3-VL-4B/8B) on long-document QA. The model uses tool calls to retrieve document pages from Qdrant (as images) and must produce a correct final answer.

## Approach

New self-contained example `examples/agentic_doc_rl/` following the `search_r1_tool_call` pattern. Reuses `geo3k_vlm_multi_turn/rollout.generate()` unchanged. Only new code: the env, reward, config, and launch script.

## Files

```
examples/agentic_doc_rl/
  __init__.py
  env_doc_search.py     # DocSearchEnv(BaseInteractionEnv)
  reward.py             # pluggable rule_based / llm_judge / both
  config.yaml           # runtime configuration
  run_qwen3_vlm_4B.sh   # launch script
```

## Data Flow

```
slime train.py
  └─ geo3k_vlm_multi_turn/rollout.generate()   [reused unchanged]
       ├─ SGLang: generate tokens → response_text
       └─ DocSearchEnv.step(response_text)
             parse <tool_call> from text
             dispatch → Qdrant query
             return obs: {images: [PIL...], obs_str: "pages X, Y"}
             format_observation() → role:"tool", content:[<image><image>text]
             rollout encodes images, appends to token context (loss_mask=0)
```

## Section 1: DocSearchEnv (`env_doc_search.py`)

### Tool dispatch in `step()`

```
name == "search_database"    → embed query (JinaV4) → Qdrant vector search → page images
name == "get_specific_pages" → Qdrant scroll by page numbers → page images
name == "final_answer"       → done=True, extract answer for reward
no tool call found            → done=True (free-form response, penalised by reward)
unknown tool                 → error string observation, continue
max_turns reached             → done=True
```

### `format_observation()`

Returns a tool-role message with images prepended:
```python
{"role": "tool", "content": "<image><image>Found content on pages 3, 7."}
```
`_encode_observation_for_generation()` in the existing rollout handles image encoding and token appending. Observation tokens get `loss_mask=0`.

### Qdrant client lifecycle

`QdrantClient` and `JinaV4Model` are module-level singletons, initialised once at first `build_env()` call. Avoids reloading the ~1.5GB embedding model per sample during rollout.

### Tool set

Starting with 2 tools:
```python
TOOLS = [SEARCH_DATABASE_TOOL, GET_SPECIFIC_PAGES_TOOL]
```
Adding a tool later requires:
1. Add OpenAI schema dict to `TOOLS`
2. Add `elif name == "new_tool":` branch in `step()`

## Section 2: Reward (`reward.py`)

### Interface

```python
def reward_func(samples, infos, **kwargs) -> list[float]:
    ...
```

### Rule-based (default)

- Extracts `final_answer` argument from the trajectory
- Scores with F1 token overlap against `ground_truth`
- Returns raw F1 float in `[0.0, 1.0]`
- Fast, zero external calls

### LLM judge (optional)

- Reuses `evaluate_response()` from `AgenticMemory/utils.py`
- Calls judge model via OpenAI-compatible endpoint
- Configured via `config.yaml`: `judge_url`, `judge_model`, `judge_api_key`

### `both` mode

- Rule-based score used as the training reward
- LLM judge score logged to wandb as a diagnostic metric
- Lets you compare both signals without paying full judge cost during training
- Switch which one drives training by changing `reward_type` in config

## Section 3: Config (`config.yaml`)

```yaml
# Environment
max_turns: 6                          # shorter than SFT's 15; increase once basic tool use is learned
qdrant_url: "http://localhost:6333"
collection_name: "mmlongdoc"
embed_device: "cuda:0"

# Tools
tools: [search_database, get_specific_pages]

# Reward
reward_type: rule_based               # options: rule_based | llm_judge | both
judge_url: "http://localhost:30000"
judge_model: "Qwen3-VL-30B-A3B-Instruct"
judge_api_key: "EMPTY"
```

## Section 4: Launch script (`run_qwen3_vlm_4B.sh`)

Key arguments:

```bash
# Wiring
--custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
--custom-rm-path                examples.agentic_doc_rl.reward.reward_func
--custom-config-path            examples/agentic_doc_rl/config.yaml

# Multimodal
--multimodal-keys '{"image": "images"}'

# Start from SFT checkpoint
--hf-checkpoint /path/to/sft_checkpoint

# Prompt data (reuse SFT parquet — already has messages + tools columns)
--prompt-data /path/to/mmlongdoc_train.parquet
--input-key messages
--tool-key tools
```

## Prerequisites at runtime

- Qdrant server running on port 6333 (binary or Docker)
- Documents pre-indexed into Qdrant (same index used by AgenticMemory SFT collection)
- SFT checkpoint from `AgenticMemory/run_agentic_sft.sh` as starting point
- (Optional, for llm_judge) SGLang judge model server running

## Design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Rollout reuse | `geo3k_vlm_multi_turn/rollout.generate()` unchanged | Already handles multimodal tool-role obs, image encoding, loss masking |
| Tool call format | Qwen native `<tool_call>` tags | Consistent with how `apply_chat_template` formats tools for Qwen models |
| Qdrant client | Module-level singleton | Avoids reloading 1.5GB JinaV4 model per sample |
| Default reward | F1 token overlap | Better than exact match for phrase/sentence answers in long-doc QA |
| `max_turns` | 6 (vs SFT 15) | Shorter episodes = more RL updates per hour; increase once tool use is stable |
| Starting tools | 2 (search + get_pages) | Simpler RL env first; scaffolding tools (memory, reflection) easy to add later |
