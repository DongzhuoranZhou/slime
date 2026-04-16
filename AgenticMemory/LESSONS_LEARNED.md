# Lessons Learned — AgenticMemory SFT Pipeline

Chronological record of bugs hit, root causes, and fixes during the first end-to-end
trajectory distillation + SFT run (2026-04-15).

> **Log paths:** Paths written as `logs/...` in this doc are shorthand for
> `$AGENTIC_MEMORY_LOG_DIR/...` (default `/lc3T/AgenticMemory/logs`). Scripts read
> that env var.

---

## 1. SGLang: Qwen3-VL-8B-Instruct produces gibberish output

**Symptom**
```json
"content": "++;\n;\n);\n;\n;\n}",
"finish_reason": "length"
```
Every prompt returns garbage regardless of input.

**Root cause**
Upstream SGLang bug in `Qwen3VLForCausalLM` (dense 8B class) at commit `dce8b06`.
Confirmed by `git stash` test — clean code produces the same output.
The MoE 30B class (`Qwen3MoeVLForCausalLM`) is unaffected.

**Fix**
Use `Qwen2.5-VL-7B-Instruct` as the student model instead.
`Qwen2VLForConditionalGeneration` class is stable at this commit.

**Sanity check before any eval**
```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<MODEL>","messages":[{"role":"user","content":"Say hello."}],"max_tokens":20}' \
  | python3 -m json.tool | grep '"content"'
# Must be non-null English text with finish_reason: stop
```

---

## 2. SGLang cannot load Megatron `.distcp` checkpoints

**Symptom**
```
ValueError: Unrecognized model in <path>. Should have a model_type key in config.json
```

**Root cause**
Slime saves checkpoints in Megatron's `.distcp` distributed format.
SGLang expects HuggingFace format with a valid `config.json`.

**Fix**
Convert with the bridge tool before serving:
```bash
PYTHONPATH=/workspace/Megatron-LM \
python tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir  /lc3T/<MODEL>_agentic_sft/iter_XXXXXXX \
  --output-dir /lc3T/<MODEL>_agentic_sft_hf \
  --origin-hf-dir /workspace/.cache/huggingface/hub/<MODEL> \
  -f
```

**Notes**
- Script is at `tools/convert_torch_dist_to_hf_bridge.py` (slime root), NOT `AgenticMemory/tools/`.
- Use `PYTHONPATH=/workspace/Megatron-LM` (NOT `/root/Megatron-LM` — that is the training node's copy).
- Args: `--input-dir`, `--output-dir`, `--origin-hf-dir`, `-f`.

---

## 3. SGLang fails with `RuntimeError: Unimplemented model type: qwen2_5_vl_text`

**Symptom**
Server starts but crashes on the warmup request:
```
RuntimeError: Unimplemented model type: qwen2_5_vl_text
  File ".../rotary_embedding.py", line 1741, in get_rope_index
```

**Root cause**
When `convert_torch_dist_to_hf_bridge.py` regenerates `config.json` via HuggingFace's
`AutoConfig.save_pretrained()`, it adds `text_config.model_type: qwen2_5_vl_text`.
The original HF model has no `text_config.model_type` field.
SGLang's `get_rope_index` handles `qwen2_5_vl` but not `qwen2_5_vl_text`.

**Fix**
Remove `model_type` from `text_config` after every conversion:
```bash
python3 -c "
import json
path = '/lc3T/<MODEL>_hf/config.json'
with open(path) as f: cfg = json.load(f)
cfg.get('text_config', {}).pop('model_type', None)
with open(path, 'w') as f: json.dump(cfg, f, indent=2)
print('done')
"
```
This makes the converted config identical in structure to the original HF model config.

---

## 4. SFT model outputs `content: null` — complete collapse

**Symptom**
After SFT, every curl returns:
```json
"content": null,
"tool_calls": null,
"finish_reason": "stop"
```
The model outputs EOS as its first token for any input.

**Root cause (loss mask — 0 trainable tokens)**
Qwen2.5-VL's chat template has **no `tool_calls` rendering**.
When `format_trajectories.py` stored tool calls in the structured `tool_calls` field,
`apply_chat_template` silently ignored them. Every assistant turn became:
```
<|im_start|>assistant\n<|im_end|>\n   (5 tokens)
```
The loss mask trained on only the last 2 tokens (`<|im_end|>\n`) per assistant turn.
The model learned in ~5 steps: "after assistant prompt → output EOS immediately".

**Diagnosis**
```
verify_parquet.py --tokenizer-type qwen3   → Masked (trainable) tokens: 0 / 1892
verify_parquet.py --tokenizer-type qwen    → Masked (trainable) tokens: 8 / 1955
                                             (8 = only EOS tokens)
```
Loss curve: hits 0 at step 5 (not gradual — instant collapse).

**Fix in `format_trajectories.py`**
Store tool calls as `Action:` JSON text in `content`, NOT in structured `tool_calls`:
```python
action_text = "Action:\n" + json.dumps({"name": tc["name"], "arguments": arguments})
assistant_msg = {"role": "assistant", "content": action_text}
# No tool_calls field
```
This matches the inference format (smolagents `Action:` format) AND is rendered
correctly by `apply_chat_template` as plain text content.

**Result after fix**
```
Masked (trainable) tokens: 388 / 2335   ← healthy
Loss curve: 0.8 → 0.1 over 100 steps   ← healthy
```

**Key insight**
- Qwen2.5-VL chat template: only renders string/list `content`, ignores `tool_calls`
- Qwen3-VL chat template: natively supports structured `tool_calls`
- If using Qwen2.5-VL as student, always use text-format tool calls in training data

---

## 5. SFT model stuck in infinite loop: `original_question` not in `reflect_on_search` schema

**Symptom**
The SFT model calls `reflect_on_search` with an `original_question` argument every step,
gets the error "Argument original_question is not in the tool's input schema", and loops
until `max_steps` on every question.

**Root cause — schema inconsistency in tool design**
`update_search_memory` had `original_question` as a **required** parameter.
`reflect_on_search` did NOT have `original_question`.
Both tools are semantically "search memory" tools called in sequence.

The teacher model (Qwen3-VL-30B) learned the pattern:
"memory tools → pass `original_question`"
and carried it over from `update_search_memory` to `reflect_on_search`.
This is a **tool schema design flaw**, not a model capability limit —
a 235B model would make the same mistake with this schema.

The SFT student then memorized the teacher's wrong behavior.

**Fix — make `original_question` optional in `update_search_memory`**
```python
# tools.py — SearchMemoryTool.inputs
"original_question": {
    "type": "string",
    "description": "The original user question. Pass on the FIRST call only — omit on subsequent calls.",
    "nullable": True,   # was: required (no nullable flag)
}

# SearchMemoryTool.forward signature
def forward(self, query, pages_visited, relevant_information,
            original_question: Optional[str] = None) -> str:
```
By making it optional, the model no longer treats `original_question` as a mandatory
field for all memory-related tools, eliminating the cross-tool hallucination.

**What does NOT fix this**
- `--min-score 1.0` filtering: the teacher can get the right answer despite the wrong
  tool call (tool errors and moves on). Score does not correlate with schema correctness.
- Adding `original_question` to `ReflectionTool` as ignored parameter: masks the symptom,
  the model still learns the wrong calling pattern.

**Correct data-side fix**
After fixing the tool schema, re-collect trajectories. Optionally also filter existing
trajectories that contain `reflect_on_search` calls with `original_question`:
```python
def has_schema_error(row):
    for step in row.get("steps", []):
        for tc in step.get("tool_calls") or []:
            if tc["name"] == "reflect_on_search":
                args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                if "original_question" in args:
                    return True
    return False
```

---

## 6. Wrong checkpoint selected for eval — model collapse at iter_110

**Symptom**
SFT checkpoint at `iter_0000110` passes the sanity check with `content: null`.
Earlier checkpoints (iter_36, iter_49) also collapse.

**Root cause**
The loss mask bug (Issue 4) was active during this training run.
With 0 trainable tokens, the model had no gradient signal but the optimizer still
modified weights due to weight decay and other regularization. After many steps this
destroyed the model.

**Fix**
Retrain with `format_trajectories.py` fixed (Issue 4 fix). Use `_v2.parquet`.
The new loss curve (0.8 → 0.1 over 110 steps) confirms healthy training.

---

## Summary: correct pipeline order

```
1. Fix tool schemas (optional original_question in update_search_memory)
2. Collect trajectories with 30B teacher  → logs/trajectories_vN.jsonl
3. Format to parquet (Action: text format) → logs/trajectories_vN.parquet
4. Verify: python sft/verify_parquet.py --tokenizer-type qwen
       → Masked tokens should be > 100 per row (not 0 or 8)
5. SFT: run_agentic_sft.sh
       → Loss should decrease gradually (not hit 0 at step 5)
6. Convert checkpoint: PYTHONPATH=/workspace/Megatron-LM python tools/convert_torch_dist_to_hf_bridge.py
7. Fix config.json: remove text_config.model_type
8. Sanity check curl: content must be non-null Action: JSON
9. Run eval_async.py
```

---

## 7. `original_question` in `update_search_memory` causes SFT model to hallucinate on `reflect_on_search`

**Symptom**
After SFT, the model calls `reflect_on_search` with `original_question` every step,
gets "Argument original_question is not in the tool's input schema", and loops until `max_steps`.

**Root cause — parameter in schema teaches wrong generalization**
`original_question` was a parameter of `update_search_memory` (even when marked optional/nullable).
The teacher passed it on **100% of calls** (146/146 in v3, 205/205 in v1).
The student learned: "memory tools → always pass `original_question`"
and generalized this to `reflect_on_search`, which never had the parameter.

Making it `nullable` did not help — the teacher still passed it every time.

**What does NOT fix this**
- Making `original_question` optional/nullable in the schema — teacher still passes it 100%
- Adding `original_question` as ignored parameter to `ReflectionTool` — masks symptom

**Real fix — remove the parameter from the schema entirely**
`original_question` is set directly on `SearchMemoryTool` by the caller at construction time:
```python
search_mem_tool = SearchMemoryTool()
search_mem_tool.original_question = question  # caller sets it, not the model
```
With no `original_question` in the schema, the teacher never passes it,
the student never learns the pattern, and `reflect_on_search` is called correctly.

**Data implication**
Any trajectory JSONL collected while `original_question` was in the schema is stale.
Check with:
```python
# count update_search_memory calls that include original_question
python3 -c "
import json
count, total = 0, 0
with open('logs/trajectories.jsonl') as f:
    for line in f:
        for step in json.loads(line).get('steps', []):
            for tc in step.get('tool_calls') or []:
                if tc['name'] == 'update_search_memory':
                    total += 1
                    args = tc.get('arguments', {})
                    if isinstance(args, str): args = json.loads(args)
                    if 'original_question' in args: count += 1
print(f'original_question present: {count}/{total}')
"
```
If count > 0 → re-collect with fixed schema before training.

**Correct pipeline order (updated)**
```
1. Remove original_question from update_search_memory schema (done in tools.py)
2. Re-collect trajectories → v4.jsonl  (v1/v2/v3 are all stale on this point)
3. Verify: 0/N calls have original_question in update_search_memory
4. Format → parquet, verify trainable tokens, train
```

---

## 8. Teacher accuracy collapses to 14% — VLM never receives page images

**Symptom**
Teacher eval on longdocurl train-set 0–99 drops from the established 67% to 14%.
Trajectories look well-formed: ~7 steps per question, reflection runs, final_answer produced.
But the findings recorded via `update_search_memory` are suspiciously on-topic paraphrases of the
original query, e.g.:

```
Query: "What should be installed in a high-traffic area to control passage and allow plant material to grow?"
Pages: 48, 10, 9
Findings: Page 48 mentions 'Plant material should be allowed to grow in high-traffic areas...'
          Page 10 discusses 'Permeable surfaces allow water and plant material to pass through...'
          Page 9 states 'In high-traffic areas, permeable surfaces are recommended...'
```

Three "different pages" with near-identical phrasing that exactly restates the query =
classic VLM hallucination pattern when images are missing from the prompt.

**Root cause — tool return type not recognized by `process_tool_calls`**
The customized smolagents agent override at
[`src/smolagents/agents.py:1823`](src/smolagents/agents.py) routes tool outputs into
`memory_step.observations_images` **only** when the result's Python type is in
`[AgentImage, AgentImageList, AgentAudio]`:

```python
if tool_call_result_type in [AgentImage, AgentImageList, AgentAudio]:
    if tool_call_result_type == AgentImageList:
        memory_step.observations_images = tool_call_result._raw_list
```

`SearchResults` and `SpecificPagesResults` in `tools.py` were declared as
`class SearchResults(AgentType)` (the plain base class). The `isinstance` check was
therefore **False**, and the PIL images were silently dropped. The VLM saw only the
`to_string()` summary ("Found relevant content on pages 9, 10, 48") and confabulated
the findings. Reflection happily consumed the confabulated text, declared "sufficient
information gathered" after one search, and produced a wrong answer.

**Fix**
`SearchResults` and `SpecificPagesResults` must extend `AgentImageList`, not `AgentType`:

```python
# tools.py
from smolagents.agent_types import AgentImageList

class SearchResults(AgentImageList):
    def __init__(self, value, query, doc_name, pages):
        AgentImageList.__init__(self, value)   # ← sets self._raw_list from PIL list
        self._query = query
        self._doc_name = doc_name
        self._pages = pages

    def to_string(self):
        pages_str = ", ".join(map(str, self._pages))
        return f"Search results for '{self._query}' in '{self._doc_name}': Found relevant content on pages {pages_str}."

class SpecificPagesResults(AgentImageList):
    def __init__(self, value, doc_name, pages):
        AgentImageList.__init__(self, value)
        ...
```

With this change, `process_tool_calls` recognises the tool output as an image list,
reads `._raw_list`, and appends the PIL images to `memory_step.observations_images` —
the VLM's next prompt now contains the real page images.

**What does NOT fix this**
- Adding a `_raw_list` attribute to `SearchResults` while keeping `AgentType` as base —
  the `isinstance` check gates on the class, not the attribute.
- Overriding `to_raw()` to return the image list — `process_tool_calls` does not call
  `to_raw()`; it reads `._raw_list` directly after the `isinstance` check.
- Changing the image resize (`max_image_size`) — images were dropped before resize
  ever mattered.

**Diagnostic signal**
When images are flowing correctly, the custom override prints
`sucessfully loaded N images.` (typo preserved from original source) at
[`src/smolagents/agents.py:1831`](src/smolagents/agents.py) on every
`search_database` / `get_specific_pages` call. If that print line is absent from
logs during an eval, `SearchResults`/`SpecificPagesResults` are not being routed as
`AgentImageList` — the subclass contract is broken.

**Key insight — `isinstance` gates are a type-contract**
smolagents uses **explicit `isinstance` checks** (not duck-typed `hasattr`) to decide
how to render tool outputs. Inheriting from the *wrong* base class fails silently at
runtime — no exception, no warning, just a drop in accuracy. Any new tool that returns
images MUST extend `AgentImage` (single) or `AgentImageList` (multiple), otherwise its
output will be stringified into the text prompt and the images lost.

This is a sibling to §5/§7 (tool schema design) — both are "contract violations that
only surface through eval accuracy, never through syntax or import errors."
