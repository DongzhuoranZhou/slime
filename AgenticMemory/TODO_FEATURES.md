# TODO Features

Tracking features we've scoped but not yet implemented. Each entry records the
question, the answer from analysis, and concrete starting points in the code.

---

## 1. RL for Qwen3-VL-8B via smolagents + slime

**Question:** Can we do RL (e.g. GRPO / PPO) on Qwen3-VL-8B for the agentic
document-QA task, reusing the existing smolagents tools + slime training?

**Short answer:** Yes — all the infrastructure exists. The missing piece is a
smolagents-aware **rollout function** that (a) lets the current student model
drive the agent loop, (b) scores the final answer, and (c) packages the whole
multi-step trajectory into one training example slime can consume.

### Why it's doable

slime is built around three pluggable axes:

| Axis | What slime gives you | What we need to add |
|------|----------------------|---------------------|
| Training | Megatron-LM trainer + GRPO/PPO/SFT losses | Nothing — reuse `--loss-type ppo` / `grpo` |
| Rollout  | SGLang server auto-reloaded with the latest student weights after each step | A `generate_rollout()` function that runs the smolagents loop against that server |
| Reward   | `--custom-rm-path` hook | Wrap the existing `evaluate_response()` (gpt-4o grader) |
| Weight sync | Already handled by `RolloutManager` after every training step | Nothing |

So the full multi-step trajectory runs against the **current** student weights
every iteration, and we train on the tokens the student actually emitted.

### Components: status and pointers

| Component | Status | Where to look |
|-----------|--------|---------------|
| Rollout function | **Needs writing.** Runs `SearchAgent.run(task)` against slime-hosted SGLang, returns one `Sample` per trajectory containing the assistant turns + tool turns + final answer. | Reference: [`slime/rollout/sft_rollout.py`](slime/rollout/sft_rollout.py) and [`slime/rollout/sglang_rollout.py`](slime/rollout/sglang_rollout.py). |
| Reward function | **Mostly done.** `evaluate_response()` already returns a 0/1 score via gpt-4o. Wrap it to match slime's reward-model signature. | [`AgenticMemory/utils.py`](utils.py) `evaluate_response` + `slime/rollout/rm_hub/`. |
| Data format | **Needs design.** Question: is one trajectory one training example, or do we split per-turn? SFT already treats it as one example (see `format_trajectories.py`). For on-policy RL, same-trajectory is usually right because the reward is on the final answer, and GAE handles credit assignment across turns. | [`AgenticMemory/format_trajectories.py`](format_trajectories.py) for the SFT packing convention. |
| Weight sync | **Done.** `RolloutManager` pushes new weights to SGLang after each `--train-backend megatron` step. | [`slime/ray/rollout.py`](slime/ray/rollout.py). |
| Prompt dataset | **Reusable.** Same `longdocurl` / `mmlongdoc` JSONL used for SFT and eval. | `PathManager.get_dataset_jsonl(...)`. |

### Key design question (decide before writing code)

**How to represent a multi-step trajectory as a single training example?**

- **Option A — one flat conversation.** Concatenate the full `messages` list
  (system + user + assistant tool-calls + tool outputs + final assistant turn).
  Mask loss on everything except the assistant's own tokens. Reward applied to
  the whole sequence. This is what the current SFT pipeline already does, and
  it's the smallest delta.
- **Option B — step-level training.** Each assistant step is its own example
  with its own reward. Requires per-step reward shaping or a value function
  that attributes credit — more flexible, more complex.

**Recommendation:** Start with Option A. It matches the SFT data path, shares
the same `--multimodal-keys` plumbing, and the GRPO objective with trajectory-
level reward is well-studied for agentic tasks.

### Starting point

1. Copy [`examples/agentic_doc_rl/`](../examples/agentic_doc_rl/) as the
   skeleton — that example already wires an agent into slime's rollout.
2. Plug in our `SearchAgent` + tools from `AgenticMemory/`.
3. Point `--rollout-function-path` at the new module.
4. Point `--custom-rm-path` at a thin wrapper over `evaluate_response`.
5. Train with `--loss-type grpo --global-batch-size 8` on the 100-question
   overfit set first to confirm the loop closes, then scale up.

### Open risks

- **Reward sparsity.** A 0/1 grader over 15-step trajectories gives a very
  sparse signal. If GRPO stalls, either (a) add shaped sub-rewards (search hit
  the right page?), or (b) warm-start from the SFT checkpoint so the base
  policy is already competent.
- **Throughput.** Each trajectory is ~15 SGLang calls plus image encoding.
  Measure rollout step time on 8 GPUs before committing to long runs.

---

## 2. Configurable search `top_k`

**Question:** `SearchDatabaseTool` currently hardcodes `limit=3`. We want the
number of returned pages to be either (a) fixed by us, or (b) chosen by the
agent per call.

**Both modes should exist** — they answer different questions.

### Where the hardcode lives

[`AgenticMemory/tools.py:122`](tools.py#L122):

```python
search_results = self.client.query_points(
    collection_name=self.collection_name,
    query=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
    using=self.embed_model_name,
    limit=3, # TODO: make this configurable
    ...
)
```

### Mode A — fixed by us (SFT / ablation lever)

Add a constructor argument and surface it to the CLI.

```python
class SearchDatabaseTool(Tool):
    def __init__(self, client, embed_model, collection_name, embed_model_name,
                 top_k: int = 3, max_image_size: int | None = None):
        ...
        self.top_k = top_k

    def forward(self, query, doc_name, excluded_pages=None):
        ...
        search_results = self.client.query_points(
            ...,
            limit=self.top_k,
            ...
        )
```

Then in every call site (`search_agent.py`, `collect_trajectories_async.py`,
`sft/eval_async.py`) take `--top-k` / `--search-top-k` from argparse and pass
it into the tool.

**Use cases:**
- Ablations: does `top_k=5` or `top_k=10` help the baseline student?
- Teacher-specific tuning: maybe the 30B teacher benefits from `top_k=5` while
  the 8B student overloads on context.
- Controlling image-token budget deterministically (important for the 30B VLM
  OOM case we already hit).

### Mode B — chosen by the agent per call (RL action)

Expose `top_k` as a tool input the model fills in:

```python
inputs = {
    "query": {...},
    "doc_name": {...},
    "excluded_pages": {...},
    "top_k": {
        "type": "integer",
        "description": (
            "Number of pages to return. Use small k (1-3) for focused lookups, "
            "larger k (5-10) for broad exploration. Default 3."
        ),
        "nullable": True,
    },
}

def forward(self, query, doc_name, excluded_pages=None, top_k=None):
    k = top_k or self.top_k          # fall back to constructor default
    k = max(1, min(k, 20))            # clamp — agent may emit nonsense
    ...
```

**Use cases:**
- Lets a trained agent adapt retrieval breadth to question difficulty.
- Becomes a new *action dimension* once we move to RL: the policy can learn
  to broaden search when its first pass fails.

### Trade-offs

| Aspect | Mode A (fixed) | Mode B (agent-chosen) |
|--------|----------------|------------------------|
| Reproducibility | High — identical context per run | Lower — agent may pick different k |
| Token budget | Bounded by us | Unbounded unless we clamp |
| Training signal | None (inert hyperparameter) | Gradient can flow through the choice (under RL) |
| Complexity | Minimal | Need clamps + prompt guidance |
| Good for | Ablations, SFT, baseline eval | RL, deployed agent |

### Recommendation

Implement **Mode A first** — it's a 5-line change and immediately unblocks
clean ablations. Then add Mode B as an additive input (backward compatible
via `nullable: True`) once RL is in flight, so the agent can exercise the
new action dimension.

### Testing

Once implemented, the sanity check is a 3-way sweep on 20 samples with the
baseline student: `top_k ∈ {1, 3, 5}`. If accuracy is flat at `top_k=3`
vs. `top_k=5`, retrieval isn't the bottleneck; if it climbs, we've found
cheap wins and the teacher run should be re-collected with the better `k`.
