# geo3k VLM SFT Training Results

## General purpose of SFT before RL

In the SFT → RL pipeline, SFT and RL serve fundamentally different roles:

**SFT** teaches the model *what to do* — it supervises the model directly on human-curated examples, shaping output format, response style, and domain-specific behavior. It is sample-efficient and stable but limited by the quality and coverage of labeled data. SFT cannot teach the model to exceed the quality of its training labels.

**RL** teaches the model *how to improve* — it uses a reward signal to push the model beyond the ceiling of the labeled data, discovering strategies (e.g. chain-of-thought reasoning, self-verification) that maximize correctness. RL is powerful but requires a stable and consistent starting point: if the model's output format is erratic, reward signals are noisy and learning is slow.

**Why SFT first:**
- RL exploration is expensive. A randomly-behaving model wastes rollout budget on malformed outputs that get reward=0 regardless of reasoning quality.
- SFT narrows the output distribution to a well-behaved region, so RL starts from a point where reward signal is informative from step 0.
- Format compliance, domain vocabulary, and response structure are cheap to teach via SFT but slow to discover via RL reward.

**What SFT should and should not do before RL:**
- Should: anchor output format, establish response structure, introduce domain-specific conventions
- Should not: over-fit to specific answers or reasoning paths — this would reduce the output diversity that RL needs to find better solutions

### For agentic tasks specifically

> Note: this section is general knowledge and does **not** apply to geo3k. geo3k is single-turn QA — no tool calls, no interaction loop. This is relevant for tasks like search-R1 where the model must call tools and process observations across multiple turns.

Agentic tasks (tool use, search, multi-turn function calling) have a much higher SFT burden than single-turn QA because the model must learn an **interaction protocol**, not just an output format.

**What SFT teaches in agentic settings:**

- **Tool call syntax**: the model must emit structurally valid tool calls (correct JSON schema, correct field names, valid parameter types). A single malformed tool call produces no execution result, breaking the entire episode. RL cannot easily recover from this — it would need to stumble into a valid call by chance.
- **When to call vs when to respond**: the model needs to learn the decision boundary — which situations warrant a tool call and which should be answered directly. This is a behavioral prior that is hard to learn from reward alone.
- **Observation ingestion**: after a tool returns a result, the model must learn to read and incorporate that result into its next action. The format of tool responses (structured JSON, search snippets, code output) is domain-specific and must be seen during SFT.
- **Multi-turn episode structure**: the model must learn the full think → act → observe → think loop. Without SFT, the model may not produce any tool calls at all, making RL reward signal zero across all rollouts with no gradient to learn from.
- **Termination behavior**: knowing when to stop calling tools and emit a final answer is a non-trivial decision the model must learn to make.

**The key failure mode without agentic SFT:** in RL rollouts, the model either never calls tools (zero reward, zero gradient for tool-calling behavior) or calls them malformed (execution errors treated as reward=0). Either way RL cannot bootstrap. SFT must give the model at least a working skeleton of the agentic loop before RL can refine the strategy.

---

## Purpose of SFT in this pipeline

This SFT step is a **format anchor**, not a reasoning teacher.

The assistant response in the training data is constructed as:
```
Answer: \boxed{<value>}
```
(see `prepare_sft_data.py`). The model is trained purely on `p(Answer: \boxed{value} | question)` — no reasoning chain, no chain-of-thought.

**What SFT teaches:** output format compliance — the model learns that geo3k-style questions should always be answered with `\boxed{}` notation.

**What SFT does NOT teach:** correctness, reasoning strategy, or problem-solving.

**Why this matters for RL:** the downstream RL reward parser extracts answers via `\boxed{}`. If the model doesn't produce this format reliably, correct reasoning yields reward=0 due to parse failure. SFT eliminates this noise so that RL reward variance reflects answer correctness only, not format compliance.

---

## Training run: Qwen3-VL-8B-Instruct, 2 epochs

### Key metrics

| Metric | Value | Interpretation |
|---|---|---|
| `train/loss` (final) | ~0.4–0.5 | Per-token NLL; low = model confident on boxed answer tokens |
| `train/grad_norm` (stable) | ~10–30 | Healthy; initial spike to ~200 at step 0 is normal (cold optimizer) |
| `rollout/response_len/mean` | 9 | **Expected** — `"Answer: \boxed{42}"` tokenizes to ~6–10 tokens |
| `rollout/truncated_ratio` | 0 | No truncation; short responses fit easily within context |
| `rollout/repetition_frac` | 0 | No repetition; short format responses are clean |

### Loss curve interpretation

- Loss drops sharply from ~3.5 → ~0.7 within the first ~200–300 steps
- Stabilizes and oscillates around 0.4–0.6 for the remainder of training
- LR decayed from 1e-5 to ~1e-6 (cosine schedule, near `--min-lr`) by step 2099
- **Convergence is fast** — the format pattern `Answer: \boxed{...}` is simple and highly consistent across samples; the model learns it within a fraction of epoch 1

### `rollout/response_len = 9` — expected, not a bug

An earlier analysis flagged this as suspicious. After checking `prepare_sft_data.py`, this is correct:
- The ground truth assistant response is `"Answer: \boxed{<value>}"`, not a full reasoning chain
- ~9 tokens is the correct tokenized length for this format
- High per-token confidence (low NLL) reflects the model has memorized this short, consistent format

---

## Verdict

SFT converged correctly and quickly (within ~300 steps out of 2099 total). The remaining steps reinforce the format without further loss reduction. The run is healthy and the model is ready for the RL phase.

The fast convergence is expected: the output distribution being learned (`Answer: \boxed{X}`) has very low entropy — most of the variation is in `X` itself, which is a short numeric or algebraic value.
