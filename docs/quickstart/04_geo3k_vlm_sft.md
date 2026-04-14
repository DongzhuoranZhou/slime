# 04 — Geo3k VLM SFT (Format Anchor Before RL)

> **Where this fits:** SFT on the same geo3k VLM task. This page covers two things: (1) the practical steps to run SFT in slime, and (2) *why* you'd run SFT before RL — both for this specific task and for the more general case of agentic/multi-turn tasks we're building toward.

**Code:** [`examples/geo3k_vlm/`](https://github.com/DongzhuoranZhou/slime/tree/dev_main/examples/geo3k_vlm)
- Launch script: [`run_geo3k_vlm_sft.sh`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/geo3k_vlm/run_geo3k_vlm_sft.sh)
- Data prep: [`prepare_sft_data.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/geo3k_vlm/prepare_sft_data.py)

---

## Why SFT Before RL

### For this task (geo3k, single-turn)

This SFT step is a **format anchor**, not a reasoning teacher.

The training data format is:
```
User:      <geometry image> + "question text"
Assistant: Answer: \boxed{270}
```

No reasoning chain. No chain-of-thought. Just the bare answer in `\boxed{}` notation.

**What SFT teaches:** that geo3k-style questions should be answered with `Answer: \boxed{value}`.

**What SFT does NOT teach:** how to solve geometry problems correctly.

**Why this matters for RL:** the downstream RL reward parser extracts answers by looking for `\boxed{}`. If the model doesn't output this format reliably, a *correct* answer still gets reward=0 because the parser fails. SFT eliminates this noise so that RL reward variance reflects answer *correctness*, not format compliance.

### For agentic tasks (what we're building toward)

> This section applies to multi-turn tool-calling tasks — not geo3k itself, but the next step in this series.

In RL rollouts, the model must emit structurally valid tool calls. A single malformed JSON breaks the entire episode and produces reward=0. RL cannot recover from this — it would need to randomly stumble into a valid call with no gradient signal.

**What SFT must teach in agentic settings:**

| What | Why it matters |
|------|---------------|
| Tool call syntax | `<tool_call>{"name":"search","arguments":{...}}</tool_call>` — wrong field name or invalid JSON = no tool execution, broken episode |
| When to call vs when to respond | Decision boundary between calling a tool and answering directly; hard to learn from reward alone |
| Observation ingestion | How to read and use tool responses (search snippets, structured JSON, etc.) |
| Multi-turn episode structure | The full think → call → observe → think loop must be internalized |
| Termination behavior | Knowing when to stop calling tools and emit a final answer |

**The key failure mode without agentic SFT:** the model either never calls tools (reward=0, zero gradient for tool-calling behavior) or calls them malformed (execution error, reward=0). Either way RL gets no gradient to learn from. SFT must give the model at least a working skeleton of the agentic loop before RL can refine the strategy.

**Summary:** SFT teaches *what to do*. RL teaches *how to improve beyond the training data ceiling*. SFT makes the RL reward signal informative from step 0.

---

## Step 1: Prepare SFT Data

The dataset [`chenhegu/geo3k_imgurl`](https://huggingface.co/datasets/chenhegu/geo3k_imgurl) has fields `problem`, `answer`, `images`. For SFT we need a `messages` column with the full conversation and the answer formatted as `\boxed{}`.

Run [`prepare_sft_data.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/geo3k_vlm/prepare_sft_data.py):

```python
from datasets import load_dataset

ds = load_dataset("chenhegu/geo3k_imgurl", split="train")

def process_sample(sample):
    sample["messages"] = [
        {"role": "user",      "content": sample["problem"]},
        {"role": "assistant", "content": f"Answer: \\boxed{{{sample['answer']}}}"}
    ]
    return sample

ds = ds.map(process_sample)
ds.to_parquet("/root/datasets/geo3k_imgurl/train_formatted.parquet")
```

Output: `train_formatted.parquet` with ~2,100 training samples.

The assistant response is intentionally short — `"Answer: \boxed{42}"` tokenizes to **~6–10 tokens**. This is not a bug; it's the design.

---

## Step 2: Launch

```bash
export WANDB_API_KEY=your_key
export SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct   # or 2B, 4B
./examples/geo3k_vlm/run_geo3k_vlm_sft.sh
```

The script downloads the model and dataset automatically. The formatted parquet must already exist at `/root/datasets/geo3k_imgurl/train_formatted.parquet` (from Step 1).

---

## SFT-Specific Arguments

SFT in slime runs through the same RL loop as GRPO, but with these arguments that disable all RL-specific computation:

| Argument | Value | What it does |
|----------|-------|--------------|
| `--rollout-function-path` | `slime.rollout.sft_rollout.generate_rollout` | Replaces LLM inference with tokenize+loss-mask. No SGLang is started. |
| `--loss-type` | `sft_loss` | Cross-entropy on assistant tokens only (no RL advantage weighting) |
| `--calculate-per-token-loss` | flag | Normalizes loss by non-masked token count, not by `global_batch_size`. Critical for VLMs where image tokens make sequence lengths highly variable. |
| `--disable-compute-advantages-and-returns` | flag | Skips GAE / advantage computation (not needed — reward is always 0 in SFT) |
| `--debug-train-only` | flag | Skips the SGLang rollout engine entirely; worker pool is never started |
| `--input-key` | `messages` | Column in the parquet file containing the `[{role, content}, ...]` conversation |
| `--apply-chat-template` | flag | Applies the model's chat template to `messages` before tokenization |

### Multimodal keys

```bash
--multimodal-keys '{"image": "images"}'
```

Same as RL — routes the `images` column through the vision encoder.

---

## Batch Size and Dataset Iteration

SFT uses the same RL loop, so batch size semantics follow the same rules:

```
rollout-batch-size == global-batch-size == 128
```

This means: fetch 128 samples → one optimizer step → repeat. No gradient accumulation.

**Why they must be equal for SFT:** the general constraint is `rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout`. For SFT: `n_samples_per_prompt=1`, `num_steps_per_rollout=1`, so both must be equal.

**Dataset iteration math** (with 2,100 training samples, batch size 128, `--num-epoch 3000`):
```
batches per epoch  = 2100 // 128 = 16
total rollouts     = 16 × 3000 = 48,000 optimizer steps
each sample seen   ≈ 48,000 × 128 / 2100 ≈ 2,926 times
```

The dataset is reshuffled at each epoch boundary when `--rollout-shuffle` is set.

---

## Optimizer

| Argument | Value | Notes |
|----------|-------|-------|
| `--lr` | `1e-5` | Peak learning rate (higher than RL; SFT is more stable) |
| `--lr-decay-style` | `cosine` | Cosine annealing |
| `--min-lr` | `1e-6` | LR floor at end of training |
| `--lr-warmup-fraction` | `0.1` | 10% of steps warming up linearly |
| `--weight-decay` | `0.1` | L2 regularization |
| `--adam-beta2` | `0.95` | Faster adaptation than 0.999; common in LLM training |

---

## Metrics

**Switch the W&B x-axis to `train/step`** (SFT doesn't have meaningful rollout steps).

### Primary (the only metrics that matter for SFT)

| Metric | What to watch |
|--------|--------------|
| `train/loss` | **Primary.** Per-token NLL on assistant tokens. Must decrease. Healthy range: drops from ~3.5 → ~0.7 in first 300 steps, stabilizes at 0.4–0.6. |
| `train/grad_norm` | Should stay in range ~1–100. Spike at step 0 (~200) is normal (cold optimizer). Near-zero = vanishing gradient or broken loss mask. |

### Data pipeline sanity checks

| Metric | Expected value | Notes |
|--------|---------------|-------|
| `rollout/response_len/mean` | **~9** | `"Answer: \boxed{42}"` is ~6–10 tokens. If this is near 0, the loss mask is broken. **This is correct behavior, not a bug.** |
| `rollout/truncated_ratio` | **0** | Ground-truth responses are short; no truncation should happen. |

### Ignore for SFT

| Metric | Why to ignore |
|--------|--------------|
| `rollout/zero_std/count_0` | Always = full batch size (reward is hardcoded to 0 in SFT rollout). No information. |
| `rollout/zero_std/count_1` | Always 0 for same reason. |
| `perf/*` | Infrastructure diagnostics only. Use if debugging slow steps. |
| `eval/*` | Not configured in the SFT script (`--eval-prompt-data` is not set). |
| `train/kl_loss`, `train/pg_clipfrac` | RL-specific metrics; will not appear in SFT runs. |

---

## Expected Training Behavior

**Loss curve:** Drops sharply from ~3.5 → ~0.7 within the first ~300 steps, then stabilizes and oscillates around 0.4–0.6 for the rest of training.

**Why convergence is fast:** The target output distribution (`Answer: \boxed{X}`) has very low entropy. Most of the variation is in `X` itself, which is a short numeric value. The model learns this pattern within a fraction of epoch 1.

**Verdict:** SFT converges quickly and correctly. The remaining ~47,700 steps mostly reinforce the format without further loss reduction. The run is healthy; the model is ready for RL.

---

## Parallelism (same as RL)

```bash
--tensor-model-parallel-size 4
--sequence-parallel
--pipeline-model-parallel-size 1
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1
--use-dynamic-batch-size --max-tokens-per-gpu 4096
```

Dynamic batch size is still important for SFT — image tokens still vary in length even though we're not doing inference.
