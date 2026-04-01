# Search-R1 Training Metrics

## Performance Evaluation

### Primary Metric

| Metric | Description |
|--------|-------------|
| `eval/nq_test` | Mean EM (Exact Match) reward on the held-out test set under greedy decoding. This is the standard benchmark number reported in Search-R1 papers. Should increase steadily over training. |

### Training Signal Health

| Metric | Description |
|--------|-------------|
| `rollout/zero_std/count_1.0` | Number of prompt groups where **all** `n-samples-per-prompt` responses scored 1.0 (all correct). An increasing trend means the model is reliably solving more prompts. |
| `rollout/zero_std/count_0.0` | Number of prompt groups where **all** responses scored 0.0 (all wrong). A decreasing trend confirms the reward signal is working. |

#### What `zero_std` means

Each group = all `n-samples-per-prompt` responses generated for the same prompt.

`zero_std` tracks groups where every response in a group received the **same reward** (zero variance). These groups contribute **zero gradient** to GRPO because the advantage is 0 — the model cannot learn from them.

- High `count_0.0` early in training is normal (model hasn't learned yet).
- `count_1.0` emerging over time means the model has mastered certain prompts.
- The ideal training state is **low zero_std counts overall**, meaning most prompt groups have mixed rewards (some correct, some not), which gives GRPO maximum learning signal.
- `count_0.1` indicates groups where all samples got partial credit (search tool used but answer wrong). May rise temporarily as the model learns to call the search tool before learning to answer correctly.

## Understanding the WandB X-Axis ("Step")

The wandb x-axis labeled **"Step"** is a raw **call counter** — it increments by 1 for every `wandb.log()` call and has no semantic meaning on its own.

Per rollout iteration, slime makes ~4–5 separate `wandb.log()` calls from different subsystems:

| Call site | Metrics logged |
|-----------|---------------|
| `slime/ray/rollout.py` | `rollout/*` (reward, response length, zero_std, etc.) |
| `slime/utils/train_metric_utils.py` | `perf/*` (timing, throughput) |
| `slime/backends/megatron_utils/data.py` | data processing stats |
| `slime/backends/megatron_utils/model.py` | `train/*` (loss, KL, lr) — once per gradient step |

**Consequence:** if you set `--num-rollout 5`, you will see exactly **5 data points** for each metric, but they will be spread across ~17–20 wandb steps on the x-axis (5 rollouts × ~4 log calls each). The gap between consecutive `train/*` points will be ~4–5 wandb steps, not 1.

The slight variation in gap size (e.g., 3 between early points, 5 between later ones) occurs because some logging paths (e.g., perf timers) are silent on the very first rollout when no data has accumulated yet.

### Use these as the x-axis instead

Change the wandb x-axis from "Step" to one of these meaningful metrics:

| Metric | Meaning |
|--------|---------|
| `train/step` | Gradient update index: `rollout_id × steps_per_rollout + step_id` |
| `rollout/step` | Rollout iteration index (equals `rollout_id` by default) |

With the default config (`--rollout-batch-size 8`, `--n-samples-per-prompt 2`, `--global-batch-size 16`), there is 1 gradient step per rollout, so `train/step` and `rollout/step` both go 0 → 1 → 2 → … → `num_rollout - 1`.

---

## Diagnostic Metrics

| Metric | Description |
|--------|-------------|
| `eval/nq_test-truncated_ratio` | Fraction of eval responses hitting `--rollout-max-response-len`. If high, the model is not finishing its answer within the token budget. |
| `rollout/response_len/mean` | Mean response length during training rollouts. Should increase as the model learns to perform multi-turn search (longer = more search turns). |
| `train/kl_loss` | KL divergence from the reference model. Should stay small (< 0.1). A spike indicates the policy is drifting too far. |
| `train/pg_clipfrac` | Fraction of clipped policy gradient updates. If consistently > 0.5, the learning rate may be too high. |
