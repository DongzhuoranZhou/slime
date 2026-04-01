# geo3k VLM Multi-Turn — Parameters & Metrics

## Important Parameters

### Multi-turn control

| Parameter | Where set | Default | Description |
|-----------|-----------|---------|-------------|
| `max_turns` | `geo3k_vlm_multi_turn_config.yaml` | `3` | **Maximum number of env interaction turns per rollout.** The loop in `rollout.py:329` iterates `for turn_idx in range(config["max_turns"])`. Each turn = one model generation + one env step. Increase this to allow more reasoning steps; the rollout ends early if the env signals `done=True` before reaching the limit. |
| `rollout_interaction_env_path` | `geo3k_vlm_multi_turn_config.yaml` | `examples.geo3k_vlm_multi_turn.env_geo3k` | Python module path (or `.py` file path) to the interaction environment. Swap this to use a different env without touching rollout code. |

### Token budget (how long each response can be)

Two independent limits control how many tokens can be generated across all turns combined. They are applied as a shared **budget** that decrements across turns (`rollout.py:185-189`):

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `rollout_max_response_len` | `--rollout-max-response-len` | `None` | Per-generation `max_tokens` passed to SGLang. In multi-turn this is the budget for the *entire episode* (all turns share it). In `run_geo3k_vlm_multi_turn.py` this is set to `4096`. |
| `rollout_max_context_len` | `--rollout-max-context-len` | `None` | Hard cap on the total context window (prompt + all turns). Takes priority over `--rollout-max-response-len` when set. Must not exceed `max_position_embeddings` in the model config. |

> **Rule:** if `--rollout-max-context-len` is set, the token budget = `max_context_len - prompt_length`. Otherwise the budget = `max_response_len - prompt_length`. When budget reaches 0 mid-episode the sample is marked `TRUNCATED` and the loop exits.

### Batch sizes and training steps

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `rollout_batch_size` | `--rollout-batch-size` | required | Number of **prompts** per rollout step. Total samples produced = `rollout_batch_size × n_samples_per_prompt`. |
| `n_samples_per_prompt` | `--n-samples-per-prompt` | `1` | How many independent responses to generate per prompt. Set to ≥ 2 for GRPO (needs variance across a group). In the current config: `2`. |
| `global_batch_size` | `--global-batch-size` | auto | Training batch size (in samples, not prompts). Must equal `rollout_batch_size × n_samples_per_prompt ÷ num_steps_per_rollout`. In the current config: `8` (= 4 prompts × 2 samples). |
| `num_rollout` | `--num-rollout` | required | Total number of rollout iterations (= training steps if 1 gradient step per rollout). |

### Sampling / generation quality

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `rollout_temperature` | `--rollout-temperature` | — | Sampling temperature. `1.0` in current config. Lower = more greedy; `0` = fully greedy (use for eval). |
| `rollout_top_p` | `--rollout-top-p` | `1.0` | Nucleus sampling. Usually left at `1.0` for RL training. |
| `rollout_top_k` | `--rollout-top-k` | `-1` | Top-k sampling. `-1` = disabled. |

### Evaluation

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `eval_interval` | `--eval-interval` | — | Run eval every N rollout steps. Set to `2` in current config. |
| `eval_prompt_data` | `--eval-prompt-data` | — | Dataset + slice for eval. Format: `name path@[start:end]`. Current config uses the first 64 samples of the train set. |
| `n_samples_per_eval_prompt` | `--n-samples-per-eval-prompt` | — | Responses per eval prompt. Set to `1` (greedy, via `--eval-top-k 1`). |
| `eval_max_response_len` | `--eval-max-response-len` | — | Token limit for eval generation. Set to `4096` in current config. |

### GRPO loss

| Parameter | CLI flag | Current value | Description |
|-----------|----------|--------------|-------------|
| `eps_clip` | `--eps-clip` | `0.2` | Lower PPO clip bound. |
| `eps_clip_high` | `--eps-clip-high` | `0.28` | Upper PPO clip bound (asymmetric clipping). |
| `kl_loss_coef` | `--kl-loss-coef` | `0.00` | KL penalty coefficient in the loss. `0` = no explicit KL loss. |
| `kl_coef` | `--kl-coef` | `0.00` | KL coefficient for advantage shaping. `0` = pure reward signal. |
| `entropy_coef` | `--entropy-coef` | `0.00` | Entropy bonus. `0` = disabled. |

---

## Where metrics come from

W&B metric namespaces are defined in `slime/utils/wandb_utils.py:150-158`:

```python
wandb.define_metric("rollout/*",    step_metric="rollout/step")
wandb.define_metric("multi_turn/*", step_metric="rollout/step")
wandb.define_metric("passrate/*",   step_metric="rollout/step")
wandb.define_metric("eval/*",       step_metric="eval/step")
wandb.define_metric("perf/*",       step_metric="rollout/step")
wandb.define_metric("train/*",      step_metric="train/step")
```

---

## Rollout metrics (`rollout/*`)

Logged every rollout step in `slime/ray/rollout.py:1163` (`_log_rollout_data`).

| Metric | Description |
|--------|-------------|
| `rollout/response_len/mean` | Mean response token length across all samples in the batch. Should grow as the model learns to reason across turns. |
| `rollout/response_len/max` | Maximum response length. Watch for saturation at `--rollout-max-response-len`. |
| `rollout/response_len/min` | Minimum response length. |
| `rollout/response_len/std` | Standard deviation of response lengths. |
| `rollout/truncated_ratio` | Fraction of samples that hit `--rollout-max-response-len`. High values mean the model is not finishing within the token budget. |
| `rollout/repetition_frac` | Fraction of responses containing repetitive text. Spike signals degenerate generation. |
| `rollout/zero_std/count_1.0` | Groups where all `n-samples-per-prompt` responses scored 1.0 (all correct). Rising trend = model mastering prompts. |
| `rollout/zero_std/count_0.0` | Groups where all responses scored 0.0 (all wrong). Decreasing trend confirms reward signal is working. |
| `rollout/error_cat/{category}` | Fraction of samples per reward category (only if `--log-reward-category` is set). |

### What `zero_std` means

Each group = all `n-samples-per-prompt` responses for the same prompt. `zero_std` tracks groups where every response received the **same reward** (zero variance). These groups contribute **zero gradient** to GRPO because the advantage is 0.

- High `count_0.0` early in training is normal.
- `count_1.0` emerging over time means the model has mastered those prompts.
- The ideal training state is **low zero_std counts overall** — most groups have mixed rewards, giving GRPO maximum learning signal.

---

## Eval metrics (`eval/*`)

Logged at eval intervals (controlled by `--eval-interval`) in `slime/ray/rollout.py:1130` (`_log_eval_rollout_data`).

| Metric | Description |
|--------|-------------|
| `eval/geo3k_eval` | Mean reward across eval samples (greedy decoding). Primary benchmark number — should increase over training. |
| `eval/geo3k_eval-truncated_ratio` | Fraction of eval responses hitting `--eval-max-response-len`. |
| `eval/geo3k_eval/response_len/mean` | Mean eval response length. |
| `eval/geo3k_eval/response_len/max` | Max eval response length. |
| `eval/geo3k_eval-pass@k` | Pass-rate metrics (only if `--log-passrate` is set). |

---

## Training metrics (`train/*`)

Logged per gradient step in `slime/backends/megatron_utils/model.py:651`.

| Metric | Description |
|--------|-------------|
| `train/loss` | Policy gradient loss. |
| `train/kl_loss` | KL divergence from reference model. Should stay small (< 0.1). A spike means the policy is drifting. |
| `train/pg_clipfrac` | Fraction of clipped policy gradient updates. Consistently > 0.5 suggests LR is too high. |
| `train/grad_norm` | Gradient norm. Spikes may indicate instability. |
| `train/lr` | Current learning rate. |

---

## Performance metrics (`perf/*`)

Logged per rollout step in `slime/utils/train_metric_utils.py:48`.

| Metric | Description |
|--------|-------------|
| `perf/rollout_time` | Wall-clock time for the rollout step (seconds). |
| `perf/tokens_per_gpu_per_sec` | Throughput in tokens/GPU/s. |
| `perf/effective_tokens_per_gpu_per_sec` | Same, counting only non-truncated tokens. |
| `perf/actor_train_tflops` | Training TFLOPs. |
| `perf/step_time` | Total step time (rollout + train). |
| `perf/wait_time_ratio` | Fraction of step time spent waiting (should be low). |

---

## Understanding the W&B X-Axis ("Step")

The wandb x-axis labeled **"Step"** is a raw call counter — it increments by 1 for every `wandb.log()` call and has no semantic meaning.

Per rollout iteration, slime makes ~4–5 separate `wandb.log()` calls:

| Call site | Metrics logged |
|-----------|---------------|
| `slime/ray/rollout.py` | `rollout/*`, `eval/*` |
| `slime/utils/train_metric_utils.py` | `perf/*` (timing, throughput) |
| `slime/backends/megatron_utils/data.py` | data processing stats |
| `slime/backends/megatron_utils/model.py` | `train/*` (loss, KL, lr) |

**Consequence:** consecutive data points for the same metric will be spaced ~4–5 wandb steps apart on the raw x-axis.

### Use these as the x-axis instead

Change the wandb x-axis from "Step" to one of:

| Metric | Meaning |
|--------|---------|
| `rollout/step` | Rollout iteration index. |
| `eval/step` | Eval iteration index. |
| `train/step` | Gradient update index. |

---

## Why you only see system metrics

`--use-wandb` is only added to the training args if `WANDB_API_KEY` is set at launch time (see `run_geo3k_vlm_multi_turn.py:52`). If the key is missing, W&B still captures system metrics by default but no training/rollout/eval metrics are logged. Ensure the env var is exported before running:

```bash
export WANDB_API_KEY=your_key_here
```
