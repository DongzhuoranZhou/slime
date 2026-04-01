# WandB Metrics Guide for geo3k VLM SFT

## TL;DR

For SFT, the only metrics that tell you whether training is working are `train/loss` and `train/grad_norm`.

---

## `train/` — Primary signal

| Metric | What to look for |
|---|---|
| `train/loss` | Should decrease monotonically. This is per-token NLL (`-log_prob` over response tokens) because `--calculate-per-token-loss` is set. If it plateaus early or increases, SFT is broken. |
| `train/grad_norm` | Should stay in a stable range (roughly 1–100). Spikes → instability; near-zero → vanishing gradient or zero loss mask. |
| `train/lr-pg_0` | Learning rate schedule. Warmup then cosine decay. Sanity-check only. |

Only `train/loss` and `train/grad_norm` are logged for SFT — no RL-specific keys like `train/ppo_kl` or `train/pg_clipfrac`.

---

## `rollout/` — Partially useful, mostly noise for SFT

### Ignore: `rollout/zero_std/count_0`, `count_1`

These count groups where all rewards are identical. In `sft_rollout.py`, `sample.reward = 0` is hardcoded, so `count_0` always equals the full batch size and `count_1` is always 0. These carry no information for SFT.

### Useful: `rollout/response_len/mean`

Reflects ground-truth response lengths from the training data. If unexpectedly low (near 0), the loss mask is being computed incorrectly and almost nothing is being trained on. Good sanity check for the data pipeline.

### Useful: `rollout/truncated_ratio`

Should be 0 for SFT — ground-truth sequences are not generated, so truncation should not happen.

---

## `perf/` — Infrastructure diagnostics only

`perf/` means **computational performance** (hardware/throughput), not model quality. Irrelevant for evaluating whether SFT is learning.

| Metric | Meaning |
|---|---|
| `perf/actor_train_time` | Wall-clock seconds for the training step |
| `perf/actor_train_tflops` | GPU FLOP/s during training (GPU utilization proxy) |
| `perf/actor_train_tok_per_s` | Tokens trained per second |
| `perf/step_time` | Total time = train + data wait |
| `perf/wait_time_ratio` | Fraction of time the trainer was blocked waiting for data |

Only useful when debugging infrastructure:
- High `wait_time_ratio` → data loading bottleneck
- Low `actor_train_tflops` → underutilized GPUs (poor sequence packing)
- Unstable `step_time` → memory pressure or node issues

`perf/log_probs_tflops` and `perf/ref_log_probs_tflops` are RL-only and will not appear in SFT runs.

---

## `eval/` — N/A

The SFT script does not set `--eval-prompt-data` or `--eval-config`, so no eval metrics are logged.

---

## Summary

| Panel | Relevant for SFT? | Notes |
|---|---|---|
| `train/loss` | **Yes — primary** | Must decrease |
| `train/grad_norm` | **Yes** | Must stay stable |
| `train/lr-pg_0` | Sanity only | |
| `rollout/response_len/mean` | Yes — data pipeline check | |
| `rollout/truncated_ratio` | Yes — should be 0 | |
| `rollout/zero_std/count_*` | **No** | Hardcoded reward=0 in SFT |
| `perf/*` | No (infra only) | Use for debugging slow steps |
| `eval/*` | No | Not configured in SFT script |
