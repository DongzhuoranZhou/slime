# 03 — Geo3k VLM RL (Single-Turn VLM GRPO)

> **Where this fits:** The first VLM experiment. The task switches from open-domain text QA (Search-R1) to image-based geometry reasoning. The RL algorithm (GRPO) is the same; the new complexity is image tokens, multimodal data loading, and variable sequence lengths. Single-turn: no tool calls, no environment interaction — the model sees an image + question and outputs an answer directly.

**Code:** [`examples/geo3k_vlm/`](https://github.com/DongzhuoranZhou/slime/tree/dev_main/examples/geo3k_vlm)
**Dataset:** [`chenhegu/geo3k_imgurl`](https://huggingface.co/datasets/chenhegu/geo3k_imgurl) on HuggingFace

---

## What It Is

Train a VLM (Qwen3-VL or Qwen2.5-VL) on the [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) dataset with GRPO. Each sample is an image of a geometry problem; the model must output the numerical answer in `\boxed{}` format. Reward = binary 0/1 from the default math RM.

This experiment is primarily about **learning the VLM training setup**: model download, checkpoint conversion, multimodal keys, dynamic batch sizing, parallelism configuration, and reading VLM-specific W&B metrics.

---

## Prerequisites

**cudnn version:** Must be `9.16.0.29` to prevent a severe performance regression in conv3d (torch 2.9 issue):
```bash
pip install nvidia-cudnn-cu12==9.16.0.29
```

**On B200 nodes:** flash-attention 3 is not supported. Add to your run script:
```bash
--sglang-mm-attention-backend sdpa
--attn-implementation flash_attention_2
```

---

## Setup

The run script handles model and dataset download automatically. Just set the environment variables:

```bash
export WANDB_API_KEY=your_key
export SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct   # or 2B, 4B for smaller runs
export SLIME_SCRIPT_NUM_GPUS=8
```

### Dataset

The dataset is downloaded automatically by the script from `chenhegu/geo3k_imgurl`. It has three columns:
- `problem` — geometry problem text
- `answer` — ground truth answer string (e.g., `"270"`)
- `images` — image data (list)

The `--multimodal-keys '{"image": "images"}'` argument tells the rollout engine to route the `images` column through the vision encoder instead of the text embedding layer. **This is the key VLM-specific argument.**

---

## Launch

```bash
# Default: Qwen3-VL-8B-Instruct, 8 GPUs
./examples/geo3k_vlm/run_geo3k_vlm.sh

# Different model size
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Instruct ./examples/geo3k_vlm/run_geo3k_vlm.sh

# External Ray cluster (multi-node)
SLIME_SCRIPT_EXTERNAL_RAY=1 ./examples/geo3k_vlm/run_geo3k_vlm.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLIME_SCRIPT_MODEL_NAME` | `Qwen3-VL-8B-Instruct` | Model to train |
| `SLIME_SCRIPT_DATASET_NAME` | `chenhegu/geo3k_imgurl` | HuggingFace dataset |
| `SLIME_SCRIPT_NUM_GPUS` | `8` | Number of GPUs |
| `SLIME_SCRIPT_EXTERNAL_RAY` | `0` | Set to `1` for external Ray cluster |

### Supported models

| Model | Notes |
|-------|-------|
| Qwen3-VL-2B/4B/8B-Instruct | Recommended starting points |
| Qwen3-VL-30B-A3B-Instruct | MoE; may need extra `provider.*` args in `model_provider.py` |
| Qwen3-VL-235B-A22B-Instruct | Large MoE |
| Qwen3-VL-*-Thinking | Thinking variants |
| Qwen2.5-VL-3B/7B/32B/72B-Instruct | Previous generation |
| Qwen3.5-35B-A3B | See [`run_geo3k_qwen35.sh`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/geo3k_vlm/run_geo3k_qwen35.sh) |

---

## Key Parameters

### Why dynamic batch size matters for VLMs

Images tokenize to different lengths (varying resolution, aspect ratio). With a fixed batch size and padding, most GPU memory is wasted on padding tokens. Dynamic batching packs sequences to fill a token budget per GPU:

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu 4608    # tune based on GPU memory
```

This is critical for training efficiency on VLM tasks.

### Parallelism (Megatron)

| Argument | Value | Notes |
|----------|-------|-------|
| `--tensor-model-parallel-size` | `4` | Split weight matrices across 4 GPUs |
| `--pipeline-model-parallel-size` | `1` | No pipeline parallelism |
| `--sequence-parallel` | flag | Splits sequence dimension across TP ranks → less activation memory |
| `--recompute-granularity full` | + uniform + 1 layer | Recompute activations in backward pass instead of storing them; trades compute for memory |

### GRPO

| Argument | Value | Notes |
|----------|-------|-------|
| `--advantage-estimator` | `grpo` | Group-relative advantage |
| `--n-samples-per-prompt` | `2` | Minimum for GRPO to have within-group variance |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.28` | Asymmetric PPO clipping |
| `--kl-loss-coef` | `0.00` | No explicit KL penalty (rely on clipping alone) |

### Rollout

| Argument | Value | Notes |
|----------|-------|-------|
| `--rollout-max-response-len` | (set in script) | Token budget for generation |
| `--rollout-temperature` | `1.0` | Sampling temperature during training rollouts |
| `--eval-interval` | N | Run eval every N rollout steps |

---

## Reward Model

The default math RM is used (binary 0/1). Three variants were tested — geo3k-specific RM with tolerance=0.05, with tolerance=0.0, and the default math RM. All performed similarly; the default is simpler.

**Important — non-binary reward precision bug:** If you write a custom RM that returns fractional values (e.g., 0.9 for partial credit), floating-point precision under fp32 means that `reward - mean` may not equal zero even when all samples in a group have the same reward. This creates spurious gradient signal.

Fix: switch the reward tensor dtype to `float16` in `slime/ray/rollout.py` (`_post_process_rewards`) to truncate precision artifacts. Or just use binary rewards.

---

## Metrics

**Switch the W&B x-axis from "Step" to `rollout/step`** (see [Overview glossary](./00_overview.md#glossary)).

### Primary

| Metric | What to watch |
|--------|--------------|
| `eval/geo3k_eval` | **Primary.** Mean reward on eval set (greedy decoding, `eval-top-k 1`). Should increase over training. |

### Training health

| Metric | What to watch |
|--------|--------------|
| `rollout/zero_std/count_0.0` | Groups where all responses scored 0. Should decrease. |
| `rollout/zero_std/count_1.0` | Groups where all responses scored 1. Should increase. |
| `rollout/response_len/mean` | Mean response length. Should grow as model learns to reason. |
| `rollout/truncated_ratio` | Fraction hitting token limit. Should be low. |

### KL / gradient health

| Metric | What to watch |
|--------|--------------|
| `train/kl_loss` | KL divergence from reference. Keep below ~0.1. Spike = policy drifting too fast. |
| `train/pg_clipfrac` | Fraction of clipped updates. Consistently >0.5 → LR too high. |
| `train/grad_norm` | Gradient norm. Spikes may indicate instability. |

### Eval diagnostics

| Metric | What to watch |
|--------|--------------|
| `eval/geo3k_eval-truncated_ratio` | Fraction of eval responses hitting token limit. High = model not finishing. |
| `eval/geo3k_eval/response_len/mean` | Mean eval response length. |

### Performance (infra only, not model quality)

| Metric | Meaning |
|--------|---------|
| `perf/rollout_time` | Wall-clock time per rollout step |
| `perf/tokens_per_gpu_per_sec` | Throughput |
| `perf/wait_time_ratio` | Fraction of step time waiting for data; should be low |

---

## Megatron Bridge Note

Slime uses [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) to support multimodal models. Not all Megatron arguments are forwarded automatically. For large MoE models (e.g., Qwen3-VL-30B-A3B), you may need to add extra settings in [`slime/backends/megatron_utils/model_provider.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/slime/backends/megatron_utils/model_provider.py):

```python
provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff
provider.freeze_language_model = False
provider.freeze_vision_model = False
```
