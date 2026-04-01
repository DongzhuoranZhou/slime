# SFT Training Parameters

Reference for the key parameters in `run_geo3k_vlm_sft_zhipu_cluster.sh`.

---

## Data Flow: How `rollout-batch-size` and `global-batch-size` interact

Even in SFT mode, slime runs its standard RL loop:

```
fetch batch from parquet → tokenize + build loss mask → optimizer step → repeat
```

The two batch sizes control different stages:

| Parameter | Controls | Stage |
|-----------|----------|-------|
| `--rollout-batch-size` | How many samples are fetched from the parquet file per loop iteration | Data loading |
| `--global-batch-size` | How many samples per optimizer step (Megatron GBS) | Training |

When both are equal (`128`), each fetch produces exactly one optimizer step.
If `global-batch-size < rollout-batch-size`, slime does multiple optimizer steps per fetch (gradient accumulation).

> Note: `global-batch-size` is counted in **samples** (not prompts). Since SFT uses `n_samples_per_prompt=1`, samples = prompts.

**SFT recommendation:** Set `rollout_batch_size` and `global_batch_size` to the same value and do not configure `n_samples_per_prompt`. This is equivalent to training one batch right after reading one batch (`num-steps-per-rollout` defaults to `1`). The general slime constraint is `(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)`; for SFT with `n_samples_per_prompt=1` and `num-steps-per-rollout=1` this simplifies to `rollout_batch_size = global_batch_size`.
(Reference: [`docs/en/examples/qwen3-4b-base-openhermes.md`](../../docs/en/examples/qwen3-4b-base-openhermes.md))

---

## Dataset Iteration

With 2,100 training samples and `--rollout-batch-size 128`:

```
rollouts_per_epoch  = 2100 // 128 = 16   (used only to estimate total rollouts)
total_rollouts      = 16 × 3000 = 48,000

# No samples are ever dropped — batches that span epoch boundaries
# take the tail of the current epoch + the head of the reshuffled next epoch.
times each sample is seen ≈ 48,000 × 128 / 2100 ≈ 2,926 (~3,000)
```

The dataset is reshuffled at each epoch boundary when `--rollout-shuffle` is set (different sample order each pass, same total coverage).

---

## SFT-Specific Arguments

| Argument | Value | What it does |
|----------|-------|--------------|
| `--rollout-function-path` | `slime.rollout.sft_rollout.generate_rollout` | Replaces the default RL rollout (LLM inference) with a simple tokenize+mask pass — no SGLang inference is triggered |
| `--input-key` | `messages` | Column in the parquet file that contains the `[{role, content}, ...]` list |
| `--apply-chat-template` | flag | Applies the model's chat template to `messages` before tokenization |
| `--rollout-shuffle` | flag | Reshuffles the dataset at each epoch boundary |
| `--num-epoch` | `3000` | Total passes over the dataset; converted internally to `num_rollout = num_epoch × (dataset_size // rollout_batch_size)` |
| `--rollout-batch-size` | `128` | Samples fetched per loop iteration |
| `--global-batch-size` | `128` | Samples per optimizer step |
| `--loss-type` | `sft_loss` | Cross-entropy loss on assistant tokens only (no RL advantage weighting) |
| `--calculate-per-token-loss` | flag | Normalizes loss by number of non-masked tokens instead of by `global_batch_size`. More stable when sequence lengths vary (e.g., images of different sizes) |
| `--disable-compute-advantages-and-returns` | flag | Skips GAE / advantage computation — not needed for SFT since `sample.reward = 0` always |
| `--debug-train-only` | flag | Skips the SGLang rollout engine entirely; the worker pool is never started |

---

## Optimizer Arguments

| Argument | Value | Notes |
|----------|-------|-------|
| `--optimizer` | `adam` | Standard Adam |
| `--lr` | `1e-5` | Peak learning rate |
| `--lr-decay-style` | `cosine` | Cosine annealing from peak to `--min-lr` |
| `--min-lr` | `1e-6` | Floor LR at end of training |
| `--lr-warmup-fraction` | `0.1` | 10% of total steps spent warming up linearly |
| `--weight-decay` | `0.1` | L2 regularization |
| `--adam-beta1` | `0.9` | First moment decay |
| `--adam-beta2` | `0.95` | Second moment decay (0.95 instead of 0.999 — common in LLM training for faster adaptation) |

---

## Parallelism Arguments (Megatron)

| Argument | Value | Notes |
|----------|-------|-------|
| `--tensor-model-parallel-size` | `4` | Split each layer's weight matrices across 4 GPUs |
| `--pipeline-model-parallel-size` | `1` | No pipeline parallelism — all layers on each TP group |
| `--sequence-parallel` | flag | Splits the sequence dimension across TP ranks to reduce activation memory |
| `--context-parallel-size` | `1` | No ring-attention context parallelism |
| `--expert-model-parallel-size` | `1` | MoE expert parallelism (1 = no split; relevant for MoE variants) |
| `--recompute-granularity full` + `--recompute-method uniform` + `--recompute-num-layers 1` | — | Recomputes activations during the backward pass instead of storing them — trades compute for memory |
| `--use-dynamic-batch-size` + `--max-tokens-per-gpu 4096` | — | Packs sequences to fill 4096 tokens per GPU instead of padding to a fixed length. Critical for VLM where image tokens make sequence lengths highly variable |

---

## Multimodal Keys

```bash
MULTIMODAL_KEYS='{"image": "images"}'
```

Maps the parquet column name (`images`) to the modality type (`image`). The rollout engine uses this to route image tensors through the vision encoder rather than the text embedding layer.
