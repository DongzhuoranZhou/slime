# Quickstart Series: Path to Multi-Turn Agentic VLM Training

## What This Series Is

We're building toward **multi-turn agentic VLM training** — training a vision-language model to use tools, search for information, and reason across multiple interaction turns.

Before getting there, we spent time on four foundational experiments to understand how to run RL and SFT, what the key parameters do, and how to read evaluation metrics. This series documents each experiment as a standalone page so you can reproduce it or use it as a reference.

**Repo:** [DongzhuoranZhou/slime](https://github.com/DongzhuoranZhou/slime/tree/dev_main) — branch `dev_main`

---

## Learning Sequence

```
[01] Search-R1 (custom)         — text-only multi-turn RL, custom <search> tags
        ↓
[02] Search-R1 Tool Calling     — same task, standard JSON tool calls (the modification step)
        ↓
[03] Geo3k VLM RL               — step up to VLM, single-turn image QA with GRPO
        ↓
[04] Geo3k VLM SFT              — format-anchor SFT on the same VLM task
```

The sequence is not arbitrary:
- **01 → 02**: Learn how multi-turn RL works in slime, then see what it takes to switch from a custom format to standard tool calls. This is the critical modification for any future agentic VLM work.
- **02 → 03**: Move from text-only to VLM. The core GRPO loop is the same; the new complexity is image tokens, multimodal keys, and variable sequence lengths.
- **03 → 04**: Understand SFT as a prerequisite for RL. The SFT page explains *why* you'd run SFT before RL in an agentic pipeline — which is the actual next step.

---

## Shared Prerequisites

> **Read first:** The official slime quick start is at [`docs/en/get_started/quick_start.md`](../en/get_started/quick_start.md). It covers the full parameter reference, colocated vs. disaggregated training, dynamic sampling, and multi-node setup. This series focuses on the specific experiments; that doc explains the framework in depth.

All experiments run inside the Docker image `slimerl/slime:latest` on a GPU node.

**Download Megatron-LM** (not bundled on the zhipu cluster — must be cloned manually):
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM
```
Megatron is never `pip install`-ed; it is added to `PYTHONPATH` at runtime. The Docker image includes it at `/root/Megatron-LM/`, but on the zhipu cluster clone it to `/workspace/Megatron-LM/`.

**SGLang:** No download needed. It is a Python dependency bundled in the `slimerl/slime:latest` image and installed automatically.

**Install slime:**
```bash
# Fresh environment
git clone https://github.com/DongzhuoranZhou/slime.git --branch dev_main /workspace/slime
cd /workspace/slime
pip install -e . --no-deps
```

> On the zhipu cluster, slime is at `/workspace/src/clean_code_for_rl/slime_0224_2026/slime/`. Run `pip install -e . --no-deps --break-system-packages` to point Python imports to this workspace instead of the image's pre-installed copy.

**W&B setup:**
```bash
# Public W&B
export WANDB_API_KEY=your_key

# Zhipu internal W&B
export WANDB_API_KEY="local-643dfe5b804d35d8ab5aeba347f30f4ee0c6ca6f"
export WANDB_BASE_URL=https://wandb.glm.ai
```

**Proxy (zhipu cluster only — needed for HuggingFace downloads, unset before training):**
```bash
export http_proxy="http://httpproxy.glm.ai:8888"
export https_proxy="http://httpproxy.glm.ai:8888"
# Unset before launching training (breaks inter-node Ray communication)
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
```

**Model conversion (HF → Megatron format):**

All slime training uses Megatron checkpoints. The general pattern is:
```bash
cd /workspace/slime
source scripts/models/<model>.sh           # sets MODEL_ARGS
PYTHONPATH=/workspace/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /workspace/<ModelName> \
    --save /workspace/<ModelName>_torch_dist
```
Each experiment page shows the exact commands for its model.

---

## Glossary

**GRPO** — Group Relative Policy Optimization. The RL algorithm used throughout. For each prompt, it generates `n-samples-per-prompt` responses, scores them all, and uses the relative reward within the group as the advantage signal. Groups where all responses score the same contribute zero gradient (see `zero_std`).

**Rollout** — One iteration of the RL loop: generate responses for a batch of prompts → score them → compute advantages → gradient step. `--num-rollout` sets the total count.

**`zero_std`** — A W&B diagnostic. `rollout/zero_std/count_0.0` = number of prompt groups where every response scored 0 (no learning signal). `count_1.0` = groups where every response scored 1. The ideal state is low counts on both — mixed rewards give GRPO the most gradient.

**W&B x-axis gotcha** — The raw "Step" axis in W&B increments by 1 for every `wandb.log()` call, and slime makes 4–5 separate log calls per rollout iteration from different subsystems. Consecutive points for the same metric are spaced ~4–5 steps apart. **Always switch the x-axis to `rollout/step` or `train/step`** for meaningful plots.

**`--custom-generate-function-path`** — The slime hook for replacing the default single-turn SGLang rollout with a custom multi-turn loop. This is what makes Search-R1 and all agentic experiments work.

**`--custom-rm-path`** — The slime hook for a custom reward function. Used whenever the default math RM doesn't fit the task.

---

## Pages in This Series

| Page | Experiment | Model | Task |
|------|-----------|-------|------|
| [01 Search-R1](./01_search_r1.md) | Multi-turn RL, custom tags | Qwen2.5-3B | NQ + HotpotQA (open-domain QA) |
| [02 Search-R1 Tool Calling](./02_search_r1_tool_call.md) | Multi-turn RL, standard tool calls | Qwen3-4B | NQ + HotpotQA |
| [03 Geo3k VLM RL](./03_geo3k_vlm_rl.md) | Single-turn VLM RL | Qwen3-VL-8B | Geometry3K (image QA) |
| [04 Geo3k VLM SFT](./04_geo3k_vlm_sft.md) | SFT format anchor | Qwen3-VL-8B | Geometry3K |
| [05 W&B Setup](./05_wandb_setup.md) | Zhipu internal W&B setup | — | All experiments |
