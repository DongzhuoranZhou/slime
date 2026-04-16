# Experiment Report — AgenticMemory SFT Pipeline
**Date:** 2026-04-15  
**Goal:** Validate end-to-end trajectory distillation + SFT pipeline for long-document VLM agents.

> **Log paths:** Paths written as `logs/...` in this doc are shorthand for
> `$AGENTIC_MEMORY_LOG_DIR/...` (default `/lc3T/AgenticMemory/logs`). Scripts read
> that env var; override it on a different machine.

---

## Setup

**Task:** Long-document QA on the `longdocurl` dataset.  
Each question requires multi-step retrieval: the agent searches a Qdrant vector DB of document page images, reflects on findings, and produces a final answer.

**Teacher model:** Qwen3-VL-30B-A3B-Instruct (SGLang, `--tp 4`)  
**Student models:** Qwen2.5-VL-7B-Instruct, Qwen3-VL-4B-Instruct  
**Eval questions:** Questions 0–99 from `longdocurl` (same set for training and eval — intentional overfitting test)  
**Grader:** GPT-4o via GLM gateway

---

## Pipeline

```
Qdrant (document page embeddings, JinaV4)
  + SGLang (Qwen3-VL-30B-A3B-Instruct, --tp 4)
    │
    │  collect_trajectories_async.py
    │  --num-samples 100 --workers 2 --max-image-size 768
    │
    ▼
logs/trajectories_overfit100_v4.jsonl      (100 trajs, 67% score=1.0, 0 schema errors)
    │
    │  format_trajectories.py --min-score 0.5
    │
    ▼
logs/trajectories_overfit100_v4.parquet    (66 rows)
    │
    ├── SLIME_SCRIPT_MODEL_NAME=Qwen2.5-VL-7B-Instruct  bash sft/run_agentic_sft.sh
    └── SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Instruct    bash sft/run_agentic_sft.sh
         (3 epochs, lr=1e-5, TP=4, batch_size=4)
    │
    ▼
eval_async.py --num-samples 100 --workers 2 --max-image-size 768
```

---

## Results

### Accuracy on training set (questions 0–99)

| Model | Baseline | SFT (v4) | Delta |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | 14% | **34%** | **+20%** |
| Qwen2.5-VL-7B-Instruct | 11% | **48%** | **+37%** |
| Teacher (Qwen3-VL-30B) | — | 67% | — |

### Avg steps per question

| Model | Baseline | SFT (v4) |
|---|---|---|
| Qwen3-VL-4B | 8.9 | 5.2 |
| Qwen2.5-VL-7B | 4.7 | 5.3 |

### Loss curves

Both models showed healthy convergence: loss 0.9 → ~0.1 over ~50 steps (3 epochs × 66 samples / batch 4).

---

## Key findings

**1. Pipeline is valid.**  
Both student models dramatically improved over baseline after SFT on 66 teacher trajectories. The 7B model reached 48% — more than 4× the baseline — confirming that the data format, loss masking, and SFT setup are all correct.

**2. Larger student benefits more.**  
7B gained +37 points vs 4B's +20 points from the same training data. Consistent with distillation literature: more model capacity → better absorption of teacher behavior.

**3. Step efficiency improved for 4B, not 7B.**  
4B: 8.9 → 5.2 steps (learned to find answers faster).  
7B: 4.7 → 5.3 steps (baseline was already efficient; SFT added slight overhead from more thorough search).

**4. This is an overfitting test, not a generalisation test.**  
Eval questions are identical to training questions. The 48% represents memorisation of teacher trajectories, not generalisation to unseen questions. Generalisation requires 1K+ samples and a held-out eval set.

---

## Bugs fixed during this run

All bugs documented in detail in `LESSONS_LEARNED.md`. Summary:

| # | Bug | Fix |
|---|---|---|
| 1 | Qwen3-VL-8B gibberish output in SGLang | Use Qwen2.5-VL-7B (stable class) |
| 2 | SGLang can't load Megatron `.distcp` checkpoints | Convert via `convert_torch_dist_to_hf_bridge.py` |
| 3 | `RuntimeError: qwen2_5_vl_text` after conversion | Remove `text_config.model_type` from `config.json` |
| 4 | SFT model outputs `content: null` (0 trainable tokens) | Store tool calls as `Action:` text in `content`, not structured `tool_calls` |
| 5 | `reflect_on_search` called with `original_question` → infinite loop | Remove `original_question` from `update_search_memory` schema entirely |
| 6 | Wrong checkpoint selected (iter_110 collapsed) | Use earlier checkpoint; retrain on v4 data |
| 7 | GPU OOM during collection (15 images in context) | Add `--max-image-size 768` to collector |

---

## Data versioning

| File | Trajs | Score=1.0 | Usable | Notes |
|---|---|---|---|---|
| `trajectories_overfit100.jsonl` (v1) | 151 | 80.1% | ✗ | `original_question` in schema → stale |
| `trajectories_overfit100_v2.parquet` | 151 | — | ✗ | Action: format fixed but same stale data |
| `trajectories_overfit100_v3.jsonl` | 100 | 67% | ✗ | OOM-crashed; also stale schema |
| `trajectories_overfit100_v4.jsonl` | 100 | 67% | **✓** | Clean: no `original_question`, `--max-image-size 768` |

---

## Next steps (to validate generalisation)

### Step 1 — Held-out eval on questions 100–199

The 48% SFT accuracy was measured on the **training questions (0–99)**. This could be memorisation of specific answers rather than learning the search-reflect-answer strategy. Evaluate on questions 100–199 (never seen during training) to find out.

**1a — Start SGLang with base 7B weights:**
```bash
pkill -f sglang && sleep 5

no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1" \
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
python -m sglang.launch_server \
  --model-path /workspace/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct \
  --tp 4 --port 30000 --dtype bfloat16 --mem-fraction-static 0.85
```

**1b — Baseline eval on held-out questions 100–199:**
```bash
python sft/eval_async.py \
  --dataset longdocurl --backend sglang \
  --sglang-url http://localhost:30000 \
  --model-name Qwen2.5-VL-7B-Instruct \
  --embed-device cuda:7 --workers 2 \
  --offset 100 --num-samples 100 \
  --max-image-size 768 \
  --output logs/eval_baseline_qwen25_7b_heldout100.jsonl
```

**1c — Kill SGLang, start SFT v4 checkpoint:**
```bash
pkill -f sglang && sleep 5

no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1" \
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
python -m sglang.launch_server \
  --model-path /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_v4_hf \
  --tp 4 --port 30000 --dtype bfloat16 --mem-fraction-static 0.85
```

**1d — SFT eval on held-out questions 100–199:**
```bash
python sft/eval_async.py \
  --dataset longdocurl --backend sglang \
  --sglang-url http://localhost:30000 \
  --model-name Qwen2.5-VL-7B-Instruct \
  --embed-device cuda:7 --workers 2 \
  --offset 100 --num-samples 100 \
  --max-image-size 768 \
  --output logs/eval_sft_qwen25_7b_v4_heldout100.jsonl
```

**1e — Compare:**
```bash
python sft/check_trajectories.py logs/eval_baseline_qwen25_7b_heldout100.jsonl
python sft/check_trajectories.py logs/eval_sft_qwen25_7b_v4_heldout100.jsonl
```

**Interpreting results:**

| Held-out SFT vs Held-out Baseline | Meaning |
|---|---|
| SFT >> Baseline (e.g. +15%+) | Model learned the strategy — generalises |
| SFT ≈ Baseline | Model memorised answers only — scale data to 1K+ |
| SFT < Baseline | Overfitting hurt generalisation — reduce epochs |

---

### Step 2 — Scale up (if held-out delta is positive)

Collect 1K trajectories, train, eval on a fresh held-out set (questions 200–300):

```bash
# Collect 1K training trajectories (questions 0–999)
python sft/collect_trajectories_async.py \
  --dataset longdocurl --backend sglang \
  --sglang-url http://localhost:30000 \
  --model-name Qwen3-VL-30B-A3B-Instruct \
  --embed-device cuda:7 --workers 2 --num-samples 1000 \
  --max-image-size 768 \
  --output logs/trajectories_1k_v1.jsonl

# Format
python sft/format_trajectories.py \
  --input logs/trajectories_1k_v1.jsonl \
  --output logs/trajectories_1k_v1.parquet --min-score 0.5

# Train
SLIME_SCRIPT_MODEL_NAME=Qwen2.5-VL-7B-Instruct \
SLIME_SCRIPT_PARQUET=/workspace/src/clean_code_for_rl/slime_0224_2026/slime/AgenticMemory/logs/trajectories_1k_v1.parquet \
bash sft/run_agentic_sft.sh

# Eval on held-out questions 1000–1099
python sft/eval_async.py \
  --offset 1000 --num-samples 100 \
  --output logs/eval_sft_qwen25_7b_1k_heldout100.jsonl \
  ... (same flags as above)
```
