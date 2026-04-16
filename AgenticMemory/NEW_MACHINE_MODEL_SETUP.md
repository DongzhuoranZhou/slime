# New Machine / Model Setup Guide

End-to-end checklist for setting up the AgenticMemory distillation pipeline on a new GPU machine,
or for swapping in a different teacher / student model.

---

## 0. Prerequisites

| Requirement | Notes |
|-------------|-------|
| GPU machine | Tested on 8× A100 80 GB. 30B teacher needs ≥ 2× A100 (TP=4 recommended). |
| CUDA 12.x | Required by SGLang |
| Python 3.10+ | `python3 --version` |
| Qdrant running | Start before any collection / eval script |
| Dataset indexed | Page images embedded into Qdrant (run once, reuse) |

---

## 1. Environment Setup

### 1.1 Clone and install

```bash
# Clone slime (contains AgenticMemory as submodule)
git clone https://github.com/DongzhuoranZhou/slime.git
cd slime
git submodule update --init AgenticMemory

# Install slime
pip install -r requirements.txt
pip install -e .

# Install AgenticMemory dependencies
cd AgenticMemory
pip install -e ".[dev]"
```

### 1.2 Python environments

Two separate environments are needed:

| Env | Purpose | Key package |
|-----|---------|-------------|
| `.venv` (local) | Jina embeddings + agent scripts | `transformers==4.57.1` (jina-embeddings-v4 remote code) |
| Docker / system Python | slime SFT training | `transformers` already installed |

> **Never mix the two.** The `.venv` transformers version is pinned for jina-embeddings-v4;
> the training stack uses a different version.

```bash
# Create .venv inside AgenticMemory
cd AgenticMemory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # installs transformers==4.57.1 + qdrant-client + smolagents etc.
```

### 1.3 `.env` file

Create `AgenticMemory/.env` (never commit this):

```dotenv
# Qdrant
QDRANT_URL=http://localhost:6333

# Teacher model API (SGLang gateway or GLM API)
API_BASE=http://localhost:30000/v1
API_KEY=EMPTY

# Eval model (GPT-4o via GLM or OpenAI)
EVAL_API_BASE=https://open.bigmodel.cn/api/paas/v4/
EVAL_API_KEY=<your_key>
EVAL_MODEL=gpt-4o-2024-11-20

# Where logs / trajectories / parquets are written.
# Default (if unset): /lc3T/AgenticMemory/logs
AGENTIC_MEMORY_LOG_DIR=/lc3T/AgenticMemory/logs
```

> Throughout the rest of this guide, paths shown as `logs/foo.jsonl` are shorthand
> for `$AGENTIC_MEMORY_LOG_DIR/foo.jsonl`. All scripts read this env var; if you
> unset it they fall back to `/lc3T/AgenticMemory/logs`. On a new machine without
> that mountpoint, just `export AGENTIC_MEMORY_LOG_DIR=/your/path` before running.

---

## 2. Start Qdrant

```bash
# Option A — Docker (recommended)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:latest

# Option B — binary
./qdrant --config-path config/config.yaml   # default port 6333
```

Verify:
```bash
curl http://localhost:6333/collections
```

---

## 3. Index Documents into Qdrant (one-time)

> Skip if the collection already exists — `curl http://localhost:6333/collections` to check.

```bash
cd AgenticMemory
source .venv/bin/activate

# Step 1: parse document pages into images
python scripts/2_parse_doc_info.py --dataset longdocurl

# Step 2: embed and upload to Qdrant
python scripts/3_index_to_qdrant.py \
    --dataset longdocurl \
    --embed-device cuda:0 \
    --collection longdocurl
```

---

## 4. Start Teacher SGLang Server

The teacher generates trajectories. **Use TP that divides the vision encoder's attention heads.**

### Qwen3-VL-30B-A3B (recommended teacher)
The vision encoder has **16 attention heads** → TP must divide 16.
Valid: `--tp 1 2 4 8 16`. **Do NOT use --tp 6 or --tp 3.**

```bash
# Recommended: TP=4 on 4× A100 80 GB
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tp 4 \
    --port 30000 \
    --trust-remote-code \
    --chat-template qwen2-vl \
    --mem-fraction-static 0.85
```

Wait until you see `Server is ready` before running collection.

### GPU OOM note
Correct tool design → agents do 5 searches × 3 images = **15 high-res images** in context.
The 30B vision encoder (~73 GB) OOMs without image resizing.
**Always pass `--max-image-size 768`** to collection and eval scripts.

---

## 5. Collect Trajectories (Teacher)

```bash
cd AgenticMemory
source .venv/bin/activate

python sft/collect_trajectories_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --embed-device cuda:7 \
    --workers 4 \
    --num-samples 100 \
    --max-image-size 768 \
    --output logs/trajectories_teacher_train100.jsonl
```

Sanity check:
```bash
python sft/check_trajectories.py logs/trajectories_teacher_train100.jsonl
```
Expect: ~80%+ score=1.0, 0 schema errors.

---

## 6. Format Trajectories for SFT

```bash
python sft/format_trajectories.py \
    --input  logs/trajectories_teacher_train100.jsonl \
    --output logs/train_trajectories_30B.parquet
```

---

## 7. SFT Training (Student)

### 7B student (recommended)
```bash
cd /path/to/slime   # back to slime root

SLIME_SCRIPT_MODEL_NAME=Qwen2.5-VL-7B-Instruct \
SLIME_SCRIPT_PARQUET=/path/to/AgenticMemory/logs/train_trajectories_30B.parquet \
SLIME_SCRIPT_SAVE_PATH=/lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft \
bash AgenticMemory/sft/run_agentic_sft.sh
```

### 4B student (faster iteration)
```bash
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Instruct \
SLIME_SCRIPT_PARQUET=... \
bash AgenticMemory/sft/run_agentic_sft.sh
```

> Default `MODEL_NAME` in `run_agentic_sft.sh` is `Qwen3-VL-4B-Instruct`.
> Always pass `SLIME_SCRIPT_MODEL_NAME` explicitly to avoid training the wrong size.

---

## 8. Convert Checkpoint for SGLang

After training completes, convert Megatron checkpoint to HuggingFace format:

```bash
# Find the latest iteration — do NOT use the placeholder "iter_XXXXXXX" literally
ls /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft/
# e.g. iter_0000047

python tools/convert_checkpoint.py \
    --model-name Qwen2.5-VL-7B-Instruct \
    --input /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft/iter_0000047 \
    --output /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_hf
```

---

## 9. Eval: Baseline vs SFT

### Step A — Start SGLang with BASE weights
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp 1 --port 30000 --trust-remote-code \
    --chat-template qwen2-vl
```

```bash
cd AgenticMemory && source .venv/bin/activate

python sft/eval_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen2.5-VL-7B-Instruct \
    --embed-device cuda:7 \
    --workers 4 \
    --num-samples 100 \
    --max-image-size 768 \
    --output logs/eval_baseline_qwen25_7b_train100.jsonl
```

### Step B — Kill SGLang, restart with SFT checkpoint
```bash
pkill -f sglang

python -m sglang.launch_server \
    --model-path /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_hf \
    --tp 1 --port 30000 --trust-remote-code \
    --chat-template qwen2-vl
```

```bash
python sft/eval_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen2.5-VL-7B-Instruct \
    --embed-device cuda:7 \
    --workers 4 \
    --num-samples 100 \
    --max-image-size 768 \
    --output logs/eval_sft_qwen25_7b_train100.jsonl
```

### Step C — Held-out eval (questions 100-199)
Repeat Steps A and B with `--offset 100 --num-samples 100` and different output paths:
```
logs/eval_baseline_qwen25_7b_heldout100.jsonl
logs/eval_sft_qwen25_7b_heldout100.jsonl
```

---

## 10. Sanity Checks

```bash
# Check trajectory quality
python sft/check_trajectories.py logs/trajectories_teacher_train100.jsonl

# Check eval output
python sft/check_trajectories.py logs/eval_sft_qwen25_7b_train100.jsonl --example-id 0

# Quick score summary
python -c "
import json
data = [json.loads(l) for l in open('logs/eval_sft_qwen25_7b_train100.jsonl')]
scores = [r['score'] for r in data if r.get('score', -1) != -1]
print(f'{len(scores)} samples, accuracy={sum(scores)/len(scores):.1%}')
"
```

---

## 11. Known Issues / Gotchas

| Issue | Fix |
|-------|-----|
| SGLang `AssertionError: 16 is not divisible by N` | TP must divide 16 for Qwen3-VL-30B. Use `--tp 4`. |
| GPU OOM during collection | Add `--max-image-size 768` to collection/eval commands. |
| `--num-samples` count exceeds requested | Fixed in v4: slice before `filter_existing_results`. |
| Schema hallucination: `original_question` in `reflect_on_search` | Fixed in v4: removed from tool schema entirely. |
| SFT model outputs `Action:\n{JSON}` format | Expected — smolagents text format. Not a collapse. |
| Checkpoint `iter_XXXXXXX` not found | Replace with actual iteration, e.g. `iter_0000047`. |
| `DongzhuoranZhou/AgenticMemory` not found | This fork was deleted; use `ZhuYuqicheng/AgenticMemory`. |

---

## 12. Data Versioning

| Version | Notes |
|---------|-------|
| v1 | 151 trajectories (num-samples bug), original_question in schema |
| v2 | 100 trajectories, original_question still in update_search_memory schema |
| v3 | 23 trajectories (OOM crash at step 24), original_question removed from schema but teacher still passed it |
| v4 | 100 clean trajectories, original_question fully removed, max-image-size 768 |

**Always use v4+ data for training.**
