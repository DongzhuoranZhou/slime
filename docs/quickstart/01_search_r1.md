# 01 — Search-R1 (Custom Multi-Turn RL)

> **Where this fits:** The first experiment. Learn how multi-turn RL works in slime using a text-only model that searches Wikipedia to answer open-domain questions. The model emits custom XML-style tags (`<search>`, `<answer>`) — not standard tool calls. See [02 Search-R1 Tool Calling](./02_search_r1_tool_call.md) for the standard-format version.

**Code:** [`examples/search-r1/`](https://github.com/DongzhuoranZhou/slime/tree/dev_main/examples/search-r1)
**Based on:** [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)

---

## What It Is

Search-R1 trains a language model to do **multi-turn retrieval-augmented QA** with RL. Each episode:
1. Model receives a question
2. Model thinks, then emits `<search>query</search>`
3. The environment runs the query against a search backend and returns snippets
4. Model thinks again, then emits `<answer>final answer</answer>`
5. Reward = Exact Match (EM) against ground truth (0 or 1)

Up to `max_turns` search rounds are allowed per episode (default: 2). This is implemented via a **custom generate function** (`--custom-generate-function-path`) that replaces slime's default single-turn rollout.

---

## Setup

### 1. Install slime and dependencies

```bash
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e . --no-deps
pip install chardet   # required for Search-R1
```

### 2. Prepare training data

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1
pip install -e . --no-deps
pip install tensordict

WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

# Train set (NQ + HotpotQA)
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources nq,hotpotqa

# (Optional) Test set
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
```

### 3. Download and convert Qwen2.5-3B

```bash
hf download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B

cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B \
    --save /root/Qwen2.5-3B_torch_dist
```

---

## Search Backend

Configure the backend in [`generate_with_search.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/search-r1/generate_with_search.py) via the `SEARCH_R1_CONFIGS` dict:

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    "search_backend": "local",   # "local" or "google"
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
    },
    "google": {
        "api_key": "your_serper_dev_key",
        "snippet_only": True,
    },
    "return_logprob": True,   # needed for TIS (optional)
    "format_score": 0.2,
}
```

### Option A: Local dense retrieval

Requires ~132 GB of disk space (60-70 GB download). Uses a separate conda env to avoid conflicts with the training env.

```bash
# 1. Create retriever environment
conda create -n retriever python=3.10 -y
conda activate retriever
conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

# 2. Download index and corpus (saves to /lc3T/Index on zhipu cluster)
save_path=/lc3T/Index
python /root/slime/examples/search-r1/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

# 3. Start retrieval server (keep running in background)
conda activate retriever
python /root/slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $save_path/e5_Flat.index \
    --corpus_path $save_path/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

Startup takes 1-2 minutes; uses ~5-7 GB GPU memory. To check if it's running: `lsof -i :8000`.

> **Important:** Deactivate the retriever conda env before running training: `conda deactivate`

### Option B: Google Search (serper.dev)

Set `"search_backend": "google"` and provide an API key from [serper.dev](https://serper.dev). No local server needed.

---

## Launch

```bash
cd /root/slime
bash examples/search-r1/run_qwen2.5_3B.sh
```

The run script wires two custom hooks into slime's training loop:

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
```

These are the only two things needed to implement multi-turn RL in slime: a **custom generate function** (the multi-turn rollout loop) and a **custom reward function** (EM scorer).

### Key GRPO parameters

| Argument | Value | Notes |
|----------|-------|-------|
| `--advantage-estimator` | `grpo` | Group-relative advantage |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.28` | Asymmetric PPO clipping |
| `--kl-loss-coef` | `0.001` | KL penalty; keep small |
| `--kl-loss-type` | `low_var_kl` | Variance-reduced KL estimator |
| `--n-samples-per-prompt` | varies | ≥2 for GRPO to have group variance |
| `--rollout-max-response-len` | 8192 | Token budget across all turns |

---

## Metrics

**Switch the W&B x-axis from "Step" to `rollout/step`** — raw Step is a log-call counter, not a training step index (see [Overview glossary](./00_overview.md#glossary)).

| Metric | What to watch |
|--------|--------------|
| `eval/nq_test` | **Primary.** Mean EM on held-out NQ test set (greedy decoding). Should increase. |
| `rollout/zero_std/count_0.0` | Groups where all samples scored 0. Should decrease as model learns. |
| `rollout/zero_std/count_1.0` | Groups where all samples scored 1. Should increase over time. |
| `rollout/response_len/mean` | Mean response length. Should grow as model learns to do multi-turn search. |
| `train/kl_loss` | KL from reference model. Keep below ~0.1; spike = policy drifting. |
| `train/pg_clipfrac` | Fraction of clipped updates. Consistently >0.5 → LR too high. |
| `eval/nq_test-truncated_ratio` | Fraction of eval responses hitting token limit. High = model not finishing. |

---

## Optional: TIS (Trajectory Importance Sampling)

TIS corrects for the train/inference distribution mismatch when using SGLang with different sampling than training.

**Enable in `generate_with_search.py`:**
```python
"return_logprob": True,   # must be True
```

**Enable in `run_qwen2.5_3B.sh`:**
```bash
GRPO_ARGS=(
   ...
   --use-tis
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

TIS adds compute overhead. When `return_logprob=True`, response postprocessing is automatically disabled to keep token/logp alignment.

---

## How slime Extensions Work

This example uses the two primary extension points in slime:

| CLI flag | File | Purpose |
|----------|------|---------|
| `--custom-generate-function-path` | `generate_with_search.generate` | Replaces the entire rollout loop. Receives a batch of prompts, returns completed trajectories with rewards. |
| `--custom-rm-path` | `generate_with_search.reward_func` | Custom reward function. Called with (prompt, response) pairs; returns scalar rewards. |

All multi-turn, tool-calling, and agentic experiments in slime follow this same two-hook pattern.

---

## Troubleshooting

**Ray process stuck:**
```bash
rm -rf /root/.cache
# if still stuck:
ray stop --force && pkill -9 ray && pkill -9 python
```

**Conda env issues:** Make sure you `conda deactivate` the retriever env before running training. The training script must use the base Python env (or the slime Docker env), not the retriever env.

**Retrieval server not responding:** `lsof -i :8000` to check PID; `nvidia-smi` to check GPU availability.
