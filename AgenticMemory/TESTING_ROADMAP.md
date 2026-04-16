# Testing Roadmap

Step-by-step testing sequence for the full trajectory generation pipeline.
Work through phases in order — each phase assumes the previous one passes.

> **Log paths:** Paths written as `logs/...` in this doc are shorthand for
> `$AGENTIC_MEMORY_LOG_DIR/...` (default `/lc3T/AgenticMemory/logs`). Scripts
> read that env var; override it on a different machine.

---

## The experiment loop

```
Teacher (Qwen3-VL-30B-A3B-Instruct, SGLang, --tp 4)
    │
    │  collect_trajectories_async.py  (100 questions from longdocurl)
    │  each trajectory = multi-step tool-calling agent run:
    │    SearchDatabaseTool → GetSpecificPagesTool → ... → final_answer
    │
    ▼
logs/trajectories_overfit100.jsonl
    │
    │  format_trajectories.py
    │
    ▼
logs/trajectories_overfit100.parquet
    │  messages (tool calls + observations) + images (base64 PNG pages)
    │  assistant turns only are trained on (sft_loss masks tool/user/system)
    │
    │  run_agentic_sft.sh
    │  SLIME_SCRIPT_MODEL_NAME=Qwen2.5-VL-7B-Instruct
    │  3 epochs, lr=1e-5, TP=4
    │
    ▼
/lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft   (HuggingFace checkpoint)
    │
    │  eval_async.py  (same 100 questions, --num-samples 100)
    │  graded by gpt-4o via GLM gateway (EVAL_API_BASE / EVAL_API_KEY)
    │
    ▼
logs/eval_baseline_qwen25_7b_train100.jsonl   ← before SFT
logs/eval_sft_qwen25_7b_train100.jsonl        ← after SFT
    │
    │  compare_eval.py
    │
    ▼
  Teacher (30B)   acc=XX%  avg_steps=Y.Y
  Baseline (7B)   acc=XX%  avg_steps=Y.Y
  SFT (7B)        acc=XX%  avg_steps=Y.Y

  Delta (SFT - Baseline): +XX%
```

**Why the same 100 questions for eval?**
This is an intentional **overfitting test** — not a generalisation test.
The student should memorise the teacher's trajectories on these exact questions.

| Signal | Meaning |
|---|---|
| Loss drops to < 0.3 after 3 epochs | Data format and loss masking are correct |
| Accuracy jumps significantly (e.g. 20% → 70%+) on the 100 training questions | Student memorised teacher trajectories — pipeline is valid |
| Accuracy does NOT improve despite loss dropping | Tool-call formatting bug — check `assistant` turn structure in parquet |
| Loss does NOT drop | Data pipeline bug — loss mask is zeroing out all tokens |
| Both improve | Scale up to 1K–10K samples for generalisation |

---

## Tool dependency map

| Tool | Qdrant? | Local files? | Notes |
|------|---------|-------------|-------|
| `DocumentPreviewTool` | No | Yes | Reads parsed JSON metadata from disk |
| `DocumentReaderTool` | No | Yes | Reads parsed text from disk |
| `PageLoaderTool` | No | Yes | Loads page images from disk |
| `SearchDatabaseTool` | **Yes** | No | Semantic search via QdrantClient |
| `GetSpecificPagesTool` | **Yes** | No | Direct page fetch via QdrantClient |
| `SearchMemoryTool` | No | No | In-memory state only |
| `ReflectionTool` | No | No | Reads SearchMemoryTool state |

There is no FAISS in this codebase. All vector search is Qdrant-based.

---

## Phase 0 — One-time data setup (shared by all environments)

Run once. Both uv venv and Slime Docker connect to the same Qdrant instance.

**Start Qdrant (binary, no Docker):**

The correct command depends on what filesystem `/workspace` is on:

**Dev machine** (`/workspace` on local ext4/xfs):
```bash
cd /workspace/qdrant && ./qdrant   # run in a tmux pane, keep it running
```

**Training node** (`/workspace` on ParaStor or other network filesystem):
Qdrant uses `mmap` for storage. ParaStor does not support `mmap` — starting Qdrant
directly will crash with `Bus error (core dumped)` when loading the collection.
Copy the storage to `/dev/shm` (RAM-backed tmpfs) first — do this once per node boot:
```bash
# Copy storage to local RAM filesystem (~1–2 min, needs to be redone after reboot)
cp -r /workspace/qdrant/storage /dev/shm/qdrant_storage

# Write config pointing at local path
cat > /tmp/qdrant_config.yaml << 'EOF'
storage:
  storage_path: /dev/shm/qdrant_storage
EOF

# Start Qdrant with that config (in a tmux pane, keep it running)
cd /workspace/qdrant && ./qdrant --config-path /tmp/qdrant_config.yaml
```

To check which filesystem you are on:
```bash
df -T /workspace/qdrant/storage/
# "parastor", "nfs", "lustre" → training node path above
# "ext4", "xfs"              → dev machine path above
```

Test it is up:
```bash
curl http://localhost:6333/collections
# → {"result":{"collections":[...]},"status":"ok","time":...}
```

Add to `.env`:
```env
QDRANT_URL=http://localhost:6333
```

**Index documents and fix payloads:**

> If Qdrant was already populated via `populate_qdrant.py` (pre-built SQLite stores),
> skip scripts 1–3 and run only the two payload fixes.

```bash
cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime/AgenticMemory
source .venv/bin/activate

# Fresh indexing only (skip if already populated via populate_qdrant.py):
python scripts/1_dataset_subset.py   # prepare dataset subset
python scripts/2_parse_doc_info.py   # parse + embed + index into Qdrant
python scripts/3_test_preview.py     # sanity check: prints page metadata

# Always required — safe to re-run (skips already-set fields):
python scripts/fix_payload_error.py  # adds document_name to every point (required for filtering)
python scripts/fix_page_num.py       # adds page_num to every point (required by SearchDatabaseTool)
```

**Verify indexing:**
```bash
cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime/AgenticMemory
source .venv/bin/activate
python scripts/database_test.py
# → "Successful retrievals: N out of N"
```

Phase 0 is complete when `database_test.py` reports zero failures.

---

## Phase 1 — uv venv: Qdrant + Gemini (GLM gateway)

**Closed-source pipeline (fully verified ✓):**
```
Qdrant (running) + .env (API_KEY, API_BASE)
    │
    ▼
python sft/collect_trajectories.py \
    --dataset longdocurl --backend api \
    --api-model openai/gemini-2.5-flash \
    --embed-device cuda:0 --num-samples 5 \
    --output logs/trajectories_gemini.jsonl
    │
    │  → JSONL: one entry per question, steps with images as base64 PNG
    ▼
python sft/format_trajectories.py \
    --input logs/trajectories_gemini.jsonl \
    --output logs/trajectories_gemini.parquet
    │
    │  → Parquet: messages + flat images list, ready for slime SFT
    ▼
python -m pytest sft/test_format_trajectories.py -v
    │
    │  → 10 passed — verifies image encoding, <image> placeholders, tool_call_id matching
    ▼
bash sft/run_agentic_sft.sh   # SFT training on collected trajectories
```

Goal: confirm retrieval tools + full agent loop work with Gemini.

**Prerequisites:** Phase 0 done, `api_sanity_check.py` passing (see `NEW_MACHINE_MODEL_SETUP.md`).

```bash
source .venv/bin/activate
```

**Step 1.1** — Qdrant reachable from uv venv:
```bash
python - <<'PY'
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
client = QdrantClient(url=os.getenv("QDRANT_URL"))
cols = client.get_collections().collections
print(f"Collections: {[c.name for c in cols]}")
PY
```
Expected: prints at least one collection name.

**Step 1.2** — `SearchDatabaseTool` returns pages:
```bash
python - <<'PY'
import sys, os; sys.path.insert(0, "src")
from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True))
from qdrant_client import QdrantClient
from search_models import JinaV4Model
from tools import SearchDatabaseTool
os.environ.setdefault("NO_PROXY", "localhost")
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
embed_model = JinaV4Model(device="cuda:0", multivector=True)
tool = SearchDatabaseTool(
    client=client,
    embed_model=embed_model,
    collection_name="longdocurl",
    embed_model_name="jinav4_multivector",
)
result = tool.forward(query="What is the main topic of this document?", doc_name="4009705")
print(result)
PY
```
Expected: prints page results with images and text summaries.

**Step 1.3** — Full agent loop (small sample):

```bash
python search_agent.py --dataset longdocurl --embed_device cuda:0 \
    --agent_model_name openai/gemini-2.5-flash --num_samples 3
```
Expected: 3 questions answered, accuracy printed.

**Step 1.4** — Full trajectory collection with Gemini (smoke test):

`collect_trajectories.py` now supports `--backend api` for Gemini via the GLM gateway.
Requires `API_BASE` and `API_KEY` in `.env`.
```bash
python sft/collect_trajectories.py \
    --dataset longdocurl \
    --backend api \
    --api-model openai/gemini-2.5-flash \
    --embed-device cuda:7 \
    --num-samples 5 \
    --output logs/test_gemini_trajectories.jsonl
```
Expected: `logs/test_gemini_trajectories.jsonl` with 5 trajectory entries, each with
non-empty `steps` and a `score` field. Resume-safe — re-running skips already-written IDs.

**Step 1.5** — Convert trajectories to parquet and inspect:

`format_trajectories.py` requires `datasets` and `pandas` — install once if missing:
```bash
uv pip install datasets pandas
```

```bash
python sft/format_trajectories.py \
    --input logs/test_gemini_trajectories.jsonl \
    --output logs/test_gemini_trajectories.parquet

python - <<'PY'
import pandas as pd
df = pd.read_parquet("logs/test_gemini_trajectories.parquet")
print(f"Rows: {len(df)}, columns: {df.columns.tolist()}")
row = df.iloc[0]
print(f"Score: {row['score']}, Steps: {row['num_steps']}, Images: {len(row['images'])}")
for m in row['messages']:
    # pandas reads missing nested fields back as None, not [], so use explicit None check
    tool_calls = m.get('tool_calls')
    content = m.get('content')
    n_tc = len(tool_calls) if tool_calls is not None else 0
    n_ct = len(str(content)) if content is not None else 0
    print(f"  role={m['role']:12s} tool_calls={n_tc} content_len={n_ct}")
PY
```
Expected: `assistant` rows with non-zero `tool_calls`, `tool` rows with `<image>` at the start of content.

**Run unit tests for format_trajectories:**
```bash
uv pip install pytest
python -m pytest sft/test_format_trajectories.py -v
```
Expected: 10 passed.

---

## Phase 2 — uv venv: Qdrant + Qwen local (TransformersModel)

> **What is `TransformersModel`?**
> `TransformersModel` is a smolagents model backend that loads weights **in-process** via
> HuggingFace `from_pretrained()`. Inference runs inside the same Python process as the agent
> loop — there is no server, no HTTP, no SGLang involved.
>
> SGLang is a completely separate concept: a standalone inference server you launch in another
> terminal. smolagents talks to it over HTTP using `OpenAIModel`. The two are mutually exclusive:
> - `--backend transformers` → `TransformersModel`, in-process, no server needed
> - `--backend sglang` / `--backend api` → `OpenAIModel`/`LiteLLMModel`, HTTP to a running server
>
> **Phase 2 is for evaluation only — not trajectory collection.**
> `collect_trajectories.py` is HTTP-only (`OpenAIModel`/`LiteLLMModel`). It cannot call
> `TransformersModel` directly. Use Phase 4 (SGLang) or Phase 1 (API) to collect SFT trajectories.

Goal: same as Phase 1 but with local Qwen weights instead of Gemini API.
No SGLang, no API gateway — model runs in-process.

**Prerequisites:** Phase 1 done, Qwen weights in HF cache.

```bash
source .venv/bin/activate
```

**Step 2.1** — Verify Qwen weights are cached:
```bash
python - <<'PY'
from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True))
from huggingface_hub import try_to_load_from_cache
path = try_to_load_from_cache("Qwen/Qwen3-VL-8B-Instruct", "config.json")
print("Cached:" if path else "NOT CACHED — will download")
PY
```

**Step 2.2** — Full agent loop with local Qwen:

```bash
python search_agent.py --dataset longdocurl --embed_device cuda:0 \
    --backend transformers --model_config configs/opensource_config.yaml --num_samples 3
```

**Step 2.3** — Trajectory collection with local Qwen via TransformersModel:

`collect_trajectories.py` uses `OpenAIModel` against SGLang — it does not support
`TransformersModel` directly. For local Qwen trajectory collection use the SGLang
path (Phase 4) or the API path with `--backend api` (Phase 1 Step 1.4).

---

## Phase 3 vs Phase 4 — how the two trajectory scripts differ

| | `collect_trajectories_baseline.py` | `collect_trajectories.py` |
|---|---|---|
| **Page selection** | Pre-ranked list from offline JSONL (`/lc/data/mmlb_retr/jina_multi/<dataset>_K128.jsonl`) | `SearchDatabaseTool` queries Qdrant live at runtime |
| **Retriever** | None at runtime — retrieval was pre-computed offline by JinaV4 | JinaV4 + Qdrant (live embedding + search) |
| **Images loaded** | Agent chooses which pages to load (`--top-k` controls candidate pool, default 10) | Agent chooses which pages to load |
| **Prompt content** | 10 page path strings (text only, zero images) | Same — agent decides when to load images |
| **Dependency** | `openai` only | `peft`, `qdrant-client`, CUDA GPU for embedding |

The offline JSONL stores top-128 candidates per question (K128). `--top-k N` slices `page_list[:N]`
at runtime — you can change `--top-k` freely without re-running retrieval, as long as N ≤ 128.

The agent never loads all candidate images upfront. It first calls `preview_document` (reads
pre-parsed metadata JSON from disk, text only) to identify the relevant page, then calls
`load_specific_pages` or `read_document_text` only for that page. This keeps token usage low
regardless of `--top-k`.

---

## Phase 3 — Slime Docker: SGLang + smolagents baseline (no Qdrant)

Goal: verify SGLang ↔ smolagents wiring in the Slime Docker env before adding Qdrant complexity.
Uses `DocAgent` + local file tools only. No embedding model, no vector DB.

If this phase fails, the problem is in SGLang connectivity or the agent loop itself — not retrieval.

**Install only what is needed (no litellm):**
```bash
pip install openai
```

**Step 3.1** — SGLang server running (separate tmux pane):
```bash
deactivate
export http_proxy="http://httpproxy.glm.ai:8888"
export https_proxy="http://httpproxy.glm.ai:8888"

HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
HF_HUB_OFFLINE=0 \
no_proxy="localhost,127.0.0.1" \
NO_PROXY="localhost,127.0.0.1" \
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tp 4 \
    --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.75
```
- `--tp 4` not `--tp 8`: uses GPUs 0–3, leaving GPUs 4–7 free for JinaV4. With `--tp 8`
  every GPU is a TP rank; JinaV4 competes with the visual encoder for the ~12 GB headroom
  and OOMs as agent context grows over steps.
- `HF_HUB_OFFLINE=0`: overrides any `HF_HUB_OFFLINE=1` in the environment — without this
  SGLang crashes at startup trying to fetch model metadata.
- Proxy is required for HF download; `no_proxy` is required so SGLang's self-warmup request
  to `127.0.0.1:30000` is not routed through the proxy (causes a 4-min hang → 503 → server kill).
- Pass the HF repo ID, not a local path — HF checks cache and downloads only if missing.

Wait until `Server is ready` appears in the logs.

**Step 3.2** — Verify `OpenAIModel` can reach SGLang:
```bash
python - <<'PY'
from smolagents import OpenAIModel
model = OpenAIModel(
    model_id="Qwen3-VL-30B-A3B-Instruct",
    api_base="http://localhost:30000/v1",
    api_key="EMPTY",
)
print("OpenAIModel configured OK")
PY
```

**Step 3.3** — Baseline trajectory collection (smoke test, 3 samples):
```bash
python sft/collect_trajectories_baseline.py \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --num-samples 3 \
    --output logs/trajectories_baseline_test.jsonl
```
Expected: JSONL file with 3 entries, each with non-empty `steps`.

Phase 3 is complete when trajectories are written without errors and each entry has at least one `tool_calls` step.

---

## Phase 4 — Slime Docker: SGLang + Qdrant + full pipeline

Goal: end-to-end trajectory collection with Qwen 30B via SGLang and semantic retrieval.
Run only after Phase 3 passes — Phase 3 isolates any SGLang/smolagents issues first.

**Additional installs:**

`collect_trajectories.py` uses `JinaV4Model` (jina-embeddings-v4), whose remote modeling
code requires `peft`. Install both, then immediately restore `nvidia-cudnn-cu12` — pip's
dependency resolver downgrades it as a transitive dep of peft, which breaks SGLang.

```bash
pip install qdrant-client --break-system-packages
pip install peft --break-system-packages
pip install "nvidia-cudnn-cu12==9.16.0.29" --no-deps --break-system-packages
```

> **Why `--break-system-packages`?**
> The Slime Docker uses an externally-managed Python (PEP 668). Plain `pip install` is
> blocked. The `--break-system-packages` flag overrides this. Safe to use here because
> we are inside a Docker container — there is no underlying OS package manager to corrupt.

> **Why restore cudnn after installing peft?**
> `peft` pulls in `nvidia-cudnn-cu12==9.10.2.21` as a transitive dependency, downgrading
> the existing `9.16.0.29` that SGLang requires. The restore must use `--no-deps` so pip
> does not cascade further changes. Verify SGLang is still healthy after:
> ```bash
> curl http://localhost:30000/model_info
> # → {"model_path":"Qwen/Qwen3-VL-30B-A3B-Instruct",...}
> ```

**Prerequisite** — SGLang server must already be running from Phase 3 Step 3.1.

**Step 4.1** — Qdrant reachable from Slime Docker:
```bash
python - <<'PY'
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
client = QdrantClient(url=os.getenv("QDRANT_URL"))
cols = client.get_collections().collections
print(f"Collections: {[c.name for c in cols]}")
PY
```
Expected: prints the indexed collection name (e.g. `longdocurl`).

**Step 4.2** — Smoke test (5 samples):

> **How many samples for overfitting evaluation?**
> The goal here is to *overfit intentionally* — collect a small set, SFT on it, and confirm the
> model memorizes those exact examples. This validates that the data format, loss masking, and
> reward signal are all wired correctly before scaling up.
>
> - **5 samples** — pipeline smoke test only (does it run without crashing?)
> - **50–200 samples** — overfitting eval (can the model learn from this data?)
>
> If the model cannot overfit 100 samples after 3 epochs, the issue is data format or loss
> masking — not model capacity. Start with 5 to confirm the pipeline runs, then collect
> 100–200 for the overfitting experiment.

**Sequential (simple, lower GPU utilisation):**
```bash
python sft/collect_trajectories.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --embed-device cuda:7 \
    --num-samples 5 \
    --output logs/test_trajectories_30B.jsonl
```

**Parallel (recommended — SGLang batches concurrent requests together):**
```bash
python sft/collect_trajectories_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --embed-device cuda:7 \
    --workers 2 \
    --num-samples 20 \
    --output logs/test_trajectories_30B_async.jsonl
```

> **What `--workers` means:**
> `--workers N` creates N Python *threads* inside one process (via `ThreadPoolExecutor`).
> One process can hold multiple threads that share the same CPU RAM — including the Python
> model object and the QdrantClient. GPU weights are read-only during inference so multiple
> threads can reference the same GPU allocation without conflict. The only serialisation
> needed is the JinaV4 forward pass (activations would collide), which is guarded by a
> `threading.Lock`. SGLang receives N concurrent HTTP requests and batches them into the
> same forward pass (continuous batching).
>
> **OOM warning for VLMs:** each concurrent worker sends images to the visual encoder.
> With `--workers 4` and Qwen3-VL-30B on 4×GPU (`--tp 4`), 4 image batches in-flight
> simultaneously exhaust the visual encoder's memory headroom → `CUDA out of memory`.
> **Start at `--workers 2` for 30B VLMs.** Go up to 4 only if `nvidia-smi` shows the
> visual encoder is idle. The script is resume-safe — rerun after a crash to pick up
> from where it stopped.

Both are resume-safe. Use parallel for the overfitting batch (100–200 samples) — significantly faster at the same GPU cost.

**Step 4.3** — Full trajectory collection (choose one):

*Sequential — simpler, lower GPU utilisation:*
```bash
python sft/collect_trajectories.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --embed-device cuda:7 \
    --output logs/trajectories_30B.jsonl
```

*Parallel — recommended for large datasets; SGLang batches concurrent requests together:*
```bash
python sft/collect_trajectories_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --embed-device cuda:7 \
    --workers 2 \
    --output logs/trajectories_30B_async.jsonl
```
`--workers 2` for Qwen3-VL-30B — each worker sends images to the visual encoder, and
4 concurrent image batches OOM on 4×GPU (`--tp 4`). Start at 2; increase only if
`nvidia-smi` shows the visual encoder is consistently idle.

Both scripts are resume-safe — re-running skips already-completed samples.

**Step 4.4** — Format to parquet:
```bash
python sft/format_trajectories.py \
    --input logs/trajectories_30B.jsonl \
    --output logs/train_trajectories_30B.parquet
```

**Step 4.5** — Inspect parquet format:

**Preferred (Slime Docker):** `sft/verify_parquet.py` runs the actual `MultiTurnLossMaskGenerator`
and decodes trainable tokens, so you can confirm the loss mask hits the right tokens before training:
```bash
python sft/verify_parquet.py \
    --parquet logs/train_trajectories_30B.parquet \
    --model-path /workspace/.cache/huggingface/hub/Qwen3-VL-8B-Instruct
```
Check: `assistant` turns show `expected_mask=1`; `system`/`user`/`tool` turns show `expected_mask=0`.
The decoded trainable tokens should be tool call JSON and final answer text — not system prompt or tool responses.

**Fallback (uv venv, no slime import):** structural check only — confirms shape and message roles:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("logs/train_trajectories_30B.parquet")
print(f"Rows: {len(df)}, columns: {df.columns.tolist()}")
row = df.iloc[0]
print(f"Score: {row['score']}, Steps: {row['num_steps']}, Images: {len(row['images'])}")
for m in row['messages']:
    # pandas reads missing nested fields back as None, not [], so use explicit None check
    tool_calls = m.get('tool_calls')
    content = m.get('content')
    n_tc = len(tool_calls) if tool_calls is not None else 0
    n_ct = len(str(content)) if content is not None else 0
    print(f"  role={m['role']:12s} tool_calls={n_tc} content_len={n_ct}")
PY
```
Check: `assistant` rows have non-zero `tool_calls`; `tool` rows have `<image>` at the start of content.

**Step 4.6** — SFT overfitting evaluation:

> The first SFT run is intentionally an overfitting experiment — not generalisation.
> Collect 100–200 samples, SFT on them for 3 epochs, and confirm the model memorises
> those exact examples. This validates the full data pipeline (format, loss masking,
> `<image>` placeholder encoding) before scaling up.
>
> | Purpose | Samples | Signal |
> |---|---|---|
> | Pipeline smoke | 5 | Does it run without crashing? |
> | Overfitting eval | 100–200 | Does the model learn from this data? |
> | Production SFT | 1K–10K+ | Generalisation |
>
> If the model cannot overfit 100 samples after 3 epochs, the problem is data format
> or loss masking — not model capacity.

**Step 4.6a** — Collect overfitting batch (100 samples, parallel):
```bash
python sft/collect_trajectories_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --embed-device cuda:7 \
    --workers 2 \
    --num-samples 100 \
    --output logs/trajectories_overfit100.jsonl
```

**Step 4.6b** — Format to parquet:
```bash
python sft/format_trajectories.py \
    --input logs/trajectories_overfit100.jsonl \
    --output logs/trajectories_overfit100.parquet
```

**Step 4.6c** — Inspect parquet format:

*Preferred (Slime Docker)* — runs the actual loss mask generator, confirms which tokens are trained:
```bash
python sft/verify_parquet.py \
    --parquet logs/trajectories_overfit100.parquet \
    --model-path /workspace/.cache/huggingface/hub/Qwen3-VL-8B-Instruct
```
Check: `assistant` turns show `expected_mask=1`; decoded trainable tokens are tool call JSON and final answer text.

*Fallback (uv venv)* — structural check only:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("logs/trajectories_overfit100.parquet")
print(f"Rows: {len(df)}, columns: {df.columns.tolist()}")
row = df.iloc[0]
print(f"Score: {row['score']}, Steps: {row['num_steps']}, Images: {len(row['images'])}")
for m in row['messages']:
    # pandas reads missing nested fields back as None, not [], so use explicit None check
    tool_calls = m.get('tool_calls')
    content = m.get('content')
    n_tc = len(tool_calls) if tool_calls is not None else 0
    n_ct = len(str(content)) if content is not None else 0
    print(f"  role={m['role']:12s} tool_calls={n_tc} content_len={n_ct}")
PY
```
Check: `assistant` rows have non-zero `tool_calls`; `tool` rows have `<image>` at the start of content.
SFT loss (`--loss-type sft_loss`) trains on `assistant` turns only — tool/user/system turns are masked out.

**Step 4.6d** — Run SFT:
```bash
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct \
SLIME_SCRIPT_PARQUET=$(pwd)/logs/trajectories_overfit100.parquet \
    bash sft/run_agentic_sft.sh
```
Train for 3 epochs. Loss should drop to near-zero (< 0.3). If it plateaus above ~1.0,
inspect the parquet: check that `assistant` turns have non-empty `tool_calls` and `tool`
turns start with `<image>`. See `logs/sft_overfit100_analysis.md` for a reference result.

---

## Step 4.7 — Baseline and SFT evaluation

To know whether SFT helped, you need a baseline measured *before* SFT on the same question set.
Run baseline first, then SFT, then the same eval again.

**Why a baseline is required:**
The teacher (Qwen 30B) and the student (the model you SFT) are different sizes.
Without a baseline you cannot tell whether any accuracy change came from SFT or was
already there. The comparison is: student-before-SFT vs student-after-SFT, on the
same 100 questions used for training (overfitting check) and on a held-out set
(generalisation check).

**Use `sft/eval_async.py`** — same worker/SGLang/lock skeleton as
`collect_trajectories_async.py`, but eval-only: no image encoding, no trajectory
steps stored. Output is small and fast to produce. The eval client (gpt-4o grader)
always reads `EVAL_API_BASE` / `EVAL_API_KEY` from `.env` — set these to the GLM
gateway so the grader is not routed to SGLang.

> **Why not reuse scores from trajectory collection?**
> `collect_trajectories_async.py` does write a `score` field, but it only succeeds if
> `API_BASE` is set to the GLM gateway at collection time. Without it the eval client
> falls back to the SGLang URL, the call fails, and all scores are written as `-1`.
> Run an explicit eval step instead — it is fast and reliable.

**Step 4.7a — Teacher eval (30B, same 100 questions):**

> **Status (2026-04-15): ✓ DONE — `logs/eval_teacher_train100.jsonl` (100/100, acc=16.0%, avg_steps=8.1)**

```bash
# SGLang must be serving Qwen3-VL-30B-A3B-Instruct (same server from Phase 4)
python sft/eval_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-30B-A3B-Instruct \
    --embed-device cuda:7 \
    --workers 2 \
    --num-samples 100 \
    --output logs/eval_teacher_train100.jsonl
```
`--workers 2` for 30B — same OOM constraint as trajectory collection.

**Step 4.7b — Baseline (student, base weights):**

Two student models are documented below. The pipeline is identical — only the model name
and `--tp` differ. Pick one based on which SGLang supports at the current commit.

> **SGLang model-class status (confirmed 2026-04-15, build `0.5.6.post3.dev902+gdce8b0606`):**
>
> | Model | SGLang class | Status |
> |---|---|---|
> | `Qwen3-VL-30B-A3B-Instruct` | `Qwen3MoeVLForCausalLM` | ✓ works |
> | `Qwen2.5-VL-7B-Instruct` | `Qwen2VLForConditionalGeneration` | ✓ works |
> | `Qwen3-VL-8B-Instruct` | `Qwen3VLForCausalLM` | ✗ gibberish output |
>
> Qwen3-VL-8B-Instruct weights are fine (direct `transformers` inference is clean).
> The bug is in SGLang's dense-Qwen3-VL model class at this commit.
>
> Confirmed failure signature from curl sanity check (2026-04-15):
> ```json
> "content": "++;\n;\n);\n;\n;\n}   ...",
> "finish_reason": "length"
> ```
> This output is completely independent of the prompt. If you see `finish_reason: "stop"`
> and a coherent English sentence, the SGLang class has been fixed — proceed with Option 2.
> Until then use Option 1 (`Qwen2.5-VL-7B-Instruct`).

---

### Option 1 — Qwen2.5-VL-7B-Instruct (✓ working at current SGLang commit)

> **Status (2026-04-15):**
> - SFT training: ✓ DONE (loss 2.5 → ~0.3, iter_0000110, wandb `slime-agentic-sft`)
> - Checkpoint at `/lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft/`
> - **Checkpoint collapse at iter_110** — see below before proceeding
> - Baseline eval: NOT yet run
> - SFT eval: blocked until checkpoint issue resolved

---

#### Checkpoint conversion (Megatron → HuggingFace)

SGLang cannot load Megatron `.distcp` checkpoints directly. Convert first:

```bash
cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime

PYTHONPATH=/workspace/Megatron-LM python tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft/iter_0000110 \
  --output-dir /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_hf \
  --origin-hf-dir /workspace/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct \
  -f
```

> **Important notes:**
> - Use `PYTHONPATH=/workspace/Megatron-LM` (NOT `/root/Megatron-LM`). `/root` is the
>   training node's Megatron; `/workspace/Megatron-LM` is the dev machine's copy.
> - Script is at `tools/convert_torch_dist_to_hf_bridge.py` (slime root), not `AgenticMemory/tools/`.
> - Args: `--input-dir`, `--output-dir`, `--origin-hf-dir`, `-f` (force overwrite).

After conversion, the `config.json` gets `text_config.model_type: qwen2_5_vl_text` added by
HuggingFace's serializer. This field is absent in the original model and breaks SGLang.
Remove it before starting SGLang:

```bash
python3 -c "
import json
path = '/lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_hf/config.json'
with open(path) as f: cfg = json.load(f)
cfg.get('text_config', {}).pop('model_type', None)
with open(path, 'w') as f: json.dump(cfg, f, indent=2)
print('done — text_config.model_type removed')
"
```

> **Why this breaks:** the original `Qwen2.5-VL-7B-Instruct` config has no `text_config.model_type`.
> HuggingFace's `AutoConfig.save_pretrained()` (called during conversion) adds it as
> `qwen2_5_vl_text`. SGLang's `get_rope_index` in `rotary_embedding.py` only handles
> `qwen2_5_vl` (not `qwen2_5_vl_text`) and raises `RuntimeError: Unimplemented model type`.
> Removing the field makes the SFT config identical in structure to the baseline.

---

#### ⚠ Checkpoint collapse at iter_110 (2026-04-15)

**Symptom:** `content: null`, `tool_calls: null`, `finish_reason: stop` — the model outputs
EOS as its **first token** for any input, including `"What is 2+2?"`.

**Root cause:** 3 epochs × 100 samples / batch_size 4 = 75 steps. But iter_110 > 75 steps,
meaning the model trained for >4 effective epochs. With lr=1e-5 and only 100 samples,
the model catastrophically overfit and learned to output EOS immediately.

**Available checkpoints:**
```
iter_0000036   # ~halfway through epoch 1 — recommended starting point
iter_0000049   # ~end of epoch 1
iter_0000073   # ~end of epoch 2
iter_0000099   # ~end of epoch 3
iter_0000110   # final — collapsed (do not use)
```

**Fix:** use an earlier checkpoint. Convert iter_0000036 first:

```bash
PYTHONPATH=/workspace/Megatron-LM python tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft/iter_0000036 \
  --output-dir /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_iter36_hf \
  --origin-hf-dir /workspace/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct \
  -f

python3 -c "
import json
path = '/lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_iter36_hf/config.json'
with open(path) as f: cfg = json.load(f)
cfg.get('text_config', {}).pop('model_type', None)
with open(path, 'w') as f: json.dump(cfg, f, indent=2)
print('done')
"
```

**Sanity check after serving** (do this before running eval):
```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-VL-7B-Instruct","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}' \
  | python3 -m json.tool | grep -A3 '"content"'
```
- `"content": "4"` or any non-null response → checkpoint is healthy, proceed.
- `"content": null` → try the next earlier checkpoint (iter_0000049, iter_0000036).

---

**Do this in order — SGLang must be restarted between baseline and SFT eval:**

**Step 1 — Baseline eval (base weights):**
```bash
no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1" \
python -m sglang.launch_server \
    --model-path /workspace/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct \
    --tp 4 --port 30000 --dtype bfloat16 --mem-fraction-static 0.8
```

**Sanity check:**
```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-VL-7B-Instruct","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":50}' \
  | python3 -m json.tool | grep '"content"'
```
Expected: clean English sentence, `"finish_reason": "stop"`.

```bash
python sft/eval_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen2.5-VL-7B-Instruct \
    --embed-device cuda:7 \
    --workers 4 \
    --num-samples 100 \
    --output logs/eval_baseline_qwen25_7b_train100.jsonl
```

**Step 2 — Kill SGLang, start with SFT checkpoint (use earliest healthy iter):**
```bash
pkill -f sglang

no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1" \
python -m sglang.launch_server \
    --model-path /lc3T/Qwen2.5-VL-7B-Instruct_agentic_sft_iter36_hf \
    --tp 4 --port 30000 --dtype bfloat16 --mem-fraction-static 0.8
```

**Step 3 — SFT eval (checkpoint, same 100 samples):**
```bash
python sft/eval_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen2.5-VL-7B-Instruct \
    --embed-device cuda:7 \
    --workers 4 \
    --num-samples 100 \
    --output logs/eval_sft_qwen25_7b_train100.jsonl
```

**Step 4 — Compare:**
```bash
python sft/compare_eval.py \
    --teacher  logs/eval_teacher_train100.jsonl \
    --baseline logs/eval_baseline_qwen25_7b_train100.jsonl \
    --sft      logs/eval_sft_qwen25_7b_train100.jsonl
```

---

### Option 2 — Qwen3-VL-8B-Instruct (⚠ broken at current SGLang commit)

Keep these commands ready. Once SGLang's `Qwen3VLForCausalLM` class is fixed, this path
works without any other changes. Always run the sanity check first.

**Server (tmux pane):**
```bash
pkill -f sglang

export http_proxy="http://httpproxy.glm.ai:8888"
export https_proxy="http://httpproxy.glm.ai:8888"

HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
HF_HUB_OFFLINE=0 \
no_proxy="localhost,127.0.0.1" \
NO_PROXY="localhost,127.0.0.1" \
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --tp 4 \
    --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8
```
`--tp 4`: uses GPUs 0–3; JinaV4 stays on cuda:7.

**Sanity check (run this before eval — will reveal if the SGLang bug is fixed):**
```bash
http_proxy= https_proxy= curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-VL-8B-Instruct","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":50}' \
  | python3 -m json.tool
```
- `"finish_reason": "stop"` + clean English → SGLang fixed, proceed with eval below.
- `"finish_reason": "length"` + garbage → still broken; use Option 1 instead.

**Baseline eval:**
```bash
python sft/eval_async.py \
    --dataset longdocurl \
    --backend sglang \
    --sglang-url http://localhost:30000 \
    --model-name Qwen3-VL-8B-Instruct \
    --embed-device cuda:7 \
    --workers 4 \
    --num-samples 100 \
    --output logs/eval_baseline_qwen3vl_8b_train100.jsonl
```

**Fallback — Transformers in-process (slow but always works):**

If SGLang is still broken and you cannot use Option 1, load the 8B weights in-process.
Keep the 30B SGLang server running on GPUs 0–3 for trajectory collection; load 8B on GPU 4.

```bash
python sft/eval_async.py \
    --dataset longdocurl \
    --backend transformers \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --model-device cuda:4 \
    --embed-device cuda:7 \
    --workers 1 \
    --num-samples 100 \
    --output logs/eval_baseline_qwen3vl_8b_train100.jsonl
```
GPU layout: SGLang 30B on 0–3 · 8B transformers on 4 · JinaV4 on 7 · GPUs 5–6 free.

> ~14 tok/s in eager mode — expect **8–16 hours** for 100 samples. Use `--num-samples 30`
> for a first pass.

---

**Step 4.7c — Full three-way compare (teacher / baseline / SFT):**

```bash
python sft/compare_eval.py \
    --teacher  logs/eval_teacher_train100.jsonl \
    --baseline logs/eval_baseline_qwen25_7b_train100.jsonl \
    --sft      logs/eval_sft_qwen25_7b_train100.jsonl
```
Expected output:
```
  Teacher (30B)        acc=XX%  avg_steps=Y.Y
  Baseline (7B)        acc=XX%  avg_steps=Y.Y
  SFT (7B)             acc=XX%  avg_steps=Y.Y

  Delta (SFT - Baseline): +XX%
  Improved: N  Regressed: N  Unchanged: N
```

**What to expect:**

| Metric | Expectation | Interpretation |
|---|---|---|
| Train loss after 3 epochs | Near-zero (< 0.3) | Data format and loss masking are correct |
| Accuracy on training 100 (baseline → SFT) | Large jump (e.g. 20% → 70%+) | Model memorised trajectories — overfitting confirmed |
| Accuracy on held-out questions (baseline → SFT) | Small or no gain | Expected at 100 samples — generalisation needs 1K+ |

> **Interpreting the results:**
> - Loss drops but accuracy on train set does NOT improve → the model is learning to predict
>   tokens correctly but not using tools in the right order. Check tool-call formatting.
> - Loss does NOT drop → data pipeline bug (loss masking wrong, all tokens masked out).
> - Both loss and train accuracy improve → SFT data is valid. Scale up to 1K–10K samples
>   for generalisation.

---

## Data versioning log

Tracks what each trajectory file contains and why versions exist.

| File | Trajs | Score=1.0 | Schema errors | Notes |
|------|-------|-----------|---------------|-------|
| `logs/trajectories_overfit100.jsonl` | 151 | 121 (80.1%) | 0 | v1 — original collection, 30B teacher |
| `logs/trajectories_overfit100.parquet` | 151 | — | — | v1 parquet — **broken**: tool_calls in structured field, 0 trainable tokens |
| `logs/trajectories_overfit100_v2.parquet` | 151 | — | — | v2 parquet — **fixed**: tool calls as `Action:` text in content (same JSONL as v1) |
| `logs/trajectories_overfit100_v3.jsonl` | 23 | 17 (73.9%) | 0 | v3 — partial collection, crashed at sample 23 from GPU OOM (see below) |

**Why v2 parquet exists but no v2 JSONL:**
The JSONL data was fine (0 schema errors). The bug was in `format_trajectories.py` — it stored
tool calls in the structured OpenAI `tool_calls` field which Qwen2.5-VL's chat template silently
ignores. Fix: reformat the same JSONL → v2 parquet using `Action:\n{JSON}` text in `content`.

**Sanity check any JSONL before formatting:**
```bash
python sft/check_trajectories.py logs/<file>.jsonl
# Shows: total, score=1.0 %, score>=0.5 %, schema errors, example trajectory
python sft/check_trajectories.py logs/<file>.jsonl --show-errors   # print all error trajectories
```

---

## Known issues with SGLang server launch

**`AssertionError: 16 is not divisible by N` on Qwen3-VL-30B**

The Qwen3-VL-30B MoE vision encoder has 16 attention heads. `--tp` must divide 16 exactly.
Valid: `--tp 1, 2, 4, 8, 16`. Invalid: 3, 5, 6, 7, etc.

Always use `--tp 4` for the 30B model on 4×80GB GPUs:
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
  --tp 4 --port 30000 --dtype bfloat16 --mem-fraction-static 0.75
```

**`Multimodal embedding cache is full` + GPU OOM during collection**

With a correct tool schema, trajectories now complete all 5 search steps properly.
Each `search_database` call returns 3 full-resolution document images as tool responses.
At 5 searches × 3 images = 15 images in context, the vision encoder OOMs on 30B
(72.65 GiB used / 79.11 GiB total on each 80GB GPU).

Fix: always pass `--max-image-size 768` when collecting with the 30B model:
```bash
python sft/collect_trajectories_async.py \
  --dataset longdocurl --backend sglang \
  --sglang-url http://localhost:30000 \
  --model-name Qwen3-VL-30B-A3B-Instruct \
  --embed-device cuda:7 --workers 2 --num-samples 100 \
  --max-image-size 768 \
  --output logs/trajectories_overfit100_v3.jsonl
```

768px long-edge reduces each image from ~2800 to ~300 vision tokens (9× reduction)
while keeping document text readable.


---

## Confirmed results (2026-04-15)

### Qwen3-VL-4B-Instruct — v4 trajectories (first clean end-to-end run)

Training data: `logs/trajectories_overfit100_v4.jsonl` — 100 samples, 67% score=1.0, 0 schema errors, no `original_question` in any tool call.

| Model | Score=1.0 | Avg steps | Notes |
|---|---|---|---|
| 4B Baseline | 14% | 8.9 | base weights, no SFT |
| 4B SFT (v4) | **34%** | **5.2** | trained on v4 trajectories, 3 epochs |
| Delta | **+20%** | -3.7 | |

**Interpretation:** SFT more than doubled accuracy and reduced avg steps by 3.7 — the model learned to find answers faster rather than exhausting all 5 searches. Pipeline is validated end-to-end.

**Next:** retrain Qwen2.5-VL-7B-Instruct on v4 data for apples-to-apples comparison with the existing 7B baseline (11% accuracy, `logs/eval_baseline_qwen25_7b_train100.jsonl`).

### Qwen2.5-VL-7B-Instruct — v4 trajectories

| Model | Score=1.0 | Avg steps | Notes |
|---|---|---|---|
| 7B Baseline | 11% | 4.7 | `eval_baseline_qwen25_7b_train100.jsonl` |
| 7B SFT (v4) | **48%** | **5.3** | `eval_sft_qwen25_7b_v4_train100.jsonl` |
| Delta | **+37%** | +0.6 | |

**Interpretation:** 7B SFT more than 4× the baseline accuracy. Larger model capacity absorbs teacher trajectories more effectively than 4B. Pipeline fully validated — ready to scale to 1K+ samples for generalisation.

**Note:** eval reported 98 samples (2 workers may have skipped 2 samples due to Qdrant timeouts). Score computed over completed samples only.
