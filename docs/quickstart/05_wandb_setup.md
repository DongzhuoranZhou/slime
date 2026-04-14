# W&B Setup on the Zhipu Cluster

> Covers how to connect slime training runs to the internal zhipu W&B instance at `https://wandb.glm.ai`. Includes both CLI-arg and env-var methods, the proxy situation, run naming, and the most common failure modes.

---

## The Two W&B Instances

| Instance | URL | When to use |
|----------|-----|-------------|
| **Zhipu internal** | `https://wandb.glm.ai` | Default for all training on the zhipu cluster. No external internet needed. |
| **Public W&B** | `https://api.wandb.ai` (default) | Only if you specifically want public W&B. Requires external internet access via proxy. |

For all day-to-day training on the cluster, use the zhipu internal instance.

---

## Credentials

| Field | Value |
|-------|-------|
| API key | `local-643dfe5b804d35d8ab5aeba347f30f4ee0c6ca6f` |
| Host URL | `https://wandb.glm.ai` |

The key prefix `local-` identifies it as a self-hosted W&B key (as opposed to a cloud W&B key which starts with a random string). The host URL must always be passed alongside the key — without it, the `local-` prefixed key is rejected by the public W&B backend.

---

## Two Ways to Pass Credentials

### Method A: CLI arguments (shell scripts)

Used in `.sh` launch scripts that call `train.py` or `train_async.py` directly via `ray job submit`:

```bash
export WANDB_KEY="local-643dfe5b804d35d8ab5aeba347f30f4ee0c6ca6f"

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-my-experiment
   --wandb-group my-model-run
   --wandb-key ${WANDB_KEY}
   --wandb-host https://wandb.glm.ai
)
```

These map directly to `wandb.login(key=args.wandb_key, host=args.wandb_host)` in [`slime/utils/wandb_utils.py`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/slime/utils/wandb_utils.py).

Reference: [`scripts/run-glm4-9B_original_with_zhipu_local_wandb_no_need_for_proxy.sh`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/scripts/run-glm4-9B_original_with_zhipu_local_wandb_no_need_for_proxy.sh)

### Method B: Environment variables (Python launch scripts)

Used in `.py` launch scripts (e.g., `run_geo3k_vlm_multi_turn.py`, `run_geo3k_vlm.sh`) that read `WANDB_API_KEY` directly:

```bash
export WANDB_API_KEY="local-643dfe5b804d35d8ab5aeba347f30f4ee0c6ca6f"
export WANDB_BASE_URL="https://wandb.glm.ai"
```

The Python launch script reads `WANDB_API_KEY` via `os.environ.get("WANDB_API_KEY")` and passes it to `--wandb-key`. `WANDB_BASE_URL` is picked up by the `wandb` library itself before `wandb.login()` is called.

Reference: [`examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py:45-53`](https://github.com/DongzhuoranZhou/slime/blob/dev_main/examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py)

### Which method to use?

| Launch mechanism | Method to use |
|-----------------|---------------|
| `.sh` script with hardcoded `WANDB_ARGS=()` | **Method A** (CLI args) |
| `.py` script that reads `os.environ.get("WANDB_API_KEY")` | **Method B** (env vars) |
| `run_geo3k_vlm.sh` / `run_geo3k_vlm_sft.sh` | **Method B** — sets `${WANDB_API_KEY}` inside the bash array |

When in doubt: set both. The CLI `--wandb-key` / `--wandb-host` args take priority inside slime; `WANDB_BASE_URL` is a fallback for the wandb library itself.

---

## Proxy Situation

The zhipu internal W&B (`wandb.glm.ai`) is on the internal network — **no proxy is needed** to reach it.

The external proxy (`http://httpproxy.glm.ai:8888`) is only needed for HuggingFace downloads and other external internet access. It must be **unset before launching training** because it blocks inter-node Ray/NCCL communication.

Correct order:
```bash
# 1. Set proxy for downloads
export http_proxy="http://httpproxy.glm.ai:8888"
export https_proxy="http://httpproxy.glm.ai:8888"

# 2. Download model / dataset
hf download Qwen/Qwen3-VL-8B-Instruct ...

# 3. Unset proxy before training (required for Ray and NCCL)
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY

# 4. Launch training (wandb.glm.ai is reachable without proxy)
./examples/geo3k_vlm/run_geo3k_vlm.sh
```

> Do NOT add `wandb.glm.ai` to `no_proxy` — it's not going through the proxy at all.

---

## W&B CLI Arguments Reference

| Argument | Default | Notes |
|----------|---------|-------|
| `--use-wandb` | off | Flag. Must be present to enable any W&B logging. If absent, only system metrics are captured by the wandb agent. |
| `--wandb-key` | `None` | API key. Required for online mode. |
| `--wandb-host` | `None` | Host URL. Required when using zhipu internal W&B. |
| `--wandb-project` | `None` | Project name. Runs are grouped under this in the W&B UI. |
| `--wandb-group` | `None` | Group name within the project. Becomes the base of the run name. |
| `--wandb-team` | `None` | W&B entity/team. Usually not needed for personal runs. |
| `--wandb-dir` | `./wandb` | Local directory for W&B run files. Redirect to `/lc3T/wandb/` if `/workspace` is tight on space. |
| `--disable-wandb-random-suffix` | off | By default, a random 6-char suffix is appended to the group/run name to avoid collisions. Add this flag for reproducible, collision-free run names (e.g., in scripts that may be re-run). |
| `--wandb-always-use-train-step` | off | If set, all metrics use `train/step` as their x-axis instead of `rollout/step`. |

---

## Run Naming

By default, the run name is: `{wandb_group}_{random_6_chars}-RANK_{rank}`

With `--disable-wandb-random-suffix`, it becomes: `{wandb_group}` (no suffix, no rank).

In slime, one run is created **per training rank** (not per job). With TP=4 and 8 GPUs, you may see multiple runs per training job. They are grouped together under the same `--wandb-group` in the W&B UI.

**Recommended naming pattern:**
```bash
--wandb-project slime-geo3k-vlm-rl        # what you're training on
--wandb-group qwen3-vl-8b-instruct        # model + variant
```

---

## Complete Example

### Shell script (Method A):

```bash
export WANDB_KEY="local-643dfe5b804d35d8ab5aeba347f30f4ee0c6ca6f"

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-geo3k-vlm-rl
   --wandb-group qwen3-vl-8b-run1
   --wandb-key ${WANDB_KEY}
   --wandb-host https://wandb.glm.ai
   --disable-wandb-random-suffix   # optional: cleaner run names
)

ray job submit ... -- python3 train_async.py ... ${WANDB_ARGS[@]}
```

### Python launch script (Method B):

```bash
export WANDB_API_KEY="local-643dfe5b804d35d8ab5aeba347f30f4ee0c6ca6f"
export WANDB_BASE_URL="https://wandb.glm.ai"

python examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py
```

---

## Common Failure Modes

### "Only system metrics in W&B, no training/rollout/eval metrics"

**Cause:** `--use-wandb` flag was not passed, or `WANDB_API_KEY` env var was empty when the Python script checked it.

**Fix:** Ensure `WANDB_API_KEY` is exported *before* running the script, not inside it:
```bash
export WANDB_API_KEY="local-643dfe5b804d35d8ab5aeba347f30f4ee0c6ca6f"
export WANDB_BASE_URL="https://wandb.glm.ai"
python examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py
```

### "wandb: ERROR: You are not authenticated"

**Cause:** `--wandb-host` is missing. The `local-` key is only valid on `wandb.glm.ai`; without the host, wandb tries to authenticate against the public `api.wandb.ai` and rejects the key.

**Fix:** Always pass `--wandb-host https://wandb.glm.ai` alongside `--wandb-key`.

### "Metrics appear at wrong x-axis positions in W&B"

**Cause:** Using the raw "Step" axis. Slime makes ~4–5 separate `wandb.log()` calls per rollout iteration, so the raw step counter is not a meaningful training index.

**Fix:** In the W&B UI, click the axis selector and switch to `rollout/step` (for rollout/eval metrics) or `train/step` (for loss/KL). See the [Overview glossary](./00_overview.md#glossary).

### "Run name collision — two runs with the same name"

**Cause:** Rerunning a script with `--disable-wandb-random-suffix` set. Without the random suffix, the group name is reused.

**Fix:** Either remove `--disable-wandb-random-suffix`, or manually change `--wandb-group` before rerunning.

---

## Offline Mode (no network during training)

If the training nodes have no network access at all:

```bash
WANDB_ARGS=(
   --use-wandb
   --wandb-project my-project
   --wandb-group my-run
   # No --wandb-key or --wandb-host
)
export WANDB_MODE=offline
```

Logs are written locally to `--wandb-dir` (default: `./wandb/`). Sync them later with:
```bash
wandb sync ./wandb/run-*/
```
